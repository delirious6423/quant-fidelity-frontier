"""Read-only HTTP collection into a portable ledger and isolated shadow model root.

No git, workflow, scheduler or publication interface exists here. The collector
owns its ledger exclusively; it checkpoints each card before attempting its tree.
A failed refresh leaves the old snapshot untouched and a blocking freshness marker.
"""
from __future__ import annotations

from contextlib import nullcontext
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time

try:
    from . import qcl_fetch as fetch
except ImportError:
    import qcl_fetch as fetch

CODE = Path(__file__).resolve().parents[1]
SHA = re.compile(r"[0-9a-f]{40,64}\Z")


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def local(directory, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('ledger paths must be relative')
    result = (directory / name).resolve()
    if not result.is_relative_to(directory.resolve()):
        raise ValueError('ledger path escapes its directory')
    return result


def valid_bytes(directory, rec):
    try:
        raw = local(directory, rec['file']).read_bytes()
        return (rec['readme_status'] in (200, 404) and len(raw) == rec['bytes']
                and hashlib.sha256(raw).hexdigest() == rec['sha256']
                and (rec['readme_status'] != 404 or raw == b''))
    except (KeyError, OSError, ValueError, TypeError):
        return False


def valid_tree(directory, rec):
    try:
        raw = local(directory, rec['tree_file']).read_bytes()
        return (rec['tree_complete'] is True and hashlib.sha256(raw).hexdigest() == rec['tree_sha256']
                and isinstance(json.loads(raw), list))
    except (KeyError, OSError, ValueError, TypeError):
        return False


def checked_meta(metadata, rid):
    if not fetch.valid_model_id(rid) or not isinstance(metadata, dict) or not SHA.fullmatch(metadata.get('sha') or ''):
        raise ValueError('invalid repository identity or immutable source revision')
    return {**metadata, 'id': rid}


def catalogue(base, model, token, stop):
    """Reuse HTTP and pagination policy; reject cycles, malformed pages and IDs."""
    import urllib.parse
    url = f'{base}/api/models?' + urllib.parse.urlencode({'filter': f'base_model:quantized:{model}', 'limit': 1000, 'full': 'true'})
    url += '&expand[]=sha&expand[]=createdAt&expand[]=lastModified&expand[]=downloads&expand[]=tags'
    seen, records = set(), {}
    while url:
        if url in seen:
            raise ValueError('catalogue pagination cycle')
        seen.add(url)
        status, body, headers = fetch.get(url, token, stop=stop)
        if status != 200:
            raise FetchFailure(status)
        page = json.loads(body)
        if not isinstance(page, list):
            raise ValueError('catalogue response is not a list')
        for item in page:
            if not isinstance(item, dict):
                raise ValueError('catalogue item is not an object')
            item = checked_meta(item, item.get('id'))
            rid = item['id']
            if rid in records and records[rid] != item:
                raise ValueError('inconsistent duplicate catalogue repository')
            records[rid] = item
        url = fetch.next_page(url, headers)
    return records


class FetchFailure(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f'HTTP {status}')


def failure(error):
    if isinstance(error, fetch.RateLimited):
        return 'rate_limited'
    if isinstance(error, fetch.FetchStopped):
        return 'rate_limited_batch_stopped'
    if isinstance(error, FetchFailure):
        return {401: 'access_denied', 403: 'access_denied', 404: 'missing_repository'}.get(error.status, f'http_{error.status}')
    return 'invalid_or_unavailable_source'  # never log HTTP bodies, tokens or local paths


def collect(model_root, ledger_dir, model_id, include_repos=(), workers=8,
            base='https://huggingface.co', token=None, install_lock=None):
    root, directory = Path(model_root).resolve(), Path(ledger_dir).resolve()
    if root.is_relative_to(CODE / 'models') or directory.is_relative_to(CODE / 'models'):
        raise ValueError('collection accepts isolated shadow roots only')
    if not fetch.valid_model_id(model_id) or not 1 <= workers <= 32:
        raise ValueError('invalid model or worker count')
    # A copied shadow must not contain links back into authoritative data.
    if any(path.is_symlink() for anchor in (root, directory) if anchor.exists() for path in anchor.rglob('*')):
        raise ValueError('shadow and ledger must contain no symlinks')
    include_repos = tuple(include_repos)
    if any(not fetch.valid_model_id(rid) for rid in include_repos):
        raise ValueError('invalid explicit repository inclusion')
    config = read(root / 'model.json', {})
    if config.get('model') != model_id:
        raise ValueError('shadow model config identity mismatch')
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / 'collector.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _collect(root, directory, model_id, include_repos, workers, base, token, install_lock)


def _collect(root, directory, model_id, include_repos, workers, base, token, install_lock):
    marker = root / 'audit/runner_fetch.json'
    idx = read(directory / 'index.json', {'model': model_id, 'repos': {}})
    if not isinstance(idx, dict) or idx.get('model') != model_id or not isinstance(idx.get('repos'), dict):
        raise ValueError('ledger identity mismatch')
    if idx.get('base', {}).get('repo', model_id) != model_id:
        raise ValueError('ledger base identity mismatch')
    if any(not isinstance(rec, dict) or rec.get('repo') != rid for rid, rec in idx['repos'].items()):
        raise ValueError('ledger repository identity mismatch')
    with install_lock() if install_lock else nullcontext():
        atomic(marker, {'status': 'blocked', 'reason': 'refresh_incomplete'})
    limited = read(directory / 'rate_limit.json')
    if limited:
        until = limited.get('resume_after', float('inf'))
        if time.time() < until:
            return {'status': 'blocked', 'changed': [], 'unresolved': [{'repo': model_id, 'status': 'rate_limit_cooldown'}]}
    stop = threading.Event()
    guard = threading.Lock()
    unresolved = []
    previous = read(root / 'snapshots/index.json', {'repos': {}})
    if previous.get('model', model_id) != model_id:
        raise ValueError('snapshot model identity mismatch')

    def checkpoint():
        atomic(directory / 'index.json', idx)

    def metadata(rid):
        status, body, _ = fetch.get(f'{base}/api/models/{rid}', token, stop=stop)
        if status != 200:
            raise FetchFailure(status)
        return checked_meta(json.loads(body), rid)

    def remember_error(rid, error):
        unresolved.append({'repo': rid, 'status': failure(error)})
        if isinstance(error, fetch.RateLimited):
            record = error.record
            delay = record.get('retry_after_seconds')
            atomic(directory / 'rate_limit.json', {'retry_after_seconds': delay,
                   'resume_after': time.time() + (delay if delay is not None else 60)})

    try:
        cat = catalogue(base, model_id, token, stop)
        # Existing captured coverage remains explicit even if a tag disappears.
        prior_cat = read(directory / 'catalogue.json', [])
        if not isinstance(prior_cat, list):
            raise ValueError('saved catalogue is invalid')
        retained = {item['id'] for item in prior_cat}
        required = set(include_repos) | set(idx['repos']) | set(previous.get('repos', {})) | retained
        for rid in sorted(required - cat.keys()):
            cat[rid] = {**metadata(rid), 'catalogue_source': 'explicit-or-retained'}
        atomic(directory / 'catalogue.json', list(cat.values()))
        base_meta = metadata(model_id)
    except Exception as error:
        remember_error(model_id, error)
        atomic(marker, {'status': 'blocked', 'unresolved': unresolved})
        return {'status': 'blocked', 'changed': [], 'unresolved': unresolved}

    def one(rid, meta, official=False):
        sha = meta['sha']
        old = idx.get('base', {}) if official else idx['repos'].get(rid, {})
        rec = dict(old) if old.get('sha') == sha else {}
        name = ('_base__' if official else '') + rid.replace('/', '__') + '@' + sha
        if not valid_bytes(directory, rec):
            status, body, _ = fetch.get(f'{base}/{rid}/raw/{sha}/README.md', token, stop=stop)
            if status not in (200, 404):
                raise FetchFailure(status)
            raw = body if status == 200 else b''
            rel = 'cards/' + name + '.md'
            atomic(directory / rel, raw)
            rec = {'repo': rid, 'sha': sha, 'file': rel, 'sha256': hashlib.sha256(raw).hexdigest(),
                   'bytes': len(raw), 'readme_status': status}
        rec.update({k: meta.get(k) for k in ('createdAt', 'lastModified', 'downloads', 'tags', 'catalogue_source')})
        with guard:
            if official:
                idx['base'] = dict(rec)
            else:
                idx['repos'][rid] = dict(rec)
            checkpoint()  # retain card even when its tree fetch fails or process dies
        if official:
            return
        if not valid_tree(directory, rec):
            files, pages = fetch.fetch_tree(base, rid, sha, token, stop)
            raw = json.dumps(files, sort_keys=True).encode()
            rel = 'trees/' + name + '.json'
            atomic(directory / rel, raw)
            sizes = [f.get('size') for f in files if f.get('type') == 'file']
            complete = all(type(n) is int and n >= 0 for n in sizes)
            rec.update(tree_file=rel, tree_sha256=hashlib.sha256(raw).hexdigest(), tree_complete=True,
                       tree_pages=pages, total_bytes_complete=complete, total_bytes=sum(sizes) if complete else None)
            with guard:
                idx['repos'][rid] = rec
                checkpoint()

    try:
        one(model_id, base_meta, official=True)
    except Exception as error:
        remember_error(model_id, error)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, rid, meta): rid for rid, meta in cat.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                remember_error(futures[future], error)
    if unresolved:
        atomic(marker, {'status': 'blocked', 'unresolved': unresolved})
        return {'status': 'blocked', 'changed': [], 'unresolved': unresolved}
    changed = sorted(rid for rid, rec in idx['repos'].items()
                     if previous.get('repos', {}).get(rid, {}).get('sha256') != rec['sha256'])
    with install_lock() if install_lock else nullcontext():
        _install(root, directory, idx, list(cat.values()), previous)
        atomic(marker, {'status': 'complete', 'changed': changed, 'unresolved': []})
    (directory / 'rate_limit.json').unlink(missing_ok=True)
    return {'status': 'complete', 'changed': changed, 'unresolved': [], 'repos': len(cat)}


def _install(root, directory, idx, cat, previous):
    """Validate using the existing import contract in an otherwise empty sandbox.

    Old connector shards cannot silently win over a new direct-fetch checkpoint.
    Files are installed first, then the snapshot pointer atomically switches.
    """
    with tempfile.TemporaryDirectory(prefix='import-', dir=directory) as name:
        stage = Path(name)
        atomic(stage / 'model.json', read(root / 'model.json'))
        atomic(stage / 'snapshots/index.json', previous)
        export = json.loads(json.dumps(idx))
        records = [export['base'], *export['repos'].values()]
        for rec in records:
            if not valid_bytes(directory, rec) or (rec is not export['base'] and not valid_tree(directory, rec)):
                raise ValueError('checkpoint failed validation before import')
            for key in ('file', 'tree_file'):
                if key in rec:
                    rel = rec[key]
                    raw = local(directory, rel).read_bytes()
                    atomic(stage / 'cache' / rel, raw)
                    atomic(root / 'cache/runner' / rel, raw)
        atomic(stage / 'cache/index.json', export)
        atomic(stage / 'cache/catalogue.json', cat)
        result = subprocess.run([sys.executable, str(CODE / 'pipeline/qcl_newmodel.py'), str(stage)],
                                capture_output=True, text=True)
        if result.returncode:
            raise ValueError('existing index importer rejected complete ledger')
        snapshot = read(stage / 'snapshots/index.json')
        for rec in [snapshot['base'], *snapshot['repos'].values()]:
            for key in ('file', 'tree_file'):
                if rec.get(key):
                    rec[key] = 'cache/runner/' + str(Path(rec[key]).relative_to('cache'))
        for stub in ('snapshots/hf_sizes_summary.json', 'snapshots/hf_branch_sizes.json', 'sets.py'):
            if not (root / stub).exists():
                atomic(root / stub, (stage / stub).read_bytes())
        atomic(root / 'cache/not_fetched.json', read(stage / 'cache/not_fetched.json'))
        atomic(root / 'snapshots/index.json', snapshot)
