#!/usr/bin/env python3
"""Scripted, local-only shadow pipeline. See docs/SCRIPTED_RUNNER.md.

No provider credentials are needed: requests are exported as complete JSON bundles,
responses may be imported independently. No command can publish or update Git.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil

from runner_state import State, digest
from runner_workspace import Workspace, CODE, read, write, write_text, install, restore
import runner_model as boundary

VERSION_FILES = ('qcl_map.py', 'autoextract.py', 'mapping_provenance.py', 'qcl_watch.py',
                 'runner_model.py', 'qcl_fmt.py', 'qcl_mapset.py', 'qcl_tiebreak.py')


def version():
    return digest({name: hashlib.sha256((CODE / 'pipeline' / name).read_bytes()).hexdigest() for name in VERSION_FILES})


_INDEX = {}


def index(root):
    """snapshots/index.json, parsed again only when its bytes change.

    Validation consults the index once per task and per claim. The file is
    still read every time (so an atomic replacement by collection is seen at
    once), but a byte-identical index is not parsed again. Timestamps are not
    trusted: two rewrites inside one clock tick could share size and mtime.
    """
    path = root / 'snapshots/index.json'
    raw = path.read_bytes()
    cached = _INDEX.get(path)
    if cached is None or cached[0] != raw:
        cached = _INDEX[path] = (raw, json.loads(raw))
    return cached[1]


def source(root, repo):
    meta = index(root)['repos'][repo]
    path = (root / meta['file']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('source path escapes shadow model')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta['sha256']:
        raise ValueError('source bytes do not match index')
    return {**meta, 'repo': repo}, raw


def records(root):
    """Read generated literal records without executing claim/source code."""
    path = root / 'verified/part8_machine.py'
    if not path.exists():
        return []
    tree = ast.parse(path.read_text())
    return [ast.literal_eval(node.value.keywords[0].value) for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'repo']


def review_bundles(root):
    model = read(root / 'model.json')['model']
    bundles = []
    for record in records(root):
        meta, raw = source(root, record['repo'])
        if record.get('source_sha256') != meta['sha256']:
            continue
        for group in ('variants', 'ladder', 'cross'):
            for entry in record.get(group) or []:
                role = 'own' if group == 'variants' else ('quoted' if group == 'cross' and entry.get('measured') is False else group)
                values = dict(entry.get('m') or {})
                if entry.get('size') is not None:
                    n = entry['size']
                    values['size'] = f"{int(n) if isinstance(n, (int, float)) and float(n).is_integer() else n} {entry.get('unit') or ''}".strip()
                for key, raw_value in values.items():
                    claim = {'repo': record['repo'], 'kind': role, 'entry': str(entry.get('name') or entry.get('variant')),
                             'key': key, 'raw': str(raw_value), 'lines': [n for n in [entry.get('L'), *(entry.get('L_extra') or [])] if n],
                             'entry_evidence': entry, 'methodology': record.get('methodology'),
                             'weights_modified': record.get('weights_modified'), 'source_sha256': record.get('source_sha256')}
                    bundles.append(boundary.prepare_review(model, meta, raw, claim, mapping_version=version(), extractor_version=version()))
    return bundles


def valid_current(w, task):
    payload = task['payload']
    root = w.model(task['model'])
    meta, _ = source(root, payload['source']['repo'])
    if meta['sha256'] != payload['source']['sha256']:
        raise ValueError('result source is no longer current')
    v = payload.get('version') or payload.get('mapping_version')
    if v != version():
        raise ValueError('result task version is no longer current')


def content_identity(bundle):
    """Exact evidence identity across byte-identical upstream revision changes.

    The full revision remains in the immutable request/response. Collection
    records the newly observed revision; only that non-content field is omitted
    from reuse comparison, never qualifiers, interpretation, lines or provenance.
    """
    copy = {k: v for k, v in bundle.items() if k != 'fingerprint'}
    copy['source'] = {k: v for k, v in bundle['source'].items() if k != 'revision'}
    return digest(copy)


def materialize(w, state, slug):
    root = w.model(slug)
    for task in state.tasks(model=slug, states=('succeeded',)):
        valid_current(w, task)
        payload, result = task['payload'], task['result']
        if payload['kind'] == 'mapping':
            mapping = boundary.validate_mapping(payload, result)
            name = mapping['repo'].replace('/', '__')
            out = root / 'maps' / payload['run']
            out.mkdir(parents=True, exist_ok=True)
            install(out / (name + '.map.json'), mapping)
            install(out / (name + '.cands.json'), {k: mapping[k] for k in ('repo', 'source_sha256', 'candidates_sha256')} | {'candidates': payload['candidates']})
        elif result['verdict'] == 'wrong':
            hp = root / 'audit/holds.json'
            holds = read(hp) if hp.exists() else []
            claim = payload['claim']
            if not any(h['repo'] == claim['repo'] and h['entry'] == claim['entry'] for h in holds):
                holds.append({'repo': claim['repo'], 'entry': claim['entry'], 'reason': result['reason']})
                write(hp, holds)


def enqueue_queues(w, state, slug, work, runs):
    root = w.model(slug)
    model = read(root / 'model.json')['model']
    for queue in sorted((work / 'pending').glob('*.txt')):
        run = queue.name[0]
        if run not in runs:
            continue
        for name in queue.read_text().splitlines():
            cpath = root / 'maps' / ('C' if run == 'C' else 'cands') / (name + '.cands.json')
            task = read(cpath)
            task['prompt'] = (work / 'tasks' / (name + '.prompt.txt')).read_text()
            meta, raw = source(root, task['repo'])
            bundle = boundary.prepare_mapping(model, meta, raw, task, run, version())
            slot = f'{slug}:map:{run}:{task["repo"]}'
            previous = state.current(slot)
            if previous and content_identity(previous['payload']) == content_identity(bundle):
                continue
            state.enqueue(slot, slug, 'map-' + run, meta['sha256'] + ':' + version(), bundle)


def model_signature(root, state, slug):
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob('*') if p.is_file()
             and not any(x in ('site', 'site_dev', '__pycache__') or x.startswith('.gatecache') for x in p.relative_to(root).parts)}
    tasks = [(t['id'], t['result']) for t in state.tasks(model=slug, states=('succeeded',), with_payload=False)
             if t['stage'].startswith('map-') or t['result'].get('verdict') == 'wrong']
    return digest([version(), files, tasks])


def reconcile(w, state):
    """One writer prepares, installs validated outputs, merges and stages review."""
    with w.lock():
        for slug in w.config['models']:
            root = w.model(slug)
            freshness = root / 'audit/runner_fetch.json'
            if freshness.exists() and read(freshness).get('status') != 'complete':
                continue  # this model is blocked; independent models can continue
            receipt = w.path / ('stage-' + slug + '.json')
            if receipt.exists() and read(receipt).get('signature') == model_signature(root, state, slug):
                continue
            baseline = w.path / ('baseline-' + slug + '.json')
            if not baseline.exists():
                write(baseline, [content_identity(b) for b in review_bundles(root)])
            # Invalidate before installing, including results from a delayed worker.
            for task in state.tasks(model=slug):
                try:
                    valid_current(w, task)
                except (ValueError, KeyError, FileNotFoundError):
                    payload = task['payload']
                    if payload['kind'] == 'mapping':
                        # Only this journal's installed outputs are owned here.
                        # Source-only mapped pins must not conceal a version bump.
                        name = payload['source']['repo'].replace('/', '__')
                        for suffix in ('.map.json', '.cands.json'):
                            (root / 'maps' / payload['run'] / (name + suffix)).unlink(missing_ok=True)
                        pins = root / 'maps' / ('mapped_c.json' if payload['run'] == 'C' else 'mapped.json')
                        if pins.exists():
                            doc = read(pins); doc.pop(payload['source']['repo'], None); write(pins, doc)
                    with state.connect() as db:
                        db.execute("UPDATE tasks SET active=0,state='blocked',token=NULL,reason='source/task invalidation' WHERE id=?", (task['id'],))
            materialize(w, state, slug)
            work = w.path / 'queues' / slug
            work.mkdir(parents=True, exist_ok=True)
            for queue in (work / 'pending').glob('*.txt'):
                queue.unlink()
            w.run(slug, 'autoextract.py', 'triage')
            w.run(slug, 'detect_copies.py')
            w.run(slug, 'qcl_mapset.py', '--record')
            w.run(slug, 'qcl_queue.py', work)
            with state.transaction():
                enqueue_queues(w, state, slug, work, 'AB')
            # Record deterministic empty-candidate maps too. Merge always withdraws stale sources.
            w.run(slug, 'qcl_mapset.py', '--record')
            cmaps = {task['payload']['source']['repo']: task['payload']['source']['sha256']
                     for task in state.tasks(model=slug, stage='map-C', states=('succeeded',))}
            cp = root / 'maps/mapped_c.json'
            write(cp, (read(cp) if cp.exists() else {}) | cmaps)
            w.run(slug, 'qcl_watch.py', 'merge')
            w.run(slug, 'gen_auto.py')
            # Existing historical skipped maps remain visible; never create new pinned skips.
            w.run(slug, 'qcl_tiebreak.py', 'queue', work, len(index(root)['repos']))
            with state.transaction():
                enqueue_queues(w, state, slug, work, 'C')
            write(receipt, {'signature': model_signature(root, state, slug)})
        fetch_blocked = any((w.model(slug) / 'audit/runner_fetch.json').exists() and read(w.model(slug) / 'audit/runner_fetch.json').get('status') != 'complete' for slug in w.config['models'])
        mapping_left = [t for t in state.tasks(with_payload=False) if t['stage'].startswith('map-') and t['state'] != 'succeeded']
        if not mapping_left and not fetch_blocked:
            all_bundles, prior = [], []
            for slug in w.config['models']:
                all_bundles.extend((slug, b) for b in review_bundles(w.model(slug)))
                prior.extend(read(w.path / ('baseline-' + slug + '.json')))
            # Freeze the same 20-item sample for an unchanged candidate, even across restarts.
            candidate = digest([version(), {slug: {
                'sources': {repo: meta['sha256'] for repo, meta in index(w.model(slug))['repos'].items()},
                'model': read(w.model(slug) / 'model.json'),
                'maps': {str(p.relative_to(w.model(slug))): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (w.model(slug) / 'maps').rglob('*') if p.is_file()}}
                for slug in w.config['models']}])
            sample_file = w.path / 'review-sample.json'
            sample = read(sample_file) if sample_file.exists() else None
            completed = {content_identity(t['payload']): t['id'] for t in state.tasks(stage='review', states=('succeeded',))}
            reusable = set(prior) | set(completed)
            identity = {b['fingerprint']: content_identity(b) for _, b in all_bundles}
            unchanged = [b['fingerprint'] for _, b in all_bundles if identity[b['fingerprint']] in reusable]
            write(w.path / 'review-reuse.json', [
                {'current_fingerprint': b['fingerprint'], 'reviewed_task': completed[identity[b['fingerprint']]],
                 'current_revision': b['source']['revision'], 'reason': 'identical complete evidence and interpretation'}
                for _, b in all_bundles if identity[b['fingerprint']] in completed])
            if not sample or sample['candidate'] != candidate or 'requests' not in sample:
                pick = boundary.sample_reviews([b for _, b in all_bundles], unchanged, w.config['input_commit'])
                sample = {'candidate': candidate, 'items': [b['fingerprint'] for b in pick],
                          'requests': [{'model': slug, 'bundle': b} for slug, b in all_bundles if b['fingerprint'] in {p['fingerprint'] for p in pick}]}
                write(sample_file, sample)
            for t in state.tasks(stage='review', with_payload=False):
                if t['state'] != 'succeeded' and t['generation'] not in sample['items']:
                    with state.connect() as db:
                        db.execute("UPDATE tasks SET active=0,state='blocked',token=NULL,reason='review sample invalidated' WHERE id=?", (t['id'],))
            with state.transaction():
                for request in sample['requests']:
                    slug, b = request['model'], request['bundle']
                    state.enqueue(f'{slug}:review:{b["fingerprint"]}', slug, 'review', b['fingerprint'], b)
        export_requests(w, state)
    return status(w, state)


def export_requests(w, state):
    out = w.path / 'requests'
    out.mkdir(exist_ok=True)
    active = {t['id'] for t in state.tasks(with_payload=False) if t['state'] != 'succeeded'}
    for p in out.glob('*.json'):
        if p.stem not in active:
            p.unlink()
    for task_id in active:
        # A task id digests its slot, generation and payload: an existing file is already exact.
        if not (out / (task_id + '.json')).exists():
            # The journal already holds the canonical encoding; write it, do not re-encode the whole card.
            write_text(out / (task_id + '.json'), '{"request":' + state.payload_text(task_id) + ',"task_id":"' + task_id + '"}\n')


def status(w, state):
    counts = state.counts()
    fetch_blockers = [slug for slug in w.config['models'] if (w.model(slug) / 'audit/runner_fetch.json').exists() and read(w.model(slug) / 'audit/runner_fetch.json').get('status') != 'complete']
    sample_blockers = []
    sample_file = w.path / 'review-sample.json'
    if sample_file.exists():
        sample = read(sample_file)
        completed = {t['generation'] for t in state.tasks(stage='review', states=('succeeded',), with_payload=False)}
        items = sample.get('items', [])
        saved = [r['bundle']['fingerprint'] for r in sample.get('requests', [])]
        if len(items) > 20 or len(set(items)) != len(items) or set(items) != set(saved):
            sample_blockers.append('malformed or incomplete durable sample')
        sample_blockers.extend(pin for pin in items if pin not in completed)
    else:
        sample_blockers.append('review sample not prepared')
    return {'mode': 'shadow', 'tasks': counts, 'fetch_blockers': fetch_blockers, 'sample_blockers': sample_blockers, 'publication': 'unavailable',
            'ready_for_release': False, 'review': 'requires exact-commit release verification separately',
            'unresolved': [{'id': t['id'], 'stage': t['stage'], 'reason': t['reason']} for t in state.tasks(with_payload=False) if t['state'] != 'succeeded']}


def complete(w, state, claim, response):
    with w.lock():
        task = state.get(claim['id'])
        if task is None or not task['active']:
            raise ValueError('unknown or inactive task')
        valid_current(w, task)
        validator = boundary.validate_mapping if task['payload']['kind'] == 'mapping' else boundary.validate_review
        result = validator(task['payload'], response)
        return state.complete(claim['id'], claim['token'], result)


def run_responses(w, state, responses, workers):
    """Offline/file adapter. Missing responses remain pending; no paid calls."""
    directory = Path(responses)
    # Claim only tasks with available files; no lease consumed merely by export.
    ids = [t['id'] for t in state.tasks(states=('pending', 'failed', 'running'), with_payload=False) if (directory / (t['id'] + '.json')).exists()]
    completed = 0
    # Concurrent file reads, serialized validated commits; no external job under lock.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while ids:
            batch = []
            # Claim every ready file fairly; missing response files do not consume leases.
            for _ in range(workers):
                claim = state.claim(limit=workers, eligible_ids=ids)
                if claim is None:
                    break
                p = directory / (claim['id'] + '.json')
                batch.append((claim, pool.submit(lambda path: boundary.parse_response(path.read_bytes()), p)))
                if claim['id'] in ids:
                    ids.remove(claim['id'])
            if not batch:
                break
            for claim, result in batch:
                try:
                    complete(w, state, claim, result.result())
                    completed += 1
                except (ValueError, TypeError, KeyError) as error:
                    state.fail(claim['id'], claim['token'], str(error), blocked=True)
    return completed


def plan(w):
    """Compute eligibility in a disposable copy; original state and data untouched."""
    result = {'mode': 'shadow', 'models': {}, 'model_calls': 0}
    with tempfile.TemporaryDirectory(prefix='qff-plan-') as directory:
        target = Path(directory) / 'plan'
        shutil.copytree(w.path, target, ignore=shutil.ignore_patterns('state.sqlite*', 'site', 'site_dev', 'orchestrator.lock'))
        scratch = Workspace(target)
        for slug in scratch.config['models']:
            root = scratch.model(slug)
            marker = root / 'audit/runner_fetch.json'
            if marker.exists() and read(marker).get('status') != 'complete':
                result['models'][slug] = {'collection': 'required or blocked', 'mapping': 'waits for current sources'}
                continue
            scratch.run(slug, 'autoextract.py', 'triage')
            scratch.run(slug, 'detect_copies.py')
            scratch.run(slug, 'qcl_mapset.py', '--record')
            doc = read(root / 'audit/mapset.json')
            result['models'][slug] = {'mapping_repos': doc['todo'], 'independent_runs': ['A', 'B'],
                                      'duplicates': doc['dup_of'], 'copied': doc['copied'],
                                      'fetch': 'network freshness not checked by offline plan',
                                      'review': 'fixed 20-item sample after merge; blind C for disagreements'}
    return result


def verify(w, state):
    reconcile(w, state)  # materialize verdict holds and regenerate before gating
    report = status(w, state)
    if report['unresolved'] or report['fetch_blockers'] or report['sample_blockers']:
        raise ValueError('incomplete work or fetch blocks shadow verification')
    with w.lock():
        current = status(w, state)
        if current['unresolved'] or current['fetch_blockers'] or current['sample_blockers']:
            raise ValueError('work or freshness changed before verification')
        before = {slug: model_signature(w.model(slug), state, slug) for slug in w.config['models']}
        for slug in w.config['models']:
            nf = w.model(slug) / 'cache/not_fetched.json'
            if nf.exists() and read(nf):
                raise ValueError('incomplete ingestion cannot be verified')
            w.run(slug, 'build.py')
        after = {slug: model_signature(w.model(slug), state, slug) for slug in w.config['models']}
        current = status(w, state)
        if before != after or current['unresolved'] or current['fetch_blockers'] or current['sample_blockers']:
            raise ValueError('inputs or freshness changed during shadow verification')
        artifacts = {slug: {str(p.relative_to(w.model(slug) / 'site')): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (w.model(slug) / 'site').rglob('*') if p.is_file()} for slug in w.config['models']}
        receipt = {'mode': 'shadow', 'input_commit': w.config['input_commit'], 'code_version': version(),
                   'model_inputs': before, 'artifacts': artifacts, 'publication_authorized': False,
                   'scope': 'per-model gated builds only; full exact-commit release verification still required'}
        write(w.path / 'shadow-verification.json', receipt)
        return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', required=True)
    sub = ap.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init'); init.add_argument('--models', nargs='+', required=True)
    sub.add_parser('plan'); sub.add_parser('status'); sub.add_parser('prepare'); sub.add_parser('verify')
    p = sub.add_parser('init-ingest'); p.add_argument('--slug', required=True); p.add_argument('--model', required=True)
    for command in ('run', 'resume'):
        p = sub.add_parser(command); p.add_argument('--responses'); p.add_argument('--workers', type=int, default=8)
    p = sub.add_parser('claim'); p.add_argument('--workers', type=int, default=8); p.add_argument('--lease', type=int, default=300)
    p = sub.add_parser('complete'); p.add_argument('claim'); p.add_argument('response')
    p = sub.add_parser('retry'); p.add_argument('task'); p.add_argument('--reason', required=True)
    p = sub.add_parser('backup'); p.add_argument('target')
    p = sub.add_parser('restore'); p.add_argument('source')
    for command in ('collect', 'ingest'):
        p = sub.add_parser(command); p.add_argument('slug'); p.add_argument('--include-repo', action='append', default=[]); p.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.command == 'init':
        w = Workspace.create(a.workspace, a.models)
    elif a.command == 'init-ingest':
        w = Workspace.create_ingest(a.workspace, a.slug, a.model)
    elif a.command == 'restore':
        w = restore(a.source, a.workspace)
    else:
        w = Workspace(a.workspace)
    if a.command == 'plan':
        print(json.dumps(plan(w), indent=2)); return
    state = State(w.path / 'state.sqlite')
    if a.command == 'verify':
        result = verify(w, state)
    elif a.command in ('run', 'resume', 'prepare'):
        result = reconcile(w, state)
        if getattr(a, 'responses', None):
            if not 1 <= a.workers <= 32:
                raise ValueError('workers must be 1..32')
            for _ in range(4):  # A/B, blind C, semantic QA, held-row regeneration
                if not run_responses(w, state, a.responses, a.workers):
                    break
                result = reconcile(w, state)
    elif a.command in ('collect', 'ingest'):
        from runner_fetch import collect
        root = w.model(a.slug)
        result = collect(root, w.path / 'ledgers' / a.slug, read(root / 'model.json')['model'], include_repos=a.include_repo, workers=a.workers, token=os.environ.get('HF_TOKEN'), install_lock=w.lock)
        if result['status'] == 'complete':
            result['runner'] = reconcile(w, state)
    elif a.command == 'claim':
        result = state.claim(lease_seconds=a.lease, limit=a.workers)
    elif a.command == 'complete':
        result = complete(w, state, read(a.claim), boundary.parse_response(Path(a.response).read_bytes()))
    elif a.command == 'retry':
        state.retry(a.task, a.reason); result = status(w, state)
    elif a.command == 'backup':
        result = {'backup': str(w.backup(a.target, state))}
    else:
        result = status(w, state)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({'error': str(error), 'publication': 'unavailable'}), file=sys.stderr)
        raise SystemExit(1)
