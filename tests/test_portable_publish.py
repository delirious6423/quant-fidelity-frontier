#!/usr/bin/env python3
"""`qcl_portable.py publish` moves a verified shadow workspace into the repository: it refuses a workspace without a current
receipt, a dirty checkout and a branch other than main; it copies exactly the shadow outputs (maps, the two generated claim
files, the merge's audit files; the snapshot index only after a collection), runs web/build.py as the data gate and commits
"Live update ..." the way the connector scan does, without pushing or deploying. Everything runs in a throwaway copy of the
repository (its own git history, one small model), so the real checkout and its data stay untouched."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / 'pipeline'))
from runner_replay import fake_response

MODEL = 'lfm2.5-24b-a2b'
GENERATED = {'verified/part8_machine.py', 'verified/part9_auto.py'}
DOCS = {'README.md', 'docs/reference/BENCHMARK_COVERAGE.md'}
IGNORE = shutil.ignore_patterns('site', 'site_dev', '__pycache__', '.gatecache*', '*.pyc')


def fixture_repo(root):
    """A copy of the code with one model, as its own git repository on a branch that is not main (the test setting)."""
    for top in ('pipeline', 'web', 'gates', 'protocol_evidence'):
        shutil.copytree(CODE / top, root / top, ignore=IGNORE)
    shutil.copytree(CODE / 'models' / MODEL, root / 'models' / MODEL, ignore=IGNORE)
    for f in ('site.json', 'watchlist.json', 'README.md', '.gitignore', 'docs/reference/BENCHMARK_COVERAGE.md'):
        (root / f).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(CODE / f, root / f)
    git(root, 'init', '-q'); git(root, 'checkout', '-q', '-b', 'fixture')
    git(root, 'add', '-A'); git(root, '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', 'commit', '-qm', 'fixture')


def git(root, *args):
    p = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True)
    assert p.returncode == 0, (args, p.stderr[-800:])
    return p.stdout


def portable(root, *args, ok=True):
    p = subprocess.run([sys.executable, str(root / 'pipeline/qcl_portable.py'), *args], capture_output=True, text=True)
    assert (p.returncode == 0) == ok, (args, p.returncode, p.stdout[-1200:], p.stderr[-1200:])
    return json.loads(p.stdout)


def answer(action):
    for path in Path(action['requests']).glob('*.json'):
        doc = json.loads(path.read_text())
        (Path(action['responses']) / path.name).write_text(json.dumps(fake_response(doc['request'])))


def commits(root):
    return git(root, 'rev-list', '--count', 'HEAD').strip()


with tempfile.TemporaryDirectory(prefix='qff-publish-') as tmp:
    repo, ws = Path(tmp) / 'repo', Path(tmp) / 'ws'
    fixture_repo(repo)
    a = portable(repo, '--workspace', str(ws), 'start', '--models', MODEL, '--remap')
    assert a['do'] == 'answer' and a['count'] > 0, a
    # 1. no receipt yet: refused, nothing committed, checkout untouched
    r = portable(repo, 'publish', '--workspace', str(ws), '--allow-branch', ok=False)
    assert r['do'] == 'refused' and r['committed'] is False and 'receipt' in r['error'], r
    assert commits(repo) == '1' and git(repo, 'status', '--porcelain').strip() == '', 'a refusal must change nothing'
    rounds = 0
    while a['do'] == 'answer':
        answer(a); a = portable(repo, '--workspace', str(ws), 'next'); rounds += 1
        assert rounds < 12, a
    assert a['do'] == 'stop', a
    receipt = json.loads((ws / 'shadow-verification.json').read_text())
    # 2. a dirty checkout and a branch other than main are refused before anything is copied
    (repo / 'scratch.txt').write_text('x\n')
    r = portable(repo, 'publish', '--workspace', str(ws), '--allow-branch', ok=False)
    assert r['do'] == 'refused' and 'local changes' in r['error'], r
    (repo / 'scratch.txt').unlink()
    r = portable(repo, 'publish', '--workspace', str(ws), ok=False)
    assert r['do'] == 'refused' and 'not main' in r['error'] and '--allow-branch' in r['error'], r
    assert commits(repo) == '1' and git(repo, 'status', '--porcelain').strip() == ''
    # 3. the publish itself
    r = portable(repo, 'publish', '--workspace', str(ws), '--allow-branch')
    assert r['do'] == 'published' and r['committed'] is True and commits(repo) == '2', r
    m = r['models'][MODEL]
    assert r['message'].startswith('Live update ') and r['message'].endswith(f"UTC: {MODEL} {m['repos']} repos, {m['values']} values"), r['message']
    assert git(repo, 'log', '-1', '--format=%s').strip() == r['message']
    assert r['push'].startswith('not pushed') and 'git push origin HEAD:refs/heads/main' in r['push'], r['push']
    assert m['copied']['maps'] > 0 and sorted(m['copied']['generated']) == sorted(GENERATED) and m['copied']['snapshots'] == 'kept', m['copied']
    changed = git(repo, 'show', '--name-only', '--format=', 'HEAD').split()
    model_files = {c for c in changed if c.startswith(f'models/{MODEL}/')}
    assert model_files and set(changed) - model_files <= DOCS, changed
    allowed = ('maps/', 'audit/review_queue.json', 'audit/holds.json', 'audit/verify_counts.json', *GENERATED)
    assert all(c[len(f'models/{MODEL}/'):].startswith(allowed) for c in model_files), sorted(model_files)
    assert not any('snapshots/' in c or 'cache/' in c or 'model.json' in c for c in changed), changed
    for rel in GENERATED:   # the committed claim files are the workspace's, byte for byte
        assert (repo / 'models' / MODEL / rel).read_bytes() == (ws / 'models' / MODEL / rel).read_bytes(), rel
    assert git(repo, 'status', '--porcelain').strip() == '', 'the publish leaves a clean checkout'
    assert json.loads((ws / 'shadow-verification.json').read_text())['model_inputs'] == receipt['model_inputs']
    # 4. the repository's own data gate passes on the result
    env = {k: v for k, v in os.environ.items() if not k.startswith('QCL_')} | {'QCL_ROOT': f'models/{MODEL}'}
    b = subprocess.run([sys.executable, 'web/build.py'], cwd=repo, env=env, capture_output=True, text=True)
    assert b.returncode == 0, b.stdout[-800:] + b.stderr[-800:]
    stats = json.loads(b.stdout.strip().splitlines()[-1].split(' | ')[0])
    assert stats['repos'] == m['repos'] and stats['values'] == m['values'], (stats, m)
    git(repo, 'checkout', '-q', '--', '.')   # build.py may rewrite audit/verify_counts.json; the test is about the commit
    # 5. the same workspace again: nothing to commit, no new commit
    r = portable(repo, 'publish', '--workspace', str(ws), '--allow-branch')
    assert r['do'] == 'stop' and r['committed'] is False and 'nothing to commit' in r['note'] and commits(repo) == '2', r
    # 6. a receipt that no longer matches the workspace is refused
    rp = ws / 'shadow-verification.json'; doc = json.loads(rp.read_text()); doc['model_inputs'] = {MODEL: 'stale'}; rp.write_text(json.dumps(doc))
    r = portable(repo, 'publish', '--workspace', str(ws), '--allow-branch', ok=False)
    assert r['do'] == 'refused' and 'changed since its receipt' in r['error'] and commits(repo) == '2', r
assert subprocess.run(['git', 'status', '--porcelain', '--', 'models'], cwd=CODE, capture_output=True, text=True).stdout == '', 'the real checkout must stay untouched'
print(f"portable publish ok: {rounds} answered rounds, {m['copied']['maps']} map files, {len(changed)} files in the Live update commit ({m['repos']} repos, {m['values']} values)")
