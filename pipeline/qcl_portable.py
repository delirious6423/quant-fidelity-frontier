#!/usr/bin/env python3
"""The QFF scan for any agent: a loop of plain commands and JSON files, no connector, subagent tool or provider needed.

  python3 pipeline/qcl_portable.py start --workspace DIR --models SLUG [SLUG ...] [--remap]
  python3 pipeline/qcl_portable.py next  --workspace DIR
  python3 pipeline/qcl_portable.py status --workspace DIR
  python3 pipeline/qcl_portable.py publish --workspace DIR [--push] [--allow-branch]

Every call prints one JSON object with "do":
  "answer"  DIR/requests/<task>.json files wait for an interpretation. For each one, write DIR/responses/<task>.json
            holding only the JSON object the request asks for (docs/runbooks/PORTABLE_SCAN.md says exactly what). Then run `then`.
  "fix"     some responses were rejected (the reason is given, fail closed). Write a corrected response, run the `retry`
            command printed for it, then run `then`.
  "stop"    nothing is left: the shadow data was merged, gated and built; `report` holds the numbers, including a
            value-level comparison with the model's committed claims when the model exists in this repository.
Everything else (workspace, queues, blind tie-breaker, QA sample, merge, gates, build) is pipeline/qcl_runner.py; this
script only decides what to say next. start/next/status never touch models/ in the repository and never run git.

`publish` is the only command that writes to the repository: it copies the verified shadow data of a workspace whose
receipt is current into models/<slug>/ (maps, the generated claim files, the merge's audit files, the snapshot index only
if the workspace collected), runs the repository's own data gate (web/build.py per model, then readme_sync.py and
qcl_benchinv.py) and commits "Live update ..." exactly like the connector scan does. Any failure restores the checkout and
commits nothing. It never force-pushes and never dispatches a deployment: the QA task reviews, tests and deploys.
DIR must be outside the repository."""
import argparse, datetime as dt, json, os, shutil, subprocess, sys
from pathlib import Path
CODE = Path(__file__).resolve().parent.parent
RUNNER = [sys.executable, str(CODE / 'pipeline/qcl_runner.py')]
HOW = ("For each DIR/requests/<task>.json: read it in full. A request with request.kind == \"mapping\" asks for the JSON object described "
       "under \"## Output\" in request.prompt: answer with candidate ids (c1, c7.2), never with a number you typed; copy request.source.sha256 "
       "into source_sha256 and request.candidates_sha256 into candidates_sha256; a prose value is one number copied exactly from its cited "
       "line. request.images, when present, lists pinned copies of the card's content images (workspace-relative files under "
       "DIR/models/<slug>/cache/images): look at them only to understand the card; a number shown only in an image is never a "
       "candidate or prose value, and a card whose quality numbers exist only in an image gets {\"no_quality_claims\": true, "
       "\"no_claims_reason\": \"... only in the image at L<n>\"} naming the image line. A request with request.kind == \"review\" asks for {\"id\": request.fingerprint, \"verdict\": \"ok\"|\"wrong\"|\"unsure\", "
       "\"reason\": one sentence citing the line}. Write only that object to DIR/responses/<task>.json. Never read another task's "
       "response or any file under DIR/models/*/maps: runs A, B and C must stay independent.")


def say(d, code=0):
    print(json.dumps(d, indent=1)); sys.exit(code)


def runner(workspace, *args):
    p = subprocess.run([*RUNNER, '--workspace', str(workspace), *args], capture_output=True, text=True)
    if p.returncode:
        say({'do': 'stop', 'error': f'qcl_runner {args[0]} failed', 'detail': (p.stdout + p.stderr)[-1200:]})
    return parse(p.stdout)


def parse(stdout):
    """qcl_runner prints one JSON document, indented; a stage script prints one JSON line last."""
    try:
        return json.loads(stdout)
    except ValueError:
        try:
            return json.loads(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {}


def outside(workspace):
    w = Path(workspace).resolve()
    if w == CODE or CODE in w.parents:
        say({'do': 'stop', 'error': 'the workspace must be a directory outside the repository'})
    return w


def pending(workspace):
    sys.path.insert(0, str(CODE / 'pipeline'))
    from runner_state import State
    tasks = State(workspace / 'state.sqlite').tasks(with_payload=False)
    waiting = [t for t in tasks if t['state'] in ('pending', 'running', 'failed')]
    blocked = [t for t in tasks if t['state'] == 'blocked']
    done = sum(t['state'] == 'succeeded' for t in tasks)
    return waiting, blocked, done


def compare(workspace, slug):
    """Value-level comparison with the committed claims of the same model, when the repository has them."""
    prod = CODE / 'models' / slug
    if not (prod / 'snapshots/index.json').exists():
        return {'note': 'no committed claims for this model in the repository; nothing to compare'}
    sys.path.insert(0, str(CODE / 'pipeline'))
    from runner_compare import compare as cmp
    return cmp(workspace / 'models' / slug, prod)['summary']


def answer_or_stop(workspace):
    waiting, blocked, done = pending(workspace)
    then = f'python3 pipeline/qcl_portable.py next --workspace {workspace}'
    if waiting:
        stages = {}
        for t in waiting:
            stages[t['stage']] = stages.get(t['stage'], 0) + 1
        say({'do': 'answer', 'requests': str(workspace / 'requests'), 'responses': str(workspace / 'responses'),
             'count': len(waiting), 'by_stage': stages, 'done_so_far': done, 'how': HOW, 'then': then})
    if blocked:
        say({'do': 'fix', 'blocked': [{'task': t['id'], 'stage': t['stage'], 'reason': t['reason'],
                                       'retry': f'python3 pipeline/qcl_runner.py --workspace {workspace} retry {t["id"]} --reason "<what was wrong and what changed>"'}
                                      for t in blocked],
             'how': 'A rejected response is never installed. Write a corrected DIR/responses/<task>.json, run its retry command, then run `then`. '
                    'If the request cannot be answered honestly (for example the card is not about this model), say so in the reason and leave it blocked; '
                    'a blocked task keeps the model from verifying, which is the correct outcome.',
             'then': then})
    receipt = runner(workspace, 'verify')
    config = json.loads((workspace / 'runner.json').read_text())
    report = {'tasks_completed': done, 'shadow_verification': str(workspace / 'shadow-verification.json'),
              'input_commit': receipt.get('input_commit'), 'models': {}}
    for slug in config['models']:
        report['models'][slug] = {'built': str(workspace / 'models' / slug / 'site'), 'versus_committed': compare(workspace, slug)}
    say({'do': 'stop', 'report': report,
         'note': 'Shadow data only: nothing was committed, pushed or published. Report the numbers above as they are.'})


# ---------------------------------------------------------------- publish
GENERATED = ('verified/part8_machine.py', 'verified/part9_auto.py')          # written by qcl_watch.py merge and gen_auto.py
MERGE_AUDIT = ('audit/review_queue.json', 'audit/holds.json')                # written by the merge, and the review holds it consumes
PINS = ('mapped.json', 'mapped_c.json')                                      # maps/ pin dictionaries: merged by key, never replaced
GATE_DOCS = ('README.md', 'docs/reference/BENCHMARK_COVERAGE.md')            # rewritten by readme_sync.py and qcl_benchinv.py
INPUT_EXCLUDES = ('maps', 'audit', *GENERATED)                               # everything else under models/<slug>/ is an input of the workspace
COMMITTER = ['-c', 'user.name=Claude', '-c', 'user.email=noreply@anthropic.com']
PUSH = 'git push origin HEAD:refs/heads/main'


def git(*args):
    return subprocess.run(['git', *args], cwd=CODE, capture_output=True, text=True)


def refuse(reason, **extra):
    say({'do': 'refused', 'error': reason, 'committed': False, **extra}, code=1)


def read_json(path):
    return json.loads(Path(path).read_text())


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def copy_model(ws_root, repo_root):
    """Copy exactly the shadow outputs into the repository's model directory; never delete anything there."""
    copied = {'maps': 0, 'pins_merged': [], 'generated': [], 'audit': [], 'snapshots': 'kept'}
    for src in sorted(p for p in (ws_root / 'maps').rglob('*') if p.is_file()):
        rel = src.relative_to(ws_root)
        dst = repo_root / rel
        if src.parent == ws_root / 'maps' and src.name in PINS:
            doc = read_json(src)
            if not isinstance(doc, dict):
                raise ValueError(f'{rel} is not a pin dictionary')
            if dst.exists():
                doc = read_json(dst) | doc                                  # the workspace's pins win; untouched pins stay
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(doc, indent=1) + '\n')
            copied['pins_merged'].append(str(rel))
        else:
            copy_file(src, dst); copied['maps'] += 1
    for rel in GENERATED:
        if (ws_root / rel).exists():
            copy_file(ws_root / rel, repo_root / rel); copied['generated'].append(rel)
    for rel in MERGE_AUDIT:
        if (ws_root / rel).exists():
            copy_file(ws_root / rel, repo_root / rel); copied['audit'].append(rel)
    marker = ws_root / 'audit/runner_fetch.json'
    if marker.exists() and read_json(marker).get('status') == 'complete':   # the workspace collected: its index and the cards it pins
        index = read_json(ws_root / 'snapshots/index.json')
        files = ['snapshots/index.json']
        for rec in [index.get('base') or {}, *index.get('repos', {}).values()]:
            for key in ('file', 'tree_file'):
                if rec.get(key):
                    path = (ws_root / rec[key]).resolve()
                    if not path.is_relative_to(ws_root.resolve()) or not path.is_file():
                        raise ValueError(f'index entry {rec[key]} is not a file inside the shadow model')
                    files.append(rec[key])
        if (ws_root / 'cache/not_fetched.json').exists():
            files.append('cache/not_fetched.json')
        for rel in files:
            copy_file(ws_root / rel, repo_root / rel)
        copied['snapshots'] = f'copied index.json and {len(files) - 1} pinned files'
    return copied


def restore(slugs):
    """Put every path publish may have touched back to HEAD; one path at a time, so an untracked doc cannot cancel the rest."""
    for path in [f'models/{slug}' for slug in slugs] + list(GATE_DOCS):
        git('checkout', '-q', '--', path)
    for slug in slugs:
        git('clean', '-fdq', '--', f'models/{slug}')


def gate(slugs):
    """The scan's data gate: web/build.py per model (every gate, fail-closed), then the generated docs. Returns the stats line per model."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('QCL_')}
    stats = {}
    live = subprocess.run([sys.executable, str(CODE / 'pipeline/qcl_live.py')], cwd=CODE, capture_output=True, text=True).stdout.split()
    for root in [f'models/{slug}' for slug in slugs] + [d for d in live if os.path.basename(d) not in slugs and not (CODE / d / 'site/data.json').exists()]:
        p = subprocess.run([sys.executable, str(CODE / 'web/build.py')], cwd=CODE, env=env | {'QCL_ROOT': root}, capture_output=True, text=True)
        if p.returncode:
            return None, {'gate': 'web/build.py', 'model': root, 'tail': (p.stdout + p.stderr)[-1200:]}
        slug = os.path.basename(root)
        if slug in slugs:
            stats[slug] = json.loads(p.stdout.strip().splitlines()[-1].split(' | ')[0])
    for script in ('pipeline/readme_sync.py', 'pipeline/qcl_benchinv.py'):                 # README table and the benchmark coverage doc
        p = subprocess.run([sys.executable, str(CODE / script)], cwd=CODE, env=env, capture_output=True, text=True)
        if p.returncode:
            return None, {'gate': script, 'tail': (p.stdout + p.stderr)[-1200:]}
    return stats, None


def push():
    for _ in range(3):   # exactly as pipeline/qcl_next.py: push HEAD to main, rebase on a rejection, never force
        if git('push', '-q', 'origin', 'HEAD:refs/heads/main').returncode == 0:
            return 'pushed'
        git('pull', '--rebase', '-q', 'origin', 'main')
    return 'push failed'


def publish(workspace, allow_branch, do_push):
    if not (workspace / 'runner.json').exists():
        refuse('no workspace here; run start first')
    config = read_json(workspace / 'runner.json')
    slugs = config['models']
    if config.get('ingestion') or not slugs:
        refuse('this workspace ingests a new model; new models reach the repository through the ingest path, not publish')
    for slug in slugs:
        if not (CODE / 'models' / slug / 'model.json').exists():
            refuse(f'the repository has no models/{slug}/; publish only updates models that exist')
    receipt_path = workspace / 'shadow-verification.json'
    if not receipt_path.exists():
        refuse('no shadow verification receipt in the workspace; run `next` until it says stop', receipt=str(receipt_path))
    previous = read_json(receipt_path)
    p = subprocess.run([*RUNNER, '--workspace', str(workspace), 'verify'], capture_output=True, text=True)
    if p.returncode:
        refuse('shadow verification does not pass now (unresolved tasks, a fetch block or a gate failure)', detail=(p.stdout + p.stderr)[-1200:])
    current = parse(p.stdout)
    if not current.get('model_inputs') or current.get('model_inputs') != previous.get('model_inputs') or current.get('input_commit') != previous.get('input_commit'):
        refuse('the workspace changed since its receipt was written; run `next` until it says stop, then publish')
    if git('rev-parse', '--is-inside-work-tree').stdout.strip() != 'true':
        refuse('the repository checkout is not a git work tree')
    status = git('status', '--porcelain=v1', '--untracked-files=normal')
    if status.returncode or status.stdout.strip():
        refuse('the checkout has local changes; commit or stash them, or use a fresh clone (nothing was changed)', status=status.stdout.strip()[:800])
    branch = git('rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
    if branch != 'main' and not allow_branch:
        refuse(f'HEAD is on {branch!r}, not main; publish commits only to main (or pass --allow-branch for a test checkout)')
    base = config.get('input_commit') or ''
    if git('cat-file', '-e', base + '^{commit}').returncode == 0:
        for slug in slugs:   # the inputs the workspace started from must still be what the repository holds
            excludes = [f':(exclude)models/{slug}/{x}' for x in INPUT_EXCLUDES]
            if git('diff', '--quiet', base, 'HEAD', '--', f'models/{slug}', *excludes).returncode:
                refuse(f'models/{slug} changed in the repository since the workspace was created from {base[:12]}; start a new workspace from the current main')
        stale_note = None
    else:
        stale_note = f'input commit {base[:12]} is not in this checkout; the inputs could not be compared'
    copied = {}
    try:
        for slug in slugs:
            copied[slug] = copy_model(workspace / 'models' / slug, CODE / 'models' / slug)
    except (ValueError, OSError) as error:
        restore(slugs); refuse(f'copy failed: {error}')
    stats, failure = gate(slugs)
    if failure:
        restore(slugs); refuse('the data gate failed; the checkout was restored and nothing was committed', **failure)
    git('add', '-A', '--', *[f'models/{slug}' for slug in slugs])
    git('add', '--', *[d for d in GATE_DOCS if (CODE / d).exists()])
    if git('diff', '--cached', '--quiet').returncode == 0:
        say({'do': 'stop', 'committed': False, 'note': 'nothing to commit: the repository already holds this data', 'models': copied})
    message = f'Live update {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC: ' + ', '.join(f'{k} {v["repos"]} repos, {v["values"]} values' for k, v in stats.items())
    c = git(*COMMITTER, 'commit', '-qm', message)
    if c.returncode:
        git('reset', '-q'); restore(slugs); refuse('git commit failed', detail=(c.stdout + c.stderr)[-800:])
    commit = git('rev-parse', 'HEAD').stdout.strip()
    changed = git('show', '--name-only', '--format=', 'HEAD').stdout.split()
    report = {'do': 'published', 'committed': True, 'commit': commit, 'message': message, 'branch': branch,
              'models': {slug: {'repos': stats[slug]['repos'], 'values': stats[slug]['values'], 'copied': copied[slug]} for slug in slugs},
              'changed_files': len(changed), 'push': push() if do_push else f'not pushed; run: {PUSH}',
              'next': 'The QA task (pipeline/qcl_qa.py) reviews, tests and deploys commits on main; publish never dispatches a deployment.'}
    if stale_note:
        report['note'] = stale_note
    say(report)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workspace')
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('start'); s.add_argument('--models', nargs='+', required=True)
    s.add_argument('--remap', action='store_true', help='forget every existing map so the whole model is interpreted again (the test setting)')
    sub.add_parser('next'); sub.add_parser('status')
    pub = sub.add_parser('publish', help='copy the verified shadow data into the repository, run its data gate and commit')
    pub.add_argument('--push', action='store_true', help='push the commit to main (rebase and retry 3 times, never force)')
    pub.add_argument('--allow-branch', action='store_true', help='publish although HEAD is not on main (test checkouts)')
    for s_ in (s, *(sub.choices[c] for c in ('next', 'status', 'publish'))):
        s_.add_argument('--workspace', default=argparse.SUPPRESS)   # also accepted after the command; SUPPRESS keeps the global value
    a = ap.parse_args()
    if not a.workspace:
        ap.error('--workspace DIR is required')
    workspace = outside(a.workspace)
    if a.command == 'start':
        if workspace.exists() and any(workspace.iterdir()):
            say({'do': 'stop', 'error': 'the workspace exists and is not empty; choose a new directory or run `next`'})
        runner(workspace, 'init', '--models', *a.models)
        if a.remap:
            for slug in a.models:
                root = workspace / 'models' / slug
                shutil.rmtree(root / 'maps', ignore_errors=True)
                (root / 'audit/skipped_maps.json').unlink(missing_ok=True)
        (workspace / 'responses').mkdir(exist_ok=True)
        runner(workspace, 'prepare')
        answer_or_stop(workspace)
    if a.command == 'next':
        if not (workspace / 'runner.json').exists():
            say({'do': 'stop', 'error': 'no workspace here; run start first'})
        runner(workspace, 'resume', '--responses', str(workspace / 'responses'))
        answer_or_stop(workspace)
    if a.command == 'status':
        waiting, blocked, done = pending(workspace)
        say({'do': 'status', 'waiting': len(waiting), 'blocked': len(blocked), 'done': done, 'runner': runner(workspace, 'status')})
    if a.command == 'publish':
        publish(workspace, a.allow_branch, a.push)


if __name__ == '__main__':
    main()
