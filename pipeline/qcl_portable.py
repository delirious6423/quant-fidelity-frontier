#!/usr/bin/env python3
"""The QFF scan for any agent: a loop of plain commands and JSON files, no connector, subagent tool or provider needed.

  python3 pipeline/qcl_portable.py start --workspace DIR --models SLUG [SLUG ...] [--remap]
  python3 pipeline/qcl_portable.py next  --workspace DIR
  python3 pipeline/qcl_portable.py status --workspace DIR

Every call prints one JSON object with "do":
  "answer"  DIR/requests/<task>.json files wait for an interpretation. For each one, write DIR/responses/<task>.json
            holding only the JSON object the request asks for (docs/runbooks/PORTABLE_SCAN.md says exactly what). Then run `then`.
  "fix"     some responses were rejected (the reason is given, fail closed). Write a corrected response, run the `retry`
            command printed for it, then run `then`.
  "stop"    nothing is left: the shadow data was merged, gated and built; `report` holds the numbers, including a
            value-level comparison with the model's committed claims when the model exists in this repository.
Everything else (workspace, queues, blind tie-breaker, QA sample, merge, gates, build) is pipeline/qcl_runner.py; this
script only decides what to say next. It never touches models/ in the repository, never runs git and never publishes.
DIR must be outside the repository."""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
CODE = Path(__file__).resolve().parent.parent
RUNNER = [sys.executable, str(CODE / 'pipeline/qcl_runner.py')]
HOW = ("For each DIR/requests/<task>.json: read it in full. A request with request.kind == \"mapping\" asks for the JSON object described "
       "under \"## Output\" in request.prompt: answer with candidate ids (c1, c7.2), never with a number you typed; copy request.source.sha256 "
       "into source_sha256 and request.candidates_sha256 into candidates_sha256; a prose value is one number copied exactly from its cited "
       "line. A request with request.kind == \"review\" asks for {\"id\": request.fingerprint, \"verdict\": \"ok\"|\"wrong\"|\"unsure\", "
       "\"reason\": one sentence citing the line}. Write only that object to DIR/responses/<task>.json. Never read another task's "
       "response or any file under DIR/models/*/maps: runs A, B and C must stay independent.")


def say(d):
    print(json.dumps(d, indent=1)); sys.exit(0)


def runner(workspace, *args):
    p = subprocess.run([*RUNNER, '--workspace', str(workspace), *args], capture_output=True, text=True)
    if p.returncode:
        say({'do': 'stop', 'error': f'qcl_runner {args[0]} failed', 'detail': (p.stdout + p.stderr)[-1200:]})
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workspace', required=True)
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('start'); s.add_argument('--models', nargs='+', required=True)
    s.add_argument('--remap', action='store_true', help='forget every existing map so the whole model is interpreted again (the test setting)')
    sub.add_parser('next'); sub.add_parser('status')
    a = ap.parse_args()
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


if __name__ == '__main__':
    main()
