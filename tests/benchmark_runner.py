#!/usr/bin/env python3
"""Repeatable orchestration benchmark for the shadow runner at real catalogue scale.

Copies one committed model root into a new external workspace, forgets every
mapping so each unique card is queued for runs A and B, then times the
deterministic orchestration only: prepare (stage scripts + enqueue + request
export), a run that consumes synthetic responses (claim, validate, persist,
materialize, merge, review sampling), and an unchanged no-op reconcile.

Responses come from the replay fixture adapter: they measure scheduling and
bookkeeping cost, never mapping quality, provider latency, tokens or cost.
Nothing is written inside the repository and no network is used.

  python3 tests/benchmark_runner.py --model lfm2.5-24b-a2b --out /tmp/qff-bench
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'pipeline'))


def runner(workspace, *args):
    cmd = [sys.executable, str(CODE / 'pipeline/qcl_runner.py'), '--workspace', str(workspace), *args]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.perf_counter() - t0
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.stderr[-800:]}')
    return seconds, json.loads(result.stdout)


def output_evidence(workspace, model):
    """Complete semantic output for an exact old/new comparison, no value normalization."""
    from qcl_runner import records
    from runner_state import State
    root = Path(workspace) / 'models' / model
    def read(path, default):
        return json.loads(path.read_text()) if path.exists() else default
    return {'records': records(root), 'holds': read(root / 'audit/holds.json', []),
            'review_queue': read(root / 'audit/review_queue.json', []),
            'sources': read(root / 'snapshots/index.json', {}),
            'maps': {str(p.relative_to(root)): read(p, None) for p in sorted((root / 'maps').rglob('*.json'))},
            'tasks': [{k: t[k] for k in ('slot', 'stage', 'generation', 'state', 'result')}
                      for t in State(Path(workspace) / 'state.sqlite').tasks(with_payload=False)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default='lfm2.5-24b-a2b')
    ap.add_argument('--out', required=True, help='new external directory')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out).absolute()
    if out.exists() or out == CODE or CODE in out.parents:
        raise ValueError('benchmark output must be a new external directory')
    out.mkdir(parents=True)
    workspace, responses = out / 'workspace', out / 'responses'
    runner_init = runner(workspace, 'init', '--models', a.model)
    root = workspace / 'models' / a.model
    shutil.rmtree(root / 'maps', ignore_errors=True)
    (root / 'audit/skipped_maps.json').unlink(missing_ok=True)
    repos = len(json.loads((root / 'snapshots/index.json').read_text())['repos'])

    prepare_s, status = runner(workspace, 'prepare')
    from runner_replay import fake_response
    from runner_state import State
    responses.mkdir()
    pending = [t for t in State(workspace / 'state.sqlite').tasks() if t['state'] != 'succeeded']
    for task in pending:
        (responses / (task['id'] + '.json')).write_text(json.dumps(fake_response(task['payload'])))
    run_s, status = runner(workspace, 'run', '--responses', str(responses), '--workers', str(a.workers))
    # Blind C and the review sample may follow; feed those rounds too, each round timed.
    rounds = [run_s]
    for _ in range(3):
        left = [t for t in State(workspace / 'state.sqlite').tasks() if t['state'] != 'succeeded']
        if not left:
            break
        for task in left:
            (responses / (task['id'] + '.json')).write_text(json.dumps(fake_response(task['payload'])))
        seconds, status = runner(workspace, 'run', '--responses', str(responses), '--workers', str(a.workers))
        rounds.append(seconds)
    noop_s, status = runner(workspace, 'prepare')
    status_s, status = runner(workspace, 'status')
    summary = {'model': a.model, 'repos': repos, 'mapping_tasks': len(pending),
               'init_seconds': round(runner_init[0], 2), 'prepare_seconds': round(prepare_s, 2),
               'run_rounds_seconds': [round(s, 2) for s in rounds], 'noop_reconcile_seconds': round(noop_s, 2),
               'status_seconds': round(status_s, 2), 'final_tasks': status['tasks'],
               'measurement': 'offline orchestration with synthetic fixture responses; no model calls, tokens or cost'}
    from runner_state import digest
    evidence = output_evidence(workspace, a.model)
    (out / 'output.json').write_text(json.dumps(evidence, sort_keys=True) + '\n')
    summary['output_sha256'] = digest(evidence)
    summary['unresolved'] = status['unresolved']
    (out / 'results.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
