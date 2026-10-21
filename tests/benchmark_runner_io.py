#!/usr/bin/env python3
"""Offline fixed-workload scheduler/collection I/O benchmark; no HTTP/model calls.

python3 tests/benchmark_runner_io.py --out <new-external-directory>
Times include actual local FULL-sync commits and atomic file fsyncs. HTTP responses
are synthetic, so this measures orchestration, not network or model latency.
"""
import argparse
import collections
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--implementation', default=str(CODE), help='checkout whose pipeline is measured')
    ap.add_argument('--repos', type=int, default=1253)
    ap.add_argument('--tasks', type=int, default=1190)
    a = ap.parse_args()
    implementation = Path(a.implementation).resolve()
    sys.path.insert(0, str(implementation / 'pipeline'))
    import runner_fetch as fetch
    from runner_state import State
    out = Path(a.out).resolve()
    if out.exists() or out.is_relative_to(CODE) or out.is_relative_to(implementation) or min(a.repos, a.tasks) < 1:
        raise ValueError('use a new external directory and positive counts')
    out.mkdir(parents=True)
    state = State(out / 'state.sqlite')
    start = time.perf_counter()
    with state.transaction():
        ids = [state.enqueue(f'm{i % 9}:A:{i}', f'm{i % 9}', 'A', 'v1', {'text': 'x' * 8192, 'i': i}) for i in range(a.tasks)]
    enqueue = time.perf_counter() - start
    start = time.perf_counter()
    remaining = set(ids)
    # Optimized selection owns only a temporary eligibility table, no long transaction.
    if hasattr(state, 'claim_scope'):
        with state.claim_scope(ids) as claim:
            while remaining:
                task = claim()
                state.complete(task['id'], task['token'], {'ok': True})
                remaining.remove(task['id'])
    else:
        while remaining:
            task = state.claim(eligible_ids=list(remaining))
            state.complete(task['id'], task['token'], {'ok': True})
            remaining.remove(task['id'])
    drain = time.perf_counter() - start
    root, ledger = out / 'shadow', out / 'ledger'
    fetch.atomic(root / 'model.json', {'model': 'test/base', 'slug': 'test', 'name': 'Test'})
    rev = 'a' * 40
    cat = [{'id': f'test/card{i}', 'sha': rev} for i in range(a.repos)]
    calls = collections.Counter()
    def get(url, token, **kwargs):
        calls['http_requests'] += 1
        if '/api/models?' in url:
            return 200, json.dumps(cat).encode(), {}
        if '/api/models/test/base' in url:
            return 200, json.dumps({'sha': rev}).encode(), {}
        if '/tree/' in url:
            return 200, b'[{"type":"file","size":123}]', {}
        return 200, b'| Quant | Mean KLD |\n|---|---|\n| Q4 | 0.125 |\n', {}
    atomic = fetch.atomic
    def tracked(path, value):
        path = Path(path)
        if path == ledger / 'index.json' or path.parent == ledger / 'checkpoints':
            calls['checkpoint_writes'] += 1
            calls['checkpoint_bytes'] += len(json.dumps(value, sort_keys=True).encode())
        return atomic(path, value)
    rounds = []
    with patch.object(fetch.fetch, 'get', get), patch.object(fetch, 'atomic', tracked):
        for _ in range(2):
            calls.clear()
            start = time.perf_counter()
            result = fetch.collect(root, ledger, 'test/base', workers=8, base='https://example.test')
            assert result['status'] == 'complete', result
            rounds.append({'seconds': round(time.perf_counter() - start, 6),
                           **{k: calls[k] for k in ('http_requests', 'checkpoint_writes', 'checkpoint_bytes')}})
    summary = {'measurement': 'offline synthetic HTTP; actual fsyncs; no model calls/tokens/cost',
               'repos': a.repos, 'tasks': a.tasks, 'mapping_model_requests': 0,
               'source_output_sha256': __import__('hashlib').sha256((root / 'snapshots/index.json').read_bytes()).hexdigest(), 'enqueue_seconds': round(enqueue, 6),
               'claim_complete_seconds': round(drain, 6), 'collection': rounds,
               'completed': state.counts()}
    (out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
