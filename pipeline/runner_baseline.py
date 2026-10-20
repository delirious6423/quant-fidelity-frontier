"""Bounded replay of the *actual* legacy dispatch branches, never its entry point.

AST-selected branches receive inert scan/agent stubs. No git, network, publication,
source mutation, or legacy top-level code is executed.
"""
import ast
import json
import math
from pathlib import Path
import time

CODE = Path(__file__).resolve().parents[1]


def dispatch_baseline(models=12):
    tree = ast.parse((CODE / 'pipeline/qcl_next.py').read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) in ("step == 'a_fetched'", "step == 'a_mapped'"):
            selected.append(node)
    if len(selected) != 2:
        raise ValueError('legacy dispatch changed; review baseline adapter')
    chunks, completed, operations = [], set(), []
    slugs = [f'model{i:02d}' for i in range(models)]
    class Handoff(Exception):
        pass
    def scan(arg):
        operations.append(arg)
        op, slug = arg.split()[:2]
        if op == 'prepare':
            return 0, {'queues': 2, 'todo': 1}
        if op == 'finish':
            return (0, {'merge': 'mock completed'}) if {(slug, 'A'), (slug, 'B')} <= completed else (1, {'error': 'pending'})
        return 0, {}
    def agents(prompts, state, step):
        chunks.append(prompts)
        completed.update(tuple(p.split(':')) for p in prompts)
        raise Handoff
    state = {'a_shards': {s: ['shard'] for s in slugs}, 'part_a': {s: {} for s in slugs}, 'notes': []}
    env = dict(scan=scan, agents=agents, s=state, math=math, CODE='', MAP_PROMPT='{slug}:{run}', minutes=lambda: 0)
    t0 = time.perf_counter()
    for turn in range(4):
        env['step'] = 'a_fetched' if turn == 0 else 'a_mapped'
        try:
            exec(compile(ast.Module(body=selected, type_ignores=[]), '<legacy-dispatch-replay>', 'exec'), env)
            break
        except Handoff:
            continue
    return {'eligible_mapping_requests': models * 2, 'logical_model_requests': sum(map(len, chunks)),
            'simulated_worker_launches': sum(map(len, chunks)), 'coordinator_handoffs': len(chunks),
            'first_dispatch': len(chunks[0]), 'first_dispatch_unstarted': models * 2 - len(chunks[0]),
            'completed': len(completed), 'completed_repeated': sum(map(len, chunks)) - len(completed),
            'scan_operations': len(operations), 'runtime_seconds': time.perf_counter() - t0,
            'actual_agent_launches': 0, 'tokens': None, 'cost': None}


def historical():
    out = {}
    for name in ('20261020T1412Z-live', '20261020T1621Z-live', '20261020T1713Z-live', '20261020T1714Z-ingest'):
        rows = [json.loads(l) for l in (CODE / f'logs/scan/{name}.jsonl').read_text().splitlines()]
        out[name] = {'records': len(rows), 'prepare_calls': sum(r['cmd'].startswith('prepare ') for r in rows),
                     'finish_calls': sum(r['cmd'].startswith('finish ') for r in rows),
                     'connector_op_lists_requested': sum(r.get('calls', 0) for r in rows),
                     'last_reported_minutes': next((r['minutes'] for r in reversed(rows) if 'minutes' in r), None)}
    return out


if __name__ == '__main__':
    print(json.dumps({'measurement': 'offline dispatch replay; stub workers, no model calls',
                      'stretch_target': '100x runtime on this fixed replay; no optimization or saving claimed',
                      'baseline': dispatch_baseline(), 'historical_log_observations': historical(),
                      'schedule_estimate': '24 + 12 + 24 + 1 = 61 coordinating sessions/day, plus weekly discovery; not billing telemetry'}, indent=2))
