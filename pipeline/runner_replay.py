#!/usr/bin/env python3
"""Offline frozen-input replay and old/new extraction shadow comparison.

All outputs go to a new external directory. Recorded responses are synthetic
candidate-ID mappings; this evaluates orchestration/provenance, not model recall.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from runner_baseline import dispatch_baseline, historical
from runner_state import State
from runner_workspace import Workspace, CODE, write, read
import qcl_runner as runner

CARD = '''# Base Model GGUF
Quantized Base/Model. The publisher measured these quants against BF16.
## Full vocabulary
Quality evaluation dataset: Wiki. Tool: llama.cpp. Context: 2048.
| Quant | Size (GB) | Mean KLD |
| --- | --- | --- |
| Q4_K_M | 4.0 | 0.1250 |
'''


def fixture(path):
    path.mkdir(parents=True)
    write(path / 'runner.json', {'version': 1, 'mode': 'shadow', 'input_commit': 'frozen-replay-v1', 'models': ['fixture']})
    root = path / 'models/fixture'
    for folder in ('cache/cards', 'snapshots', 'audit', 'verified', 'maps'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    write(root / 'model.json', {'model': 'Base/Model', 'name': 'Fixture', 'slug': 'fixture', 'max_size_gb': 400, 'others': [], 'size_note': '', 'pin_note': '', 'coverage_note': ''})
    cards = {'Publisher/normal-GGUF': CARD, 'Publisher/duplicate-GGUF': CARD,
             'Publisher/disagree-GGUF': CARD.replace('Wiki.', 'Code.'),
             'Publisher/partial-GGUF': CARD.replace('Wiki.', 'Books.'),
             'Publisher/empty-GGUF': '# Model\nNo reported numerical claims.\n',
             'Publisher/image-GGUF': '# Model\nQuality results\n![Results](results.png)\n',
             'Publisher/new-GGUF': CARD.replace('Wiki.', 'New corpus.')}
    idx = {}
    for i, (repo, text) in enumerate(cards.items()):
        file = 'cache/cards/' + repo.replace('/', '__') + '.md'
        (root / file).write_bytes(text.encode())
        idx[repo] = {'file': file, 'sha': 'a' * 40, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'bytes': len(text.encode()),
                     'lastModified': f'2026-10-{10+i:02d}T00:00:00Z', 'tags': ['gguf'], 'total_bytes': 4000000000}
    write(root / 'snapshots/index.json', {'model': 'Base/Model', 'repos': idx, 'captured_at': '2026-10-20T00:00:00Z'})
    for name in ('hf_sizes_summary.json', 'hf_branch_sizes.json'):
        write(root / 'snapshots' / name, {})
    # Use the same repository metric vocabulary as normal qcl_newmodel scaffolding.
    ns = {}
    exec((CODE / 'models/qwen3.8-flash-next/sets.py').read_text(), ns)
    (root / 'sets.py').write_text('SETS = []\nMETRICS = ' + repr(ns['METRICS']) + '\n')
    return Workspace(path)


def fake_response(bundle):
    if bundle['kind'] == 'review':
        return {'id': bundle['fingerprint'], 'verdict': 'ok', 'reason': 'Replay fixture: exact row and metric at [L7].'}
    cands = bundle['candidates']
    metric = next((cid for cid, c in cands.items() if c['header'] == 'Mean KLD'), None)
    size = next((cid for cid, c in cands.items() if c['header'] == 'Size (GB)'), None)
    role = 'quoted' if 'disagree' in bundle['source']['repo'] and bundle['run'] == 'B' else 'own'
    return {'repo': bundle['source']['repo'], 'source_sha256': bundle['source']['sha256'], 'candidates_sha256': bundle['candidates_sha256'],
            'no_quality_claims': not bool(metric), 'no_claims_reason': '' if metric else 'No text values in the frozen fixture.',
            'method': {'reference': 'BF16', 'dataset': 'Wiki', 'ctx': '2048', 'tool': 'llama.cpp', 'L': [2, 4]},
            'entries': ([{'role': role, 'name': 'Q4_K_M', 'panel': 'Full vocabulary', 'metrics': {'mean_kld': metric},
                         'size': size, 'unit': 'GB', 'basis': 'disk'}] if metric else []), 'prose': []}


def semantic_output(root):
    return {'claims': runner.records(root), 'review_queue': read(root / 'audit/review_queue.json'),
            'mapped': read(root / 'maps/mapped.json')}


def legacy_shadow(w):
    """Existing safe stage scripts, never qcl_next/qa/scan or publication."""
    slug = 'fixture'; root = w.model(slug); work = w.path / 'queues/fixture'
    for script, args in (('autoextract.py', ('triage',)), ('detect_copies.py', ()), ('qcl_mapset.py', ('--record',)), ('qcl_queue.py', (work,))):
        w.run(slug, script, *args)
    for p in (root / 'maps/cands').glob('*.cands.json'):
        task = read(p)
        if not task['candidates']:
            continue  # preserve the existing deterministic empty-map path
        for run in 'AB':
            bundle = {'kind': 'mapping', 'source': {'repo': task['repo'], 'sha256': task['source_sha256']},
                      'run': run, 'candidates': task['candidates'], 'candidates_sha256': task['candidates_sha256']}
            write(root / 'maps' / run / p.name.replace('.cands.', '.map.'), fake_response(bundle))
    w.run(slug, 'qcl_mapset.py', '--record')
    w.run(slug, 'qcl_watch.py', 'merge'); w.run(slug, 'gen_auto.py')
    w.run(slug, 'qcl_tiebreak.py', 'queue', work, 1000)
    for p in (root / 'maps/C').glob('*.cands.json'):
        task = read(p)
        bundle = {'kind': 'mapping', 'source': {'repo': task['repo'], 'sha256': task['source_sha256']},
                  'run': 'C', 'candidates': task['candidates'], 'candidates_sha256': task['candidates_sha256']}
        write(p.with_name(p.name.replace('.cands.', '.map.')), fake_response(bundle))
        (work / 'claimed').mkdir(exist_ok=True)
        (work / 'claimed/C00.txt').write_text(task['repo'].replace('/', '__') + '\n')
    w.run(slug, 'qcl_tiebreak.py', 'record', work)
    w.run(slug, 'qcl_watch.py', 'merge'); w.run(slug, 'gen_auto.py')
    return semantic_output(root)


def qa_replay(out):
    from deploy_state import assess_reviews
    from qcl_qa import entries
    from runner_model import prepare_review
    items = [{'id': 'one'}, {'id': 'two'}]
    verdict = {'id': 'one', 'verdict': 'ok', 'reason': 'Fixture evidence [L7].'}
    assessed = assess_reviews(items, 2, [[verdict]])
    state = State(out / 'qa.sqlite')
    for item in items:
        state.enqueue(item['id'], 'fixture', 'review', 'v1', item)
    c = state.claim(); state.complete(c['id'], c['token'], verdict)
    next_claim = State(out / 'qa.sqlite').claim()
    old = "R=[{'repo':'Publisher/normal-GGUF','variants':[{'name':'Q4','m':{'mean_kld':'0.1250'},'L':7,'panel':'Old panel'}]}]"
    new = old.replace('Old panel', 'New panel')
    old_entries, new_entries = entries(old), entries(new)
    legacy_detected = any(old_entries.get(k, (None,))[0] != v[0] for k, v in new_entries.items())
    source = {'repo': 'Publisher/normal-GGUF', 'sha': 'a' * 40, 'sha256': hashlib.sha256(CARD.encode()).hexdigest()}
    claim = {'repo': source['repo'], 'entry': 'Q4', 'kind': 'own', 'key': 'mean_kld', 'raw': '0.1250', 'lines': [7], 'panel': 'Old panel'}
    a = prepare_review('Base/Model', source, CARD.encode(), claim, mapping_version='v1', extractor_version='v1')
    b = prepare_review('Base/Model', source, CARD.encode(), {**claim, 'panel': 'New panel'}, mapping_version='v1', extractor_version='v1')
    return {'legacy_partial_review_ready': assessed['ready'], 'legacy_missing': assessed['counts']['missing'],
            'legacy_completed_reviews_repeated_on_new_session': 1,
            'legacy_repeat_basis': 'qcl_qa stops done on missing verdict; fresh session samples both frozen items again',
            'scripted_completed_reviews_repeated': 0, 'scripted_remaining_claim': next_claim['payload']['id'],
            'unchanged_number_changed_panel_detected_legacy': legacy_detected,
            'unchanged_number_changed_panel_detected_scripted': runner.content_identity(a) != runner.content_identity(b)}


def replay(out):
    out = Path(out).absolute()
    if out.exists() or out == CODE or CODE in out.parents:
        raise ValueError('replay output must be a new external directory')
    out.mkdir(parents=True)
    before = dispatch_baseline()
    state = State(out / 'schedule.sqlite')
    t0 = time.perf_counter()
    for i in range(12):
        for run in 'AB':
            state.enqueue(f'{i}:{run}', str(i), run, 'v1', {'model': i, 'run': run})
    max_running = 0
    while True:
        batch = [state.claim(limit=8) for _ in range(8)]
        batch = [b for b in batch if b]
        if not batch:
            break
        max_running = max(max_running, len(batch))
        for c in batch:
            state.complete(c['id'], c['token'], {'ok': True})
    after = {'logical_model_requests': 24, 'actual_agent_launches': 0, 'coordinator_handoffs': 0,
             'completed': len(state.tasks()), 'completed_repeated': 0, 'max_running': max_running,
             'runtime_seconds': time.perf_counter() - t0, 'tokens': None, 'cost': None}
    old = fixture(out / 'old')
    t0 = time.perf_counter(); expected = legacy_shadow(old); old_time = time.perf_counter() - t0
    new = fixture(out / 'new'); state = State(new.path / 'state.sqlite')
    t0 = time.perf_counter(); runner.reconcile(new, state)
    requests = 0
    while True:
        c = state.claim(limit=3)
        if c is None:
            previous = len(state.tasks()); runner.reconcile(new, state)
            if len(state.tasks()) == previous:
                break
            continue
        runner.complete(new, state, c, fake_response(c['payload'])); requests += 1
    runner.reconcile(new, state)
    new_time = time.perf_counter() - t0
    actual = semantic_output(new.model('fixture'))
    write(out / 'old-output.json', expected); write(out / 'new-output.json', actual)
    if expected != actual:
        raise AssertionError('claim/provenance shadow difference; inspect output JSON')
    ops_before = len((new.path / 'operations.jsonl').read_text().splitlines())
    runner.reconcile(new, State(new.path / 'state.sqlite'))
    repeated = len((new.path / 'operations.jsonl').read_text().splitlines()) - ops_before
    if repeated:
        raise AssertionError(f'unchanged restart repeated {repeated} stage operations')
    summary = {'measurement': 'offline synthetic recorded responses; no actual model calls', 'qa_replay': qa_replay(out), 'legacy_dispatch': before,
               'scripted_dispatch': after, 'shadow': {'equal': True, 'logical_requests': requests,
               'accepted_metric_claims': sum(len(e.get('m', {})) for r in actual['claims'] for g in ('variants', 'ladder', 'cross') for e in r[g]),
               'held': len(actual['review_queue']['disagreements']), 'unresolved_tasks': len(runner.status(new, state)['unresolved']),
               'old_stage_seconds': old_time, 'new_stage_seconds': new_time, 'restart_repeated_stage_operations': repeated,
               'output_sha256': digest_output(actual)}, 'historical': historical(), 'stretch_target': '100x runtime; not achieved or claimed'}
    write(out / 'results.json', summary)
    return summary


def digest_output(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', required=True)
    a = ap.parse_args(); print(json.dumps(replay(a.out), indent=2))
