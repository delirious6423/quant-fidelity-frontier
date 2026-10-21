#!/usr/bin/env python3
"""The portable loop (pipeline/qcl_portable.py) runs a whole model from a clean workspace to a verified shadow build with
nothing but files and commands: start exports requests, every answered round is absorbed by `next`, a rejected answer
comes back as a `fix` action with the retry command, and the final `stop` carries the comparison with the committed data.
Interpretation here is the replay fixture (synthetic, no model call); the repository's models/ directory stays untouched."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / 'pipeline'))
from runner_replay import fake_response

MODEL = 'lfm2.5-24b-a2b'


def portable(*args):
    p = subprocess.run([sys.executable, str(CODE / 'pipeline/qcl_portable.py'), *args], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    return json.loads(p.stdout)


def answer(action, broken=None):
    for path in Path(action['requests']).glob('*.json'):
        doc = json.loads(path.read_text())
        response = fake_response(doc['request'])
        if broken and path.stem == broken:
            response = dict(response, entries=[{'role': 'own', 'name': 'x', 'metrics': {'mean_kld': 'c999'}}], no_quality_claims=False)
        (Path(action['responses']) / path.name).write_text(json.dumps(response))


before = subprocess.run(['git', 'status', '--porcelain', '--', 'models'], cwd=CODE, capture_output=True, text=True).stdout
with tempfile.TemporaryDirectory(prefix='qff-portable-') as tmp:
    ws = str(Path(tmp) / 'ws')
    inside = portable('--workspace', str(CODE / 'work' / 'nope'), 'status')
    assert inside['do'] == 'stop' and 'outside the repository' in inside['error'], inside

    a = portable('--workspace', ws, 'start', '--models', MODEL, '--remap')
    assert a['do'] == 'answer' and a['count'] > 0 and a['by_stage'].get('map-A') == a['by_stage'].get('map-B') == a['count'] // 2, a
    assert 'never with a number you typed' in a['how'] and a['then'].endswith(f'next --workspace {ws}')
    first = sorted(Path(a['requests']).glob('*.json'))[0].stem
    answer(a, broken=first)                                  # one answer names a candidate that does not exist
    b = portable('--workspace', ws, 'next')
    assert b['do'] in ('answer', 'fix'), b
    # Everything valid was absorbed; the broken one is blocked and comes back as a fix action with its retry command.
    rounds = 0
    while b['do'] == 'answer':
        answer(b); b = portable('--workspace', ws, 'next'); rounds += 1
        assert rounds < 8, b
    assert b['do'] == 'fix' and [x['task'] for x in b['blocked']] == [first] and 'retry' in b['blocked'][0]['retry'], b
    doc = json.loads((Path(ws) / 'requests' / (first + '.json')).read_text())
    (Path(ws) / 'responses' / (first + '.json')).write_text(json.dumps(fake_response(doc['request'])))
    r = subprocess.run([sys.executable, str(CODE / 'pipeline/qcl_runner.py'), '--workspace', ws, 'retry', first, '--reason', 'fixture replaced the invalid candidate id'],
                       capture_output=True, text=True); assert r.returncode == 0, r.stderr[-800:]
    c = portable('--workspace', ws, 'next')
    while c['do'] == 'answer':
        answer(c); c = portable('--workspace', ws, 'next'); rounds += 1
        assert rounds < 12, c
    assert c['do'] == 'stop', c
    rep = c['report']; cmp_ = rep['models'][MODEL]['versus_committed']
    assert rep['tasks_completed'] > 0 and os.path.exists(rep['shadow_verification']) and os.path.isdir(rep['models'][MODEL]['built']), rep
    assert {'precision_vs_committed', 'recall_of_committed', 'identical', 'only_in_trial', 'only_in_committed'} <= set(cmp_), cmp_
    assert cmp_['same_repositories'] is True and cmp_['committed_values'] > 0, cmp_
    s = portable('--workspace', ws, 'status'); assert s['waiting'] == 0 and s['blocked'] == 0 and s['done'] == rep['tasks_completed'], s
    assert os.path.isfile(str(Path(ws) / 'responses' / (first + '.json')))
after = subprocess.run(['git', 'status', '--porcelain', '--', 'models'], cwd=CODE, capture_output=True, text=True).stdout
assert before == after, 'the portable loop must not touch the repository data'
print(f"portable loop ok: {rep['tasks_completed']} tasks, {rounds} answered rounds, recall of committed values {cmp_['recall_of_committed']} (fixture answers)")
