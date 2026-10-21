#!/usr/bin/env python3
"""Value-level comparison of one model's merged claims in a shadow workspace against the committed claims of the same model.

  python3 pipeline/runner_compare.py WORKSPACE/models/SLUG models/SLUG

Both sides are read with qcl_runner.records (AST, no data code is executed). A value is (repo, group, entry name, metric
key, published string). The summary gives precision and recall of the trial relative to the committed data; the lists
show every difference so a reader can judge whether it is an error or a claim the committed data missed."""
import json, sys
from pathlib import Path
CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / 'pipeline'))


def values(root):
    from qcl_runner import records
    out = {}
    for r in records(Path(root)):
        for group in ('variants', 'ladder', 'cross'):
            for e in r.get(group) or []:
                name = str(e.get('name') or e.get('variant'))
                for k, v in (e.get('m') or {}).items():
                    out[(r['repo'], group, name, k)] = str(v)
                if e.get('size') is not None:
                    out[(r['repo'], group, name, 'size')] = f"{e['size']} {e.get('unit') or ''}".strip()
    return out


def compare(trial_root, prod_root):
    trial_root, prod_root = Path(trial_root), Path(prod_root)
    t, p = values(trial_root), values(prod_root)
    same = {k for k in t if k in p and t[k] == p[k]}
    differ = {k for k in t if k in p and t[k] != p[k]}
    only_trial, only_prod = set(t) - set(p), set(p) - set(t)
    tr = set(json.load(open(trial_root / 'snapshots/index.json'))['repos'])
    pr = set(json.load(open(prod_root / 'snapshots/index.json'))['repos'])
    row = lambda k, src: {'repo': k[0], 'group': k[1], 'entry': k[2], 'key': k[3], 'value': src[k]}
    return {'summary': {'trial_values': len(t), 'committed_values': len(p), 'identical': len(same), 'same_key_different_value': len(differ),
                        'only_in_trial': len(only_trial), 'only_in_committed': len(only_prod),
                        'precision_vs_committed': round(len(same) / max(1, len(t)), 4), 'recall_of_committed': round(len(same) / max(1, len(p)), 4),
                        'repos_with_values_trial': len({k[0] for k in t}), 'repos_with_values_committed': len({k[0] for k in p}),
                        'same_repositories': tr == pr},
            'same_key_different_value': [{**row(k, t), 'committed': p[k]} for k in sorted(differ)],
            'only_in_trial': [row(k, t) for k in sorted(only_trial)],
            'only_in_committed': [row(k, p) for k in sorted(only_prod)]}


if __name__ == '__main__':
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=1))
