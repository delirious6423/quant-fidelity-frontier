#!/usr/bin/env python3
"""Check one claimed mapping queue: every name has a parseable map in the run folder, naming the right repo, and using
only candidate ids that exist for that card. Prints problems; exit 1 if any.

  python3 pipeline/qcl_mapcheck.py WORK/claimed/QUEUE.txt MODEL/maps/RUN"""
import json, os, re, sys
from pathlib import Path
import mapping_provenance as MP
q, out = sys.argv[1], sys.argv[2]
cands_dir = os.path.join(os.path.dirname(out), "cands")
root = Path(out).resolve().parents[1]
index = json.loads((root / "snapshots/index.json").read_text())["repos"]
legacy = MP.recorded_pins(root, Path(out).name)
bad = []
for name in [l.strip() for l in open(q) if l.strip()]:
    p = os.path.join(out, name + ".map.json")
    if not os.path.exists(p): bad.append(f"{name}: missing"); continue
    try: m = json.load(open(p))
    except Exception as e: bad.append(f"{name}: not JSON ({e})"); continue
    if (m.get("repo") or "").replace("/", "__") != name: bad.append(f"{name}: repo field is {m.get('repo')!r}")
    cf = MP.candidate_path(p)
    try:
        candidates = json.load(open(cf))
        MP.mapping_source(m, candidates, index, legacy, path=p)
        ids = set(MP.candidate_values(candidates))
    except (OSError, ValueError, KeyError, TypeError) as error:
        bad.append(f"{name}: invalid source provenance ({error})"); continue
    if ids is None: bad.append(f"{name}: no candidates file"); continue
    used = set(re.findall(r'"(c\d+(?:\.\d+)?)"', json.dumps(m)))
    unknown = sorted(used - ids)
    if unknown: bad.append(f"{name}: unknown candidate ids {unknown[:5]}")
print("\n".join(bad) if bad else f"ok {q}")
sys.exit(1 if bad else 0)
