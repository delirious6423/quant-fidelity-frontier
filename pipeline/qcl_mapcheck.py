#!/usr/bin/env python3
"""Check one claimed mapping queue: every name has a parseable map in the run folder, naming the right repo, and using
only candidate ids that exist for that card. Prints problems; exit 1 if any.

  python3 pipeline/qcl_mapcheck.py WORK/claimed/QUEUE.txt MODEL/maps/RUN"""
import json, os, re, sys
q, out = sys.argv[1], sys.argv[2]
cands_dir = os.path.join(os.path.dirname(out), "cands")
bad = []
for name in [l.strip() for l in open(q) if l.strip()]:
    p = os.path.join(out, name + ".map.json")
    if not os.path.exists(p): bad.append(f"{name}: missing"); continue
    try: m = json.load(open(p))
    except Exception as e: bad.append(f"{name}: not JSON ({e})"); continue
    if (m.get("repo") or "").replace("/", "__") != name: bad.append(f"{name}: repo field is {m.get('repo')!r}")
    cf = os.path.join(cands_dir, name + ".cands.json")
    ids = set(json.load(open(cf))) if os.path.exists(cf) else None
    if ids is None: bad.append(f"{name}: no candidates file"); continue
    used = set(re.findall(r'"(c\d+(?:\.\d+)?)"', json.dumps(m)))
    unknown = sorted(used - ids)
    if unknown: bad.append(f"{name}: unknown candidate ids {unknown[:5]}")
print("\n".join(bad) if bad else f"ok {q}")
sys.exit(1 if bad else 0)
