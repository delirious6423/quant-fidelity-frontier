#!/usr/bin/env python3
"""Which cards still need an LLM mapping, after triage and copy detection.

  QCL_ROOT=models/<slug> python3 pipeline/qcl_mapset.py            # writes audit/mapset.json, prints the to-map list size
  QCL_ROOT=models/<slug> python3 pipeline/qcl_mapset.py --record   # after mapping: record which card bytes now have both maps
  QCL_ROOT=models/<slug> python3 pipeline/qcl_mapset.py --baseline # once, for a hand-reviewed model: its current card bytes count as mapped

A card needs mapping when triage puts it in wave 1 (core metrics), it is not a copy of another card (byte-identical
cards map once, through their oldest repo; >= 60% copied cards are labelled as copies), and its current bytes have no
mapping yet (maps/A and maps/B record the card SHA-256 they were made from in maps/mapped.json)."""
import collections, glob, json, os, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")
t = json.load(open(f"{R}/audit/triage.json")); idx = json.load(open(f"{R}/snapshots/index.json"))["repos"]
cp = json.load(open(f"{R}/audit/copies.json")) if os.path.exists(f"{R}/audit/copies.json") else {}
mp = f"{R}/maps/mapped.json"; mapped = json.load(open(mp)) if os.path.exists(mp) else {}
def _repo_of(f):
    try: return json.load(open(f)).get("repo")
    except Exception: return None                  # unreadable or half-written map: not recorded, so the card is queued again
if "--record" in sys.argv:   # after a mapping run (or a run that stopped early): a card counts as mapped once both runs have a readable map for it
    for f in glob.glob(f"{R}/maps/A/*.map.json"):
        rp = _repo_of(f); fb = f.replace("/A/", "/B/")
        if rp in idx and os.path.exists(fb) and _repo_of(fb) == rp: mapped[rp] = idx[rp]["sha256"]
    os.makedirs(os.path.dirname(mp), exist_ok=True)   # a new model has no maps/ yet
    json.dump(mapped, open(mp, "w"), indent=1, sort_keys=True)
if "--baseline" in sys.argv:   # hand-reviewed models (Flash-Next): every card already reviewed counts as mapped at its current bytes
    for rp, m in idx.items(): mapped.setdefault(rp, m["sha256"])
    os.makedirs(os.path.dirname(mp), exist_ok=True); json.dump(mapped, open(mp, "w"), indent=1, sort_keys=True)
def copied(r): c = cp.get(r) or {}; return c.get("share", 0) >= 0.6 and c.get("claim_lines", 0) >= 4
by = collections.defaultdict(list)
for r in t["wave1"]: by[idx[r]["sha256"]].append(r)
reps, dup = [], {}
for sha, rs in by.items():
    rs.sort(key=lambda r: (r not in mapped, idx[r].get("lastModified") or "z"))   # keep an already-mapped repo as the representative
    reps.append(rs[0]); dup.update({x: rs[0] for x in rs[1:]})
todo = sorted(r for r in reps if not copied(r) and mapped.get(r) != idx[r]["sha256"])
json.dump({"todo": todo, "copied": sorted(r for r in reps if copied(r)), "dup_of": dup}, open(f"{R}/audit/mapset.json", "w"), indent=1)
print(json.dumps({"wave1": len(t["wave1"]), "unique": len(reps), "already_mapped": sum(1 for r in reps if mapped.get(r) == idx[r]["sha256"]), "todo": len(todo)}))
