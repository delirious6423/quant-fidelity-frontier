#!/usr/bin/env python3
"""Plan connector batches so every result is (a) within the connector's ~80 KB budget and (b) large enough (>= 52 KB)
to be saved to disk instead of returned inline. Cards are read with --max-bytes equal to their exact size.

  python3 qcl_plan.py SIZES.json OUT_PLAN.json [IDS.json]     # SIZES: {repo_id: size_bytes or 0 for no README}
Writes [{"ops": [...], "repos": [...]}]; each ops list is passed verbatim as hf_fs operations."""
import json, sys
LO, HI, CH, PAD = 54_000, 50_000, 50_000, ("hf://README.md", 8737)
MAXOPS = 30                                    # hf_fs hard limit per call, padding included
sizes = json.load(open(sys.argv[1]))
if len(sys.argv) > 3:                          # optional IDS.json: plan only these ids (a reused sizes file can hold other shards)
    keep = set(json.load(open(sys.argv[3]))); sizes = {r: s for r, s in sizes.items() if r in keep}
small = sorted([(r, s) for r, s in sizes.items() if 0 < s <= HI], key=lambda x: -x[1])
big = [(r, s) for r, s in sizes.items() if s > HI]
none = [r for r, s in sizes.items() if not s]
plans = []
def op_stat(r): return {"cmd": "stat", "args": [f"hf://models/{r}/README.md"]}
def op_cat(r, n, off=0):
    a = [f"hf://models/{r}/README.md", "--max-bytes", str(n)]
    if off: a += ["--offset", str(off)]
    return {"cmd": "cat", "args": a}
def pad(ops, total):
    while total < LO:
        ops.append({"cmd": "cat", "args": [PAD[0], "--max-bytes", str(PAD[1])]}); total += PAD[1]
    return ops
# first-fit decreasing into bins of [LO, HI]
bins = []
for r, s in small:
    for b in bins:
        if b["t"] + s <= HI and len(b["ops"]) + 2 + max(0, -(-(LO - b["t"] - s) // PAD[1])) <= MAXOPS: b["ops"] += [op_stat(r), op_cat(r, s)]; b["t"] += s; b["repos"].append(r); break
    else: bins.append({"ops": [op_stat(r), op_cat(r, s)], "t": s, "repos": [r]})
for b in bins: plans.append({"ops": pad(b["ops"], b["t"]), "repos": b["repos"]})
for r, s in big:                              # one call per chunk; the saver stitches consecutive files of one repo
    off = 0; first = True
    while off < s:
        n = min(CH, s - off); ops = ([op_stat(r)] if first else []) + [op_cat(r, n, off)]
        plans.append({"ops": pad(ops, n), "repos": [r], "chunk_of": r}); off += n; first = False
for i in range(0, len(none), 20):              # repos whose README stat said missing: confirm with stat only (padded)
    plans.append({"ops": pad([op_stat(r) for r in none[i:i + 20]], 0), "repos": none[i:i + 20]})
assert all(len(p["ops"]) <= MAXOPS for p in plans), "op limit"
json.dump(plans, open(sys.argv[2], "w"))
print(json.dumps({"calls": len(plans), "cards": len(sizes), "big": len(big), "bytes": sum(sizes.values())}))
