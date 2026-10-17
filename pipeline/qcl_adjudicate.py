#!/usr/bin/env python3
"""Compare a mapping run with the verified data and classify every disagreement mechanically.
  python3 qcl_adjudicate.py MAPDIR [MAPDIR2]   (two dirs = two independent runs: also reports their agreement)"""
import json, glob, sys, collections, qcl_map as Q, autoextract as A
CORE = {"kld", "top1", "ppl", "ppl_delta"}
def produced(d):
    out = {}; meta = {}
    for p in glob.glob(d + "/*.map.json"):
        m = json.load(open(p)); c = json.load(open(p.replace(".map.json", ".cands.json")))
        for e in m.get("entries", []):
            for k, cid in (e.get("metrics") or {}).items():
                cc = c.get(str(cid)); f = A.gt_family(k)
                if not cc or f not in CORE: continue
                t = A.numtoks(cc["raw"])
                if not t: continue
                key = (m["repo"], f, t[0]); out[key] = out.get(key, 0) + 1
                meta[key] = (cc, k, e.get("role"))
    return collections.Counter(out), meta
R, gt = A.ground_truth()
runs = [produced(d) for d in sys.argv[1:]]
repos = {k[0] for P, _ in runs for k in P}
G = collections.Counter((g["repo"], g["family"], g["tok"]) for g in gt if g["repo"] in repos and g["family"] in CORE)
for (P, meta), d in zip(runs, sys.argv[1:]):
    tp = sum((G & P).values()); extra = P - G
    ok = sum(n for k, n in extra.items() if A.header_family(meta[k][0]["header"]) == k[1] or A.header_family(meta[k][0]["label"]) == k[1])
    print(f"{d.split('/')[-1]}: recall {tp/sum(G.values()):.1%}, raw precision {tp/sum(P.values()):.1%}; "
          f"of {sum(extra.values())} extras, {ok} sit under a header/label of the same metric (likely correct values the ground truth omits)"
          f" -> adjusted precision {(tp+ok)/sum(P.values()):.1%}")
if len(runs) == 2:
    (P1, _), (P2, _) = runs; U = P1 | P2; I = P1 & P2
    tpu = sum((G & U).values())
    print(f"two independent runs: agree on {sum(I.values())} of {sum(U.values())} values ({sum(I.values())/sum(U.values()):.1%});"
          f" union recall {tpu/sum(G.values()):.1%}; disagreements to resolve: {sum((U - I).values())}")
