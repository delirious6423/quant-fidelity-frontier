#!/usr/bin/env python3
"""Mechanical copied-card detection: for every card, the share of its numeric claim lines that also appear verbatim
in ONE card from a DIFFERENT owner whose repo was CREATED earlier. The source is the older card with the largest
overlap (ties go to the oldest). Writes audit/copies.json; build.py treats share >= 0.6 (with >= 4 lines) as copied."""
import json, os, re, collections, sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]   # sibling modules
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")          # data root: one folder per model (models/<slug>/); code stays here
idx = json.load(open(f"{ROOT}/snapshots/index.json"))["repos"]
BASE = json.load(open(f"{ROOT}/model.json")).get("model", "Qwen/Qwen3.8-Flash-Next") if os.path.exists(f"{ROOT}/model.json") else "Qwen/Qwen3.8-Flash-Next"

def claim_lines(txt):
    out = set()
    for l in txt.split("\n"):
        s = re.sub(r"\s+", " ", l.replace("**", "").replace("`", "")).strip()
        if len(s) < 12 or not re.search(r"\d\.\d|\d%", s): continue
        if s.startswith("<") or "huggingface.co/" in s or s.startswith("hf download") or s.startswith("--"): continue
        out.add(s)
    return out

cards = {}
for rid, m in idx.items():
    try: cards[rid] = (m.get("createdAt") or m.get("lastModified") or "", claim_lines(open(f"{ROOT}/{m['file']}", encoding="utf-8", errors="replace").read()))
    except FileNotFoundError: pass
# Qwen's own base card lines are excluded as "copying" sources: reproducing the base model table is handled separately
base_lines = set()
for rid, (_, L) in cards.items():
    if rid.startswith("Qwen/"): base_lines |= L
_b = json.load(open(f"{ROOT}/snapshots/index.json")).get("base")
if _b: base_lines |= claim_lines(open(f"{ROOT}/{_b['file']}", encoding="utf-8", errors="replace").read())
where = collections.defaultdict(list)
for rid, (t, L) in cards.items():
    for s in L: where[s].append((t, rid))
res = {}
for rid, (t, L) in cards.items():
    own = rid.split("/")[0]
    L2 = [s for s in L if s not in base_lines]
    if len(L2) < 4: continue
    # overlap with every older card from another owner; the source is the card sharing the MOST claim lines
    # (ties -> the oldest), and the reported overlap is with that one card (external review F12: attributing each
    # line to its oldest holder named a partial match instead of the near-verbatim mirror)
    src = collections.Counter()
    for s in L2:
        for r in {r for (tt, r) in where[s] if r.split("/")[0] != own and tt < t}:
            src[r] += 1
    if not src: continue
    top = min(src, key=lambda r: (-src[r], cards[r][0], r))
    n = src[top]
    share = n / len(L2)
    if share >= 0.3:
        res[rid] = {"source": top, "shared_lines": n, "claim_lines": len(L2), "share": round(share, 2)}
json.dump(res, open(f"{ROOT}/audit/copies.json", "w"), indent=1)
for r, v in sorted(res.items(), key=lambda x: -x[1]["share"]):
    print(f"{v['share']:.2f}  {r:70} <- {v['source']} ({v['shared_lines']}/{v['claim_lines']})")
