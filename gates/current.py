"""Which verified entries still describe the card the index pins now.

A reviewed entry is pinned to the card it was read from (its `sha`, matched like verify.py: the index sha starts with
the entry's first 8 characters). When the hourly scan saves a newer card for that repo, the old claims are no longer on
the card the site links to. They are withdrawn (fail-closed), and the repo is listed as not yet reviewed until a new
mapping covers it, instead of stopping the whole build."""
import glob, json, os

STALE_FLAG = "Card changed after it was reviewed; its earlier claims are withdrawn until it is reviewed again"

def fresh(r, idx):
    m = idx.get(r["repo"])
    return bool(m) and str(m.get("sha", "")).startswith(str(r.get("sha", ""))[:8])

def current(R, idx):
    """R: entries from every verified/part*.py (each may carry _part). Returns the entries to use."""
    fresh_reviewed = {r["repo"] for r in R if not r.get("auto") and fresh(r, idx)}
    out, stale = [], {}
    for r in R:
        if r.get("auto"):   # no claims: an unreviewed listing simply follows the current card
            if r["repo"] not in fresh_reviewed: out.append(r if fresh(r, idx) or r["repo"] not in idx else {**r, "sha": idx[r["repo"]]["sha"]})
        elif fresh(r, idx): out.append(r)
        elif r["repo"] in idx: stale.setdefault(r["repo"], r)
    have = {r["repo"] for r in out}
    for rp, r in sorted(stale.items()):
        if rp in have: continue
        out.append({"repo": rp, "sha": idx[rp]["sha"], "fmt": r.get("fmt"), "runtime": r.get("runtime"), "weights_modified": r.get("weights_modified"),
                    "method": None, "methodology": None, "variants": [], "auto": True, "flags": [STALE_FLAG], "_part": "stale"})
    return out

def load(root):
    idx = json.load(open(f"{root}/snapshots/index.json"))["repos"]; R = []
    for p in sorted(glob.glob(f"{root}/verified/part*.py")):
        ns = {}; exec(open(p).read(), ns)
        for r in ns["R"]: r["_part"] = os.path.basename(p); R.append(r)
    return current(R, idx)
