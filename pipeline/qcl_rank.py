#!/usr/bin/env python3
"""Order the queued models in watchlist.json by a weighted score: how much there is to compare, and how new it is.

  python3 pipeline/qcl_rank.py            # print the ranking
  python3 pipeline/qcl_rank.py --write    # also reorder the queued models in watchlist.json (live/ingesting untouched)

Each queued entry carries its survey inputs (written by pipeline/qcl_discover.py, never typed by hand):
  quants      quant repos tagged base_model:quantized:<model> (union of several Hub search sorts, 80 each)
  quants_1k   of those, repos with 1,000+ downloads (quants people actually use)
  released    approximate release date (YYYY-MM-DD or YYYY-MM)

  score = demand x freshness x early                  (weights in discovery.json "ranking")
  demand    = quants + used_weight x quants_1k        a used quant counts extra
  freshness = 0.5 ** (age_days / half_life_days)      loses half its weight every half-life
  early     = min(early_max, sqrt(early_days / age))  very new: its quant count is still growing, so project it; 1 after"""
import datetime as dt, json, math, os, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WL = os.path.join(CODE, "watchlist.json")
_R = json.load(open(os.path.join(CODE, "discovery.json")))["ranking"]
HALF_LIFE, USED_W, EARLY_DAYS, EARLY_MAX = _R["half_life_days"], _R["used_weight"], _R["early_days"], _R["early_max"]

def age_days(released, today):
    d = dt.date.fromisoformat(released if len(released) == 10 else released + "-15")   # month only: mid-month
    return max(1, (today - d).days)

def score(m, today):
    a = age_days(m["released"], today)
    demand = m["quants"] + USED_W * m["quants_1k"]
    return demand * 0.5 ** (a / HALF_LIFE) * min(EARLY_MAX, max(1.0, math.sqrt(EARLY_DAYS / a))), demand, a

def rank(models, today=None):
    today = today or dt.date.today()
    has = lambda m: all(k in m for k in ("quants", "quants_1k", "released"))
    return sorted((m for m in models if has(m)), key=lambda m: -score(m, today)[0]) + [m for m in models if not has(m)]

if __name__ == "__main__":
    w = json.load(open(WL)); today = dt.date.today()
    ranked = rank([m for m in w["models"] if m["status"] == "queued"], today)
    for i, m in enumerate(ranked, 1):
        if "quants" not in m: print(f"{i:2}  {m['model']:48} no survey inputs; ranked last"); continue
        s, d, a = score(m, today)
        print(f"{i:2}  {m['model']:48} score {s:6.1f}  demand {d:3}  ({m['quants']} quants, {m['quants_1k']} used)  age {a} d")
    if "--write" in sys.argv:
        w["models"] = [m for m in w["models"] if m["status"] != "queued"] + ranked
        json.dump(w, open(WL, "w"), indent=1); open(WL, "a").write("\n")
