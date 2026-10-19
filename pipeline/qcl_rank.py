#!/usr/bin/env python3
"""Order the queued models in watchlist.json by a weighted score: how much there is to compare, and how new it is.

  python3 pipeline/qcl_rank.py            # print the ranking
  python3 pipeline/qcl_rank.py --write    # also reorder the queued models in watchlist.json (live/ingesting untouched)

Each queued entry carries its survey inputs (written by the discovery survey, never typed by hand):
  quants      quant repos tagged base_model:quantized:<model> (the Hub search counts at most 80, so 80 means 80+)
  quants_1k   of those, repos with 1,000+ downloads (quants people actually use)
  released    approximate release date (YYYY-MM-DD or YYYY-MM)

  score = demand x freshness x early
  demand    = quants + 2 x quants_1k               a used quant counts three times
  freshness = 0.5 ** (age_days / HALF_LIFE)        loses half its weight every 90 days
  early     = min(2, sqrt(14 / age_days))          under two weeks old: its quant count is still growing, so project it
                                                   (at most x2); 1 from day 14 on"""
import datetime as dt, json, math, os, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WL = os.path.join(CODE, "watchlist.json")
HALF_LIFE, EARLY_DAYS, EARLY_MAX = 90, 14, 2.0

def age_days(released, today):
    d = dt.date.fromisoformat(released if len(released) == 10 else released + "-15")   # month only: mid-month
    return max(1, (today - d).days)

def score(m, today):
    a = age_days(m["released"], today)
    demand = m["quants"] + 2 * m["quants_1k"]
    return demand * 0.5 ** (a / HALF_LIFE) * min(EARLY_MAX, max(1.0, math.sqrt(EARLY_DAYS / a))), demand, a

w = json.load(open(WL)); today = dt.date.today()
q = [m for m in w["models"] if m["status"] == "queued"]
missing = [m["slug"] for m in q if not all(k in m for k in ("quants", "quants_1k", "released"))]
ranked = sorted((m for m in q if m["slug"] not in missing), key=lambda m: -score(m, today)[0]) + [m for m in q if m["slug"] in missing]
for i, m in enumerate(ranked, 1):
    if m["slug"] in missing: print(f"{i:2}  {m['model']:48} no survey inputs; ranked last"); continue
    s, d, a = score(m, today)
    print(f"{i:2}  {m['model']:48} score {s:6.1f}  demand {d:3}  ({m['quants']}{'+' if m['quants'] >= 80 else ''} quants, {m['quants_1k']} used)  age {a} d")
if "--write" in sys.argv:
    w["models"] = [m for m in w["models"] if m["status"] != "queued"] + ranked
    json.dump(w, open(WL, "w"), indent=1); open(WL, "a").write("\n")
