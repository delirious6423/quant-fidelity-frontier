#!/usr/bin/env python3
"""git merge driver for watchlist.json, so the live-update and new-model tasks can both commit it.

  git config merge.wljson.driver "python3 pipeline/merge_watchlist.py %O %A %B"   # .gitattributes: watchlist.json merge=wljson

Per model: the furthest status wins (live > ingesting > queued), and the latest last_scan_at; last_ingest_at is the
latest of both. Other fields come from ours (%A), with theirs filling gaps. Writes the result to %A."""
import json, sys
O, A, B = sys.argv[1:4]
def ld(p):
    try: return json.load(open(p))
    except Exception: return None
a, b = ld(A), ld(B)
if a is None or b is None: sys.exit(1)          # not valid JSON on one side: let git report a conflict
RANK = {"queued": 0, "ingesting": 1, "live": 2}
bm = {m["slug"]: m for m in b.get("models", [])}
out = []
for m in a.get("models", []):
    o = bm.pop(m["slug"], {})
    r = {**o, **m}
    if RANK.get(o.get("status"), -1) > RANK.get(m.get("status"), -1): r["status"] = o["status"]
    ts = [x for x in (m.get("last_scan_at"), o.get("last_scan_at")) if x]
    if ts: r["last_scan_at"] = max(ts)
    out.append(r)
out += list(bm.values())                          # models only theirs knows about
a["models"] = out
li = [x for x in (a.get("last_ingest_at"), b.get("last_ingest_at")) if x]
if li: a["last_ingest_at"] = max(li)
json.dump(a, open(A, "w"), indent=1)
