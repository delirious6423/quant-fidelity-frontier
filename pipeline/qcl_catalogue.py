#!/usr/bin/env python3
"""Hub catalogue and delta for one model, from saved hf_fs search results (no model ever retypes them).

  python3 pipeline/qcl_catalogue.py models/<slug> RESULT_FILE...   # merge search results, write cache/delta.json

Search with `search hf://models --tag base_model:quantized:<MODEL> --limit 1000` (pad with README cats so the result
is saved to a file). Writes cache/hub_catalogue.json (merged by repo id) and cache/delta.json: repos that are new or whose
Hub `updated_at` is newer than what the index holds, minus gated repos (their cards cannot be read)."""
import json, os, sys
R, files = sys.argv[1], sys.argv[2:]
os.makedirs(f"{R}/cache", exist_ok=True)
cp = f"{R}/cache/hub_catalogue.json"
cat = {e["path"]: e for e in (json.load(open(cp)) if os.path.exists(cp) else [])}
seen = 0
for f in files:
    for r in json.load(open(f)).get("results", []):
        res = r.get("result") or {}
        if res.get("op") != "search": continue
        for e in res.get("entries", []):
            if e.get("type") == "repo" and e.get("repo_type", "model") == "model":
                cat[e["path"]] = {k: e.get(k) for k in ("path", "updated_at", "downloads", "likes", "gated", "private")}; seen += 1
json.dump(sorted(cat.values(), key=lambda e: e["path"]), open(cp, "w"), indent=1)
ip = f"{R}/snapshots/index.json"
idx = json.load(open(ip))["repos"] if os.path.exists(ip) else {}
delta, gated = [], []
for rid, e in sorted(cat.items()):
    if e.get("gated"): gated.append(rid); continue
    have = idx.get(rid)
    if not have or (e.get("updated_at") or "") > (have.get("lastModified") or ""): delta.append(rid)
json.dump(delta, open(f"{R}/cache/delta.json", "w"), indent=1)
print(json.dumps({"search_entries": seen, "catalogue": len(cat), "in_index": len(idx), "delta": len(delta), "gated": len(gated)}))
