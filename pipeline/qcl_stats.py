#!/usr/bin/env python3
"""Card-size lookups through the connector, padded so results are saved to disk (no model transcribes anything).

  python3 qcl_stats.py ops IDS.json            # print the stat batches (JSON list of op lists), 23 stats + 6 pads each
  python3 qcl_stats.py parse SIZES.json RESULT_FILE...   # merge sizes from saved results into SIZES.json"""
import json, re, sys
PAD = {"cmd": "cat", "args": ["hf://README.md", "--max-bytes", "8737"]}
if sys.argv[1] == "ops":
    ids = json.load(open(sys.argv[2]))
    print(json.dumps([[{"cmd": "stat", "args": [f"hf://models/{r}/README.md"]} for r in ids[i:i + 23]] + [PAD] * 6 for i in range(0, len(ids), 23)]))
else:
    sp = sys.argv[2]
    try: sizes = json.load(open(sp))
    except Exception: sizes = {}
    n = 0
    for f in sys.argv[3:]:
        for r in json.load(open(f)).get("results", []):
            x = r.get("result") or {}
            m = re.match(r"hf://models/([^/]+/[^/]+)/README\.md$", x.get("uri") or "")
            if not m or x.get("op") != "stat": continue
            sizes[m.group(1)] = x.get("size") if (r.get("status") == "success" and x.get("exists") and x.get("type") == "file") else 0
            n += 1
    json.dump(sizes, open(sp, "w"))
    print(json.dumps({"parsed": n, "total": len(sizes), "missing_readme": sum(1 for v in sizes.values() if not v)}))
