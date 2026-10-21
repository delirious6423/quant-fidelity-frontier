#!/usr/bin/env python3
"""Print the data root of every model that is published (status "live" in watchlist.json; a model folder that is not
in the watch list counts as live). Models still being ingested are left out, so CI and Pages never build half a model.

  for d in $(python3 pipeline/qcl_live.py); do QCL_ROOT=$d python3 web/build.py; done"""
import glob, json, os
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st = {m["slug"]: m["status"] for m in json.load(open(os.path.join(CODE, "watchlist.json")))["models"]}
for mj in sorted(glob.glob(os.path.join(CODE, "models", "*", "model.json"))):
    d = os.path.dirname(mj); slug = json.load(open(mj)).get("slug") or os.path.basename(d)
    if st.get(slug, "live") == "live": print(os.path.relpath(d, CODE))
