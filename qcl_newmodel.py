#!/usr/bin/env python3
"""Turn a connector-fetched card cache into a model data root the pipeline reads (QCL_ROOT=models/<slug>).

  python3 qcl_newmodel.py models/<slug>

Merges cache/index_*.json + cache/catalogue.json into snapshots/index.json (paths relative to the data root,
sha "main" because the connector exposes no commit sha; every card is pinned by the SHA-256 of its exact bytes),
and creates empty stubs for model-specific inputs that this model does not have (sizes listing, branch sizes)."""
import glob, json, os, sys, time
R = sys.argv[1]
cat = {e["path"]: e for e in json.load(open(f"{R}/cache/catalogue.json"))}
repos = {}
for f in sorted(glob.glob(f"{R}/cache/index*.json")):
    for rid, m in json.load(open(f)).get("repos", {}).items():
        c = cat.get(rid, {})
        repos[rid] = {"sha": "main", "file": ("cache/" + m["file"]) if m.get("file") else None, "sha256": m["sha256"], "bytes": m["bytes"],
                      "lastModified": c.get("updated_at") or "", "createdAt": "", "downloads": c.get("downloads"), "likes": c.get("likes"),
                      "tags": [], "fetched_at": m.get("fetched_at"), "via": "hf-connector", "no_readme": bool(m.get("no_readme"))}
for rid, m in repos.items():                     # a repo with no README gets an empty card file so every stage can read it
    if not m["file"]:
        p = f"{R}/cache/cards/{rid.replace('/', '__')}.md"; open(p, "w").close(); m["file"] = os.path.relpath(p, R)
os.makedirs(f"{R}/snapshots", exist_ok=True); os.makedirs(f"{R}/verified", exist_ok=True); os.makedirs(f"{R}/audit", exist_ok=True)
missing = sorted(set(cat) - set(repos))
json.dump({"captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": json.load(open(f"{R}/model.json"))["model"],
           "pinning": "sha256 of card bytes (connector; no commit sha)", "repos": repos}, open(f"{R}/snapshots/index.json", "w"), indent=1)
for stub in ("hf_sizes_summary.json", "hf_branch_sizes.json"):
    if not os.path.exists(f"{R}/snapshots/{stub}"): json.dump({}, open(f"{R}/snapshots/{stub}", "w"))
if not os.path.exists(f"{R}/sets.py"):
    ns = {}; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sets.py")).read(), ns)
    open(f"{R}/sets.py", "w").write("# No curated sets yet: every set is assembled automatically from one card.\nSETS = []\nMETRICS = " + repr(ns["METRICS"]) + "\n")
print(json.dumps({"repos_in_index": len(repos), "catalogue": len(cat), "not_fetched": len(missing)}))
json.dump(missing, open(f"{R}/cache/not_fetched.json", "w"))
