#!/usr/bin/env python3
"""Turn a connector-fetched card cache into a model data root the pipeline reads (QCL_ROOT=models/<slug>).

  python3 qcl_newmodel.py models/<slug>

Merges cache/index_*.json + cache/catalogue.json into snapshots/index.json (paths relative to the data root,
sha "main" because the connector exposes no commit sha; every card is pinned by the SHA-256 of its exact bytes),
and creates empty stubs for model-specific inputs that this model does not have (sizes listing, branch sizes)."""
import glob, json, os, sys, time
R = sys.argv[1]
def _load_cat():
    """Hub listing: hub_catalogue.json (from qcl_catalogue.py) or catalogue.json (list of {path, updated_at} or the API form {catalogue: [{id, lastModified}]})."""
    out = {}
    for p in (f"{R}/cache/catalogue.json", f"{R}/cache/hub_catalogue.json"):          # the hub listing wins when both exist
        if not os.path.exists(p): continue
        d = json.load(open(p)); d = d.get("catalogue", []) if isinstance(d, dict) else d
        for e in d:
            rid = e.get("path") or e.get("id")
            if rid: out[rid] = {"updated_at": e.get("updated_at") or e.get("lastModified"), "downloads": e.get("downloads"), "likes": e.get("likes")}
    return out
cat = _load_cat()
repos = {}
if os.path.exists(f"{R}/snapshots/index.json"):   # keep what is already pinned (e.g. commit-pinned snapshots); cache entries add or replace
    repos.update(json.load(open(f"{R}/snapshots/index.json")).get("repos", {}))
_fetched = {}                                     # newest fetch of each repo across every shard's index file
for f in sorted(glob.glob(f"{R}/cache/index*.json")):
    for rid, m in json.load(open(f)).get("repos", {}).items():
        if rid not in _fetched or (m.get("fetched_at") or "") >= (_fetched[rid].get("fetched_at") or ""): _fetched[rid] = m
for rid, m in _fetched.items():
    c, old = cat.get(rid, {}), repos.get(rid, {})
    if old.get("sha256") == m["sha256"] and old.get("fetched_at") == m.get("fetched_at") and old.get("sha") != "main": continue   # commit-pinned entry, unchanged
    # lastModified records the Hub time the saved card is known to be current for. It only moves to the listing's
    # updated_at when this card was fetched after that update; otherwise a failed refetch would hide the change forever.
    up = c.get("updated_at") or ""
    lm = up if (m.get("fetched_at") or "") >= up[:19] else (old.get("lastModified") if old.get("sha256") == m["sha256"] else "") or ""
    repos[rid] = {"sha": "main", "file": ("cache/" + m["file"]) if m.get("file") else None, "sha256": m["sha256"], "bytes": m["bytes"],
                  "lastModified": lm, "createdAt": "", "downloads": c.get("downloads", old.get("downloads")), "likes": c.get("likes", old.get("likes")),
                  "tags": old.get("tags") or [], "fetched_at": m.get("fetched_at"), "via": "hf-connector", "no_readme": bool(m.get("no_readme"))}
for rid, m in repos.items():                     # a repo with no README gets an empty card file so every stage can read it
    if not m["file"]:
        p = f"{R}/cache/cards/{rid.replace('/', '__')}.md"; open(p, "w").close(); m["file"] = os.path.relpath(p, R)
os.makedirs(f"{R}/snapshots", exist_ok=True); os.makedirs(f"{R}/verified", exist_ok=True); os.makedirs(f"{R}/audit", exist_ok=True)
missing = sorted(set(cat) - set(repos))
_prev = json.load(open(f"{R}/snapshots/index.json")) if os.path.exists(f"{R}/snapshots/index.json") else {}
_out = {k: v for k, v in _prev.items() if k != "repos"}                                # keep other top-level fields (e.g. "base")
_out.update({"captured_at": _prev.get("captured_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),   # first capture; stays fixed
             "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": json.load(open(f"{R}/model.json"))["model"],
             "repos": repos})
if "source" not in _out: _out.setdefault("pinning", "sha256 of card bytes (connector; no commit sha)")   # commit-pinned roots keep "source"
json.dump(_out, open(f"{R}/snapshots/index.json", "w"), indent=1)
for stub in ("hf_sizes_summary.json", "hf_branch_sizes.json"):
    if not os.path.exists(f"{R}/snapshots/{stub}"): json.dump({}, open(f"{R}/snapshots/{stub}", "w"))
if not os.path.exists(f"{R}/sets.py"):
    ns = {}; exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qwen3.8-flash-next", "sets.py")).read(), ns)
    open(f"{R}/sets.py", "w").write("# No curated sets yet: every set is assembled automatically from one card.\nSETS = []\nMETRICS = " + repr(ns["METRICS"]) + "\n")
print(json.dumps({"repos_in_index": len(repos), "catalogue": len(cat), "not_fetched": len(missing)}))
json.dump(missing, open(f"{R}/cache/not_fetched.json", "w"))
