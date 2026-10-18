#!/usr/bin/env python3
"""Turn a fetched card cache into a model data root the pipeline reads (QCL_ROOT=models/<slug>).

  python3 qcl_newmodel.py models/<slug>

Merges cache/index*.json + cache/catalogue.json into snapshots/index.json (paths relative to the data root,
preserving direct-fetch commit SHAs; connector cards use "main" and every card retains its exact-byte SHA-256),
and creates empty stubs for model-specific inputs that this model does not have (sizes listing, branch sizes)."""
import glob, hashlib, json, os, sys, time
R = sys.argv[1]
def _card_path(filename):
    """Connector paths are cache-relative; qcl_fetch paths include its --out directory."""
    candidates = [filename] if os.path.isabs(filename) else [os.path.join(R, "cache", filename), filename, os.path.join(R, filename)]
    for p in candidates:
        if os.path.isfile(p):
            if os.path.commonpath([os.path.abspath(R), os.path.abspath(p)]) != os.path.abspath(R):
                raise ValueError(f"card is outside model root: {filename}")
            return os.path.relpath(p, R)
    raise FileNotFoundError(f"cached card not found: {filename}")
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
cached_base = None
_prev = json.load(open(f"{R}/snapshots/index.json")) if os.path.exists(f"{R}/snapshots/index.json") else {}
repos.update(_prev.get("repos", {}))   # keep what is already pinned; cache entries add or replace
_fetched = {}                                     # newest fetch of each repo across every shard's index file
for f in sorted(glob.glob(f"{R}/cache/index*.json")):
    cached = json.load(open(f))
    if cached.get("base"): cached_base = cached["base"]
    for rid, m in cached.get("repos", {}).items():
        if rid not in _fetched or (m.get("fetched_at") or "") >= (_fetched[rid].get("fetched_at") or ""): _fetched[rid] = m
for rid, m in _fetched.items():
    c, old = cat.get(rid, {}), repos.get(rid, {})
    path = _card_path(m["file"]) if m.get("file") else None
    card = open(os.path.join(R, path), "rb").read() if path else b""
    if hashlib.sha256(card).hexdigest() != m["sha256"] or m.get("bytes", len(card)) != len(card):
        raise ValueError(f"cached card bytes do not match ledger: {rid}")
    direct = "readme_status" in m
    # lastModified records the Hub time the saved card is known to be current for. A direct fetch records it itself;
    # for a connector fetch it moves to the listing's updated_at only when the card was fetched after that update,
    # otherwise a failed refetch would hide the change from the next scan forever.
    up = c.get("updated_at") or ""
    lm = m.get("lastModified") or (up if (m.get("fetched_at") or "") >= up[:19] else (old.get("lastModified") if old.get("sha256") == m["sha256"] else "")) or ""
    repos[rid] = {**m, "sha": m.get("sha") or "main", "file": path, "sha256": m["sha256"], "bytes": len(card),
                  "lastModified": lm, "createdAt": m.get("createdAt") or old.get("createdAt") or "",
                  "downloads": m.get("downloads", c.get("downloads", old.get("downloads"))), "likes": m.get("likes", c.get("likes", old.get("likes"))),
                  "tags": m.get("tags") or old.get("tags") or [], "fetched_at": m.get("fetched_at"), "via": m.get("via") or ("hf-api" if direct else "hf-connector"),
                  "no_readme": bool(m.get("no_readme") or (direct and m["readme_status"] != 200))}
    if m.get("tree_file"): repos[rid]["tree_file"] = _card_path(m["tree_file"])
for rid, m in repos.items():                     # a repo with no README gets an empty card file so every stage can read it
    if not m["file"]:
        os.makedirs(f"{R}/cache/cards", exist_ok=True)
        p = f"{R}/cache/cards/{rid.replace('/', '__')}.md"; open(p, "w").close(); m["file"] = os.path.relpath(p, R)
os.makedirs(f"{R}/snapshots", exist_ok=True); os.makedirs(f"{R}/verified", exist_ok=True); os.makedirs(f"{R}/audit", exist_ok=True)
missing = sorted(set(cat) - set(repos))
model = json.load(open(f"{R}/model.json"))["model"]
base = _prev.get("base")
base_path = f"cache/cards/_base__{model.replace('/', '__')}.md"
if cached_base:
    base_path = _card_path(cached_base["file"])
    card = open(os.path.join(R, base_path), "rb").read()
    if hashlib.sha256(card).hexdigest() != cached_base["sha256"] or cached_base.get("bytes", len(card)) != len(card):
        raise ValueError("cached base card bytes do not match ledger")
    base = {**cached_base, "file": base_path, "bytes": len(card)}
elif os.path.isfile(os.path.join(R, base_path)):    # older qcl_fetch caches captured the base only at main
    card = open(os.path.join(R, base_path), "rb").read()
    base = {"repo": model, "sha": "main", "file": base_path, "sha256": hashlib.sha256(card).hexdigest(),
            "bytes": len(card), "via": "hf-api"}
_out = {k: v for k, v in _prev.items() if k not in ("repos", "base")}                 # keep other top-level fields (e.g. "source")
_out.update({"captured_at": _prev.get("captured_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),   # first capture; stays fixed
             "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": model, "repos": repos, **({"base": base} if base else {})})
if "source" not in _out: _out.setdefault("pinning", "sha256 of card bytes; commit sha when available")   # commit-pinned roots keep "source"
json.dump(_out, open(f"{R}/snapshots/index.json", "w"), indent=1)
for stub in ("hf_sizes_summary.json", "hf_branch_sizes.json"):
    if not os.path.exists(f"{R}/snapshots/{stub}"): json.dump({}, open(f"{R}/snapshots/{stub}", "w"))
if not os.path.exists(f"{R}/sets.py"):
    ns = {}; exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qwen3.8-flash-next", "sets.py")).read(), ns)
    open(f"{R}/sets.py", "w").write("# No curated sets yet: every set is assembled automatically from one card.\nSETS = []\nMETRICS = " + repr(ns["METRICS"]) + "\n")
print(json.dumps({"repos_in_index": len(repos), "catalogue": len(cat), "not_fetched": len(missing)}))
json.dump(missing, open(f"{R}/cache/not_fetched.json", "w"))
