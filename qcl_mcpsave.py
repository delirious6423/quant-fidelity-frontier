#!/usr/bin/env python3
"""Save model cards from a persisted Hugging Face connector result (hf_fs batch) byte-for-byte.

  python3 qcl_mcpsave.py OUT_DIR RESULT_FILE [RESULT_FILE ...]   (files in call order; chunks of one card may span files)

Batch convention (what the fetch agents send): for each card, consecutive ops
  [stat URI] (optional, gives the true size) , cat URI , cat URI --offset 20000 , cat URI --offset 40000 ...
Results come back in input order. Consecutive cat results of one URI are concatenated; a card is saved only if it is
complete (last chunk not truncated) and, when a stat was included, its UTF-8 length equals the stat size.
Prints JSON {"saved": n, "retry": [...], "need_chunks": {repo: next_offset}}. Nothing is ever retyped by a model."""
import hashlib, json, os, re, sys, time
out, files = sys.argv[1], sys.argv[2:]
os.makedirs(f"{out}/cards", exist_ok=True)
ip = os.environ.get("QCL_INDEX") or f"{out}/index.json"
idx = json.load(open(ip)) if os.path.exists(ip) else {"repos": {}}
d = {"results": [r for f in files for r in json.load(open(f)).get("results", [])]}
groups = []                         # [(rid, size_from_stat, [results...])]
for r in d.get("results", []):
    x = r.get("result") or {}
    m = re.match(r"hf://models/([^/]+/[^/]+)/README\.md$", x.get("uri") or "")
    if not m: continue              # padding or other ops
    rid = m.group(1)
    if not groups or groups[-1][0] != rid: groups.append([rid, None, []])
    if x.get("op") == "stat":
        if x.get("exists") is False or r.get("status") != "success": groups[-1][1] = 0
        else: groups[-1][1] = x.get("size")
    else: groups[-1][2].append(r)
saved, retry, need = 0, [], {}
for rid, size, rs in groups:
    if size == 0 and not rs:
        idx["repos"][rid] = {"file": None, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest(), "no_readme": True, "via": "hf-connector",
                             "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        saved += 1; continue
    if not rs or any(r.get("status") != "success" or r.get("output_truncated") or "content" not in (r.get("result") or {}) for r in rs):
        err = " ".join(json.dumps(r.get("error") or (r.get("result") or {}).get("error") or "") for r in rs)
        if re.search(r"not.?found|does not exist|404|No such", err, re.I):
            idx["repos"][rid] = {"file": None, "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest(), "no_readme": True, "via": "hf-connector",
                                 "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            saved += 1
        else: retry.append(rid)
        continue
    last = rs[-1]["result"]
    if last.get("truncated"):
        need[rid] = last.get("next_offset"); continue
    text = "".join(r["result"]["content"] for r in rs)
    b = text.encode("utf-8")
    if size is not None and len(b) != size:
        retry.append(rid); continue          # e.g. a chunk boundary split a multi-byte character
    fn = f"{out}/cards/{rid.replace('/', '__')}.md"
    open(fn, "wb").write(b)
    idx["repos"][rid] = {"file": os.path.relpath(fn, out), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
                         "size_checked": size is not None or len(rs) == 1, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "via": "hf-connector"}
    saved += 1
json.dump(idx, open(ip, "w"), indent=1)
print(json.dumps({"saved": saved, "retry": retry, "need_chunks": need}))
