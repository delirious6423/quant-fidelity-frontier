#!/usr/bin/env python3
"""Incremental ingestion: the work happens before anyone asks for the site.

  python3 qcl_watch.py delta  INDEX_NEW.json         # repos that are new or whose card sha changed vs snapshots/index.json
  python3 qcl_watch.py merge  MAPDIR_A [MAPDIR_B]     # apply mappings, keep only gate-passing entries, quarantine the rest, build

merge:
  1. apply every *.map.json (values copied from source cells by code; two runs -> keep values both runs agree on,
     the rest go to the review queue);
     apply reviewed maps/resolved overrides only when their manifest pins both the current source and mapping bytes;
  2. write verified/part8_machine.py (machine entries replace older entries for the same repo);
  3. run the gates; any machine entry named in a gate failure is removed and written to audit/review_queue.json,
     then the gates run again, so published data stays fail-closed while the rest of the delta ships;
  4. build.
"""
import glob, hashlib, json, os, re, subprocess, sys, time
import qcl_map as Q
import qcl_fmt as F

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]   # sibling modules
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")          # data root: one folder per model (models/<slug>/); code stays here
OUT = f"{ROOT}/verified/part8_machine.py"

def load_verified(skip_machine=True):
    R = []
    for p in sorted(glob.glob(f"{ROOT}/verified/part*.py")):
        if skip_machine and p == OUT: continue
        ns = {}; exec(open(p).read(), ns); R += ns["R"]
    return R

def applied_from(d):
    out = {}
    for m in sorted(glob.glob(f"{d}/*.map.json")):
        cf = m.replace(".map.json", ".cands.json")
        if not os.path.exists(cf): cf = os.path.join(os.path.dirname(os.path.dirname(m)), "cands", os.path.basename(cf))   # models/<slug>/maps/cands/
        a = Q.apply(json.load(open(m)), json.load(open(cf))); out[a["repo"]] = a
    return out

def agree(a, b):
    """Keep a value only if the other run also filed the same published string, under the same metric family and
    the same role, somewhere on the card (entry names differ between runs, so they are not compared)."""
    import autoextract as AX
    def fam(k):
        if k == "mean_kl_mnats": return k
        if k.startswith(("kld:", "top1:", "ppl:", "kl_mnats:")): return k
        if k.startswith("bench:"): return "bench:" + re.sub(r"[^a-z0-9]", "", k[6:].lower())   # runs spell benchmark names differently
        return AX.gt_family(k) if AX.gt_family(k) != "other" else k
    role = {"variants": "own", "ladder": "ladder", "cross": "cross"}
    sb = {(role[k], e.get("panel"), e.get("measured", True), fam(mk), v) for k in role for e in b[k] for mk, v in e["m"].items()}
    out = dict(a, variants=[], ladder=[], cross=[], weights_modified=(None if a.get("weights_modified") is None and b.get("weights_modified") is None else bool(a.get("weights_modified") or b.get("weights_modified")))); held = []
    for k in role:
        for e in a[k]:
            keep = {mk: v for mk, v in e["m"].items() if (role[k], e.get("panel"), e.get("measured", True), fam(mk), v) in sb}
            held += [(a["repo"], e.get("name") or e.get("variant"), mk, v) for mk, v in e["m"].items() if mk not in keep]
            if keep: out[k].append(dict(e, m=keep))
    return out, held

def resolved_from(allowed_repos):
    """Load reviewed overrides pinned to the current card and exact resolver-map bytes.

    manifest.json: {"repos": {repo: {"source_sha256": ..., "mapping_sha256": ..., ...provenance}}}.
    Missing, changed or invalid overrides do not resolve any independent-run disagreements.
    """
    directory = f"{ROOT}/maps/resolved"
    try:
        with open(f"{directory}/manifest.json") as f: manifest = json.load(f)
    except (OSError, ValueError): return {}, {}
    records = manifest.get("repos", {}) if isinstance(manifest, dict) else {}
    if not isinstance(records, dict): return {}, {}
    with open(f"{ROOT}/snapshots/index.json") as f: idx = json.load(f)["repos"]
    out, provenance = {}, {}
    for rp in sorted(set(allowed_repos) & set(records)):
        record = records[rp]
        source_sha = idx.get(rp, {}).get("sha256")
        if not isinstance(record, dict) or not source_sha or record.get("source_sha256") != source_sha: continue
        name = rp.replace("/", "__")
        path = f"{directory}/{name}.map.json"
        try:
            with open(path, "rb") as f: raw = f.read()
            if hashlib.sha256(raw).hexdigest() != record.get("mapping_sha256"): continue
            mapping = json.loads(raw)
            if mapping.get("repo") != rp: continue
            candidate_path = f"{directory}/{name}.cands.json"
            if not os.path.exists(candidate_path): candidate_path = f"{ROOT}/maps/cands/{name}.cands.json"
            with open(candidate_path, "rb") as f: candidate_raw = f.read()
            if record.get("candidates_sha256") and hashlib.sha256(candidate_raw).hexdigest() != record["candidates_sha256"]: continue
            cands = json.loads(candidate_raw)
            applied = Q.apply(mapping, cands)
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError): continue
        out[rp] = applied
        provenance[rp] = {**record, "mapping_file": os.path.relpath(path, ROOT)}
    return out, provenance

def write(entries, base):
    idx = json.load(open(f"{ROOT}/snapshots/index.json"))["repos"]
    by = {r["repo"]: r for r in base}
    lines = ["# AUTO-GENERATED by qcl_watch.py merge (machine-mapped; values copied by code). Do not edit by hand.",
             "R = []", "def repo(**k): R.append(k); return k", ""]
    for rp, a in sorted(entries.items()):
        o = by.get(rp, {})
        lines.append("repo(**" + repr(dict(repo=rp, sha=idx[rp]["sha"], fmt=o.get("fmt") or F.fmt_of(rp, idx[rp].get("tags")), runtime=o.get("runtime"),
                     weights_modified=a.get("weights_modified") if a.get("weights_modified") is not None else o.get("weights_modified"), method=o.get("method"),
                     methodology=a["methodology"] if isinstance(a.get("methodology"), dict) else None,
                     variants=a["variants"], ladder=a["ladder"], cross=a["cross"], flags=a["flags"] or [])) + ")")
    open(OUT, "w").write("\n".join(lines) + "\n")

def failing(entries):
    bad = set()
    for g in ("verify.py", "verify_columns.py"):
        out = subprocess.run(["python3", f"{CODE}/gates/{g}"], capture_output=True, text=True).stdout
        for l in out.splitlines():
            if re.search(r"FAIL|MISMATCH|ROWLABEL", l):
                for rp, a in entries.items():
                    if rp in l:
                        for k in ("variants", "ladder", "cross"):
                            for e in a[k]:
                                nm = str(e.get("name") or e.get("variant"))
                                if nm in l: bad.add((rp, k, nm))
    return bad

def merge(dirs):
    t0 = time.time()
    runs = [applied_from(d) for d in dirs]
    held = []
    if len(runs) == 2:
        entries = {}
        for rp in runs[0]:
            if rp in runs[1]: entries[rp], h = agree(runs[0][rp], runs[1][rp]); held += h
    else: entries = runs[0]
    # The original independent maps remain untouched; a source-pinned review can supersede their agreement.
    resolved, provenance = resolved_from(entries) if len(runs) == 2 else ({}, {})
    entries.update(resolved)
    resolved_held = [dict(zip(("repo", "entry", "key", "raw"), h), resolver=provenance[h[0]]) for h in held if h[0] in resolved]
    held = [h for h in held if h[0] not in resolved]
    base = load_verified()
    # machine entries replace the same repos' older entries: drop those repos from the base parts' view via the build's duplicate rule
    write(entries, base)
    quarantined = []
    for _ in range(3):
        bad = failing(entries)
        if not bad: break
        for rp, k, nm in bad:
            entries[rp][k] = [e for e in entries[rp][k] if str(e.get("name") or e.get("variant")) != nm]
            quarantined.append({"repo": rp, "kind": k, "entry": nm, "reason": "failed a gate"})
        write(entries, base)
    json.dump({"quarantined": quarantined, "disagreements": [dict(zip(("repo", "entry", "key", "raw"), h)) for h in held],
               "resolved_disagreements": resolved_held, "resolved_repos": len(resolved)},
              open(f"{ROOT}/audit/review_queue.json", "w"), indent=1)
    nvals = sum(len(e["m"]) for a in entries.values() for k in ("variants", "ladder", "cross") for e in a[k])
    print(f"merged {len(entries)} repos, {nvals} values; resolved {len(resolved)} repos; quarantined {len(quarantined)} entries; held {len(held)} disagreements; {time.time()-t0:.1f}s")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "delta":
        new = json.load(open(sys.argv[2]))["repos"]; old = json.load(open(f"{ROOT}/snapshots/index.json"))["repos"]
        d = sorted(r for r, m in new.items() if old.get(r, {}).get("sha") != m.get("sha"))
        print(json.dumps(d))
    elif cmd == "merge": merge(sys.argv[2:] or [f"{ROOT}/maps/A", f"{ROOT}/maps/B"])   # default: the maps stored with the model
