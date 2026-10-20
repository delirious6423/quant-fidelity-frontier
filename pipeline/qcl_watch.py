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
import mapping_provenance as MP

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]   # sibling modules
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")          # data root: one folder per model (models/<slug>/); code stays here
OUT = f"{ROOT}/verified/part8_machine.py"
# model.json "agree_strict_corpus": per-corpus keys (kld:code vs kld:wiki) must match between runs. Off by default: older
# models' runs name the corpus inconsistently, and requiring it there would hold back ~500 values both runs agree on.
STRICT_CORPUS = bool((json.load(open(f"{ROOT}/model.json")) if os.path.exists(f"{ROOT}/model.json") else {}).get("agree_strict_corpus"))

def load_verified(skip_machine=True):
    R = []
    for p in sorted(glob.glob(f"{ROOT}/verified/part*.py")):
        if skip_machine and p == OUT: continue
        ns = {}; exec(open(p).read(), ns); R += ns["R"]
    return R

def applied_from(d):
    out = {}
    legacy = MP.recorded_pins(ROOT, os.path.basename(os.path.normpath(d)))
    for m in sorted(glob.glob(f"{d}/*.map.json")):
        cf = MP.candidate_path(m)
        try:
            mapping = json.load(open(m)); candidates = json.load(open(cf))
            pin = MP.mapping_source(mapping, candidates, Q.A.IDX, legacy, path=m)
            if Q.source_pin(mapping["repo"]) != pin: raise ValueError("mapped source changed before application")
            a = Q.apply(mapping, MP.candidate_values(candidates), source_sha256=pin)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"withheld stale/invalid mapping {os.path.basename(m)}: {error}", file=sys.stderr)
            continue
        out[a["repo"]] = a
    return out

def panel_key(p):
    """Panel titles are typed by each run; compare them without markdown heading marks, case or spacing."""
    return re.sub(r"\s+", " ", re.sub(r"^#+\s*|[*`]", "", str(p).strip())).strip().lower() if p else None

def agree(a, b):
    """Keep a value only if the other run also filed the same published string, under the same metric family, the
    same role and the same panel, somewhere on the card (entry names differ between runs, so they are not compared).
    Panels: two different named panels disagree (e.g. a legacy top-k panel vs the current full-vocabulary one). A run
    that names no panel does not contradict one that does; the named panel is kept, so a value is never moved into the
    default panel by the run that did not separate them."""
    import autoextract as AX
    def fam(k):
        if k == "mean_kl_mnats": return k
        if globals().get("STRICT_CORPUS", True) and k.startswith(("kld:", "top1:", "ppl:", "kl_mnats:")):   # per-corpus panels kept apart (opt-in per model)
            f, c = k.split(":", 1); return f + ":" + re.sub(r"[^a-z0-9]", "", c.lower())
        if k.startswith("bench:"): return "bench:" + re.sub(r"[^a-z0-9]", "", k[6:].lower())   # runs spell benchmark names differently
        return AX.gt_family(k) if AX.gt_family(k) != "other" else k
    role = {"variants": "own", "ladder": "ladder", "cross": "cross"}
    sb = {}   # (role, measured, family, string) -> {normalized panel: panel as B wrote it}
    for k in role:
        for e in b[k]:
            for mk, v in e["m"].items(): sb.setdefault((role[k], e.get("measured", True), fam(mk), v), {})[panel_key(e.get("panel"))] = e.get("panel")
    out = dict(a, variants=[], ladder=[], cross=[], weights_modified=(None if a.get("weights_modified") is None and b.get("weights_modified") is None else bool(a.get("weights_modified") or b.get("weights_modified")))); held = []
    for k in role:
        for e in a[k]:
            pa = panel_key(e.get("panel")); keep = {}; adopt = set()
            for mk, v in e["m"].items():
                pb = sb.get((role[k], e.get("measured", True), fam(mk), v), {})
                if pa in pb or (pa is not None and None in pb): keep[mk] = v
                elif pa is None and len(pb) == 1: keep[mk] = v; adopt.add(next(iter(pb.values())))
            if len(adopt) > 1:   # A named no panel and B split its values across panels: keep only the unpanelled matches
                keep = {mk: v for mk, v in keep.items() if None in sb.get((role[k], e.get("measured", True), fam(mk), v), {})}; adopt = set()
            held += [(a["repo"], e.get("name") or e.get("variant"), mk, v) for mk, v in e["m"].items() if mk not in keep]
            if keep: out[k].append(dict(e, m=keep, **({"panel": next(iter(adopt))} if adopt else {})))
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
            candidate_pin = record.get("candidates_sha256")
            if (not isinstance(candidate_pin, str) or not re.fullmatch(r"[a-f0-9]{64}", candidate_pin)
                    or hashlib.sha256(candidate_raw).hexdigest() != candidate_pin): continue
            cands = json.loads(candidate_raw)
            if candidate_path.startswith(directory):   # resolver-written candidates: their raw strings are re-checked on the card line
                gen_p = f"{ROOT}/maps/cands/{name}.cands.json"; gen = json.load(open(gen_p)) if os.path.exists(gen_p) else {}
                gen = gen.get("candidates", gen) if isinstance(gen, dict) else {}
                cc = cands.get("candidates", cands)
                for cid, c in list(cc.items()):
                    if gen.get(cid) == c: continue
                    lines = Q.A_lines(rp); L = c.get("L")
                    if not (isinstance(L, int) and 0 < L <= len(lines) and (Q.exact_in(str(c.get("raw", "")), lines[L - 1]) or Q.exact_in(Q.clean_raw(str(c.get("raw", ""))), lines[L - 1]))): del cc[cid]; continue
                    cc[cid] = {**c, "header": "", "label": "(prose)"}   # header/label were typed by the resolver, not card wording
            applied = Q.apply(mapping, cands.get("candidates", cands), source_sha256=source_sha)
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError): continue
        out[rp] = applied
        provenance[rp] = {**record, "mapping_file": os.path.relpath(path, ROOT)}
    return out, provenance

def write(entries, base):
    idx = json.load(open(f"{ROOT}/snapshots/index.json"))["repos"]
    by = {r["repo"]: r for r in base}
    def _image_flags(rp, meta, has_claims):
        """A card whose images may hold quant results or details is flagged by code (a mapping run cannot read images)."""
        try:
            import qcl_map as _Q
            card = open(os.path.join(ROOT, meta["file"]), encoding="utf-8", errors="replace").read().splitlines()
            flag = _Q.image_flag(card, has_claims)
        except (OSError, ImportError, KeyError): return []
        return [flag] if flag else []
    lines = ["# AUTO-GENERATED by qcl_watch.py merge (machine-mapped; values copied by code). Do not edit by hand.",
             "R = []", "def repo(**k): R.append(k); return k", ""]
    for rp, a in sorted(entries.items()):
        o = by.get(rp, {})
        pin = a.get("source_sha256")
        if not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin) or pin != idx[rp].get("sha256"):
            raise ValueError(f"missing or stale mapped source pin: {rp}")
        if hashlib.sha256(open(os.path.join(ROOT, idx[rp]["file"]), "rb").read()).hexdigest() != pin:
            raise ValueError(f"source bytes changed before writing claims: {rp}")
        lines.append("repo(**" + repr(dict(repo=rp, sha=idx[rp]["sha"], source_sha256=pin, fmt=o.get("fmt") or F.fmt_of(rp, idx[rp].get("tags")), runtime=o.get("runtime"),
                     weights_modified=a.get("weights_modified") if a.get("weights_modified") is not None else o.get("weights_modified"), method=o.get("method"),
                     methodology=a["methodology"] if isinstance(a.get("methodology"), dict) else None,
                     variants=a["variants"], ladder=a["ladder"], cross=a["cross"], flags=(a["flags"] or []) + _image_flags(rp, idx[rp], bool(a["variants"] or a["ladder"] or a["cross"])))) + ")")
    open(OUT, "w").write("\n".join(lines) + "\n")

def build_failing(entries):
    """build.py's own fail-closed checks (sign and plausibility of parsed values) name (repo, variant, ...) tuples."""
    import ast
    out = subprocess.run(["python3", f"{CODE}/build.py"], capture_output=True, text=True, env={**os.environ, "QCL_ROOT": ROOT})
    if out.returncode == 0: return set()
    bad = set()
    for l in (out.stdout + out.stderr).splitlines():
        if not l.startswith("("): continue
        try: t = ast.literal_eval(l)
        except Exception: continue
        if not (isinstance(t, tuple) and len(t) >= 2): continue
        rp, nm = str(t[0]), str(t[1]); raw = str(t[3]) if len(t) >= 4 else None
        # the build may label a ladder row with the repo it compares against ("not named on card"), not the repo that owns
        # the entry; then search every repo, matching by entry name
        for ork in ([rp] if rp in entries else list(entries)):
            for k in ("variants", "ladder", "cross"):
                for e in entries.get(ork, {}).get(k, []):
                    en = str(e.get("name") or e.get("variant"))
                    # by name, or by the offending raw string (the build may show a row under a different variant label)
                    if en == nm or (rp in entries and raw and raw in [str(v) for v in (e.get("m") or {}).values()]): bad.add((ork, k, en))
    return bad

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
    # Tie-breaker (run C, pipeline/qcl_tiebreak.py): a held run-A value is published when C, mapping the same card bytes
    # blind, filed the same string under the same role, family and panel. Only values held above are added; agreed
    # values are never changed, and the gates below check every value against its card line as usual.
    tiebroken = []
    if len(runs) == 2 and os.path.isdir(f"{ROOT}/maps/C"):
        idx_c = json.load(open(f"{ROOT}/snapshots/index.json"))["repos"]
        mcp = f"{ROOT}/maps/mapped_c.json"; mapped_c = json.load(open(mcp)) if os.path.exists(mcp) else {}
        held_repos = {h[0] for h in held}
        for rp, c in applied_from(f"{ROOT}/maps/C").items():
            if rp not in held_repos or rp not in entries or mapped_c.get(rp) != idx_c.get(rp, {}).get("sha256"): continue
            ac, _ = agree(runs[0][rp], c)
            for k in ("variants", "ladder", "cross"):
                for e in ac[k]:
                    nm = e.get("name") or e.get("variant")
                    tgt = next((x for x in entries[rp][k] if (x.get("name") or x.get("variant")) == nm), None)
                    new = {mk: v for mk, v in e["m"].items() if not (tgt and mk in tgt["m"])}
                    if not new: continue
                    if tgt: tgt["m"] = {**tgt["m"], **new}
                    else: entries[rp][k].append(dict(e, m=new))
                    tiebroken += [(rp, nm, mk, v) for mk, v in new.items()]
        tb = set(tiebroken); held = [h for h in held if tuple(h) not in tb]
    # The original independent maps remain untouched; a source-pinned review can supersede their agreement.
    resolved, provenance = resolved_from(entries) if len(runs) == 2 else ({}, {})
    entries.update(resolved)
    resolved_held = [dict(zip(("repo", "entry", "key", "raw"), h), resolver=provenance[h[0]]) for h in held if h[0] in resolved]
    held = [h for h in held if h[0] not in resolved]
    # audit/holds.json: entries a review found mis-attributed ({"repo", "entry", "reason"}); held back, never edited
    holds = json.load(open(f"{ROOT}/audit/holds.json")) if os.path.exists(f"{ROOT}/audit/holds.json") else []
    held_by_review = []
    for h in holds:
        a = entries.get(h["repo"])
        if not a: continue
        for k in ("variants", "ladder", "cross"):
            keep = [e for e in a[k] if str(e.get("name") or e.get("variant")) != h["entry"]]
            if len(keep) != len(a[k]): held_by_review.append({"repo": h["repo"], "kind": k, "entry": h["entry"], "reason": "review hold: " + h["reason"]}); a[k] = keep
    base = load_verified()
    # machine entries replace the same repos' older entries: drop those repos from the base parts' view via the build's duplicate rule
    write(entries, base)
    quarantined = list(held_by_review)
    for _ in range(6):   # tie-broken values may need a pass of their own
        bad = failing(entries) or build_failing(entries)   # source/column gates first, then the build's sign/plausibility checks
        if not bad: break
        for rp, k, nm in bad:
            # an entry that fails only after tie-broken values were added to it: drop those values (held again), keep the
            # values A and B agreed on; the next pass re-checks the entry as it was
            tbk = {(h[2], h[3]) for h in tiebroken if h[0] == rp and str(h[1]) == nm}
            hit = [e for e in entries[rp][k] if str(e.get("name") or e.get("variant")) == nm and tbk & set(e["m"].items())]
            if hit:
                for e in hit:
                    back = [(mk, v) for mk, v in e["m"].items() if (mk, v) in tbk]
                    e["m"] = {mk: v for mk, v in e["m"].items() if (mk, v) not in tbk}
                    held += [(rp, nm, mk, v) for mk, v in back]
                    tiebroken = [h for h in tiebroken if not (h[0] == rp and str(h[1]) == nm and (h[2], h[3]) in back)]
                entries[rp][k] = [e for e in entries[rp][k] if e["m"] or e.get("size") is not None]
                continue
            entries[rp][k] = [e for e in entries[rp][k] if str(e.get("name") or e.get("variant")) != nm]
            quarantined.append({"repo": rp, "kind": k, "entry": nm, "reason": "failed a gate"})
        write(entries, base)
    json.dump({"quarantined": quarantined, "disagreements": [dict(zip(("repo", "entry", "key", "raw"), h)) for h in held],
               "resolved_disagreements": resolved_held, "resolved_repos": len(resolved),
               "tiebroken": [dict(zip(("repo", "entry", "key", "raw"), h)) for h in tiebroken if h[0] not in resolved]},
              open(f"{ROOT}/audit/review_queue.json", "w"), indent=1)
    nvals = sum(len(e["m"]) for a in entries.values() for k in ("variants", "ladder", "cross") for e in a[k])
    print(f"merged {len(entries)} repos, {nvals} values; resolved {len(resolved)} repos; tie-broken {len(tiebroken)} values; quarantined {len(quarantined)} entries; held {len(held)} disagreements; {time.time()-t0:.1f}s")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "delta":
        new = json.load(open(sys.argv[2]))["repos"]; old = json.load(open(f"{ROOT}/snapshots/index.json"))["repos"]
        d = sorted(r for r, m in new.items() if old.get(r, {}).get("sha") != m.get("sha"))
        print(json.dumps(d))
    elif cmd == "merge": merge(sys.argv[2:] or [f"{ROOT}/maps/A", f"{ROOT}/maps/B"])   # default: the maps stored with the model
