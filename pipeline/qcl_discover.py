#!/usr/bin/env python3
"""Weekly discovery: find new base models, count their quants, admit the ones that pass the rules, re-rank the queue.

Two ways to make the Hub searches; the rules, counts, release estimate, ranking and output files are the same code either way.

  python3 pipeline/qcl_discover.py --http [--base URL] [--commit]   # automatic: searches straight from the Hub API (no connector),
                                                                   # writes the output files and prints a JSON report; --commit also
                                                                   # commits them (never pushes). HF_TOKEN, if set, raises the rate limit.
  python3 pipeline/qcl_discover.py                       # agent-driven (hf_fs connector): start, and after each round of hf_fs calls:
  python3 pipeline/qcl_discover.py --results P1 P2 ...   # the saved result paths, in the order of the op lists
  python3 pipeline/qcl_discover.py --abort               # end early: unlock and push
  Options for replays and tests: --root DIR (where watchlist.json, discovery.json, audit/ and docs/QUEUE.md live; default: the
  repository), --now ISO-UTC (the run's clock), --no-git (connector path: no sync, lock, commit or push).

Rules (discovery.json): a model is queued when it belongs to the current family of a listed lab ("labs": name prefixes),
has at least admission.min_quants quant repos, and was released within admission.max_age_months. A new model from a
listed lab outside its current family is reported as a possible new family (the owner adds its prefix to discovery.json).
Counts are code-made from search results: quant repos tagged base_model:quantized:<id> (the HTTP path reads the complete
list, paginated; the connector path unions up to five search sorts of 80 each), repos with 1,000+ downloads, and a release
estimate (the earliest Hub activity of the model's quants, never later than an earlier recorded date). The queue is then
ordered by pipeline/qcl_rank.py. Output: watchlist.json, audit/discovery.json (all inputs) and docs/QUEUE.md (the ranked
queue and everything not admitted, with reasons)."""
import argparse, datetime as dt, json, os, re, shlex, subprocess, sys, urllib.parse
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "pipeline"))
NEXT = "python3 pipeline/qcl_discover.py"
PAD = {"cmd": "cat", "args": ["hf://README.md", "--max-bytes", "8737"]}   # pads keep every result large enough to be saved to a file
QUANTISH = re.compile(r"gguf|awq|gptq|mlx|fp8|fp4|nvfp4|mxfp|int4|int8|w4a|exl\d|quant|bnb|\d-?bit|onnx|litert|qat|"
                      r"abliterat|uncensor|heretic|distill|lora|merge|-base\b|dspark|eagle|mtp|draft|preview-\d|tiny|random", re.I)
TASKS = ("text-generation", "image-text-to-text")
CAP = 80                                                     # the connector's per-search cap; a capped count gets four more sorts
MORE_SORTS = ("createdAt", "likes", "lastModified", "trendingScore")
LOCK_H = 0.92                                                # as pipeline/qcl_scan.py: an older discover.lock belongs to a dead session


# ---- the abstract searches (one per Hub listing), and their two renderings ----
def qsearch(model, srt, all_pages=False): return {"tag": f"base_model:quantized:{model}", "sort": srt, "limit": CAP, "all": all_pages}
def find_calls(cfg):
    """Op lists for the found step: each lab's newest repos (two labs per call) and the trending models of the two tasks."""
    labs = list(cfg["labs"])
    calls = [[{"author": o, "sort": "createdAt", "limit": 40} for o in labs[i:i + 2]] for i in range(0, len(labs), 2)]
    return calls + [[{"tag": t, "sort": "trendingScore", "limit": CAP}] for t in TASKS]
def hf_op(q):
    args = [f"hf://models/{q['author']}" if q.get("author") else "hf://models"] + (["--tag", q["tag"]] if q.get("tag") else [])
    return {"cmd": "search", "args": args + ["--sort", q["sort"], "--limit", str(q["limit"])]}
def hf_ops(call): return [hf_op(q) for q in call] + [PAD] * 6
def entries(path):
    """Repo entries of every search result in one saved hf_fs result file."""
    out = []
    for r in json.load(open(path)).get("results", []):
        res = r.get("result") or {}
        if res.get("op") == "search": out += [e for e in res.get("entries", []) if e.get("type") == "repo"]
    return out


# ---- the discovery itself: the same functions for both sources ----
def found(seen, known, cfg):
    """Candidates (current family of a listed lab, not quantish, not yet in the watchlist), possible new families, unlisted labs."""
    cands, newfam, unlisted = [], [], []
    for mid, e in seen.items():
        owner, name = mid.split("/", 1)
        if QUANTISH.search(name) or mid in known: continue
        if owner in cfg["labs"]:
            if any(name.startswith(p) for p in cfg["labs"][owner]): cands.append(mid)
            elif (e.get("likes") or 0) >= 50: newfam.append({"model": mid, "likes": e.get("likes"), "updated_at": e.get("updated_at")})
        elif (e.get("likes") or 0) >= 300 and e.get("task") in TASKS:
            unlisted.append({"model": mid, "likes": e.get("likes")})
    return cands, newfam, unlisted
def slug_of(model): return re.sub(r"[^a-z0-9.]+", "-", model.split("/")[1].lower()).strip("-")
def finish(root, w, counts, capped, newfam, unlisted, cfg, now):
    """Rows, admission, refresh of queued models, ranking; writes watchlist.json, audit/discovery.json, docs/QUEUE.md."""
    import qcl_rank as R
    known = {m["model"]: m for m in w["models"]}
    adm = cfg["admission"]; rows, admitted, updated, rejected = [], [], [], []
    for m, c in counts.items():
        ups = sorted((e.get("updated_at") or "")[:10] for e in c.values() if e.get("updated_at"))
        rel = min([x for x in (ups[0] if ups else None, (known.get(m) or {}).get("released")) if x] or [None])
        row = {"model": m, "quants": len(c), "quants_1k": sum(1 for e in c.values() if (e.get("downloads") or 0) >= 1000),
               "released": rel, "searched_sorts": 1 + len(MORE_SORTS) if m in capped else 1}
        rows.append(row)
        age = (now.date() - dt.date.fromisoformat(rel if len(rel) == 10 else rel + "-15")).days if rel else None
        why = ("fewer than %d quant repos" % adm["min_quants"] if row["quants"] < adm["min_quants"] else
               "no release date (no quants yet)" if age is None else
               "released more than %d months ago" % adm["max_age_months"] if age > adm["max_age_months"] * 30.44 else None)
        cur = known.get(m)
        if cur and cur["status"] != "queued": continue
        if cur:   # already queued: refresh its inputs (an owner-added model stays queued even if it misses a rule; noted)
            cur.update({k: row[k] for k in ("quants", "quants_1k", "released")}); updated.append(m)
            if why: rejected.append({**row, "reason": why + " (already queued; left in place)"})
        elif why: rejected.append({**row, "reason": why})
        else:
            w["models"].append({"model": m, "slug": slug_of(m), "status": "queued", **{k: row[k] for k in ("quants", "quants_1k", "released")}})
            admitted.append(m)
    q = R.rank([x for x in w["models"] if x["status"] == "queued"], now.date())
    w["models"] = [x for x in w["models"] if x["status"] != "queued"] + q
    today = now.date(); L = [f"# Model queue\n\nGenerated by `pipeline/qcl_discover.py` on {now:%Y-%m-%d %H:%M} UTC from Hub search results; "
                             "rules and weights are in [`discovery.json`](../discovery.json), all inputs in [`audit/discovery.json`](../audit/discovery.json).\n",
                             "## Queue (next first)\n", "| # | Model | Score | Quant repos | Used (1k+ downloads) | Released |", "|---|---|---|---|---|---|"]
    for i, x in enumerate(q, 1):
        sc = f"{R.score(x, today)[0]:.0f}" if "quants" in x else "—"
        L.append(f"| {i} | [{x['model']}](https://huggingface.co/{x['model']}) | {sc} | {x.get('quants', '—')} | {x.get('quants_1k', '—')} | {x.get('released', '—')} |")
    L += ["\n## Not admitted\n", "| Model | Quant repos | Released | Reason |", "|---|---|---|---|"]
    L += [f"| {r['model']} | {r['quants']} | {r['released'] or '—'} | {r['reason']} |" for r in sorted(rejected, key=lambda r: -r["quants"])] or ["| — | | | |"]
    L += ["\n## Possible new families (not queued until their prefix is added to `discovery.json`)\n"]
    L += [f"- [{x['model']}](https://huggingface.co/{x['model']}) ({x['likes']} likes)" for x in newfam] or ["- none"]
    L += ["\n## Popular models from labs not in `discovery.json`\n"]
    L += [f"- [{x['model']}](https://huggingface.co/{x['model']}) ({x['likes']} likes)" for x in unlisted] or ["- none"]
    audit = {"at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "rules": cfg["admission"], "ranking": cfg["ranking"], "counted": rows,
             "admitted": admitted, "not_admitted": rejected, "possible_new_families": newfam, "unlisted_labs": unlisted}
    wl = os.path.join(root, "watchlist.json")
    json.dump(w, open(wl, "w"), indent=1); open(wl, "a").write("\n")
    os.makedirs(os.path.join(root, "audit"), exist_ok=True); os.makedirs(os.path.join(root, "docs"), exist_ok=True)
    json.dump(audit, open(os.path.join(root, "audit", "discovery.json"), "w"), indent=1)
    open(os.path.join(root, "docs", "QUEUE.md"), "w").write("\n".join(L) + "\n")
    msg = f"Weekly discovery {now:%Y-%m-%d}: {len(admitted)} admitted, {len(updated)} queued models recounted"
    return {"admitted": admitted, "updated": len(updated), "not_admitted": len(rejected), "possible_new_families": newfam,
            "unlisted_labs": unlisted}, msg
OUTPUTS = "watchlist.json audit/discovery.json docs/QUEUE.md"


# ---- git (connector path: lock, commit and push; HTTP path: --commit only, never a push) ----
def sh(cmd, root): return subprocess.run(cmd, cwd=root, shell=True, capture_output=True, text=True)
def commit(msg, paths, root, identity=True):
    for p in shlex.split(paths): sh(f"git add -A -- {shlex.quote(p)}", root)
    if sh("git diff --cached --quiet", root).returncode == 0: return "nothing to commit"
    who = "-c user.name=Claude -c user.email=noreply@anthropic.com " if identity else ""
    p = sh(f"git {who}commit -qm {shlex.quote(msg)}", root)
    return "committed" if p.returncode == 0 else "commit failed: " + (p.stderr.strip().splitlines() or ["?"])[-1][:200]
def push(msg, paths, root):
    if commit(msg, paths, root) == "nothing to commit": return "nothing to commit"
    for _ in range(3):
        if sh("git push -q origin HEAD:refs/heads/main", root).returncode == 0: return "pushed"
        sh("git pull --rebase -q origin main", root)
    return "push failed"
def lock_age_h(root, now):
    lock = os.path.join(root, "discover.lock")
    if not os.path.exists(lock): return None
    try: return (now - dt.datetime.fromisoformat(open(lock).read().strip())).total_seconds() / 3600
    except ValueError: return None


# ---- the HTTP path: the same searches straight from the Hub API ----
def api_entry(m):
    """A Hub API model item in the shape of an hf_fs search entry (the shape both paths feed into found() and finish())."""
    return {"type": "repo", "repo_type": "model", "path": m.get("id") or m.get("modelId"), "updated_at": m.get("lastModified"),
            "downloads": m.get("downloads"), "likes": m.get("likes"), "gated": m.get("gated"), "private": m.get("private"),
            "task": m.get("pipeline_tag")}
def hub_search(base, token, q, counter):
    import qcl_fetch as F
    params = {**({"author": q["author"]} if q.get("author") else {}), **({"filter": q["tag"]} if q.get("tag") else {}),
              "sort": q["sort"], "direction": -1, "limit": 1000 if q.get("all") else q["limit"], "full": "true"}
    url = f"{base}/api/models?" + urllib.parse.urlencode(params) + "&expand[]=lastModified&expand[]=downloads&expand[]=likes&expand[]=gated&expand[]=pipeline_tag"
    out, seen = [], set()
    while url:
        if url in seen: raise RuntimeError("pagination cycle")
        seen.add(url); counter[0] += 1
        st, body, h = F.get(url, token)
        if st != 200: raise RuntimeError(f"Hub API HTTP {st}: {url.split('?')[0]} {json.dumps(q)}")
        page = json.loads(body)
        if not isinstance(page, list): raise RuntimeError("Hub API response is not a list")
        out += [api_entry(m) for m in page if isinstance(m, dict) and (m.get("id") or m.get("modelId"))]
        url = F.next_page(url, h) if q.get("all") else None
    return out
def run_http(a, root, cfg, now):
    import qcl_fetch as F
    token = os.environ.get("HF_TOKEN") or None
    age = lock_age_h(root, now)
    if age is not None and age < LOCK_H:
        say({"do": "stop", "report": {"skipped": "a connector-driven discovery run holds discover.lock", "age_h": round(age, 2)}})
    w = json.load(open(os.path.join(root, "watchlist.json"))); known = {m["model"]: m for m in w["models"]}
    n = [0]
    try:
        seen = {}
        for call in find_calls(cfg):
            for q in call:
                for e in hub_search(a.base, token, q, n): seen[e["path"]] = e
        cands, newfam, unlisted = found(seen, known, cfg)
        models = sorted(set(cands) | {m["model"] for m in w["models"] if m["status"] == "queued"})
        counts = {m: {e["path"]: e for e in hub_search(a.base, token, qsearch(m, "downloads", all_pages=True), n)} for m in models}
    except F.RateLimited as e:
        say({"do": "stop", "error": str(e), "rate_limited": e.record, "requests": n[0]}, 2)
    except (RuntimeError, ValueError, OSError) as e:
        say({"do": "stop", "error": f"{type(e).__name__}: {e}", "requests": n[0]}, 1)
    report, msg = finish(root, w, counts, [], newfam, unlisted, cfg, now)   # complete listings: no capped counts
    report.update({"source": "hub-api", "requests": n[0], "candidates": len(cands), "counted": len(models)})
    if a.commit: report["git"] = commit(msg, OUTPUTS, root, identity=False)
    say({"do": "stop", "report": report})


# ---- the connector path: a state machine driven by the session's hf_fs calls ----
def say(d, code=0): print(json.dumps(d)); sys.exit(code)
def run_connector(a, root, cfg, now):
    ST = os.path.join(root, "work", "discover_state.json")
    git = not a.no_git
    os.environ["QCL_LOCK"] = os.path.join(root, "discover.lock"); os.environ["QCL_PART"] = "-discover"
    def scan(x):
        p = sh(f"python3 {shlex.quote(os.path.join(CODE, 'pipeline', 'qcl_scan.py'))} {x}", root); line = (p.stdout.strip().splitlines() or ["{}"])[-1]
        try: return p.returncode, json.loads(line)
        except ValueError: return p.returncode, {"raw": line}
    def save(s): os.makedirs(os.path.dirname(ST), exist_ok=True); json.dump(s, open(ST, "w"), indent=1)
    def stop(s, note=None):
        if note: s.setdefault("notes", []).append(note)
        if git: scan("unlock --discard"); s["git_unlock"] = push("discovery unlock", "discover.lock logs/scan", root)
        s["step"] = "done"; save(s)
        say({"do": "stop", "report": {k: s.get(k) for k in ("admitted", "updated", "possible_new_families", "notes", "git", "git_unlock")}})
    def hf(calls, s, step):
        ops = [hf_ops(c) for c in calls]
        s["step"] = step; s["expect"] = len(ops); s["last_ops"] = ops; save(s)
        say({"do": "hf_fs", "op_lists": ops, "then": f"{NEXT} --results <saved path of each call, in order>"})
    results = list(a.results)
    s = json.load(open(ST)) if os.path.exists(ST) else None
    if s and s.get("step") == "done": s = None
    if a.abort: stop(s or {}, "ended early by the session (--abort)")
    if s is None and git and not a.synced:
        sh("git fetch -q origin main", root)
        if sh("git status --porcelain --untracked-files=no", root).stdout.strip() == "": sh("git checkout -q -B main origin/main", root)
        sh('git config merge.wljson.driver "python3 pipeline/merge_watchlist.py %O %A %B"', root)
        os.execvp("python3", ["python3", os.path.abspath(__file__), *sys.argv[1:], "--synced"])
    if s is None:
        if git:
            code, d = scan("lock")
            if code == 3: say({"do": "stop", "report": {"skipped": "another discovery run holds the lock", **d}})
            push("discovery lock", "discover.lock", root)
        s = {"step": "start", "notes": []}
        hf(find_calls(cfg), s, "found")
    if len(results) != s.get("expect"):
        say({"do": "hf_fs", "op_lists": s["last_ops"], "then": f"{NEXT} --results <saved path of each call, in order>",
             "error": f"expected {s.get('expect')} saved paths, got {len(results)}; make every call and pass every path in order"})
    w = json.load(open(os.path.join(root, "watchlist.json"))); known = {m["model"]: m for m in w["models"]}
    if s["step"] == "found":
        seen = {}
        for f in results:
            for e in entries(f): seen[e["path"]] = e
        cands, s["possible_new_families"], s["unlisted_labs"] = found(seen, known, cfg)
        s["models"] = sorted(set(cands) | {m["model"] for m in w["models"] if m["status"] == "queued"})
        hf([[qsearch(m, "downloads")] for m in s["models"]], s, "counted")
    if s["step"] == "counted":
        s["counts"] = {m: {e["path"]: e for e in entries(f)} for m, f in zip(s["models"], results)}
        s["capped"] = [m for m, c in s["counts"].items() if len(c) >= CAP]
        calls = [[qsearch(m, srt)] for m in s["capped"] for srt in MORE_SORTS]
        if calls: hf(calls, s, "counted2")
        s["step"] = "counted2"; results = []
    if s["step"] == "counted2":
        for i, f in enumerate(results):
            m = s["capped"][i // len(MORE_SORTS)]
            for e in entries(f): s["counts"][m][e["path"]] = e
        report, msg = finish(root, w, s["counts"], s["capped"], s["possible_new_families"], s["unlisted_labs"], cfg, now)
        s.update({"admitted": report["admitted"], "updated": report["updated"]})
        if git: s["git"] = push(msg, OUTPUTS, root)
        stop(s)
    stop(s, f"unexpected state {s.get('step')}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", nargs="*", default=[], help="connector path: saved hf_fs result files, in op-list order")
    p.add_argument("--abort", action="store_true"); p.add_argument("--synced", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--http", action="store_true", help="search the Hub API directly (no connector, no git)")
    p.add_argument("--base", default="https://huggingface.co"); p.add_argument("--commit", action="store_true", help="--http: also commit the output files (no push)")
    p.add_argument("--root", default=CODE); p.add_argument("--now"); p.add_argument("--no-git", action="store_true")
    a = p.parse_args()
    root = os.path.abspath(a.root)
    cfg = json.load(open(os.path.join(root, "discovery.json")))
    now = dt.datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else dt.datetime.now(dt.timezone.utc)
    now = (now.replace(tzinfo=dt.timezone.utc) if now.tzinfo is None else now).astimezone(dt.timezone.utc)
    (run_http if a.http else run_connector)(a, root, cfg, now)


if __name__ == "__main__": main()
