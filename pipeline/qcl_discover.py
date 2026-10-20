#!/usr/bin/env python3
"""Weekly discovery: find new base models, count their quants, admit the ones that pass the rules, re-rank the queue.
The session runs this and does only what it prints, like qcl_next.py and qcl_qa.py.

  python3 pipeline/qcl_discover.py                       # start, and after each round of hf_fs calls:
  python3 pipeline/qcl_discover.py --results P1 P2 ...   # the saved result paths, in the order of the op lists
  python3 pipeline/qcl_discover.py --abort               # end early: unlock and push

Rules (discovery.json): a model is queued when it belongs to the current family of a listed lab ("labs": name prefixes),
has at least admission.min_quants quant repos, and was released within admission.max_age_months. A new model from a
listed lab outside its current family is reported as a possible new family (the owner adds its prefix to discovery.json).
Counts are code-made from saved search results: quant repos tagged base_model:quantized:<id>, as the union of up to five
search sorts (80 each), repos with 1,000+ downloads, and a release estimate (the earliest Hub activity of the model's quants,
never later than an earlier recorded date). The queue is then ordered by pipeline/qcl_rank.py. Output: watchlist.json,
audit/discovery.json (all inputs) and docs/QUEUE.md (the ranked queue and everything not admitted, with reasons)."""
import datetime as dt, json, os, re, subprocess, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["QCL_LOCK"] = "discover.lock"; os.environ["QCL_PART"] = "-discover"
ST = os.path.join(CODE, "work", "discover_state.json"); NEXT = "python3 pipeline/qcl_discover.py"
WL = os.path.join(CODE, "watchlist.json"); CFG = json.load(open(os.path.join(CODE, "discovery.json")))
PAD = {"cmd": "cat", "args": ["hf://README.md", "--max-bytes", "8737"]}   # pads keep every result large enough to be saved to a file
QUANTISH = re.compile(r"gguf|awq|gptq|mlx|fp8|fp4|nvfp4|mxfp|int4|int8|w4a|exl\d|quant|bnb|\d-?bit|onnx|litert|qat|"
                      r"abliterat|uncensor|heretic|distill|lora|merge|-base\b|dspark|eagle|mtp|draft|preview-\d|tiny|random", re.I)
NOW = dt.datetime.now(dt.timezone.utc)

def sh(cmd): return subprocess.run(cmd, cwd=CODE, shell=True, capture_output=True, text=True)
def scan(a):
    p = sh(f"python3 pipeline/qcl_scan.py {a}"); line = (p.stdout.strip().splitlines() or ["{}"])[-1]
    try: return p.returncode, json.loads(line)
    except ValueError: return p.returncode, {"raw": line}
def say(d): print(json.dumps(d)); sys.exit(0)
def save(s): os.makedirs(os.path.dirname(ST), exist_ok=True); json.dump(s, open(ST, "w"), indent=1)
def push(msg, paths):
    import shlex
    for p in shlex.split(paths): sh(f"git add -A -- {shlex.quote(p)}")
    if sh("git diff --cached --quiet").returncode == 0: return "nothing to commit"
    sh(f"git -c user.name=Claude -c user.email=noreply@anthropic.com commit -qm {json.dumps(msg)}")
    for _ in range(3):
        if sh("git push -q origin HEAD:refs/heads/main").returncode == 0: return "pushed"
        sh("git pull --rebase -q origin main")
    return "push failed"
def stop(s, note=None):
    if note: s.setdefault("notes", []).append(note)
    scan("unlock --discard"); s["git_unlock"] = push("discovery unlock", "discover.lock logs/scan")
    s["step"] = "done"; save(s)
    say({"do": "stop", "report": {k: s.get(k) for k in ("admitted", "updated", "possible_new_families", "notes", "git", "git_unlock")}})
def hf(ops, s, step):
    s["step"] = step; s["expect"] = len(ops); s["last_ops"] = ops; save(s)
    say({"do": "hf_fs", "op_lists": ops, "then": f"{NEXT} --results <saved path of each call, in order>"})
def entries(path):
    out = []
    for r in json.load(open(path)).get("results", []):
        res = r.get("result") or {}
        if res.get("op") == "search": out += [e for e in res.get("entries", []) if e.get("type") == "repo"]
    return out
def qop(model, srt): return [{"cmd": "search", "args": ["hf://models", "--tag", f"base_model:quantized:{model}", "--sort", srt, "--limit", "80"]}] + [PAD] * 6
def slug_of(model): return re.sub(r"[^a-z0-9.]+", "-", model.split("/")[1].lower()).strip("-")

args = sys.argv[1:]
results = [x for x in args[args.index("--results") + 1:] if not x.startswith("--")] if "--results" in args else []
s = json.load(open(ST)) if os.path.exists(ST) else None
if s and s.get("step") == "done": s = None
if "--abort" in args: stop(s or {}, "ended early by the session (--abort)")
if s is None and "--synced" not in args:
    sh("git fetch -q origin main")
    if sh("git status --porcelain --untracked-files=no").stdout.strip() == "": sh("git checkout -q -B main origin/main")
    sh('git config merge.wljson.driver "python3 pipeline/merge_watchlist.py %O %A %B"')
    os.execvp("python3", ["python3", os.path.abspath(__file__), *args, "--synced"])
if s is None:
    code, d = scan("lock")
    if code == 3: say({"do": "stop", "report": {"skipped": "another discovery run holds the lock", **d}})
    push("discovery lock", "discover.lock")
    s = {"step": "start", "notes": []}
    labs = list(CFG["labs"])
    ops = [[{"cmd": "search", "args": [f"hf://models/{o}", "--sort", "createdAt", "--limit", "40"]} for o in labs[i:i + 2]] + [PAD] * 6
           for i in range(0, len(labs), 2)]
    ops += [[{"cmd": "search", "args": ["hf://models", "--tag", t, "--sort", "trendingScore", "--limit", "80"]}] + [PAD] * 6
            for t in ("text-generation", "image-text-to-text")]
    hf(ops, s, "found")

if len(results) != s.get("expect"):
    say({"do": "hf_fs", "op_lists": s["last_ops"], "then": f"{NEXT} --results <saved path of each call, in order>",
         "error": f"expected {s.get('expect')} saved paths, got {len(results)}; make every call and pass every path in order"})

w = json.load(open(WL)); known = {m["model"]: m for m in w["models"]}
if s["step"] == "found":
    seen = {}
    for f in results:
        for e in entries(f): seen[e["path"]] = e
    cands, newfam, unlisted = [], [], []
    for mid, e in seen.items():
        owner, name = mid.split("/", 1)
        if QUANTISH.search(name) or mid in known: continue
        if owner in CFG["labs"]:
            if any(name.startswith(p) for p in CFG["labs"][owner]): cands.append(mid)
            elif (e.get("likes") or 0) >= 50: newfam.append({"model": mid, "likes": e.get("likes"), "updated_at": e.get("updated_at")})
        elif (e.get("likes") or 0) >= 300 and e.get("task") in ("text-generation", "image-text-to-text"):
            unlisted.append({"model": mid, "likes": e.get("likes")})
    queued = [m["model"] for m in w["models"] if m["status"] == "queued"]
    s["models"] = sorted(set(cands) | set(queued)); s["possible_new_families"] = newfam; s["unlisted_labs"] = unlisted
    hf([qop(m, "downloads") for m in s["models"]], s, "counted")

if s["step"] == "counted":
    s["counts"] = {m: {e["path"]: e for e in entries(f)} for m, f in zip(s["models"], results)}
    s["capped"] = [m for m, c in s["counts"].items() if len(c) >= 80]
    ops = [qop(m, srt) for m in s["capped"] for srt in ("createdAt", "likes", "lastModified", "trendingScore")]
    if ops: hf(ops, s, "counted2")
    s["step"] = "counted2"; results = []

if s["step"] == "counted2":
    for i, f in enumerate(results):
        m = s["capped"][i // 4]
        for e in entries(f): s["counts"][m][e["path"]] = e
    adm = CFG["admission"]; rows, admitted, updated, rejected = [], [], [], []
    for m, c in s["counts"].items():
        ups = sorted((e.get("updated_at") or "")[:10] for e in c.values() if e.get("updated_at"))
        rel = min([x for x in (ups[0] if ups else None, (known.get(m) or {}).get("released")) if x] or [None])
        row = {"model": m, "quants": len(c), "quants_1k": sum(1 for e in c.values() if (e.get("downloads") or 0) >= 1000),
               "released": rel, "searched_sorts": 5 if m in s["capped"] else 1}
        rows.append(row)
        age = (NOW.date() - dt.date.fromisoformat(rel if len(rel) == 10 else rel + "-15")).days if rel else None
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
    sys.path.insert(0, os.path.join(CODE, "pipeline")); import qcl_rank as R
    q = R.rank([x for x in w["models"] if x["status"] == "queued"])
    w["models"] = [x for x in w["models"] if x["status"] != "queued"] + q
    json.dump(w, open(WL, "w"), indent=1); open(WL, "a").write("\n")
    os.makedirs(os.path.join(CODE, "audit"), exist_ok=True)
    json.dump({"at": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "rules": CFG["admission"], "ranking": CFG["ranking"], "counted": rows,
               "admitted": admitted, "not_admitted": rejected, "possible_new_families": s["possible_new_families"],
               "unlisted_labs": s["unlisted_labs"]}, open(os.path.join(CODE, "audit", "discovery.json"), "w"), indent=1)
    today = NOW.date(); L = [f"# Model queue\n\nGenerated by `pipeline/qcl_discover.py` on {NOW:%Y-%m-%d %H:%M} UTC from Hub search results; "
                             "rules and weights are in [`discovery.json`](../discovery.json), all inputs in [`audit/discovery.json`](../audit/discovery.json).\n",
                             "## Queue (next first)\n", "| # | Model | Score | Quant repos | Used (1k+ downloads) | Released |", "|---|---|---|---|---|---|"]
    for i, x in enumerate(q, 1):
        sc = f"{R.score(x, today)[0]:.0f}" if "quants" in x else "—"
        L.append(f"| {i} | [{x['model']}](https://huggingface.co/{x['model']}) | {sc} | {x.get('quants', '—')} | {x.get('quants_1k', '—')} | {x.get('released', '—')} |")
    L += ["\n## Not admitted\n", "| Model | Quant repos | Released | Reason |", "|---|---|---|---|"]
    L += [f"| {r['model']} | {r['quants']} | {r['released'] or '—'} | {r['reason']} |" for r in sorted(rejected, key=lambda r: -r["quants"])] or ["| — | | | |"]
    L += ["\n## Possible new families (not queued until their prefix is added to `discovery.json`)\n"]
    L += [f"- [{x['model']}](https://huggingface.co/{x['model']}) ({x['likes']} likes)" for x in s["possible_new_families"]] or ["- none"]
    L += ["\n## Popular models from labs not in `discovery.json`\n"]
    L += [f"- [{x['model']}](https://huggingface.co/{x['model']}) ({x['likes']} likes)" for x in s["unlisted_labs"]] or ["- none"]
    open(os.path.join(CODE, "docs", "QUEUE.md"), "w").write("\n".join(L) + "\n")
    s.update({"admitted": admitted, "updated": len(updated)})
    s["git"] = push(f"Weekly discovery {NOW:%Y-%m-%d}: {len(admitted)} admitted, {len(updated)} queued models recounted",
                    "watchlist.json audit/discovery.json docs/QUEUE.md")
    stop(s)
stop(s, f"unexpected state {s.get('step')}")
