#!/usr/bin/env python3
"""Deterministic steps of the hourly scan (docs/HOURLY.md). The session only makes connector calls and starts Haiku
agents between these commands; everything that decides, splits, merges, builds or tests is here.

  python3 pipeline/qcl_scan.py lock                  # take the run lock (exit 3 if another run holds a fresh one)
  python3 pipeline/qcl_scan.py elapsed [MIN]         # minutes since this run took the lock; exit 1 once past MIN (default 40)
  python3 pipeline/qcl_scan.py plan                  # which models to scan / ingest; creates model.json for an ingest
  python3 pipeline/qcl_scan.py ingest-state SLUG     # where a (possibly interrupted) ingest stands: catalogue, fetch or map
  python3 pipeline/qcl_scan.py fetched SLUG SHARD    # record that a fetch agent finished this shard
  python3 pipeline/qcl_scan.py checkpoint SLUG       # mark an unfinished ingest "ingesting" so the next run resumes it
  python3 pipeline/qcl_scan.py search-ops SLUG       # the padded hf_fs search op lists for one model
  python3 pipeline/qcl_scan.py catalogue SLUG [--full] FILE..  # merge search results; split the delta into fetch shards
  python3 pipeline/qcl_scan.py prepare SLUG          # index, triage, copies, what to map, mapping queues
  python3 pipeline/qcl_scan.py finish SLUG           # check every claimed queue, record, merge, catalogue the rest
  python3 pipeline/qcl_scan.py publish [--ingested SLUG]   # build all, assemble, run all tests, update watch list + PROGRESS
  python3 pipeline/qcl_scan.py unlock

Every command prints one JSON line. Exit code 0 = go on; 2 = stop this model (nothing to do); 1 = failure: stop and report."""
import datetime as dt, glob, json, os, subprocess, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WL = os.path.join(CODE, "watchlist.json"); LOCK = os.path.join(CODE, "scan.lock")
PAD = {"cmd": "cat", "args": ["hf://README.md", "--max-bytes", "8737"]}
FORMATS = ["gguf", "exl3", "nvfp4", "mlx", "awq", "gptq", "fp8", "int4", "int8", "mxfp4", "w4a16", "autoround", "bnb", "2bit", "3bit", "4bit", "5bit", "6bit", "8bit", "abliterated", "uncensored", "heretic", "mtp"]
now = dt.datetime.now(dt.timezone.utc)
LOCK_H = 0.92        # a run stops starting work at 40-45 min and ends by ~55 min; a lock older than 55 min belongs to a dead session

LOG = os.path.join(CODE, "work", "scanlog.jsonl")   # every command's result line; `unlock` files it under logs/scan/ for the commit
def out(d, code=0):
    print(json.dumps(d))
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        rec = {"at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "cmd": " ".join(sys.argv[1:3]), "exit": code}
        rec.update({k: v for k, v in d.items() if k in ("delta", "ingest", "resume", "phase", "shards", "search_entries", "catalogue", "in_index", "gated",
                    "todo", "cards", "queues", "merge", "error", "rc", "tests", "progress_line", "minutes", "locked", "page_full", "slug", "status", "bad", "pending")})
        if "tail" in d: rec["tail"] = str(d["tail"])[-600:]
        if "built" in d: rec["built"] = {k: [v.get("repos"), v.get("values")] for k, v in d["built"].items()}
        if "op_lists" in d: rec["calls"] = len(d["op_lists"])
        open(LOG, "a").write(json.dumps(rec).replace(CODE + "/", "").replace(os.path.expanduser("~"), "~") + "\n")   # no local paths in a public repo
    except Exception: pass
    sys.exit(code)
def run(cmd, env=None, check=True):
    p = subprocess.run(cmd, cwd=CODE, shell=True, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and p.returncode != 0: out({"error": cmd, "rc": p.returncode, "tail": (p.stdout + p.stderr)[-1500:]}, 1)
    return p
def wl(): return json.load(open(WL))
def M(slug): return os.path.join(CODE, "models", slug)
def entry(slug): return next(m for m in wl()["models"] if m["slug"] == slug)
def env(slug): return {"QCL_ROOT": M(slug)}

def max_gb(model):
    """Size plausibility ceiling for build.py: BF16 is ~2 bytes per parameter, so 2.5 GB per billion parameters when the id
    names them (e.g. 550B -> 1375 GB); 1000 GB otherwise (the default 400 GB rejected GLM-5.3-Flash's 642 GB BF16 files)."""
    import re
    m = re.search(r"(\d+(?:\.\d+)?)B(?![a-z])", model.split("/")[-1])
    return max(400, round(2.5 * float(m.group(1)))) if m else 1000

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "lock":
    if os.path.exists(LOCK):
        age = (now - dt.datetime.fromisoformat(open(LOCK).read().strip())).total_seconds() / 3600
        if age < LOCK_H: out({"locked": True, "age_h": round(age, 2), "note": "another run is in progress; stop"}, 3)
    open(LOCK, "w").write(now.isoformat()); out({"locked": False, "took": now.isoformat()})
elif cmd == "elapsed":
    lim = float(sys.argv[2]) if len(sys.argv) > 2 else 40
    mins = (now - dt.datetime.fromisoformat(open(LOCK).read().strip())).total_seconds() / 60 if os.path.exists(LOCK) else 0
    out({"minutes": round(mins, 1), "limit": lim, "over": mins > lim}, 1 if mins > lim else 0)
elif cmd == "unlock":
    if os.path.exists(LOG) and "--discard" not in sys.argv:   # a run that fetched cards must publish (or record a failed publish) before it unlocks
        recs = [json.loads(l) for l in open(LOG) if l.strip()]
        live = {m["slug"] for m in wl()["models"] if m["status"] == "live"}   # Part B (ingest) catalogues publish only when the model is done
        last_cat = max((i for i, r in enumerate(recs) if r["cmd"].startswith("catalogue") and r.get("shards") and r["cmd"].split()[-1] in live), default=-1)
        if last_cat >= 0 and not any(r["cmd"].startswith("publish") for r in recs[last_cat:]):
            print(json.dumps({"error": "cards were fetched this run but publish was not run; run `qcl_scan.py publish` now (or `unlock --discard` after discarding the changes)"})); sys.exit(1)
    if os.path.exists(LOCK): os.remove(LOCK)
    for f in glob.glob(f"{CODE}/logs/scan/*-partial.jsonl"):   # superseded by the full log below
        os.remove(f)
    if os.path.exists(LOG):    # keep this run's command log with the unlock commit (logs/scan/<UTC>.jsonl), so a later check can see what happened
        os.makedirs(f"{CODE}/logs/scan", exist_ok=True); os.replace(LOG, f"{CODE}/logs/scan/{now:%Y%m%dT%H%M}Z.jsonl")
    out({"unlocked": True})
elif cmd == "plan":
    w = wl(); live = [m["slug"] for m in w["models"] if m["status"] == "live"]
    last = dt.datetime.fromisoformat(w["last_ingest_at"]); due = (now - last).total_seconds() / 3600 >= w["min_hours_between_ingests"]
    q = next((m for m in w["models"] if m["status"] == "ingesting"), None)     # an unfinished ingest resumes first, without waiting
    resume = bool(q)
    q = q or next((m for m in w["models"] if m["status"] == "queued"), None)
    ingest = None
    if (due or resume) and q:
        ingest = q["slug"]; d = M(ingest); os.makedirs(f"{d}/cache/cards", exist_ok=True)
        if not os.path.exists(f"{d}/model.json"):
            nm = q["model"].split("/")[-1]
            json.dump({"model": q["model"], "name": nm, "slug": ingest,
                       "size_note": "Sizes are as published on each card: download size, GPU memory or a single file. GiB figures are converted to GB. Each row says which size it is.",
                       "pin_note": f"Cards for this model were captured through the Hugging Face connector from {now:%d %b %Y} and are pinned by the SHA-256 of their exact text, not by commit. Source links point to the current card, which may have changed since capture.",
                       "coverage_note": "claims made only off-site or only in images, repos whose cards are gated (access denied to our account), and repos the Hub search did not return",
                       "others": [], "max_size_gb": max_gb(q["model"])}, open(f"{d}/model.json", "w"), indent=1)
    out({"delta": live, "ingest": ingest, "resume": resume, "ingest_due": due, "next_queued": q["slug"] if q else None})
elif cmd == "ingest-state":
    slug = sys.argv[2]; c = f"{M(slug)}/cache"
    if not os.path.exists(f"{c}/hub_catalogue.json"): out({"slug": slug, "phase": "catalogue"})
    left = sorted(os.path.basename(f)[4:-5] for f in glob.glob(f"{c}/ids_*.json") if not os.path.exists(f"{c}/done_{os.path.basename(f)[4:-5]}"))
    if left: out({"slug": slug, "phase": "fetch", "shards": left})
    out({"slug": slug, "phase": "map"})
elif cmd == "fetched":
    slug, sh = sys.argv[2], sys.argv[3]; open(f"{M(slug)}/cache/done_{sh}", "w").write(now.isoformat()); out({"slug": slug, "shard": sh, "done": True})
elif cmd == "checkpoint":
    slug = sys.argv[2]; w = wl()
    for m in w["models"]:
        if m["slug"] == slug and m["status"] == "queued": m["status"] = "ingesting"
    json.dump(w, open(WL, "w"), indent=1)
    if os.path.exists(LOG) and os.path.exists(LOCK):   # the run's log so far goes with every checkpoint, so a run that dies later still shows how far it got
        os.makedirs(f"{CODE}/logs/scan", exist_ok=True)
        st = dt.datetime.fromisoformat(open(LOCK).read().strip())
        import shutil; shutil.copy(LOG, f"{CODE}/logs/scan/{st:%Y%m%dT%H%M}Z-partial.jsonl")
    out({"slug": slug, "status": entry(slug)["status"], "next": f"git add models/{slug} watchlist.json logs/scan && git commit -qm 'Ingest checkpoint: {slug}' && push; the next run resumes"})
elif cmd == "search-ops":
    # Every op list = one search (limit 80, about 21K chars; 90 measured 78.5K total) + 6 README pads (52K chars). That keeps each result between
    # ~52K (always saved to a file, never inline) and ~77K (inside the connector's 80K shared output budget).
    # delta runs: the 80 most recently modified quants; ingest runs or a full recent page: --full (several sorts + format words)
    slug = sys.argv[2]; tag = f"base_model:quantized:{entry(slug)['model']}"
    def op(*extra): return [{"cmd": "search", "args": ["hf://models", *extra, "--tag", tag, "--limit", "80"]}] + [PAD] * 6
    if "--full" in sys.argv:
        ops = [op("--sort", srt) for srt in ("lastModified", "createdAt", "downloads", "likes", "id")]
        ops += [op(f, "--sort", "lastModified") for f in FORMATS]
    else:
        ops = [op("--sort", "lastModified")]
    out({"slug": slug, "mode": "full" if "--full" in sys.argv else "recent", "calls": len(ops), "op_lists": ops,
         "note": "call hf_fs once per op list, exactly as given; every result must be saved to a file (never transcribe inline output)"})
elif cmd == "catalogue":
    slug = sys.argv[2]; full = "--full" in sys.argv
    files = [a for a in sys.argv[3:] if not a.startswith("--")]
    trunc, oldest = [], None
    for f in files:
        for r in json.load(open(f)).get("results", []):
            res = r.get("result") or {}
            if res.get("truncated") or r.get("output_truncated"): trunc.append(f)
            ups = [e.get("updated_at") or "" for e in res.get("entries", [])]
            if ups: oldest = min([oldest] + ups) if oldest else min(ups)
    n_entries = sum(len((r.get("result") or {}).get("entries", [])) for f in files for r in json.load(open(f)).get("results", []))
    if n_entries == 0:   # no search results at all: the calls were not made or the paths are wrong; never record an empty catalogue
        out({"error": "no search results in the given files; make the hf_fs calls and pass their saved paths", "files": len(files)}, 1)
    w = wl(); me = entry(slug); last = me.get("last_scan_at") or ""
    page_full = (not full) and bool(trunc) and (not last or (oldest or "") > last)   # the recent page may not reach back to the last scan
    if page_full and "--accept-partial" not in sys.argv:
        out({"page_full": True, "oldest_in_page": oldest, "last_scan_at": last, "next": "rerun search-ops with --full, then catalogue with all files"}, 1)
    p = run(f"python3 pipeline/qcl_catalogue.py {M(slug)} " + " ".join(files))
    info = json.loads(p.stdout.strip().splitlines()[-1])
    for m in w["models"]:
        if m["slug"] == slug: m["last_scan_at"] = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    json.dump(w, open(WL, "w"), indent=1)
    delta = json.load(open(f"{M(slug)}/cache/delta.json")); ts = now.strftime("%Y%m%d%H"); shards = []
    for i in range(0, len(delta), 100):
        s_ = f"{ts}{chr(97 + i // 100)}"; json.dump(delta[i:i + 100], open(f"{M(slug)}/cache/ids_{s_}.json", "w")); shards.append(s_)
    out({**info, "truncated_results": len(trunc), "shards": shards, "next": "fetch each shard with a Haiku agent (docs/agents/FETCH.md)" if shards else "nothing changed"}, 0 if shards else 2)
elif cmd == "prepare":
    slug = sys.argv[2]; e = env(slug)
    run(f"python3 pipeline/qcl_newmodel.py {M(slug)}")
    run("python3 pipeline/autoextract.py triage", e); run("python3 pipeline/detect_copies.py", e)
    ms = json.loads(run("python3 pipeline/qcl_mapset.py --record", e).stdout.strip().splitlines()[-1])   # --record first: maps from an interrupted run count
    W = os.path.join(CODE, "work", slug); subprocess.run(f"rm -rf {W}", shell=True)
    q = json.loads(run(f"python3 pipeline/qcl_queue.py {W}", e).stdout.strip().splitlines()[-1])
    if not q["queues"]:   # nothing to map, but fetched cards still change the index: re-merge (stale entries are re-pinned or quarantined) and relist unreviewed repos
        mg = run("python3 pipeline/qcl_watch.py merge", e).stdout.strip().splitlines()[-1]; run("python3 pipeline/gen_auto.py", e)
        out({**ms, **q, "merge": mg, "next": "no cards to map; merged and relisted, go to publish"}, 2)
    out({**ms, **q, "work": os.path.relpath(W, CODE), "next": "start Haiku workers per docs/agents/MAP.md for RUN=A and RUN=B"}, 0)
elif cmd == "finish":
    slug = sys.argv[2]; e = env(slug); W = os.path.join(CODE, "work", slug)
    pending = sorted(os.listdir(f"{W}/pending")) if os.path.isdir(f"{W}/pending") else []
    if pending: out({"error": "queues still pending; start more workers", "pending": pending}, 1)
    bad = []
    for q in sorted(glob.glob(f"{W}/claimed/*.txt")):   # none when every card was already mapped by an earlier run
        r = os.path.basename(q)[0]
        p = run(f"python3 pipeline/qcl_mapcheck.py {q} {M(slug)}/maps/{r}", check=False)
        if p.returncode != 0:
            bad.append({"queue": os.path.basename(q), "problems": p.stdout.strip().splitlines()[:8]})
            os.replace(q, f"{W}/pending/{os.path.basename(q)}")          # back to pending for a fresh worker
    if bad: out({"error": "some maps failed the check; their queues are back in pending; rerun workers then finish", "bad": bad}, 1)
    run("python3 pipeline/qcl_mapset.py --record", e)
    mg = run("python3 pipeline/qcl_watch.py merge", e).stdout.strip().splitlines()[-1]
    run("python3 pipeline/gen_auto.py", e)
    out({"merge": mg})
elif cmd == "publish":
    ing = sys.argv[sys.argv.index("--ingested") + 1] if "--ingested" in sys.argv else None
    if ing:   # the new model must build and pass its UI tests before it goes live
        run(f"python3 build.py", env(ing)); run(f"python3 tests/test_ui.py", {"QCL_SITE": f"models/{ing}/site"})
        w = wl()
        for m in w["models"]:
            if m["slug"] == ing: m["status"] = "live"   # from queued or ingesting
        w["last_ingest_at"] = now.isoformat(); json.dump(w, open(WL, "w"), indent=1)
    stats = {}
    for mj in sorted(glob.glob(f"{CODE}/models/*/model.json")):
        slug = json.load(open(mj))["slug"]
        if next((m["status"] for m in wl()["models"] if m["slug"] == slug), "live") != "live": continue
        p = run("python3 build.py", env(slug)); stats[slug] = json.loads(p.stdout.strip().splitlines()[-1].split(" | ")[0])
    run("python3 assemble.py")
    run("python3 tests/test_gates.py"); run("python3 tests/test_claims.py"); run("python3 tests/test_mcp.py"); run("python3 tests/test_discovery.py")
    for t in ("test_model_panels", "test_mapping_panels", "test_combined_columns", "test_qualifiers", "test_current"): run(f"python3 tests/{t}.py")
    for slug in stats: run("python3 tests/test_ui.py", {"QCL_SITE": f"models/{slug}/site"})
    line = f"- {now:%Y-%m-%d %H:%M} UTC hourly scan: " + "; ".join(f"{s} {v['repos']} repos, {v['values']} values" for s, v in stats.items()) + (f"; ingested {ing}" if ing else "")
    with open(f"{CODE}/PROGRESS.md", "a") as f: f.write(line + "\n")
    out({"built": stats, "tests": "all passed", "progress_line": line, "next": "git add -A, commit, push, dispatch pages.yml, then unlock"})
else:
    print(__doc__); sys.exit(1)
