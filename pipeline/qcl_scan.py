#!/usr/bin/env python3
"""Deterministic steps of the hourly scan (docs/HOURLY.md). The session only makes connector calls and starts Haiku
agents between these commands; everything that decides, splits, merges, builds or tests is here.

  python3 pipeline/qcl_scan.py lock                  # take the run lock (exit 3 if another run holds a fresh one)
  python3 pipeline/qcl_scan.py plan                  # which models to scan / ingest; creates model.json for an ingest
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

def out(d, code=0): print(json.dumps(d)); sys.exit(code)
def run(cmd, env=None, check=True):
    p = subprocess.run(cmd, cwd=CODE, shell=True, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if check and p.returncode != 0: out({"error": cmd, "rc": p.returncode, "tail": (p.stdout + p.stderr)[-1500:]}, 1)
    return p
def wl(): return json.load(open(WL))
def M(slug): return os.path.join(CODE, "models", slug)
def entry(slug): return next(m for m in wl()["models"] if m["slug"] == slug)
def env(slug): return {"QCL_ROOT": M(slug)}

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "lock":
    if os.path.exists(LOCK):
        age = (now - dt.datetime.fromisoformat(open(LOCK).read().strip())).total_seconds() / 3600
        if age < 2: out({"locked": True, "age_h": round(age, 2), "note": "another run is in progress; stop"}, 3)
    open(LOCK, "w").write(now.isoformat()); out({"locked": False, "took": now.isoformat()})
elif cmd == "unlock":
    if os.path.exists(LOCK): os.remove(LOCK)
    out({"unlocked": True})
elif cmd == "plan":
    w = wl(); live = [m["slug"] for m in w["models"] if m["status"] == "live"]
    last = dt.datetime.fromisoformat(w["last_ingest_at"]); due = (now - last).total_seconds() / 3600 >= w["min_hours_between_ingests"]
    q = next((m for m in w["models"] if m["status"] == "queued"), None)
    ingest = None
    if due and q:
        ingest = q["slug"]; d = M(ingest); os.makedirs(f"{d}/cache/cards", exist_ok=True)
        if not os.path.exists(f"{d}/model.json"):
            nm = q["model"].split("/")[-1]
            json.dump({"model": q["model"], "name": nm, "slug": ingest,
                       "size_note": "Sizes are as published on each card: download size, GPU memory or a single file. GiB figures are converted to GB. Each row says which size it is.",
                       "pin_note": f"Cards for this model were captured through the Hugging Face connector from {now:%d %b %Y} and are pinned by the SHA-256 of their exact text, not by commit. Source links point to the current card, which may have changed since capture.",
                       "coverage_note": "claims made only off-site or only in images, repos whose cards are gated (access denied to our account), and repos the Hub search did not return",
                       "others": []}, open(f"{d}/model.json", "w"), indent=1)
    out({"delta": live, "ingest": ingest, "ingest_due": due, "next_queued": q["slug"] if q else None})
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
    ms = json.loads(run("python3 pipeline/qcl_mapset.py", e).stdout.strip().splitlines()[-1])
    W = os.path.join(CODE, "work", slug); subprocess.run(f"rm -rf {W}", shell=True)
    q = json.loads(run(f"python3 pipeline/qcl_queue.py {W}", e).stdout.strip().splitlines()[-1])
    out({**ms, **q, "work": os.path.relpath(W, CODE), "next": "start Haiku workers per docs/agents/MAP.md for RUN=A and RUN=B" if q["queues"] else "no cards to map"}, 0 if q["queues"] else 2)
elif cmd == "finish":
    slug = sys.argv[2]; e = env(slug); W = os.path.join(CODE, "work", slug)
    pending = sorted(os.listdir(f"{W}/pending")) if os.path.isdir(f"{W}/pending") else []
    if pending: out({"error": "queues still pending; start more workers", "pending": pending}, 1)
    bad = []
    for q in sorted(glob.glob(f"{W}/claimed/*.txt")):
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
            if m["slug"] == ing: m["status"] = "live"
        w["last_ingest_at"] = now.isoformat(); json.dump(w, open(WL, "w"), indent=1)
    stats = {}
    for mj in sorted(glob.glob(f"{CODE}/models/*/model.json")):
        slug = json.load(open(mj))["slug"]
        if next((m["status"] for m in wl()["models"] if m["slug"] == slug), "live") != "live": continue
        p = run("python3 build.py", env(slug)); stats[slug] = json.loads(p.stdout.strip().splitlines()[-1].split(" | ")[0])
    run("python3 assemble.py")
    run("python3 tests/test_gates.py"); run("python3 tests/test_claims.py"); run("python3 tests/test_mcp.py")
    for slug in stats: run("python3 tests/test_ui.py", {"QCL_SITE": f"models/{slug}/site"})
    line = f"- {now:%Y-%m-%d %H:%M} UTC hourly scan: " + "; ".join(f"{s} {v['repos']} repos, {v['values']} values" for s, v in stats.items()) + (f"; ingested {ing}" if ing else "")
    with open(f"{CODE}/PROGRESS.md", "a") as f: f.write(line + "\n")
    out({"built": stats, "tests": "all passed", "progress_line": line, "next": "git add -A, commit, push, dispatch pages.yml, then unlock"})
else:
    print(__doc__); sys.exit(1)
