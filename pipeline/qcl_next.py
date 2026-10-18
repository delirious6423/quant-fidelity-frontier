#!/usr/bin/env python3
"""The hourly scan as a state machine: the coordinating session only does what this prints, then runs it again.

  python3 pipeline/qcl_next.py                       # first call of a run, and after agents finish
  python3 pipeline/qcl_next.py --results P1 P2 ...   # after the hf_fs calls it asked for, saved paths in the same order
  python3 pipeline/qcl_next.py --abort               # the session has to stop early: checkpoint, unlock and push

Every call prints one JSON object with "do":
  "hf_fs"   call the hf_fs tool once per entry of op_lists, exactly as given; then run `then` with the saved paths in order
            (a result is saved when the tool message says "Output has been saved to <path>" or "Full output saved to: <path>")
  "agents"  start one subagent per entry of prompts (model haiku, effort high, at most max_parallel at once), wait for all
            of them, then run `then`
  "stop"    the run is over; reply with `report`
Everything else (lock, catalogue, prepare, merge, build, tests, publish, git commit and push, checkpoints, unlock) is done
here, so no step depends on the session's judgement. State for the run lives in work/next_state.json."""
import datetime as dt, glob, json, math, os, subprocess, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = "python3 pipeline/qcl_scan.py"
ST = os.path.join(CODE, "work", "next_state.json")
NEXT = "python3 pipeline/qcl_next.py"
FETCH_PROMPT = ("Follow docs/agents/FETCH.md in the repo at {code}, with S = `{shard}` and M = `models/{slug}`. "
                "Use only Bash and the hf_fs tool. A result is saved to a file when the tool message says "
                "'Output has been saved to <path>' or 'Full output saved to: <path>'; use that path.")
MAP_PROMPT = "Follow docs/agents/MAP.md in the repo at {code}, with W = `work/{slug}`, RUN = `{run}`, M = `models/{slug}`."

def sh(cmd, check=False):
    p = subprocess.run(cmd, cwd=CODE, shell=True, capture_output=True, text=True)
    if check and p.returncode: raise RuntimeError(f"{cmd}: {(p.stdout + p.stderr)[-800:]}")
    return p
def scan(args):
    p = sh(f"{SCAN} {args}"); line = (p.stdout.strip().splitlines() or ["{}"])[-1]
    try: d = json.loads(line)
    except ValueError: d = {"raw": line}
    return p.returncode, d
def git(*cmds):
    for c in cmds: sh(c)
def push(msg, paths):
    import shlex
    for p in shlex.split(paths):   # one path at a time: a missing path (e.g. scan.lock after unlock) must not cancel the rest
        if p.startswith(":!"): continue
        sh(f"git add -A -- {shlex.quote(p)}" + "".join(f" {shlex.quote(x)}" for x in shlex.split(paths) if x.startswith(":!")))
    if sh("git diff --cached --quiet").returncode == 0: return "nothing to commit"
    sh(f"git -c user.name=Claude -c user.email=noreply@anthropic.com commit -qm {json.dumps(msg)}")
    for _ in range(3):   # push the current commit to main explicitly: a scan clone may be on a detached HEAD
        if sh("git push -q origin HEAD:refs/heads/main").returncode == 0: return "pushed"
        sh("git pull --rebase -q origin main")
    s_ = load() or {}; s_.setdefault("notes", []).append(f"push failed: {msg}"); save(s_)
    return "push failed"
def load(): return json.load(open(ST)) if os.path.exists(ST) else None
def save(s): os.makedirs(os.path.dirname(ST), exist_ok=True); json.dump(s, open(ST, "w"), indent=1)
def minutes():
    _, d = scan("elapsed 999"); return d.get("minutes", 0)
def say(d): print(json.dumps(d)); sys.exit(0)
def hf(op_lists, s, step):
    s["step"] = step; s["expect"] = len(op_lists); s["last_ops"] = op_lists; save(s)
    say({"do": "hf_fs", "op_lists": op_lists, "then": f"{NEXT} --results <saved path of each call, in order>"})
def agents(prompts, s, step, par=8):
    s["step"] = step; save(s)
    say({"do": "agents", "prompts": prompts, "max_parallel": par, "model": "haiku", "effort": "high", "then": NEXT})

def finish_run(s, note=None):
    if note: s["notes"].append(note)
    s["minutes"] = minutes()
    if s.get("ingest") and s.get("b_progress"): checkpoint(s)
    code, d = scan("unlock")
    if code == 1:   # a fetch without publish: Part A was not published, discard it rather than leave the lock
        git("git checkout -- models watchlist.json PROGRESS.md", "git clean -fdq models")
        scan("unlock --discard"); s["notes"].append("Part A discarded at unlock")
    s["git_unlock"] = push(f"scan unlock ({ {'A': 'live updates', 'B': 'new model'}.get(PART, 'scan') })", f"{LOCKF} logs/scan")
    s["step"] = "done"; save(s)
    say({"do": "stop", "report": {k: s.get(k) for k in ("minutes", "part_a", "part_b", "notes", "git_unlock")}})

def checkpoint(s):
    slug = s["ingest"]; scan(f"checkpoint {slug}")
    s["part_b"]["checkpoints"] = s["part_b"].get("checkpoints", 0) + 1
    return push(f"Ingest checkpoint: {slug}", f"models/{slug} watchlist.json logs/scan")

# ---------------------------------------------------------------- run
args = sys.argv[1:]
results = [x for x in args[args.index("--results") + 1:] if not x.startswith("--")] if "--results" in args else []
PART = args[args.index("--part") + 1] if "--part" in args else "AB"   # A = live updates task, B = new-model task, AB = both
LOCKF = "ingest.lock" if PART == "B" else "scan.lock"
os.environ["QCL_LOCK"] = LOCKF; os.environ["QCL_PART"] = {"A": "-live", "B": "-ingest"}.get(PART, "")
ST = os.path.join(CODE, "work", f"next_state{os.environ['QCL_PART']}.json")
NEXT = f"python3 pipeline/qcl_next.py --part {PART}"
def nonlive_excludes():
    w = json.load(open(os.path.join(CODE, "watchlist.json")))
    return " ".join(f"':!models/{m['slug']}'" for m in w["models"] if m["status"] != "live")
s = load()
if s and s.get("step") == "done": s = None; os.remove(ST)
if "--abort" in args:   # the session must end early: save Part B progress, unlock, push
    s = s or {"notes": [], "part_a": {}, "part_b": {}, "ingest": None, "b_progress": False}
    finish_run(s, "ended early by the session (--abort)")
if s is None and "--synced" not in args:   # start of a run: work on the newest main (a clone may be a detached, older commit)
    sh("git fetch -q origin main")
    head, tip = sh("git rev-parse HEAD").stdout.strip(), sh("git rev-parse origin/main").stdout.strip()
    if tip and head != tip and sh("git status --porcelain --untracked-files=no").stdout.strip() == "":
        sh("git checkout -q -B main origin/main")
    sh('git config merge.wljson.driver "python3 pipeline/merge_watchlist.py %O %A %B"')   # both tasks commit the watch list
    os.execvp("python3", ["python3", os.path.abspath(__file__), *args, "--synced"])   # re-run with the newest code
if s is None:   # start of a run
    code, d = scan("lock")
    if code == 3: say({"do": "stop", "report": {"skipped": "another run holds the lock", **d}})
    _, p = scan("plan")
    s = {"step": "start", "live": p.get("delta", []), "ingest": p.get("ingest") if PART != "A" else None, "notes": [], "part_a": {}, "part_b": {}, "b_progress": False}
    if PART == "A": sh("git checkout -q -- watchlist.json 2>/dev/null; git clean -fdq models")   # plan may scaffold a new model: not this task's job
    if PART == "B":
        if not s["ingest"]: sh(f"rm -f {LOCKF}"); s["step"] = "done"; save(s); say({"do": "stop", "report": {"skipped": "no new model due", "next_queued": p.get("next_queued")}})
        push("ingest lock", LOCKF); s["step"] = "b_start"; save(s)
    elif s["ingest"]: push("scan lock", LOCKF)   # an ingest is real work: let a parallel run see the lock now
if s.get("step") == "start":
    ops = []
    for slug in s["live"]:
        _, o = scan(f"search-ops {slug}"); ops += o["op_lists"]
    hf(ops, s, "a_catalogue")

step = s["step"]
if step in ("a_catalogue", "b_catalogue") and len(results) != s.get("expect"):   # every requested call must have been made
    say({"do": "hf_fs", "op_lists": s["last_ops"], "error": f"expected {s.get('expect')} saved paths after --results, got {len(results)}. "
         "Make every hf_fs call below (one per op list, exactly as given), then pass all saved paths in order.",
         "then": f"{NEXT} --results <saved path of each call, in order>"})
if step == "a_catalogue":
    shards = {}
    for slug, path in zip(s["live"], results):
        code, d = scan(f"catalogue {slug} {path}")
        if code == 1 and d.get("page_full"): s["notes"].append(f"{slug}: more than 80 changed since last scan; only the newest 80 handled")
        if code == 1 and d.get("page_full"): code, d = scan(f"catalogue {slug} {path} --accept-partial")
        s["part_a"][slug] = {"delta": d.get("delta"), "gated": d.get("gated")}
        if d.get("shards"): shards[slug] = d["shards"]
    s["a_shards"] = shards
    if not shards:
        s["notes"].append("no changes on the live models")
        if s["ingest"]: s["step"] = "b_start"; save(s); step = "b_start"
        else: finish_run(s)
    else:
        if not s["ingest"]: push("scan lock", LOCKF)
        agents([FETCH_PROMPT.format(code=CODE, shard=sh_, slug=slug) for slug, ss in shards.items() for sh_ in ss], s, "a_fetched")

if step == "a_fetched":
    queues = {}
    for slug, ss in s["a_shards"].items():
        for sh_ in ss: scan(f"fetched {slug} {sh_}")
        code, d = scan(f"prepare {slug}")
        s["part_a"][slug].update({"to_map": d.get("todo"), "merge": d.get("merge")})
        if code == 0 and d.get("queues"): queues[slug] = d["queues"]
    s["a_queues"] = queues; s["map_rounds"] = 0
    step = "a_mapped" if not queues else None
    if queues:
        pr = []
        for slug, q in queues.items():
            n = max(1, min(2, math.ceil(q / 6)))
            pr += [MAP_PROMPT.format(code=CODE, slug=slug, run=r) for r in "AB" for _ in range(n)]
        agents(pr[:8], s, "a_mapped")

if step == "a_mapped":
    retry = []
    for slug in s.get("a_queues", {}):
        code, d = scan(f"finish {slug}")
        if code == 1: retry.append(slug); s["notes"].append(f"{slug}: finish: {d.get('error')}")
        else: s["part_a"][slug]["merge"] = d.get("merge")
    if retry and s["map_rounds"] < 2:
        s["map_rounds"] += 1
        agents([MAP_PROMPT.format(code=CODE, slug=slug, run=r) for slug in retry for r in "AB"], s, "a_mapped")
    code, d = scan("publish")
    if code == 0:
        s["part_a"]["published"] = d.get("progress_line")
        s["part_a"]["git"] = push(f"Hourly scan {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC: " + str(d.get("progress_line", ""))[2:200], ". ':!scan.lock' ':!ingest.lock' ':!logs/scan' " + nonlive_excludes())
        sh("gh api -X POST repos/delirious6423/quant-fidelity-frontier/actions/workflows/pages.yml/dispatches -f ref=main")
        s["part_a"]["deploy"] = "dispatched"
    else:
        s["part_a"]["publish_failed"] = {"error": d.get("error"), "tail": str(d.get("tail", ""))[-600:]}
        git("git checkout -- models watchlist.json PROGRESS.md", "git clean -fdq models")
    s["step"] = "b_start"; save(s); step = "b_start"

if step == "b_start":
    if not s["ingest"]: finish_run(s)
    if minutes() > 30: finish_run(s, "Part B skipped: less than 25 minutes left")
    scan("plan")   # recreates the new model's model.json if a Part A discard removed it
    _, d = scan(f"ingest-state {s['ingest']}"); ph = d.get("phase"); s["part_b"]["phase_at_start"] = ph
    if ph == "catalogue":
        _, o = scan(f"search-ops {s['ingest']} --full"); s["b_ops"] = o["op_lists"]; s["b_paths"] = []
        hf(s["b_ops"][:7], s, "b_catalogue")
    step = "b_fetch" if ph == "fetch" else "b_map"

if step == "b_catalogue":
    s["b_paths"] += results
    if len(s["b_paths"]) < len(s["b_ops"]):   # searches come in rounds of 7 calls
        hf(s["b_ops"][len(s["b_paths"]):len(s["b_paths"]) + 7], s, "b_catalogue")
    slug = s["ingest"]; code, d = scan(f"catalogue {slug} --full " + " ".join(s["b_paths"]))
    if code == 1: finish_run(s, f"catalogue failed: {d.get('error')}")
    s["part_b"]["catalogue"] = {"repos": d.get("catalogue"), "shards": len(d.get("shards") or []), "gated": d.get("gated")}
    s["b_progress"] = True; checkpoint(s)
    if minutes() > 25: finish_run(s, "catalogue done; fetching starts next run")
    step = "b_fetch"

if step == "b_fetch":
    slug = s["ingest"]; _, d = scan(f"ingest-state {slug}")
    if d.get("phase") == "fetch":
        if minutes() > 40: finish_run(s, "time is up; fetching continues next run")
        todo = d["shards"][:2]; s["b_shards"] = todo
        agents([FETCH_PROMPT.format(code=CODE, shard=x, slug=slug) for x in todo], s, "b_fetched", par=2)
    step = "b_map"

if step == "b_fetched":
    slug = s["ingest"]
    for x in s.get("b_shards", []): scan(f"fetched {slug} {x}")
    s["b_progress"] = True; s["part_b"]["shards_fetched"] = s["part_b"].get("shards_fetched", 0) + len(s.get("b_shards", [])); checkpoint(s)
    _, d = scan(f"ingest-state {slug}")
    if d.get("phase") == "fetch" or minutes() > 35: finish_run(s, "fetched up to 2 shards this run; the next run continues")
    step = "b_map"

if step == "b_map":
    slug = s["ingest"]
    if minutes() > 40: finish_run(s, "time is up; mapping continues next run")
    code, d = scan(f"prepare {slug}")
    s["part_b"]["to_map"] = d.get("todo")
    if code == 1: finish_run(s, f"prepare failed: {d.get('error')}: {str(d.get('tail', ''))[-300:]}")
    if code == 0 and d.get("queues"):
        n = max(1, min(4, math.ceil(d["queues"] / 4)))
        agents([MAP_PROMPT.format(code=CODE, slug=slug, run=r) for r in "AB" for _ in range(n)], s, "b_mapped")
    step = "b_finish"

if step == "b_mapped":
    slug = s["ingest"]; s["b_progress"] = True; checkpoint(s)
    pend = [f for f in os.listdir(f"{CODE}/work/{slug}/pending")] if os.path.isdir(f"{CODE}/work/{slug}/pending") else []
    if pend:
        if minutes() > 40: finish_run(s, f"{len(pend)} mapping queues left; the next run continues")
        agents([MAP_PROMPT.format(code=CODE, slug=slug, run=r) for r in sorted({p[0] for p in pend})], s, "b_mapped")
    step = "b_finish"

if step == "b_finish":
    slug = s["ingest"]
    code, d = scan(f"finish {slug}")
    if code == 1:
        s["b_progress"] = True; s["b_retries"] = s.get("b_retries", 0) + 1
        pend = sorted(os.listdir(f"{CODE}/work/{slug}/pending")) if os.path.isdir(f"{CODE}/work/{slug}/pending") else []
        if pend and s["b_retries"] <= 2 and minutes() < 40:   # maps that failed the check go back to workers once or twice
            agents([MAP_PROMPT.format(code=CODE, slug=slug, run=r) for r in sorted({p[0] for p in pend})], s, "b_mapped")
        finish_run(s, f"finish: {d.get('error')}; the next run re-queues")
    code, d = scan(f"publish --ingested {slug}")
    if code == 0:
        s["part_b"]["live"] = d.get("progress_line")
        s["part_b"]["git"] = push(f"Hourly scan {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC: ingested {slug}", f"models/{slug} watchlist.json PROGRESS.md")
        sh("gh api -X POST repos/delirious6423/quant-fidelity-frontier/actions/workflows/pages.yml/dispatches -f ref=main")
        s["part_b"]["deploy"] = "dispatched"; s["b_progress"] = False
        finish_run(s)
    s["part_b"]["publish_failed"] = {"error": d.get("error"), "tail": str(d.get("tail", ""))[-600:]}
    git("git checkout -- PROGRESS.md"); s["b_progress"] = True
    finish_run(s, "new model built but failed a test; it stays unpublished")

finish_run(s)
