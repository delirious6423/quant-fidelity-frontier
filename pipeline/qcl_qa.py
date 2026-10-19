#!/usr/bin/env python3
"""QA worker: the only job that publishes the site. The session runs this and does what it prints, like qcl_next.py.

  python3 pipeline/qcl_qa.py           # start, and again after the review agents finish
  python3 pipeline/qcl_qa.py --abort   # end early: unlock and push

Run: sync to the newest main; take qa.lock; if nothing changed since the last approved commit (audit/deployed.json)
and no model waits in "review", stop. Otherwise sample up to 20 of the values added since then (a new model's values
count as added), ask Haiku reviewers to check them against their cards (docs/agents/REVIEW.md), hold every value judged
wrong (audit/holds.json, never edited), re-merge, promote models in "review" to "live", run every test, and only then
commit, push, deploy and record the approved commit. A failed test publishes nothing."""
import datetime as dt, glob, json, os, random, subprocess, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["QCL_LOCK"] = "qa.lock"; os.environ["QCL_PART"] = "-qa"
ST = os.path.join(CODE, "work", "qa_state.json"); QA = os.path.join(CODE, "work", "qa")
DEPLOYED = os.path.join(CODE, "audit", "deployed.json")
SAMPLE = 20; PER_AGENT = 10
NEXT = "python3 pipeline/qcl_qa.py"
REVIEW_PROMPT = "Follow docs/agents/REVIEW.md in the repo at {code}, with I = `work/qa/items_{k}.json` and V = `work/qa/verdict_{k}.json`."

def sh(cmd): return subprocess.run(cmd, cwd=CODE, shell=True, capture_output=True, text=True)
def scan(a):
    p = sh(f"python3 pipeline/qcl_scan.py {a}"); line = (p.stdout.strip().splitlines() or ["{}"])[-1]
    try: return p.returncode, json.loads(line)
    except ValueError: return p.returncode, {"raw": line}
def say(d): print(json.dumps(d)); sys.exit(0)
def save(s): os.makedirs(os.path.dirname(ST), exist_ok=True); json.dump(s, open(ST, "w"), indent=1)
def push(msg, paths):
    import shlex
    ex = [x for x in shlex.split(paths) if x.startswith(":!")]
    for p in shlex.split(paths):
        if not p.startswith(":!"): sh(f"git add -A -- {shlex.quote(p)} " + " ".join(shlex.quote(x) for x in ex))
    if sh("git diff --cached --quiet").returncode == 0: return "nothing to commit"
    sh(f"git -c user.name=Claude -c user.email=noreply@anthropic.com commit -qm {json.dumps(msg)}")
    for _ in range(3):
        if sh("git push -q origin HEAD:refs/heads/main").returncode == 0: return "pushed"
        sh("git pull --rebase -q origin main")
    return "push failed"
def wl(): return json.load(open(os.path.join(CODE, "watchlist.json")))
def stop(s, note=None, pushed_lock=True):
    if note: s.setdefault("notes", []).append(note)
    scan("unlock --discard")
    s["git_unlock"] = push("QA unlock", "qa.lock logs/scan") if pushed_lock else "lock never pushed"
    s["step"] = "done"; save(s)
    say({"do": "stop", "report": {k: s.get(k) for k in ("reviewed", "verdicts", "held", "promoted", "published", "deploy", "notes", "git_unlock")}})

def entries(src):
    """(repo, kind, entry, key) -> (raw, lines) for every value in a part8_machine.py source text."""
    ns = {}; exec(src, ns); out = {}
    for r in ns.get("R", []):
        for kind in ("variants", "ladder", "cross"):
            for e in r.get(kind) or []:
                nm = str(e.get("name") or e.get("variant")); Ls = [e.get("L")] + list(e.get("L_extra") or [])
                for k, v in (e.get("m") or {}).items(): out[(r["repo"], kind, nm, k)] = (str(v), [x for x in Ls if x])
                if e.get("size") is not None: out[(r["repo"], kind, nm, "size")] = (f"{int(e['size']) if float(e['size']).is_integer() else e['size']} {e.get('unit') or ''}".strip() if isinstance(e['size'], (int, float)) else f"{e['size']} {e.get('unit') or ''}".strip(), [x for x in Ls if x])   # 166.0 -> "166", as cards print it
    return out

args = sys.argv[1:]
s = json.load(open(ST)) if os.path.exists(ST) else None
if s and s.get("step") == "done": s = None
if "--abort" in args: stop(s or {}, "ended early by the session (--abort)")
if s is None and "--synced" not in args:   # newest main, newest code
    sh("git fetch -q origin main")
    if sh("git status --porcelain --untracked-files=no").stdout.strip() == "": sh("git checkout -q -B main origin/main")
    sh('git config merge.wljson.driver "python3 pipeline/merge_watchlist.py %O %A %B"')
    os.execvp("python3", ["python3", os.path.abspath(__file__), *args, "--synced"])

if s is None:
    code, d = scan("lock")
    if code == 3: say({"do": "stop", "report": {"skipped": "another QA run holds the lock", **d}})
    head = sh("git rev-parse HEAD").stdout.strip()
    base = (json.load(open(DEPLOYED)) if os.path.exists(DEPLOYED) else {}).get("commit")
    review = [m["slug"] for m in wl()["models"] if m["status"] == "review"]
    changed = [l for l in sh(f"git diff --name-only {base} {head} -- models build.py assemble.py discovery.py web").stdout.split()
               if "/verified/" in l or "/snapshots/" in l or not l.startswith("models/")] if base else ["(first run)"]   # data or site code
    s = {"step": "start", "head": head, "base": base, "review": review, "notes": []}
    if not changed and not review: sh("rm -f qa.lock"); s["step"] = "done"; save(s); say({"do": "stop", "report": {"skipped": "nothing new since the last published commit", "commit": head}})
    push("QA lock", "qa.lock")
    # values added since the approved commit (a model in review: all of its values)
    items = []
    for p8 in sorted(glob.glob(f"{CODE}/models/*/verified/part8_machine.py")):
        slug = p8.split("/models/")[1].split("/")[0]
        st = next((m["status"] for m in wl()["models"] if m["slug"] == slug), "live")
        if st not in ("live", "review"): continue
        rel = os.path.relpath(p8, CODE)
        old = {} if (st == "review" or not base) else entries(sh(f"git show {base}:{rel}").stdout or "R=[]")
        if not base and st != "review": continue   # first run: everything live was already reviewed when it was published
        idx = json.load(open(f"{CODE}/models/{slug}/snapshots/index.json"))["repos"]
        for key, (raw, Ls) in entries(open(p8).read()).items():
            if old.get(key, (None,))[0] == raw: continue
            rp, kind, nm, k = key
            if rp not in idx: continue
            items.append({"slug": slug, "repo": rp, "kind": {"variants": "own"}.get(kind, kind), "group": kind, "entry": nm, "key": k, "raw": raw, "card": f"models/{slug}/{idx[rp]['file']}", "lines": Ls})
    random.Random(head).shuffle(items)
    pick = items[:SAMPLE]
    for i, it in enumerate(pick): it["id"] = f"q{i}"
    s["new_values"] = len(items); s["reviewed"] = len(pick)
    if not pick: s["step"] = "reviewed"; save(s)
    else:
        os.makedirs(QA, exist_ok=True)
        for f in glob.glob(f"{QA}/*.json"): os.remove(f)
        chunks = [pick[i:i + PER_AGENT] for i in range(0, len(pick), PER_AGENT)]
        for k, c in enumerate(chunks): json.dump(c, open(f"{QA}/items_{k}.json", "w"), indent=1)
        json.dump(pick, open(f"{QA}/sample.json", "w"), indent=1)
        s["chunks"] = len(chunks); s["step"] = "reviewed"; save(s)
        say({"do": "agents", "prompts": [REVIEW_PROMPT.format(code=CODE, k=k) for k in range(len(chunks))], "max_parallel": 2,
             "model": "haiku", "effort": "high", "then": NEXT})

if s["step"] == "reviewed":
    pick = json.load(open(f"{QA}/sample.json")) if os.path.exists(f"{QA}/sample.json") and s.get("reviewed") else []
    by = {it["id"]: it for it in pick}; verdicts = {"ok": 0, "wrong": 0, "unsure": 0, "missing": 0}; held = []
    got = {}
    for f in glob.glob(f"{QA}/verdict_*.json"):
        try: got.update({v["id"]: v for v in json.load(open(f)) if isinstance(v, dict) and v.get("id") in by})
        except Exception: pass
    for i, it in by.items():
        v = got.get(i)
        if not v: verdicts["missing"] += 1; continue
        verdicts[v.get("verdict") if v.get("verdict") in verdicts else "unsure"] += 1
        if v.get("verdict") == "wrong": held.append((it, str(v.get("reason", ""))[:300]))
    s["verdicts"] = verdicts; s["held"] = len(held)
    touched = set()
    for it, why in held:   # hold the entry; the number itself is never edited
        hp = f"{CODE}/models/{it['slug']}/audit/holds.json"
        hs = json.load(open(hp)) if os.path.exists(hp) else []
        if not any(h["repo"] == it["repo"] and h["entry"] == it["entry"] for h in hs):   # holds are per entry (row)
            hs.append({"repo": it["repo"], "entry": it["entry"], "reason": f"QA review {dt.datetime.now(dt.timezone.utc):%Y-%m-%d}: {it['key']}: {why}"})
            json.dump(hs, open(hp, "w"), indent=1); touched.add(it["slug"])
    for slug in touched:
        sh(f"QCL_ROOT=models/{slug} python3 pipeline/qcl_watch.py merge"); sh(f"QCL_ROOT=models/{slug} python3 pipeline/gen_auto.py")
    # promote finished new models, then run every test; a failure publishes nothing
    w = wl(); promoted = [m["slug"] for m in w["models"] if m["status"] == "review"]
    for m in w["models"]:
        if m["status"] == "review": m["status"] = "live"
    json.dump(w, open(os.path.join(CODE, "watchlist.json"), "w"), indent=1)
    code, d = scan("publish")
    if code != 0 and promoted:   # keep the live site moving: retry without the new model(s)
        s["notes"].append(f"new model failed the tests, left in review: {promoted}: {str(d.get('tail', ''))[-300:]}")
        w = wl()
        for m in w["models"]:
            if m["slug"] in promoted: m["status"] = "review"
        json.dump(w, open(os.path.join(CODE, "watchlist.json"), "w"), indent=1); promoted = []
        code, d = scan("publish")
    if code != 0:
        sh("git checkout -- . && git clean -fdq models")
        stop(s, f"tests failed, nothing published: {d.get('error')}: {str(d.get('tail', ''))[-600:]}")
    try:   # the record compares with the committed watch list, so a re-run of this step still reports the promotion
        before = {m["slug"]: m["status"] for m in json.loads(sh("git show HEAD:watchlist.json").stdout)["models"]}
        promoted = sorted({*promoted, *(m["slug"] for m in wl()["models"] if m["status"] == "live" and before.get(m["slug"]) == "review")})
    except Exception: pass
    s["promoted"] = promoted; s["published"] = d.get("progress_line")
    os.makedirs(os.path.dirname(DEPLOYED), exist_ok=True)
    json.dump({"commit": s["head"], "at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "reviewed": s.get("reviewed"), "verdicts": verdicts,
               "held": len(held), "promoted": promoted}, open(DEPLOYED, "w"), indent=1)
    s["git"] = push(f"QA {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC: reviewed {s.get('reviewed', 0)} new values, held {len(held)}"
                    + (f", promoted {', '.join(promoted)}" if promoted else "") + "; published",
                    ". ':!qa.lock' ':!scan.lock' ':!ingest.lock' ':!logs/scan'")
    if s["git"] == "pushed":
        r = sh("gh api -X POST repos/delirious6423/quant-fidelity-frontier/actions/workflows/pages.yml/dispatches -f ref=main")
        s["deploy"] = "dispatched" if r.returncode == 0 else f"dispatch failed: {r.stderr[-200:]}"
    stop(s)
stop(s, f"unexpected state {s.get('step')}")
