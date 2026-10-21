#!/usr/bin/env python3
"""QA worker: the only job that publishes the site. The session runs this and does what it prints, like qcl_next.py.

  python3 pipeline/qcl_qa.py           # start, and again after the review agents finish
  python3 pipeline/qcl_qa.py --abort   # end early: unlock and push

Run: sync to the newest main; take qa.lock; if nothing changed since the last approved commit (audit/deployed.json)
and no model waits in "review", stop. Otherwise sample up to 20 of the values added since then (a new model's values
count as added), ask Haiku reviewers to check them against their cards (docs/runbooks/agents/REVIEW.md), hold every value judged
wrong (audit/holds.json, never edited), re-merge, promote models in "review" to "live", run every test, and only then
commit, push and request deployment. The deployed receipt advances only after an
exact-commit Pages workflow succeeds. Missing or uncertain reviews publish nothing."""
import datetime as dt, glob, json, os, random, subprocess, sys
try:
    from . import deploy_state as release
except ImportError:
    import deploy_state as release
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = os.path.join(CODE, "work", "qa_state.json"); QA = os.path.join(CODE, "work", "qa")
DEPLOYED = os.path.join(CODE, "audit", "deployed.json")
PENDING = os.path.join(CODE, "audit", "pending_release.json")
REPOSITORY = "delirious6423/quant-fidelity-frontier"
SAMPLE = 20; PER_AGENT = 10
NEXT = "python3 pipeline/qcl_qa.py"
REVIEW_PROMPT = "Follow docs/runbooks/agents/REVIEW.md in the repo at {code}, with I = `work/qa/items_{k}.json` and V = `work/qa/verdict_{k}.json`."

def sh(cmd): return subprocess.run(cmd, cwd=CODE, shell=True, capture_output=True, text=True)
def command(args): return subprocess.run(args, cwd=CODE, capture_output=True, text=True)
def scan(a):
    p = sh(f"python3 pipeline/qcl_scan.py {a}"); line = (p.stdout.strip().splitlines() or ["{}"])[-1]
    try: return p.returncode, json.loads(line)
    except ValueError: return p.returncode, {"raw": line}
def say(d): print(json.dumps(d)); sys.exit(0)
def save(s): os.makedirs(os.path.dirname(ST), exist_ok=True); json.dump(s, open(ST, "w"), indent=1)

def require_clean_checkout():
    """A fresh QA run must not absorb unrelated edits into its release commit."""
    status = command(["git", "status", "--porcelain=v1", "--untracked-files=normal"])
    if status.returncode != 0:
        raise ValueError("Could not inspect the checkout; no release attempted.")
    if status.stdout.strip():
        raise ValueError("The checkout has local changes. Commit or stash them, or use a fresh clone before starting QA. No files were changed.")

def synchronize():
    """Advance only a clean checkout without discarding unpublished commits."""
    require_clean_checkout()
    if command(["git", "fetch", "-q", "origin", "main"]).returncode != 0:
        raise ValueError("Could not synchronize main; no release attempted.")
    ancestry = command(["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"])
    if ancestry.returncode != 0:
        raise ValueError("Local commits or incomplete history prevent a safe synchronization. Preserve this checkout and use a fresh clone; no release attempted.")
    if command(["git", "checkout", "-q", "-B", "main", "origin/main"]).returncode != 0:
        raise ValueError("Could not check out synchronized main; no release attempted.")
    if command(["git", "config", "merge.wljson.driver", "python3 pipeline/merge_watchlist.py %O %A %B"]).returncode != 0:
        raise ValueError("Could not configure the watchlist merge driver; no release attempted.")

def changed_since_deployment(base, head):
    """A failed comparison is an error, never proof that publication is current."""
    if not isinstance(head, str) or not release.SHA.fullmatch(head):
        raise ValueError("Could not establish the current commit; no release attempted.")
    if base is None:
        return ["(first run)"]
    if not isinstance(base, str) or not release.SHA.fullmatch(base):
        raise ValueError("The deployed receipt has an invalid commit; review audit/deployed.json before retrying.")
    diff = command(["git", "diff", "--name-only", "-z", base, head, "--"])
    if diff.returncode != 0:
        raise ValueError("Could not compare the deployed commit with the current tree. For a shallow clone run `git fetch --unshallow origin`, otherwise run `git fetch origin main`, then retry QA. No release attempted.")
    return release.publication_changes(diff.stdout.split("\0"))

def push(msg, paths, allow_rebase=True):
    import shlex
    clean_index = command(["git", "diff", "--cached", "--quiet"]).returncode
    if clean_index == 1: return "index already contains staged changes; preserved without committing or pushing"
    if clean_index != 0: return "could not inspect staged changes; no commit or push attempted"
    ex = [x for x in shlex.split(paths) if x.startswith(":!")]
    for p in shlex.split(paths):
        if p.startswith(":!"): continue
        if not os.path.exists(os.path.join(CODE, p)) and command(["git", "ls-files", "--error-unmatch", "--", p]).returncode != 0: continue
        if command(["git", "add", "-A", "--", p, *ex]).returncode != 0: return "stage failed"
    changed = command(["git", "diff", "--cached", "--quiet"]).returncode
    if changed == 0: return "nothing to commit"
    if changed != 1: return "staged diff failed"
    if command(["git", "-c", "user.name=Claude", "-c", "user.email=noreply@anthropic.com", "commit", "-qm", msg]).returncode != 0: return "commit failed"
    for _ in range(3):
        if command(["git", "push", "-q", "origin", "HEAD:refs/heads/main"]).returncode == 0: return "pushed"
        if not allow_rebase: return "push failed; candidate must be revalidated after synchronization"
        if command(["git", "pull", "--rebase", "-q", "origin", "main"]).returncode != 0: return "rebase failed"
    return "push failed"
def wl(): return json.load(open(os.path.join(CODE, "watchlist.json")))
def stop(s, note=None, pushed_lock=True):
    if note: s.setdefault("notes", []).append(note)
    scan("unlock --discard")
    s["git_unlock"] = push("QA unlock and release receipt", "qa.lock logs/scan audit/pending_release.json audit/deployed.json") if pushed_lock else "no further push attempted"
    s["step"] = "done"; save(s)
    say({"do": "stop", "report": {k: s.get(k) for k in ("reviewed", "verdicts", "held", "promoted", "published", "readme", "deploy", "notes", "git_unlock")}})

def dispatch(pending):
    target = release.resolve_target(pending)["target_commit"]
    result = command(["gh", "api", "-X", "POST", f"repos/{REPOSITORY}/actions/workflows/pages.yml/dispatches",
                      "-f", "ref=main", "-f", f"inputs[expected_sha]={target}"])
    pending["status"] = "dispatched" if result.returncode == 0 else "dispatch_failed"
    pending["dispatch_at"] = release.utc_now()
    if result.returncode != 0: pending["error"] = result.stderr[-300:]
    release.write_json(PENDING, pending)
    return pending["status"]

def dispatch_if_current(pending):
    """Request Pages only for a commit that is still the tip of main on the remote. The workflow refuses any other
    commit (and a SHA that never reached the remote), so asking would only add a failed deploy-pages run."""
    target = release.resolve_target(pending)["target_commit"]
    tip = command(["git", "ls-remote", "origin", "refs/heads/main"])
    names = tip.stdout.split()[:1] if tip.returncode == 0 else []
    if names == [target]: return dispatch(pending)
    pending["status"] = "dispatch_failed"; pending["dispatch_at"] = release.utc_now()
    pending["error"] = "main on the remote does not name the tested candidate; no deploy requested, the next QA run starts a new candidate."
    release.write_json(PENDING, pending)
    return pending["status"]

def reconcile_pending():
    """Run before skip detection, including after a fresh automation clone."""
    pending = release.read_json(PENDING)
    if pending is None: return False
    if not isinstance(pending, dict) or pending.get("version") != 1 or pending.get("workflow") != release.WORKFLOW:
        raise ValueError("Pending release receipt has an unsupported schema or workflow.")
    if pending.get("status") not in release.TERMINAL | {"prepared", "dispatched", "pending"}:
        raise ValueError("Pending release receipt has an unknown status.")
    if pending.get("status") in release.TERMINAL:
        return pending.get("status") != "succeeded"
    introduced = command(["git", "log", "-1", "--format=%H", "--", "audit/pending_release.json"])
    pending = release.resolve_target(pending, introduced.stdout.strip() if introduced.returncode == 0 else None)
    remote = command(["gh", "api", "--paginate", "--slurp", "-X", "GET",
                      f"repos/{REPOSITORY}/actions/workflows/pages.yml/runs", "-f", "event=workflow_dispatch", "-f", "per_page=100"])
    if remote.returncode != 0:
        say({"do": "stop", "report": {"deploy": "Workflow status unavailable; deployed receipt unchanged", "error": remote.stderr[-300:]}})
    try: payload = json.loads(remote.stdout)
    except ValueError: payload = None
    result = release.workflow_result(pending, payload)
    if result["outcome"] == "unknown":
        say({"do": "stop", "report": {"deploy": result}})
    if pending["status"] == "prepared" and result["outcome"] == "pending" and "run" not in result:
        # The process may have died after pushing the durable receipt but before
        # dispatch. Retry only while the branch still names that exact candidate.
        head = command(["git", "rev-parse", "HEAD"]).stdout.strip()
        if head == pending["target_commit"]:
            status = dispatch(pending)
        else:
            pending.update(status="failed", error="Main advanced before the pending candidate was dispatched; a new candidate must be tested.")
            release.write_json(PENDING, pending)
            status = "failed"
        recorded = push("QA pending release receipt", "audit/pending_release.json")
        say({"do": "stop", "report": {"deploy": status, "receipt": recorded, "commit": pending["target_commit"]}})
    result = release.reconcile(PENDING, DEPLOYED, payload, pending["target_commit"])
    recorded = push("QA release workflow result", "audit/pending_release.json audit/deployed.json")
    if recorded not in ("pushed", "nothing to commit") or result["outcome"] == "pending":
        say({"do": "stop", "report": {"deploy": result, "receipt": recorded}})
    return result["outcome"] != "succeeded"

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

def main(args=None):
    os.environ["QCL_LOCK"] = "qa.lock"; os.environ["QCL_PART"] = "-qa"
    args = sys.argv[1:] if args is None else args
    s = json.load(open(ST)) if os.path.exists(ST) else None
    if s and s.get("step") == "done": s = None
    if "--abort" in args: stop(s or {}, "ended early by the session (--abort)")
    if s is None:
        try:
            if "--synced" in args: require_clean_checkout()
            else: synchronize()
        except ValueError as error:
            say({"do": "stop", "report": {"error": str(error)}})
        if "--synced" not in args:   # newest main, newest code
            os.execvp("python3", ["python3", os.path.abspath(__file__), *args, "--synced"])

    if s is None:
        try: retry_failed_release = reconcile_pending()
        except (ValueError, OSError, TypeError, AttributeError) as error:
            say({"do": "stop", "report": {"error": "Pending release cannot be verified; deployed receipt unchanged", "detail": str(error)}})
        try:
            current = command(["git", "rev-parse", "HEAD"])
            head = current.stdout.strip() if current.returncode == 0 else None
            deployed = release.read_json(DEPLOYED)
            if deployed is not None and (not isinstance(deployed, dict) or not isinstance(deployed.get("commit"), str)
                                         or not release.SHA.fullmatch(deployed["commit"])):
                raise ValueError("The deployed receipt is malformed; review audit/deployed.json before retrying.")
            base = deployed["commit"] if deployed is not None else None
            changed = changed_since_deployment(base, head)
            review = [m["slug"] for m in wl()["models"] if m["status"] == "review"]
        except (ValueError, OSError, TypeError) as error:
            say({"do": "stop", "report": {"error": str(error)}})
        s = {"step": "start", "head": head, "base": base, "review": review, "notes": []}
        if not changed and not review and not retry_failed_release: s["step"] = "done"; save(s); say({"do": "stop", "report": {"skipped": "nothing new since the last published commit", "commit": head}})
        code, d = scan("lock")
        if code == 3: say({"do": "stop", "report": {"skipped": "another QA run holds the lock", **d}})
        if code != 0: say({"do": "stop", "report": {"error": "QA lock could not be acquired", **d}})
        locked = push("QA lock", "qa.lock")
        if locked not in ("pushed", "nothing to commit"): stop(s, "Could not persist QA lock: " + locked, pushed_lock=False)
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
        try: pick = release.read_json(f"{QA}/sample.json") if s.get("reviewed") else []
        except (OSError, ValueError): pick = None
        batches = []
        for f in sorted(glob.glob(f"{QA}/verdict_*.json")) if s.get("reviewed") else []:
            try: batches.append(release.read_json(f))
            except (OSError, ValueError): batches.append(None)
        reviewed = release.assess_reviews(pick, s.get("reviewed"), batches)
        verdicts, held = reviewed["counts"], reviewed["held"]
        s["verdicts"] = verdicts; s["held"] = len(held)
        if not reviewed["ready"]:
            # The session (and this state) ends with the run, so there is nothing to resume: release the lock; the next run re-reviews.
            stop(s, f"blocked, nothing published: every sampled value needs one explained verdict (unresolved {reviewed['unresolved']}, issues {reviewed['issues']})")
        touched = set()
        for it, why in held:   # hold the entry; the number itself is never edited
            hp = f"{CODE}/models/{it['slug']}/audit/holds.json"
            hs = json.load(open(hp)) if os.path.exists(hp) else []
            if not any(h["repo"] == it["repo"] and h["entry"] == it["entry"] for h in hs):   # holds are per entry (row)
                hs.append({"repo": it["repo"], "entry": it["entry"], "reason": f"QA review {dt.datetime.now(dt.timezone.utc):%Y-%m-%d}: {it['key']}: {why}"})
                release.write_json(hp, hs)
            touched.add(it["slug"])  # Also retry a merge if an earlier run died after saving its holds.
        for slug in touched:
            import shlex
            for action in ("pipeline/qcl_watch.py merge", "pipeline/gen_auto.py"):
                applied = sh(f"QCL_ROOT={shlex.quote('models/' + slug)} python3 {action}")
                if applied.returncode != 0:
                    stop(s, f"blocked, nothing published: a held entry could not be quarantined in {slug}: {applied.stderr[-300:]}")
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
            stop(s, f"tests failed; local changes preserved and no release attempted: {d.get('error')}: {str(d.get('tail', ''))[-600:]}")
        try:   # the record compares with the committed watch list, so a re-run of this step still reports the promotion
            before = {m["slug"]: m["status"] for m in json.loads(sh("git show HEAD:watchlist.json").stdout)["models"]}
            promoted = sorted({*promoted, *(m["slug"] for m in wl()["models"] if m["status"] == "live" and before.get(m["slug"]) == "review")})
        except Exception: pass
        s["promoted"] = promoted; s["published"] = d.get("progress_line")
        rs = sh("python3 pipeline/readme_sync.py")   # README model table from the data just tested, in the same commit
        if rs.returncode != 0: stop(s, "README synchronization failed; no release attempted.")
        s["readme"] = (rs.stdout.strip().splitlines() or [rs.stderr.strip()[-200:]])[-1]
        pending = release.new_pending(s)
        release.write_json(PENDING, pending)
        s["git"] = push(f"QA {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC: reviewed {s.get('reviewed', 0)} new values, held {len(held)}"
                        + (f", promoted {', '.join(promoted)}" if promoted else "") + "; awaiting Pages verification",
                        ". ':!qa.lock' ':!scan.lock' ':!ingest.lock' ':!logs/scan'", allow_rebase=False)
        if s["git"] != "pushed":
            pending.update(status="push_failed", error=s["git"])
            release.write_json(PENDING, pending)
            # An unlock push must not accidentally push/rebase the rejected candidate.
            stop(s, "Release push failed; deployed receipt unchanged. Synchronize and revalidate before retrying.", pushed_lock=False)
        candidate = command(["git", "rev-parse", "HEAD"])
        pending = release.resolve_target(pending, candidate.stdout.strip() if candidate.returncode == 0 else None)
        s["deploy"] = dispatch_if_current(pending)
        s["notes"].append("Deployed receipt remains unchanged until the exact Pages workflow succeeds.")
        stop(s)
    stop(s, f"unexpected state {s.get('step')}")


if __name__ == "__main__":
    main()
