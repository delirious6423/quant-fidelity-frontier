#!/usr/bin/env python3
"""Run the CI battery of .github/workflows/verify.yml on this machine, step for step, without GitHub Actions.

  python3 pipeline/ci_local.py [--jobs 4] [--only plan|safety|model|site ...] [--models SLUG ...] [--from TEXT]
                               [--pip] [--log-dir DIR] [--receipt FILE] [--list] [--keep-going]

The workflow file is the single source of truth: this runner parses it and executes every `run:` step of every job in
the workflow's own order (plan, safety, then the per-model jobs side by side, then site; safety goes first because the
jobs share one checkout here and some safety suites assert it stays untouched while builds write audit files), with the same shell
(`bash -e`), the same environment variables (`MODEL_DIR` for a model job, `GITHUB_OUTPUT`, and a private `RUNNER_TEMP`
per step, so side-by-side model jobs never share a scratch file) and the same cwd. Nothing is copied from the workflow by hand, so the local run cannot drift from what Actions would run.

What is skipped, because it only makes sense on a hosted runner: every `uses:` step (checkout, Python setup, caches,
artifact upload and download), the artifact restore step of the site job (the model jobs leave models/<slug>/site in
place here), package installs (`pip install`, `playwright install`, the MCP 1.x venv) unless --pip is given, and the
venv-based MCP 1.x check when that venv does not exist. Everything else runs exactly as written.

Output: one line per step (OK / FAIL / SKIP with seconds), the step's full log under --log-dir (a temporary
directory by default, printed at start), a summary, and exit status 1 if any step failed. --receipt writes a JSON
receipt (commit, host-independent step list, outcomes, seconds) that a person or agent can attach to a pull request
or a QA note. --list prints the steps and exits. --keep-going runs the site job even when a model or safety step
failed (by default it is skipped, as on Actions, where `site` needs every other job)."""
import argparse, concurrent.futures as cf, json, os, re, shutil, subprocess, sys, tempfile, time

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(CODE, ".github", "workflows", "verify.yml")
INSTALL_RE = re.compile(r"^\s*(pip install\b|python -m playwright install\b|python3 -m venv\b|/tmp/qff-mcp-v1/bin/pip install\b)")
VENV_PY = "/tmp/qff-mcp-v1/bin/python"


def load_workflow():
    try:
        import yaml
    except ImportError:
        sys.exit("pip install pyyaml (the runner reads the workflow file)")
    with open(WORKFLOW, encoding="utf-8") as f:
        return yaml.safe_load(f)


def live_models():
    p = subprocess.run([sys.executable, os.path.join(CODE, "pipeline", "qcl_live.py")], cwd=CODE, capture_output=True, text=True)
    if p.returncode:
        sys.exit("pipeline/qcl_live.py failed:\n" + p.stderr)
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def run_steps(job_name, job, allow_pip=False):
    """The runnable steps of one job: [{name, script, skipped_reason}] in the workflow's order."""
    out = []
    for i, step in enumerate(job.get("steps", []), 1):
        name = step.get("name") or step.get("run", "").strip().splitlines()[0][:60] if step.get("run") else step.get("uses", f"step {i}")
        if "uses" in step:
            out.append({"name": name, "skip": f"uses {step['uses']} (hosted runner only)"}); continue
        script = step.get("run", "")
        if job_name == "site" and name.startswith("Restore every live model"):
            out.append({"name": name, "skip": "artifact restore (the model jobs leave models/<slug>/site in place locally)"}); continue
        lines = script.splitlines()
        kept = []
        for line in lines:
            m = re.match(r"^(\s*)pip install [^&]+&&\s*(.*)$", line)   # `pip install X && CMD` keeps CMD
            if m and not allow_pip:
                line = m.group(1) + m.group(2)
            elif INSTALL_RE.match(line) and not allow_pip:
                continue
            if line.strip().startswith(VENV_PY) and not os.path.exists(VENV_PY):
                out.append({"name": name + " (MCP 1.x venv check)", "skip": f"{VENV_PY} not present; run with --pip to create it"})
                continue
            kept.append(line)
        if not any(l.strip() for l in kept):
            out.append({"name": name, "skip": "package install (run with --pip to execute)"}); continue
        out.append({"name": name, "script": "\n".join(kept) + "\n", "if": step.get("if")})
    return out


def execute(step, env, log_path, cwd=CODE):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"## {step['name']}\n{step['script']}\n--- output ---\n"); log.flush()
        p = subprocess.run(["bash", "-e"], input=step["script"], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return p.returncode, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="model jobs (and safety) run side by side, this many at once")
    ap.add_argument("--only", nargs="*", choices=["plan", "safety", "model", "site"], help="run these jobs only")
    ap.add_argument("--models", nargs="*", metavar="SLUG", help="model jobs for these live models only")
    ap.add_argument("--from", dest="from_text", metavar="TEXT", help="site job: start at the first step whose name contains TEXT")
    ap.add_argument("--pip", action="store_true", help="also run the package-install lines (pip, playwright, the MCP 1.x venv)")
    ap.add_argument("--log-dir", help="where step logs go (default: a new temporary directory)")
    ap.add_argument("--receipt", help="write a JSON receipt of the run here")
    ap.add_argument("--list", action="store_true", help="print the steps that would run and exit")
    ap.add_argument("--keep-going", action="store_true", help="run the site job even after a failed safety or model step")
    a = ap.parse_args()
    wf = load_workflow()
    jobs = wf["jobs"]
    only = set(a.only or ["plan", "safety", "model", "site"])
    log_dir = a.log_dir or tempfile.mkdtemp(prefix="qff-ci-")
    os.makedirs(log_dir, exist_ok=True)
    models = live_models()
    if a.models:
        chosen = {m for m in a.models}
        models = [d for d in models if os.path.basename(d) in chosen]
        if len(models) != len(chosen):
            sys.exit(f"not live or unknown: {sorted(chosen - {os.path.basename(d) for d in models})}")
    plan = [("plan", s, {}) for s in run_steps("plan", jobs["plan"], a.pip)] if "plan" in only else []
    safety = [("safety", s, {}) for s in run_steps("safety", jobs["safety"], a.pip)] if "safety" in only else []
    model_steps = run_steps("model", jobs["model"], a.pip)
    per_model = [(f"model {os.path.basename(d)}", s, {"MODEL_DIR": d}) for d in models for s in model_steps] if "model" in only else []
    site = run_steps("site", jobs["site"], a.pip) if "site" in only else []
    if a.from_text:
        idx = next((i for i, s in enumerate(site) if a.from_text.lower() in s["name"].lower()), None)
        if idx is None: sys.exit(f"no site step contains {a.from_text!r}")
        site = site[idx:]
    site = [("site", s, {}) for s in site]
    everything = plan + safety + per_model + site
    if a.list:
        for job, s, _ in everything:
            if "skip" in s and s["skip"].startswith("uses "): continue   # hosted-runner plumbing, never run locally
            print(f"{'SKIP' if 'skip' in s else 'RUN ':4} {job:28} {s['name']}" + (f"  [{s['skip']}]" if "skip" in s else ""))
        return 0
    print(f"ci_local: {len([1 for _, s, _ in everything if 'skip' not in s])} steps, logs in {log_dir}", flush=True)
    base_env = {**os.environ, "GITHUB_OUTPUT": os.path.join(log_dir, "github_output.txt"), "CI": "true"}
    results, counter = [], [0]

    def run_one(job, step, extra):
        if "skip" in step:
            results.append({"job": job, "step": step["name"], "outcome": "skipped", "reason": step["skip"], "seconds": 0})
            if not step["skip"].startswith("uses "): print(f"SKIP {job:28} {step['name']}  [{step['skip']}]", flush=True)
            return True
        counter[0] += 1
        log_path = os.path.join(log_dir, f"{counter[0]:03d}-{re.sub(r'[^A-Za-z0-9]+', '-', job + '-' + step['name'])[:80]}.log")
        runner_tmp = tempfile.mkdtemp(prefix="job-", dir=log_dir)   # $RUNNER_TEMP: private to this job, as on a hosted runner
        rc, secs = execute(step, {**base_env, **extra, "RUNNER_TEMP": runner_tmp}, log_path)
        ok = rc == 0
        results.append({"job": job, "step": step["name"], "outcome": "success" if ok else "failure", "seconds": secs, "log": log_path})
        print(f"{'OK  ' if ok else 'FAIL'} {job:28} {step['name']}  {secs}s" + ("" if ok else f"  -> {log_path}"), flush=True)
        if not ok:
            with open(log_path, encoding="utf-8", errors="replace") as f: tail = f.read().splitlines()[-12:]
            for line in tail: print("     | " + line, flush=True)
        return ok

    t0 = time.time()
    ok = all(run_one(*x) for x in plan)
    # On Actions every job has its own checkout. Here they share one, so the safety suites (some assert that the
    # checkout stays untouched) run before the model jobs, whose builds write audit files; the model jobs then run
    # side by side, each with its own RUNNER_TEMP, and the site job last.
    ok = all(run_one(*x) for x in safety) and ok
    def job_runner(label, steps):
        return all(run_one(*x) for x in steps)
    groups = {}
    for job, s, extra in per_model:
        groups.setdefault(job, []).append((job, s, extra))
    with cf.ThreadPoolExecutor(max(1, a.jobs)) as ex:
        outcomes = list(ex.map(lambda kv: job_runner(*kv), groups.items()))
    ok = ok and all(outcomes)
    if site:
        if ok or a.keep_going:
            ok = all(run_one(*x) for x in site) and ok
        else:
            print("site: skipped because an earlier job failed (as on Actions; --keep-going overrides)", flush=True)
            for job, s, _ in site:
                results.append({"job": job, "step": s["name"], "outcome": "skipped", "reason": "an earlier job failed", "seconds": 0})
    total = round(time.time() - t0, 1)
    failed = [r for r in results if r["outcome"] == "failure"]
    ran = [r for r in results if r["outcome"] == "success"]
    print(f"ci_local: {len(ran)} ok, {len(failed)} failed, {len(results) - len(ran) - len(failed)} skipped, {total}s wall; logs in {log_dir}", flush=True)
    for r in failed: print(f"  FAIL {r['job']}: {r['step']} -> {r['log']}")
    if a.receipt:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=CODE, capture_output=True, text=True).stdout.strip() != ""
        receipt = {"workflow": os.path.relpath(WORKFLOW, CODE), "commit": head, "dirty": dirty, "models": [os.path.basename(d) for d in models],
                   "jobs": sorted(only), "ok": not failed, "seconds": total, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "steps": [{k: v for k, v in r.items() if k != "log"} for r in results]}
        with open(a.receipt, "w", encoding="utf-8") as f: json.dump(receipt, f, indent=1); f.write("\n")
        print(f"receipt: {a.receipt}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
