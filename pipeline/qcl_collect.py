#!/usr/bin/env python3
"""Collect the model cards of every live model straight from the public Hugging Face API: no connector, no agent.

  python3 pipeline/qcl_collect.py [--models SLUG ...] [--all-live] [--workers 8] [--base URL] [--root DIR]
  python3 pipeline/qcl_collect.py --summary LINES.jsonl      # one line for a commit message, from the JSON lines of a run

For each model (default: every live model, as pipeline/qcl_live.py lists them; the model id comes from models/<slug>/model.json,
else watchlist.json) it runs the fetch-once ledger of pipeline/qcl_fetch.py into models/<slug>/cache (one catalogue call, cards
and file trees only for repos whose commit changed), then pipeline/qcl_newmodel.py, which merges the cache into
snapshots/index.json. One JSON line per model on stdout: slug, catalogue, new, changed, unchanged, unresolved, seconds (plus
rate_limited / failed / error when something did not complete). Safe to run every hour: a model with no change on the Hub
leaves every file byte-identical. HF_TOKEN from the environment raises the rate limit; it is never printed or saved, and HTTP
error bodies never reach stdout. A rate limit (cache/rate_limit.json) is reported for that model and the others continue.
.github/workflows/collect.yml runs this hourly and commits models/*/cache and models/*/snapshots/index.json."""
import argparse, contextlib, glob, io, json, os, re, subprocess, sys, time
from types import SimpleNamespace

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "pipeline"))
import qcl_fetch as fetch


def live_slugs(root):
    """Same rule as pipeline/qcl_live.py: status "live" in watchlist.json; a model folder not in the watch list counts as live."""
    wl = os.path.join(root, "watchlist.json")
    st = {m["slug"]: m["status"] for m in json.load(open(wl))["models"]} if os.path.exists(wl) else {}
    out = []
    for mj in sorted(glob.glob(os.path.join(root, "models", "*", "model.json"))):
        slug = json.load(open(mj)).get("slug") or os.path.basename(os.path.dirname(mj))
        if st.get(slug, "live") == "live": out.append(slug)
    return out


def model_id(root, slug):
    mj = os.path.join(root, "models", slug, "model.json")
    if os.path.exists(mj):
        model = json.load(open(mj)).get("model")
        if model: return model
    wl = os.path.join(root, "watchlist.json")
    if os.path.exists(wl):
        for m in json.load(open(wl))["models"]:
            if m["slug"] == slug: return m["model"]
    raise ValueError(f"no model id for {slug}: neither models/{slug}/model.json nor watchlist.json names it")


def sanitize(text):
    """Error text without HTTP bodies: qcl_fetch appends the response as a bytes repr after 'HTTP <status>: '."""
    text = re.sub(r"(HTTP \d+): b['\"].*$", r"\1", str(text), flags=re.S)
    return re.sub(r"\s+", " ", text).strip()[:300]


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f: return json.load(f)
    except (OSError, ValueError): return default


def ledger_repos(index):
    return {rid: rec.get("sha256") for rid, rec in (index or {}).get("repos", {}).items()}


def collect_one(root, slug, workers, base, token):
    """Fetch one model's ledger, then merge it into the snapshot. Returns the JSON record for that model."""
    t0 = time.time()
    out = f"models/{slug}/cache"   # relative to root: the ledger's card paths must contain no local path (public repo)
    rec = {"slug": slug, "catalogue": None, "new": 0, "changed": 0, "unchanged": 0, "unresolved": None, "seconds": None}
    index_path = f"{out}/index.json"
    before_bytes = open(index_path, "rb").read() if os.path.exists(index_path) else None
    before = ledger_repos(read_json(index_path))
    log = io.StringIO()
    try:
        model = model_id(root, slug)
        rec["model"] = model
        args = SimpleNamespace(model=model, out=out, include_repo=[], workers=workers, base=base)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                fetch.fetch_model(args, token)
            finally:
                for line in log.getvalue().splitlines():
                    if line.strip(): print(f"  {slug}: {sanitize(line)}", file=sys.__stderr__, flush=True)
        if os.path.exists(f"{out}/rate_limit.json"): os.remove(f"{out}/rate_limit.json")   # as qcl_fetch's CLI does after success
    except fetch.RateLimited as error:
        fetch.write_json(f"{out}/rate_limit.json", error.record)
        rec["rate_limited"] = {k: error.record.get(k) for k in ("retry_after", "retry_after_seconds", "observed_at")}
    except ValueError as error:   # identity refusal (model.json or ledger names another model): nothing was fetched
        rec["error"] = sanitize(error); rec["seconds"] = round(time.time() - t0, 1); return rec
    except RuntimeError as error:
        msg = sanitize(error)
        m = re.match(r"(\d+) repositories failed", msg)
        if m: rec["failed"] = int(m.group(1))   # gated or missing repos; everything else is checkpointed in the ledger
        else: rec["error"] = msg
    except Exception as error:   # noqa: BLE001 - one model must never stop the others
        rec["error"] = f"{type(error).__name__}: {sanitize(error)}"
    # Only the ledger's write time changed: keep the old bytes, so an unchanged model changes no file.
    after_raw = read_json(index_path)
    if before_bytes is not None and isinstance(after_raw, dict):
        strip = lambda d: {k: v for k, v in d.items() if k != "captured_at"}
        if strip(json.loads(before_bytes)) == strip(after_raw):
            with open(index_path, "wb") as f: f.write(before_bytes)
    after = ledger_repos(after_raw)
    cat = read_json(f"{out}/catalogue.json", [])
    rec["catalogue"] = len(cat)
    rec["new"] = sum(1 for rid in after if rid not in before)
    rec["changed"] = sum(1 for rid, h in after.items() if rid in before and before[rid] != h)
    rec["unchanged"] = sum(1 for rid, h in after.items() if rid in before and before[rid] == h)
    ledger = (after_raw or {}).get("repos", {})   # unresolved: catalogued repos whose pinned commit is not in the ledger (gated, failed, rate-limited)
    rec["unresolved"] = sum(1 for m in cat if isinstance(m, dict) and ledger.get(m.get("id"), {}).get("sha") != m.get("sha"))
    # Whatever was checkpointed is valid and pinned: merge it (a partial ledger after a 429 resumes next hour).
    p = subprocess.run([sys.executable, os.path.join(CODE, "pipeline", "qcl_newmodel.py"), f"models/{slug}"],
                       cwd=root, capture_output=True, text=True)
    if p.returncode:
        rec["error"] = "qcl_newmodel failed: " + sanitize((p.stdout + p.stderr)[-400:])
    else:
        try: rec["index"] = json.loads(p.stdout.strip().splitlines()[-1]).get("repos_in_index")
        except (ValueError, IndexError): pass
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def summary(records):
    """One line for the commit message: only what changed, plus what did not complete."""
    parts = []
    for r in records:
        bits = []
        if r.get("new"): bits.append(f"+{r['new']}")
        if r.get("changed"): bits.append(f"~{r['changed']}")
        if r.get("rate_limited"): bits.append("rate-limited")
        if r.get("failed"): bits.append(f"{r['failed']} failed")
        if r.get("error"): bits.append("error")
        if bits: parts.append(f"{r['slug']} {' '.join(bits)}")
    return "; ".join(parts) if parts else "no card changed"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=[], help="slugs under models/ (default: every live model)")
    ap.add_argument("--all-live", action="store_true", help="every live model (the default when --models is not given)")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--base", default="https://huggingface.co")
    ap.add_argument("--root", default=CODE, help="repository root holding models/ and watchlist.json (tests use a copy)")
    ap.add_argument("--summary", metavar="LINES", help="print one summary line from a file of this script's JSON lines, then exit")
    a = ap.parse_args()
    if a.summary:
        print(summary([json.loads(l) for l in open(a.summary) if l.strip()])); return 0
    if not 1 <= a.workers <= 32: ap.error("--workers must be between 1 and 32")
    root = os.path.abspath(a.root)
    slugs = a.models or live_slugs(root)
    if a.all_live: slugs = sorted(set(slugs) | set(live_slugs(root)))
    for slug in slugs:
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,95}", slug) or ".." in slug: ap.error(f"bad slug: {slug!r}")
    os.chdir(root)   # the ledger records card paths relative to here (models/<slug>/cache/cards/...), like fetch.yml
    token = os.environ.get("HF_TOKEN") or None
    records = []
    for slug in slugs:
        rec = collect_one(root, slug, a.workers, a.base, token)
        records.append(rec); print(json.dumps(rec), flush=True)
    return 1 if any("error" in r for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
