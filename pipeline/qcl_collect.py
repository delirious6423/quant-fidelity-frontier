#!/usr/bin/env python3
"""Collect the model cards of every live model straight from the public Hugging Face API: no connector, no agent.

  python3 pipeline/qcl_collect.py [--models SLUG ...] [--all-live] [--workers 8] [--wait-429 SECONDS] [--base URL] [--root DIR] [--strict-images] [--no-images]
  python3 pipeline/qcl_collect.py --summary LINES.jsonl      # one line for a commit message, from the JSON lines of a run

For each model (default: every live model, as pipeline/qcl_live.py lists them; the model id comes from models/<slug>/model.json,
else watchlist.json) it runs the fetch-once ledger of pipeline/qcl_fetch.py into models/<slug>/cache (one catalogue call, cards
and file trees only for repos whose commit changed), then pipeline/qcl_newmodel.py, which merges the cache into
snapshots/index.json. When no card or tree changed, the ledger and catalogue keep their previous bytes and the merge is
skipped; Hub metadata that moved (downloads, likes) is taken over at most once a day. One JSON line per model on stdout: slug, catalogue, new, changed, unchanged, unresolved, seconds (plus
rate_limited / failed / error when something did not complete). After a model's cards are installed, pipeline/qcl_images.py
pins the content images its cards embed (models/<slug>/cache/images, coverage in models/<slug>/audit/images.json); the
record carries them under "images" (cards, referenced, pinned, new, unresolved, seconds) and --strict-images makes the run
exit non-zero when any image is unresolved. Safe to run every hour: a model with no change on the Hub leaves every file
byte-identical. HF_TOKEN from the environment raises the rate limit; it is never printed or saved, and HTTP
error bodies never reach stdout. A rate limit (cache/rate_limit.json) is reported for that model and the others continue; with
--wait-429 the run sleeps the Retry-After (within that budget per run) and resumes the model, so one limit does not cost the
hour. A ledger the connector path wrote (no model identity, no pinned commits) is replaced by an empty HTTP ledger bound to the
model: the snapshot index pins those cards already, and the HTTP path re-collects them at their current commit.
.github/workflows/collect.yml runs this hourly and commits models/*/cache (cards, trees and images), models/*/snapshots/index.json
and models/*/audit/images.json."""
import argparse, contextlib, glob, io, json, os, re, subprocess, sys, time
from types import SimpleNamespace

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "pipeline"))
import qcl_fetch as fetch
import qcl_images as images


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


def content(index):
    """What the ledger pins: every repo's commit, card bytes, card file and tree file, and the base card. Downloads, likes
    and lastModified are Hub metadata that moves every hour without any card changing."""
    base = index.get("base") or {}
    return ({rid: tuple(rec.get(k) for k in ("sha", "sha256", "file", "tree_file")) for rid, rec in index.get("repos", {}).items()},
            tuple(base.get(k) for k in ("sha", "sha256", "file")))


def strip(index):
    return {k: v for k, v in index.items() if k != "captured_at"}   # the ledger's write time alone is never a change


def hours_since(stamp):
    try: return (time.time() - time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")) + time.timezone) / 3600
    except (TypeError, ValueError): return float("inf")


METADATA_HOURS = 24   # metadata-only churn (downloads, likes) is let through at most once a day
METADATA_KEYS = ("createdAt", "lastModified", "downloads", "likes", "tags")   # what fetch_repo copies from the listing


def refresh_metadata(index, cat):
    """Copy the listing's metadata onto the ledger records of the same repo (a record keeps the values of its own fetch)."""
    for m in cat:
        rec = index.get("repos", {}).get(m.get("id")) if isinstance(m, dict) else None
        if rec is None: continue
        for k in METADATA_KEYS:
            if k in m or k in rec: rec[k] = m.get(k)


def connector_era(ledger):
    """A cache ledger written by the connector path (pipeline/qcl_mcpsave.py): no model identity and no pinned commits."""
    if not isinstance(ledger, dict) or "model" in ledger or not isinstance(ledger.get("repos"), dict): return False
    recs = list(ledger["repos"].values())
    return all(isinstance(r, dict) and "sha" not in r for r in recs)


def resume(first, again, waited):
    """The record of a model whose collection resumed after a rate limit: counts add up, the rest is the resumed run's."""
    rec = dict(again)
    for k in ("new", "changed"): rec[k] = (first.get(k) or 0) + (again.get(k) or 0)
    rec["seconds"] = round((first.get("seconds") or 0) + (again.get("seconds") or 0), 1)
    rec["waited"] = round(waited, 1)
    return rec


def collect_one(root, slug, workers, base, token):
    """Fetch one model's ledger, then merge it into the snapshot. Returns the JSON record for that model."""
    t0 = time.time()
    out = f"models/{slug}/cache"   # relative to root: the ledger's card paths must contain no local path (public repo)
    rec = {"slug": slug, "catalogue": None, "new": 0, "changed": 0, "unchanged": 0, "unresolved": None, "seconds": None}
    index_path, cat_path = f"{out}/index.json", f"{out}/catalogue.json"
    before_bytes = open(index_path, "rb").read() if os.path.exists(index_path) else None
    cat_bytes = open(cat_path, "rb").read() if os.path.exists(cat_path) else None
    before = ledger_repos(read_json(index_path))
    log = io.StringIO()
    try:
        model = model_id(root, slug)
        rec["model"] = model
        if connector_era(read_json(index_path)):
            # Cards the connector saved carry no pinned commit, so nothing in that ledger resumes over HTTP; the snapshot
            # index already pins them. Start the HTTP ledger for this model, bound to its identity.
            fetch.write_json(index_path, {"model": model, "repos": {}}); rec["ledger"] = "connector-era ledger replaced"
            before = {}
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
    # Stats come from the fresh ledger and catalogue, before any restore below.
    after_raw, cat = read_json(index_path), read_json(cat_path, [])
    after = ledger_repos(after_raw)
    rec["catalogue"] = len(cat)
    rec["new"] = sum(1 for rid in after if rid not in before)
    rec["changed"] = sum(1 for rid, h in after.items() if rid in before and before[rid] != h)
    rec["unchanged"] = sum(1 for rid, h in after.items() if rid in before and before[rid] == h)
    ledger = (after_raw or {}).get("repos", {})   # unresolved: catalogued repos whose pinned commit is not in the ledger (gated, failed, rate-limited)
    rec["unresolved"] = sum(1 for m in cat if isinstance(m, dict) and ledger.get(m.get("id"), {}).get("sha") != m.get("sha"))
    # No card or tree changed: keep the previous bytes of the ledger and catalogue and skip the merge, so the model changes
    # no file. Hub metadata (downloads, likes) moves every hour; it is taken over once the capture is older than METADATA_HOURS.
    skip_merge = False
    prev_raw = json.loads(before_bytes) if before_bytes is not None else None
    if isinstance(prev_raw, dict) and isinstance(after_raw, dict) and content(prev_raw) == content(after_raw):
        if hours_since(prev_raw.get("captured_at")) <= METADATA_HOURS:
            same = strip(prev_raw) == strip(after_raw) and cat_bytes == open(cat_path, "rb").read()
            rec["metadata"] = "unchanged" if same else "deferred"
            skip_merge = True
    if isinstance(after_raw, dict) and not skip_merge and isinstance(cat, list):
        refresh_metadata(after_raw, cat)   # the ledger records carry the listing's metadata of their own fetch; take the fresh one
        if isinstance(prev_raw, dict) and strip(prev_raw) == strip(after_raw) and cat_bytes == open(cat_path, "rb").read():
            rec["metadata"] = "unchanged"; skip_merge = True   # nothing moved at all: not even captured_at changes
        else:
            fetch.write_json(index_path, after_raw)
    if skip_merge:
        with open(index_path, "wb") as f: f.write(before_bytes)
        if cat_bytes is not None:
            with open(cat_path, "wb") as f: f.write(cat_bytes)
    # Whatever was checkpointed is valid and pinned: merge it (a partial ledger after a 429 resumes next hour).
    p = None if skip_merge else subprocess.run([sys.executable, os.path.join(CODE, "pipeline", "qcl_newmodel.py"), f"models/{slug}"],
                                               cwd=root, capture_output=True, text=True)
    if p is None:
        pass
    elif p.returncode:
        rec["error"] = "qcl_newmodel failed: " + sanitize((p.stdout + p.stderr)[-400:])
    else:
        try: rec["index"] = json.loads(p.stdout.strip().splitlines()[-1]).get("repos_in_index")
        except (ValueError, IndexError): pass
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def summary(records):
    """One line for the commit message: only what changed, plus what did not complete, plus the image coverage."""
    parts, img = [], []
    for r in records:
        bits = []
        if r.get("new"): bits.append(f"+{r['new']}")
        if r.get("changed"): bits.append(f"~{r['changed']}")
        if r.get("rate_limited"): bits.append("rate-limited")
        if r.get("failed"): bits.append(f"{r['failed']} failed")
        if r.get("error"): bits.append("error")
        i = r.get("images") if isinstance(r.get("images"), dict) else None
        if i is not None:
            img.append(i)
            if i.get("new"): bits.append(f"img +{i['new']}")
            if i.get("unresolved"): bits.append(f"{i['unresolved']} img unresolved")
            if i.get("rate_limited"): bits.append("img rate-limited")
            if i.get("error"): bits.append("img error")
        if bits: parts.append(f"{r['slug']} {' '.join(bits)}")
    text = "; ".join(parts) if parts else "no card changed"
    if img: text += f"; images {sum(i.get('pinned') or 0 for i in img)}/{sum(i.get('referenced') or 0 for i in img)} pinned"
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=[], help="slugs under models/ (default: every live model)")
    ap.add_argument("--all-live", action="store_true", help="every live model (the default when --models is not given)")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--base", default="https://huggingface.co")
    ap.add_argument("--root", default=CODE, help="repository root holding models/ and watchlist.json (tests use a copy)")
    ap.add_argument("--summary", metavar="LINES", help="print one summary line from a file of this script's JSON lines, then exit")
    ap.add_argument("--no-images", action="store_true", help="cards only: skip the image step (pipeline/qcl_images.py)")
    ap.add_argument("--strict-images", action="store_true", help="exit 1 when any model has an unresolved image")
    ap.add_argument("--max-bytes", type=int, default=images.DEFAULT_MAX_BYTES, help="largest image accepted (pipeline/qcl_images.py)")
    ap.add_argument("--wait-429", type=float, default=0, metavar="SECONDS",
                    help="after a rate limit, wait its Retry-After and resume that model, up to this many seconds per run (default: never wait)")
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
    waited = 0.0
    for slug in slugs:
        rec = collect_one(root, slug, a.workers, a.base, token)
        while rec.get("rate_limited") and a.wait_429 > 0:
            secs = (rec["rate_limited"].get("retry_after_seconds") or 60.0) + 1.0
            if waited + secs > a.wait_429:
                print(f"  {slug}: rate limited; the wait budget of {a.wait_429:g}s is spent, resume next run", file=sys.stderr, flush=True); break
            print(f"  {slug}: rate limited; waiting {secs:.0f}s before resuming", file=sys.stderr, flush=True)
            time.sleep(secs); waited += secs
            rec = resume(rec, collect_one(root, slug, a.workers, a.base, token), waited)
        if not a.no_images and os.path.exists(f"models/{slug}/snapshots/index.json"):   # the cards are installed: pin their images
            rec["images"] = images.collect_images(root, slug, a.workers, a.base.rstrip("/"), token, a.max_bytes)
        records.append(rec); print(json.dumps(rec), flush=True)
    failed = [r for r in records if "error" in r or "error" in (r.get("images") or {})]
    if a.strict_images: failed += [r for r in records if (r.get("images") or {}).get("unresolved")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
