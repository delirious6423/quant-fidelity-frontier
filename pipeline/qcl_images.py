#!/usr/bin/env python3
"""Fetch the content images of every pinned card once, byte-for-byte: the charts and tables that often hold a quant's benchmark numbers.

  python3 pipeline/qcl_images.py [--models SLUG ...] [--all-live] [--workers 8] [--strict] [--max-bytes 8000000] [--base URL] [--root DIR]

For each model it reads every card that models/<slug>/snapshots/index.json pins, finds the content images with
pipeline/card_images.py (badges and logos are left out), resolves a relative URL against the card's Hub location
(https://huggingface.co/<repo>/resolve/<sha or main>/<path>; an absolute URL is used as given; a data: URI is decoded
in place) and fetches each URL once into models/<slug>/cache/images/<sha256>.<ext>, with the ledger
models/<slug>/cache/images/index.json keyed by URL ({sha256, bytes, content_type, file, fetched_at, repos, L}).
Resumable: a URL whose ledger entry has its file present with the recorded SHA-256 is never requested again; a URL no card
references any more leaves the ledger (and its file, when nothing else pins it). The HTTP helper is pipeline/qcl_fetch.py get
(3 attempts with backoff on 5xx, timeouts and connection errors; a 429 stops the batch and its Retry-After is recorded,
never slept on). Bytes are stored exactly as received. HF_TOKEN is sent to the Hub origin only (--base) and is never
printed or saved; no response body reaches stdout.
Coverage per model: models/<slug>/audit/images.json (cards, references, referenced, pinned). Every URL that is not pinned
is classified, with evidence, into exactly one of three lists there (entries {url, repo, L, references, status, attempts,
reason}, sorted by URL, no timestamp, so an unchanged situation writes identical bytes hour to hour):
- unresolved: a retryable failure, the only class --strict fails on: the helper gave up (5xx, timeout or connection error
  on every attempt), a 429 stopped the batch (that URL and the ones not requested after it), any other HTTP answer, or
  an unexpected exception;
- gone: a definitive HTTP 401, 403, 404 or 410: the file is not available to anyone with this access (deleted, or a gated
  repository);
- refused: by this repository's policy: not an image content type, or larger than --max-bytes (bytes and content_type of
  the response are recorded, the body is never stored), a URL with credentials or a non-HTTP(S) scheme, a redirect the
  helper refuses, or a data: URI that does not decode.
One JSON line on stdout (slug, cards, referenced, pinned, new, unresolved, gone, refused, seconds). --strict exits non-zero
only when a model has an unresolved image: a scheduled run stays red for what a later run may still fetch, never for a
file nobody can fetch. pipeline/qcl_collect.py runs this step after the cards of a model are installed."""
import argparse, base64, binascii, concurrent.futures as cf, glob, hashlib, json, os, re, sys, threading, time, urllib.parse

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "pipeline"))
import qcl_fetch as fetch
import card_images

HUB = "https://huggingface.co"
TRIES = 3                      # attempts per URL inside qcl_fetch.get (5xx, timeouts, connection errors)
DEFAULT_MAX_BYTES = 8_000_000
GONE = frozenset({401, 403, 404, 410})   # a definitive answer: the file is not available with this access
EXT = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "gif": "gif", "webp": "webp", "svg+xml": "svg", "bmp": "bmp", "avif": "avif",
       "tiff": "tiff", "x-icon": "ico", "vnd.microsoft.icon": "ico"}
MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"),
         (b"BM", "image/bmp"))
URL_SAFE = "%/:?=&#@!$'()*+,;~[]"
FILE_RE = re.compile(r"[0-9a-f]{64}\.[a-z0-9]{1,8}")


# ---------------------------------------------------------------- what the cards reference
def live_slugs(root):
    wl = os.path.join(root, "watchlist.json")
    st = {m["slug"]: m["status"] for m in json.load(open(wl))["models"]} if os.path.exists(wl) else {}
    out = []
    for mj in sorted(glob.glob(os.path.join(root, "models", "*", "model.json"))):
        slug = json.load(open(mj)).get("slug") or os.path.basename(os.path.dirname(mj))
        if st.get(slug, "live") == "live": out.append(slug)
    return out


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f: return json.load(f)
    except (OSError, ValueError): return default


def resolve(base, repo, sha, url):
    """The URL to request for one image reference: absolute as given, relative against the card's Hub location."""
    url = url.strip()
    if url.lower().startswith("data:"): return url
    parts = urllib.parse.urlsplit(url)
    if parts.scheme or url.startswith("//"):
        resolved = url if parts.scheme else "https:" + url
    else:
        resolved = urllib.parse.urljoin(f"{base}/{repo}/resolve/{sha or 'main'}/", url)
    resolved = urllib.parse.urldefrag(resolved).url
    return urllib.parse.quote(resolved, safe=URL_SAFE)


def card_references(model_dir, index, base):
    """Every content image reference of every card the index pins: {url (resolved), repo, L, alt, raw}."""
    refs = []
    for repo, rec in sorted((index or {}).get("repos", {}).items()):
        if not rec.get("file"): continue
        path = os.path.join(model_dir, rec["file"])
        if os.path.commonpath([os.path.abspath(model_dir), os.path.abspath(path)]) != os.path.abspath(model_dir):
            raise ValueError(f"card path escapes the model root: {rec['file']}")
        try:
            with open(path, "rb") as f: lines = f.read().decode("utf-8", errors="replace").split("\n")
        except OSError:
            continue
        for n, alt, raw in card_images.card_images(lines):
            refs.append({"url": resolve(base, repo, rec.get("sha"), raw), "repo": repo, "L": n, "alt": alt, "raw": raw})
    return refs


def by_url(refs):
    out = {}
    for r in refs:
        out.setdefault(r["url"], []).append(r)
    return out


# ---------------------------------------------------------------- the ledger
def ledger_path(model_dir):
    return os.path.join(model_dir, "cache", "images", "index.json")


def load_ledger(model_dir, model):
    ledger = read_json(ledger_path(model_dir), {}) or {}
    if not isinstance(ledger, dict) or not isinstance(ledger.get("images"), dict):
        ledger = {"images": {}}
    if ledger.get("model") not in (None, model):
        raise ValueError(f"existing image ledger belongs to {ledger.get('model')!r}, not {model!r}")
    return {"model": model, "pinning": "sha256 of the image bytes as received; URL resolved against the card's pinned commit",
            "images": {u: r for u, r in ledger["images"].items() if isinstance(r, dict)}}


def pinned_file(model_dir, rec):
    """The image file of a ledger record when it is present inside the model root with the recorded SHA-256, else None."""
    file = rec.get("file")
    if not isinstance(file, str) or not file or not isinstance(rec.get("sha256"), str): return None
    path = os.path.abspath(os.path.join(model_dir, file))
    if os.path.commonpath([os.path.abspath(model_dir), path]) != os.path.abspath(model_dir) or not os.path.isfile(path): return None
    with open(path, "rb") as f: raw = f.read()
    if hashlib.sha256(raw).hexdigest() != rec["sha256"] or rec.get("bytes") not in (None, len(raw)): return None
    return path


def pinned_images(model_dir, repo, meta, raw, prefix="", base=HUB):
    """[{url, sha256, file, bytes, content_type, L, alt}] for the pinned content images of one card (the mapping bundle's
    `images`); `prefix` is put before the model-root-relative file path (the runner uses models/<slug>/)."""
    ledger = read_json(ledger_path(model_dir), {}) or {}
    images = ledger.get("images") if isinstance(ledger, dict) else None
    if not isinstance(images, dict): return []
    lines = raw.decode("utf-8", errors="replace").split("\n")
    out, seen = [], set()
    for n, alt, ref in card_images.card_images(lines):
        url = resolve(base, repo, meta.get("sha"), ref)
        rec = images.get(url)
        if not isinstance(rec, dict) or (url, n) in seen or pinned_file(model_dir, rec) is None: continue
        seen.add((url, n))
        out.append({"url": url, "sha256": rec["sha256"], "file": prefix + rec["file"], "bytes": rec.get("bytes"),
                    "content_type": rec.get("content_type"), "L": n, "alt": alt})
    return out


def serialized(value):
    return json.dumps(value, indent=1).encode()


def write_if_changed(path, value):
    data = serialized(value)
    try:
        with open(path, "rb") as f:
            if f.read() == data: return False
    except OSError: pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f: f.write(data)
    os.replace(tmp, path)
    return True


# ---------------------------------------------------------------- one image
def image_type(headers, body):
    """(content type as received, sniffed type) — the second from the first bytes, when they carry an image signature."""
    ct = (headers.get("Content-Type") or headers.get("content-type") or "").split(";")[0].strip().lower()
    sniffed = next((t for magic, t in MAGIC if body.startswith(magic)), None)
    if sniffed is None and body[:4] == b"RIFF" and body[8:12] == b"WEBP": sniffed = "image/webp"
    return ct, sniffed


def extension(content_type):
    sub = content_type.split("/", 1)[1] if "/" in content_type else ""
    return EXT.get(sub) or (re.sub(r"[^a-z0-9]", "", sub)[:8] or "img")


def decode_data_uri(url):
    m = re.match(r"data:([^;,]*)(;base64)?,(.*)$", url, re.S | re.I)
    if not m: raise ValueError("malformed data: URI")
    ct = (m.group(1) or "").strip().lower()
    try:
        body = base64.b64decode(m.group(3), validate=True) if m.group(2) else urllib.parse.unquote_to_bytes(m.group(3))
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"data: URI does not decode: {error}") from None
    return ct, body


def fetch_image(url, base, token, max_bytes, stop):
    """One image: ("pinned", body, content_type, sniffed) or ("refused" | "http", status, reason, evidence), where evidence
    carries the bytes and content_type of a response this policy refused. Raises RateLimited / FetchStopped (429 stops the
    batch) and RuntimeError (the helper gave up after TRIES attempts, or refused a redirect)."""
    if url.lower().startswith("data:"):
        ct, body = decode_data_uri(url)
        headers = {"Content-Type": ct or "application/octet-stream"}
        status = "inline"
    else:
        try:
            same_origin = fetch.origin(url) == fetch.origin(base)
        except RuntimeError as error:
            return "refused", None, str(error), {}
        status, body, headers = fetch.get(url, token if same_origin else None, tries=TRIES, stop=stop)
        if status != 200:
            return "http", status, f"HTTP {status}", {}
    ct, sniffed = image_type(headers, body)
    evidence = {"bytes": len(body), **({"content_type": ct} if ct else {})}
    if not ct.startswith("image/") and sniffed is None:
        return "refused", status, f"not an image: content type {ct or 'missing'}", evidence
    if len(body) > max_bytes:
        return "refused", status, f"larger than --max-bytes: {len(body)} > {max_bytes} bytes", evidence
    return "pinned", body, ct, sniffed


# ---------------------------------------------------------------- one model
def collect_images(root, slug, workers=8, base=HUB, token=None, max_bytes=DEFAULT_MAX_BYTES):
    """Pin every content image the model's cards reference. Returns the stdout record (without the slug)."""
    t0 = time.time()
    model_dir = os.path.join(root, "models", slug)
    rec = {"cards": 0, "referenced": 0, "pinned": 0, "new": 0, "unresolved": 0, "gone": 0, "refused": 0, "seconds": None}
    index = read_json(os.path.join(model_dir, "snapshots", "index.json"))
    if not isinstance(index, dict):
        rec["error"] = "no snapshot index"; rec["seconds"] = round(time.time() - t0, 1); return rec
    model = (read_json(os.path.join(model_dir, "model.json"), {}) or {}).get("model") or index.get("model")
    try:
        ledger = load_ledger(model_dir, model)
        refs = card_references(model_dir, index, base)
    except ValueError as error:
        rec["error"] = str(error); rec["seconds"] = round(time.time() - t0, 1); return rec
    groups = by_url(refs)
    rec["cards"] = len({r["repo"] for r in refs}); rec["references"] = len(refs); rec["referenced"] = len(groups)
    images_dir = os.path.join(model_dir, "cache", "images")
    lock = threading.Lock()
    stop = threading.Event()
    failures, rate_limited = {"unresolved": {}, "gone": {}, "refused": {}}, None   # every URL not pinned lands in exactly one

    def uses(url):
        rs = groups[url]
        return {"repos": sorted({r["repo"] for r in rs}), "L": {repo: sorted({r["L"] for r in rs if r["repo"] == repo}) for repo in sorted({r["repo"] for r in rs})}}

    kept = {}
    for url in sorted(groups):
        old = ledger["images"].get(url)
        if isinstance(old, dict) and pinned_file(model_dir, old) is not None:
            kept[url] = {**{k: old[k] for k in ("sha256", "bytes", "content_type", "file", "fetched_at") if k in old},
                         **({"sniffed": old["sniffed"]} if old.get("sniffed") else {}), **uses(url)}
    ledger["images"] = kept
    todo = [url for url in sorted(groups) if url not in kept]

    def checkpoint():
        write_if_changed(ledger_path(model_dir), {**ledger, "images": {u: ledger["images"][u] for u in sorted(ledger["images"])}})

    def one(url):
        return url, fetch_image(url, base, token, max_bytes, stop)

    def record(cls, url, status, attempts, reason, **evidence):
        failures[cls][url] = {"status": status, "attempts": attempts, "reason": reason, **evidence}

    with cf.ThreadPoolExecutor(max(1, workers)) as ex:
        futures = {ex.submit(one, url): url for url in todo}
        for fut in cf.as_completed(futures):
            url = futures[fut]
            try:
                _, result = fut.result()
            except fetch.RateLimited as error:
                rate_limited = rate_limited or error
                record("unresolved", url, 429, 1, "HTTP 429: the batch stopped; Retry-After recorded, resume a later run")
                continue
            except fetch.FetchStopped:
                record("unresolved", url, None, 0, "not requested: the batch stopped after HTTP 429")
                continue
            except RuntimeError as error:   # the helper gave up (5xx, timeouts or connection errors on every attempt), or refused a redirect
                reason = re.sub(r":\s*\S+$", "", str(error)) or "gave up"
                if reason.startswith("refusing"): record("refused", url, None, 1, reason)   # policy: no request can change it
                else: record("unresolved", url, None, TRIES if reason.startswith("gave up") else 1, reason)
                continue
            except ValueError as error:     # a data: URI that does not decode: no request can change it
                record("refused", url, None, 0, str(error))
                continue
            except Exception as error:      # noqa: BLE001 - one image must never stop the others
                record("unresolved", url, None, None, f"{type(error).__name__}")
                continue
            if result[0] == "http":         # a definitive answer is gone; any other status may still change, so it stays unresolved
                record("gone" if result[1] in GONE else "unresolved", url, result[1], 1, result[2]); continue
            if result[0] == "refused":
                record("refused", url, result[1], 1 if result[1] is not None else 0, result[2], **result[3]); continue
            _, body, ct, sniffed = result
            digest = hashlib.sha256(body).hexdigest()
            file = f"cache/images/{digest}.{extension(ct if ct.startswith('image/') else sniffed)}"
            with lock:
                path = os.path.join(model_dir, file)
                if not os.path.exists(path):
                    os.makedirs(images_dir, exist_ok=True)
                    tmp = f"{path}.tmp"
                    with open(tmp, "wb") as f: f.write(body)     # exactly the bytes received
                    os.replace(tmp, path)
                entry = {"sha256": digest, "bytes": len(body), "content_type": ct or sniffed, "file": file,
                         "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **uses(url)}
                if sniffed and sniffed != ct: entry["sniffed"] = sniffed
                ledger["images"][url] = entry
                rec["new"] += 1
                if rec["new"] % 25 == 0: checkpoint()   # a run cut short keeps what it pinned; the files are already on disk
    # Files no entry pins any more (a card dropped the image, or changed it) leave the cache with their entries.
    referenced_files = {os.path.abspath(os.path.join(model_dir, e["file"])) for e in ledger["images"].values()}
    if os.path.isdir(images_dir):
        for name in os.listdir(images_dir):
            path = os.path.abspath(os.path.join(images_dir, name))
            if FILE_RE.fullmatch(name) and path not in referenced_files and os.path.isfile(path):
                os.remove(path)
    checkpoint()
    rec["pinned"] = len(ledger["images"])
    def entries(cls):
        return [{"url": url, "repo": groups[url][0]["repo"], "L": groups[url][0]["L"], "references": len(groups[url]), **failures[cls][url]}
                for url in sorted(failures[cls])]

    report = {"model": model, "cards": rec["cards"], "references": rec["references"], "referenced": rec["referenced"], "pinned": rec["pinned"],
              **{cls: entries(cls) for cls in ("unresolved", "gone", "refused")}}
    if rate_limited:
        report["rate_limited"] = {k: rate_limited.record.get(k) for k in ("url", "retry_after")}
        rec["rate_limited"] = {k: rate_limited.record.get(k) for k in ("retry_after", "retry_after_seconds", "observed_at")}
    write_if_changed(os.path.join(model_dir, "audit", "images.json"), report)
    for cls, found in failures.items(): rec[cls] = len(found)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=[], help="slugs under models/ (default: every live model)")
    ap.add_argument("--all-live", action="store_true", help="every live model (the default when --models is not given)")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--strict", action="store_true", help="exit 1 when any image is unresolved (retryable); gone and refused images do not fail")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES); ap.add_argument("--base", default=HUB, help="the Hub origin (tests use a local server)")
    ap.add_argument("--root", default=CODE, help="repository root holding models/ and watchlist.json (tests use a copy)")
    a = ap.parse_args()
    if not 1 <= a.workers <= 32: ap.error("--workers must be between 1 and 32")
    if a.max_bytes < 1: ap.error("--max-bytes must be positive")
    root = os.path.abspath(a.root)
    slugs = a.models or live_slugs(root)
    if a.all_live: slugs = sorted(set(slugs) | set(live_slugs(root)))
    for slug in slugs:
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,95}", slug) or ".." in slug: ap.error(f"bad slug: {slug!r}")
    token = os.environ.get("HF_TOKEN") or None
    base = a.base.rstrip("/")
    short = []
    for slug in slugs:
        rec = collect_images(root, slug, a.workers, base, token, a.max_bytes)
        print(json.dumps({"slug": slug, **rec}), flush=True)
        if rec.get("error") or (a.strict and rec["unresolved"]): short.append(slug)
    return 1 if short else 0


if __name__ == "__main__":
    sys.exit(main())
