#!/usr/bin/env python3
"""Fetch-once source ledger for one base model, straight from the Hugging Face API (no browser).

  python3 qcl_fetch.py ORG/MODEL [--out DIR] [--workers 8] [--base https://huggingface.co]

1. one catalogue call: every repo tagged base_model:quantized:ORG/MODEL (sha, createdAt, lastModified, downloads, tags)
2. per repo, in parallel: raw README.md at the pinned sha, and the file tree at that sha (sizes)
3. writes OUT/cards/<owner>__<name>@<sha12>.md, OUT/trees/..json, OUT/catalogue.json, OUT/index.json (sha256 per card)
Resumable: anything already in index.json at the same sha is skipped. Polite: bounded workers,
the first 429 stops the batch and records Retry-After without sleeping. In-flight requests may
finish, but no worker starts another request. Transient 5xx retries wait at most 60 seconds.
Set HF_TOKEN for higher rate limits (sent only to the requested origin). Public
cross-origin redirects drop credentials; HTTPS never redirects to HTTP."""
import argparse, concurrent.futures as cf, datetime as dt, email.utils, hashlib, json, math, os, re, sys, threading, time, urllib.error, urllib.parse, urllib.request


def valid_model_id(model):
    return (isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", model)
            and ".." not in model)


def read_identity_object(path, label):
    try:
        with open(path, encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read existing {label} {path}: {error}") from None
    if not isinstance(value, dict):
        raise ValueError(f"existing {label} must be an object: {path}")
    return value


def load_ledger(out, model, expected_slug=None):
    """Bind a reused destination to one base model before any request or mutation."""
    if not valid_model_id(model):
        raise ValueError("model must be a Hugging Face owner/repository ID")
    config_path = os.path.join(os.path.dirname(os.path.abspath(out)), "model.json")
    if os.path.lexists(config_path):
        config = read_identity_object(config_path, "model config")
        if config.get("model") != model:
            raise ValueError(f"existing model config identity does not match requested {model!r}: {config_path}")
        if expected_slug is not None and config.get("slug") != expected_slug:
            raise ValueError(f"existing model config slug does not match destination {expected_slug!r}: {config_path}")
    index_path = os.path.join(out, "index.json")
    if not os.path.lexists(index_path):
        return {"model": model, "repos": {}}
    index = read_identity_object(index_path, "ledger")
    if not valid_model_id(index.get("model")):
        raise ValueError(f"existing ledger model is missing or invalid; identity must be reviewed before reuse: {index_path}")
    if index["model"] != model:
        raise ValueError(f"existing ledger model {index['model']!r} does not match requested {model!r}: {index_path}")
    if not isinstance(index.get("repos"), dict) or any(not isinstance(record, dict) for record in index["repos"].values()):
        raise ValueError(f"existing ledger repos must be an object of records: {index_path}")
    if "base" in index and (not isinstance(index["base"], dict) or index["base"].get("repo") != model):
        raise ValueError(f"existing ledger base card identity does not match requested {model!r}: {index_path}")
    return index


def origin(url):
    """Compare complete HTTP origins; credentials and protocol downgrades are never implicit."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("expected an HTTP(S) URL without embedded credentials")
    return parsed.scheme, parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_stopped(getattr(req, "_qff_stop", None))
        previous, following = origin(req.full_url), origin(newurl)
        if previous[0] == "https" and following[0] != "https":
            raise RuntimeError("refusing HTTPS-to-HTTP redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected._qff_stop = getattr(req, "_qff_stop", None)
        if redirected is not None and previous != following:
            # Hugging Face may redirect public files to a CDN. Fetch the public
            # bytes, but never forward the Hub token to that different origin.
            for name in list(redirected.headers) + list(redirected.unredirected_hdrs):
                if name.lower() in ("authorization", "cookie", "proxy-authorization"):
                    redirected.remove_header(name)
        return redirected


OPENER = urllib.request.build_opener(SafeRedirectHandler())


def retry_after_seconds(value, now=None):
    """Parse both Retry-After forms; malformed values use normal backoff."""
    if not value:
        return None
    try:
        seconds = float(value)
        return max(0, seconds) if math.isfinite(seconds) else None
    except (TypeError, ValueError):
        try:
            when = email.utils.parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
            return max(0, when.timestamp() - (time.time() if now is None else now))
        except (TypeError, ValueError, OverflowError):
            return None


class FetchStopped(RuntimeError):
    pass


class RateLimited(RuntimeError):
    def __init__(self, url, retry_after):
        self.record = {"url": url, "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "retry_after": retry_after, "retry_after_seconds": retry_after_seconds(retry_after)}
        super().__init__(f"HTTP 429: batch stopped; Retry-After={retry_after!r}; resume a later run: {url}")


def check_stopped(stop):
    if stop is not None and stop.is_set():
        raise FetchStopped("batch stopped after HTTP 429")


def pause(seconds, stop):
    if stop is None:
        time.sleep(seconds)
    elif stop.wait(seconds):
        raise FetchStopped("batch stopped after HTTP 429")


def get(url, token, tries=6, stop=None):
    origin(url)
    for a in range(tries):
        check_stopped(stop)
        req = urllib.request.Request(url, headers={"User-Agent": "quant-fidelity-frontier/1.0", **({"Authorization": f"Bearer {token}"} if token else {})})
        req._qff_stop = stop
        try:
            with OPENER.open(req, timeout=60) as r: return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if stop is not None:
                    stop.set()
                retry_after = e.headers.get("Retry-After")
                e.close()
                raise RateLimited(url, retry_after) from None
            if e.code in (500, 502, 503, 504):
                delay = retry_after_seconds(e.headers.get("Retry-After"))
                e.close()
                if a + 1 < tries:
                    pause(min(60, delay if delay is not None else 2 ** a), stop)
                continue
            return e.code, e.read(), dict(e.headers)
        except (urllib.error.URLError, TimeoutError):
            if a + 1 < tries:
                pause(min(30, 2 ** a), stop)
    raise RuntimeError(f"gave up after {tries} tries: {url}")

def next_page(url, headers):
    links = headers.get("Link") or headers.get("link") or ""
    nxt = [p.split(";")[0].strip("<> ") for p in links.split(",") if 'rel="next"' in p]
    if not nxt:
        return None
    target = urllib.parse.urljoin(url, nxt[0])
    if origin(target) != origin(url):
        raise RuntimeError("refusing cross-origin pagination link")
    return target


def write_json(path, value):
    # A completed result must survive interruption without truncating the ledger.
    temporary = f"{path}.tmp"
    with open(temporary, "w") as f:
        json.dump(value, f, indent=1)
    os.replace(temporary, path)


def catalogue(base, model, token, stop=None):
    out, url = [], f"{base}/api/models?" + urllib.parse.urlencode({"filter": f"base_model:quantized:{model}", "limit": 1000, "full": "true"}) + "&expand[]=sha&expand[]=createdAt&expand[]=lastModified&expand[]=downloads&expand[]=tags"
    while url:
        st, body, h = get(url, token, stop=stop)
        if st != 200: raise RuntimeError(f"catalogue HTTP {st}: {body[:200]!r}")
        out += json.loads(body)
        url = next_page(url, h)
    return out


def model_metadata(base, rid, token, stop=None):
    st, body, _ = get(f"{base}/api/models/{rid}", token, stop=stop)
    if st != 200:
        raise RuntimeError(f"model metadata {rid} HTTP {st}: {body[:200]!r}")
    metadata = json.loads(body)
    sha = metadata.get("sha")
    if not sha or sha == "main":
        raise RuntimeError(f"model metadata {rid} has no pinned commit SHA")
    return metadata


def model_sha(base, rid, token, stop=None):
    return model_metadata(base, rid, token, stop)["sha"]


def fetch_card(base, rid, sha, token, path, stop=None):
    st, body, _ = get(f"{base}/{rid}/raw/{sha}/README.md", token, stop=stop)
    if st not in (200, 404):
        raise RuntimeError(f"README {rid}@{sha} HTTP {st}: {body[:200]!r}")
    card = body if st == 200 else b""  # Only a confirmed 404 means no README.
    with open(path, "wb") as f:
        f.write(card)
    return {"sha": sha, "file": path, "sha256": hashlib.sha256(card).hexdigest(),
            "bytes": len(card), "readme_status": st}


def fetch_tree(base, rid, sha, token, stop=None):
    files, pages, seen = [], 0, set()
    url = f"{base}/api/models/{rid}/tree/{sha}?recursive=true"
    while url:
        if url in seen:
            raise RuntimeError(f"tree pagination cycle for {rid}@{sha}")
        seen.add(url)
        st, body, headers = get(url, token, stop=stop)
        if st != 200:
            raise RuntimeError(f"tree {rid}@{sha} HTTP {st}: {body[:200]!r}")
        page = json.loads(body)
        if not isinstance(page, list):
            raise RuntimeError(f"tree {rid}@{sha} returned a non-list response")
        files.extend(page)
        pages += 1
        url = next_page(url, headers)
    return files, pages


def fetch_repo(base, m, token, out, previous=None, stop=None):
    check_stopped(stop)
    rid = m["id"]
    sha = m.get("sha")
    if not sha or sha == "main":
        sha = model_sha(base, rid, token, stop)
    fn = f"{out}/cards/{rid.replace('/', '__')}@{sha[:12]}.md"
    # Previously checkpointed bytes are safe to reuse even when their old tree
    # was only a first page. Unindexed orphan cards can be recovered by recording
    # their pinned SHA, hash and successful HTTP status in the ledger first.
    card = None
    if previous and previous.get("sha") == sha and previous.get("readme_status") in (200, 404):
        try:
            with open(previous["file"], "rb") as f:
                saved = f.read()
            if hashlib.sha256(saved).hexdigest() == previous.get("sha256"):
                card = {k: previous[k] for k in ("sha", "file", "sha256", "readme_status")}
                card["bytes"] = len(saved)
        except (OSError, KeyError):
            pass
    if card is None:
        card = fetch_card(base, rid, sha, token, fn, stop)
    files, pages = fetch_tree(base, rid, sha, token, stop)
    tree_path = f"{out}/trees/{rid.replace('/', '__')}.json"
    write_json(tree_path, files)
    sizes = [f.get("size") for f in files if f.get("type") == "file"]
    sizes_complete = all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in sizes)
    return rid, {**card,
                 "catalogue_source": m.get("catalogue_source", "base_model:quantized"),
                 "createdAt": m.get("createdAt"), "lastModified": m.get("lastModified"), "downloads": m.get("downloads"), "tags": m.get("tags"),
                 "tree_file": tree_path, "tree_complete": True, "tree_pages": pages,
                 "total_bytes_complete": sizes_complete, "total_bytes": sum(sizes) if sizes_complete else None}


def fetch_model(a, token):
    # A mistyped --out must never combine two models' catalogues or sources.
    idx = load_ledger(a.out, a.model)
    os.makedirs(f"{a.out}/cards", exist_ok=True); os.makedirs(f"{a.out}/trees", exist_ok=True)
    stop = threading.Event()
    t0 = time.time()
    cat = catalogue(a.base, a.model, token, stop)
    known = {m["id"] for m in cat}
    for rid in a.include_repo:
        if rid not in known:
            metadata = model_metadata(a.base, rid, token, stop)
            cat.append({**{k: metadata[k] for k in ("sha", "createdAt", "lastModified", "downloads", "likes", "tags") if k in metadata}, "id": rid, "catalogue_source": "explicit-include"})
            known.add(rid)
    write_json(f"{a.out}/catalogue.json", cat)
    ip = f"{a.out}/index.json"

    def checkpoint():
        idx["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(ip, idx)

    # Resolve the official card's commit before reading its bytes, and save it even
    # if a later quant repository fails or the fetch is interrupted.
    base_sha = model_sha(a.base, a.model, token, stop)
    base_path = f"{a.out}/cards/_base__{a.model.replace('/', '__')}@{base_sha[:12]}.md"
    idx["base"] = {"repo": a.model, "via": "hf-api", **fetch_card(a.base, a.model, base_sha, token, base_path, stop)}
    checkpoint()
    todo = [m for m in cat if idx["repos"].get(m["id"], {}).get("sha") != m.get("sha")
            or "bytes" not in idx["repos"].get(m["id"], {})
            or not idx["repos"].get(m["id"], {}).get("tree_complete")]
    print(f"catalogue: {len(cat)} repos in {time.time()-t0:.1f}s; to fetch: {len(todo)}", flush=True)
    done, errors, rate_limited = 0, [], None
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(fetch_repo, a.base, m, token, a.out, idx["repos"].get(m["id"]), stop) for m in todo]):
            try:
                rid, rec = fut.result()
            except RateLimited as error:
                rate_limited = rate_limited or error
                write_json(f"{a.out}/rate_limit.json", rate_limited.record)
                continue
            except FetchStopped:
                continue
            except Exception as error:
                errors.append(str(error))
                print(f"  fetch failed: {error}", file=sys.stderr, flush=True)
                continue
            idx["repos"][rid] = rec; done += 1
            checkpoint()
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} fetched, {time.time()-t0:.1f}s", flush=True)
    if rate_limited:
        raise rate_limited
    if errors:
        raise RuntimeError(f"{len(errors)} repositories failed; successful results checkpointed; first error: {errors[0]}")
    print(f"done: {len(idx['repos'])} repos in ledger, {time.time()-t0:.1f}s total")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--out", default="cache")
    ap.add_argument("--include-repo", action="append", default=[], help="also fetch this exact repository ID (repeatable)")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--base", default="https://huggingface.co")
    a = ap.parse_args()
    if not 1 <= a.workers <= 32:
        ap.error("--workers must be between 1 and 32")
    try:
        fetch_model(a, os.environ.get("HF_TOKEN"))
    except RateLimited as error:
        write_json(f"{a.out}/rate_limit.json", error.record)
        raise SystemExit(str(error)) from None
    except ValueError as error:
        raise SystemExit(f"REFUSING TO FETCH: {error}") from None
    if os.path.exists(f"{a.out}/rate_limit.json"):
        os.remove(f"{a.out}/rate_limit.json")


if __name__ == "__main__": main()
