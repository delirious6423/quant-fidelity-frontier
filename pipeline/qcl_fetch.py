#!/usr/bin/env python3
"""Fetch-once source ledger for one base model, straight from the Hugging Face API (no browser).

  python3 qcl_fetch.py ORG/MODEL [--out DIR] [--workers 8] [--base https://huggingface.co]

1. one catalogue call: every repo tagged base_model:quantized:ORG/MODEL (sha, createdAt, lastModified, downloads, tags)
2. per repo, in parallel: raw README.md at the pinned sha, and the file tree at that sha (sizes)
3. writes OUT/cards/<owner>__<name>@<sha12>.md, OUT/trees/..json, OUT/catalogue.json, OUT/index.json (sha256 per card)
Resumable: anything already in index.json at the same sha is skipped. Polite: bounded workers, honours Retry-After,
exponential back-off on 429/5xx, stops cleanly after repeated 429s and resumes next run.
Set HF_TOKEN for higher rate limits (sent only to the --base host)."""
import argparse, concurrent.futures as cf, hashlib, json, os, sys, time, urllib.error, urllib.parse, urllib.request

def get(url, token, tries=6):
    for a in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "quant-fidelity-frontier/1.0", **({"Authorization": f"Bearer {token}"} if token else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r: return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                wait = float(e.headers.get("Retry-After") or 0) or min(60, 2 ** a)
                time.sleep(wait); continue
            return e.code, e.read(), dict(e.headers)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(min(30, 2 ** a))
    raise RuntimeError(f"gave up after {tries} tries: {url}")

def next_page(url, headers):
    links = headers.get("Link") or headers.get("link") or ""
    nxt = [p.split(";")[0].strip("<> ") for p in links.split(",") if 'rel="next"' in p]
    if not nxt:
        return None
    target = urllib.parse.urljoin(url, nxt[0])
    if urllib.parse.urlparse(target).netloc != urllib.parse.urlparse(url).netloc:
        raise RuntimeError("refusing cross-host pagination link")
    return target


def write_json(path, value):
    # A completed result must survive interruption without truncating the ledger.
    temporary = f"{path}.tmp"
    with open(temporary, "w") as f:
        json.dump(value, f, indent=1)
    os.replace(temporary, path)


def catalogue(base, model, token):
    out, url = [], f"{base}/api/models?" + urllib.parse.urlencode({"filter": f"base_model:quantized:{model}", "limit": 1000, "full": "true"}) + "&expand[]=sha&expand[]=createdAt&expand[]=lastModified&expand[]=downloads&expand[]=tags"
    while url:
        st, body, h = get(url, token)
        if st != 200: raise RuntimeError(f"catalogue HTTP {st}: {body[:200]!r}")
        out += json.loads(body)
        url = next_page(url, h)
    return out


def model_metadata(base, rid, token):
    st, body, _ = get(f"{base}/api/models/{rid}", token)
    if st != 200:
        raise RuntimeError(f"model metadata {rid} HTTP {st}: {body[:200]!r}")
    metadata = json.loads(body)
    sha = metadata.get("sha")
    if not sha or sha == "main":
        raise RuntimeError(f"model metadata {rid} has no pinned commit SHA")
    return metadata


def model_sha(base, rid, token):
    return model_metadata(base, rid, token)["sha"]


def fetch_card(base, rid, sha, token, path):
    st, body, _ = get(f"{base}/{rid}/raw/{sha}/README.md", token)
    if st not in (200, 404):
        raise RuntimeError(f"README {rid}@{sha} HTTP {st}: {body[:200]!r}")
    card = body if st == 200 else b""  # Only a confirmed 404 means no README.
    with open(path, "wb") as f:
        f.write(card)
    return {"sha": sha, "file": path, "sha256": hashlib.sha256(card).hexdigest(),
            "bytes": len(card), "readme_status": st}


def fetch_tree(base, rid, sha, token):
    files, pages, seen = [], 0, set()
    url = f"{base}/api/models/{rid}/tree/{sha}?recursive=true"
    while url:
        if url in seen:
            raise RuntimeError(f"tree pagination cycle for {rid}@{sha}")
        seen.add(url)
        st, body, headers = get(url, token)
        if st != 200:
            raise RuntimeError(f"tree {rid}@{sha} HTTP {st}: {body[:200]!r}")
        page = json.loads(body)
        if not isinstance(page, list):
            raise RuntimeError(f"tree {rid}@{sha} returned a non-list response")
        files.extend(page)
        pages += 1
        url = next_page(url, headers)
    return files, pages


def fetch_repo(base, m, token, out, previous=None):
    rid = m["id"]
    sha = m.get("sha")
    if not sha or sha == "main":
        sha = model_sha(base, rid, token)
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
        card = fetch_card(base, rid, sha, token, fn)
    files, pages = fetch_tree(base, rid, sha, token)
    tree_path = f"{out}/trees/{rid.replace('/', '__')}.json"
    write_json(tree_path, files)
    sizes = [f.get("size") for f in files if f.get("type") == "file"]
    sizes_complete = all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in sizes)
    return rid, {**card,
                 "catalogue_source": m.get("catalogue_source", "base_model:quantized"),
                 "createdAt": m.get("createdAt"), "lastModified": m.get("lastModified"), "downloads": m.get("downloads"), "tags": m.get("tags"),
                 "tree_file": tree_path, "tree_complete": True, "tree_pages": pages,
                 "total_bytes_complete": sizes_complete, "total_bytes": sum(sizes) if sizes_complete else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--out", default="cache")
    ap.add_argument("--include-repo", action="append", default=[], help="also fetch this exact repository ID (repeatable)")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--base", default="https://huggingface.co")
    a = ap.parse_args(); token = os.environ.get("HF_TOKEN")
    os.makedirs(f"{a.out}/cards", exist_ok=True); os.makedirs(f"{a.out}/trees", exist_ok=True)
    t0 = time.time()
    cat = catalogue(a.base, a.model, token)
    known = {m["id"] for m in cat}
    for rid in a.include_repo:
        if rid not in known:
            metadata = model_metadata(a.base, rid, token)
            cat.append({**{k: metadata[k] for k in ("sha", "createdAt", "lastModified", "downloads", "likes", "tags") if k in metadata}, "id": rid, "catalogue_source": "explicit-include"})
            known.add(rid)
    write_json(f"{a.out}/catalogue.json", cat)
    ip = f"{a.out}/index.json"; idx = json.load(open(ip)) if os.path.exists(ip) else {"model": a.model, "repos": {}}

    def checkpoint():
        idx["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(ip, idx)

    # Resolve the official card's commit before reading its bytes, and save it even
    # if a later quant repository fails or the fetch is interrupted.
    base_sha = model_sha(a.base, a.model, token)
    base_path = f"{a.out}/cards/_base__{a.model.replace('/', '__')}@{base_sha[:12]}.md"
    idx["base"] = {"repo": a.model, "via": "hf-api", **fetch_card(a.base, a.model, base_sha, token, base_path)}
    checkpoint()
    todo = [m for m in cat if idx["repos"].get(m["id"], {}).get("sha") != m.get("sha")
            or "bytes" not in idx["repos"].get(m["id"], {})
            or not idx["repos"].get(m["id"], {}).get("tree_complete")]
    print(f"catalogue: {len(cat)} repos in {time.time()-t0:.1f}s; to fetch: {len(todo)}", flush=True)
    done, errors = 0, []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(fetch_repo, a.base, m, token, a.out, idx["repos"].get(m["id"])) for m in todo]):
            try:
                rid, rec = fut.result()
            except Exception as error:
                errors.append(str(error))
                print(f"  fetch failed: {error}", file=sys.stderr, flush=True)
                continue
            idx["repos"][rid] = rec; done += 1
            checkpoint()
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} fetched, {time.time()-t0:.1f}s", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} repositories failed; successful results checkpointed; first error: {errors[0]}")
    print(f"done: {len(idx['repos'])} repos in ledger, {time.time()-t0:.1f}s total")

if __name__ == "__main__": main()
