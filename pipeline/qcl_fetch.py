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

def catalogue(base, model, token):
    out, url = [], f"{base}/api/models?" + urllib.parse.urlencode({"filter": f"base_model:quantized:{model}", "limit": 1000, "full": "true"}) + "&expand[]=sha&expand[]=createdAt&expand[]=lastModified&expand[]=downloads&expand[]=tags"
    while url:
        st, body, h = get(url, token)
        if st != 200: raise RuntimeError(f"catalogue HTTP {st}: {body[:200]!r}")
        out += json.loads(body)
        nxt = [p.split(";")[0].strip("<> ") for p in (h.get("Link") or h.get("link") or "").split(",") if 'rel="next"' in p]
        url = nxt[0] if nxt else None
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--out", default="cache")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--base", default="https://huggingface.co")
    a = ap.parse_args(); token = os.environ.get("HF_TOKEN")
    os.makedirs(f"{a.out}/cards", exist_ok=True); os.makedirs(f"{a.out}/trees", exist_ok=True)
    t0 = time.time()
    cat = catalogue(a.base, a.model, token)
    json.dump(cat, open(f"{a.out}/catalogue.json", "w"), indent=1)
    ip = f"{a.out}/index.json"; idx = json.load(open(ip)) if os.path.exists(ip) else {"model": a.model, "repos": {}}
    todo = [m for m in cat if idx["repos"].get(m["id"], {}).get("sha") != m.get("sha")]
    print(f"catalogue: {len(cat)} repos in {time.time()-t0:.1f}s; to fetch: {len(todo)}", flush=True)
    base_card = (a.model, None)
    def one(m):
        rid, sha = m["id"], m.get("sha") or "main"
        st, body, _ = get(f"{a.base}/{rid}/raw/{sha}/README.md", token)
        card = body if st == 200 else b""          # a repo without a README is recorded as an empty card
        fn = f"{a.out}/cards/{rid.replace('/', '__')}@{sha[:12]}.md"; open(fn, "wb").write(card)
        st2, tree, _ = get(f"{a.base}/api/models/{rid}/tree/{sha}?recursive=true", token)
        if st2 == 200: open(f"{a.out}/trees/{rid.replace('/', '__')}.json", "wb").write(tree)
        files = json.loads(tree) if st2 == 200 else []
        return rid, {"sha": sha, "file": fn, "sha256": hashlib.sha256(card).hexdigest(), "readme_status": st,
                     "createdAt": m.get("createdAt"), "lastModified": m.get("lastModified"), "downloads": m.get("downloads"), "tags": m.get("tags"),
                     "total_bytes": sum(f.get("size") or 0 for f in files if f.get("type") == "file")}
    done = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(one, m) for m in todo]):
            rid, rec = fut.result(); idx["repos"][rid] = rec; done += 1
            if done % 25 == 0 or done == len(todo):
                json.dump(idx, open(ip, "w"), indent=1); print(f"  {done}/{len(todo)} fetched, {time.time()-t0:.1f}s", flush=True)
    st, body, _ = get(f"{a.base}/{a.model}/raw/main/README.md", token)      # the base model's own card (for copy detection)
    if st == 200: open(f"{a.out}/cards/_base__{a.model.replace('/', '__')}.md", "wb").write(body)
    idx["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); json.dump(idx, open(ip, "w"), indent=1)
    print(f"done: {len(idx['repos'])} repos in ledger, {time.time()-t0:.1f}s total")

if __name__ == "__main__": main()
