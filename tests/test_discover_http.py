"""Discovery over the Hub API equals discovery over saved hf_fs results, on a fake local Hub (no network).

Both paths of pipeline/qcl_discover.py run against the same small set of base models and quant repos, on temporary copies
of the repository's watchlist.json and discovery.json, and must write byte-identical watchlist.json, audit/discovery.json
and docs/QUEUE.md. A 429 with Retry-After is reported, exits non-zero and writes nothing; a fresh discover.lock skips the
run; --commit commits the three files with the connector path's message and never pushes."""
import json, os, shutil, subprocess, sys, tempfile, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "pipeline", "qcl_discover.py")
NOW = "2026-10-21T03:00:00Z"
PAGE = 25   # the fake Hub's page size for limit=1000 listings (counts paginate); like the Hub, a smaller limit is one page

# ---- the fake Hub: base models of labs in discovery.json, and quant repos tagged base_model:quantized:<base> ----
def base(mid, task="text-generation", likes=10, created="2026-10-01"):
    return {"id": mid, "pipeline_tag": task, "likes": likes, "downloads": 1000 * likes, "createdAt": f"{created}T00:00:00.000Z",
            "lastModified": f"{created}T12:00:00.000Z", "trendingScore": likes, "private": False, "gated": False, "tags": [task]}
def quants(mid, n, first="2026-09-01", used=10):
    out = []
    for i in range(n):
        day = min(28, 1 + i % 28); month = int(first[5:7]) + i // 28
        when = f"{first[:5]}{month:02d}-{day:02d}"
        out.append({"id": f"quant{i}/{mid.split('/')[1]}-Q{i}-GGUF", "pipeline_tag": "text-generation", "likes": n - i, "downloads": 5000 - 100 * i if i < used else 50 + i,
                    "createdAt": f"{when}T00:00:00.000Z", "lastModified": f"{when}T06:00:00.000Z", "trendingScore": i, "private": False,
                    "gated": False, "tags": ["gguf", f"base_model:quantized:{mid}", "text-generation"]})
    return out
MODELS = [base("Qwen/Qwen3.8-9B", likes=120, created="2026-10-02"),               # current family, 60 quants: admitted
          base("Qwen/Qwen3.8-9B-GGUF", likes=5),                                   # quantish name: skipped
          base("Qwen/Qwen4-Preview", likes=80, created="2026-10-10"),              # outside the current family, 80 likes: possible new family
          base("Qwen/Qwen3.8-27B", likes=900),                                     # already live in the watchlist: untouched
          base("google/gemma-4-2B-it", likes=70, created="2026-10-05"),            # current family, 5 quants: not admitted
          base("deepseek-ai/DeepSeek-V4.1-Lite", likes=60, created="2026-03-01"),  # current family, 45 quants but old: not admitted
          base("acme-labs/Acme-7B", likes=400),                                    # unlisted lab, 400 likes: reported
          base("acme-labs/Acme-7B-vision", task="image-text-to-text", likes=350),
          base("someone/Little-1B", likes=3)]
MODELS += quants("Qwen/Qwen3.8-9B", 60, used=12) + quants("google/gemma-4-2B-it", 5) + quants("deepseek-ai/DeepSeek-V4.1-Lite", 45, first="2026-02-01", used=3)

def search(author=None, tag=None, sort="createdAt", limit=80):
    rows = [m for m in MODELS if (not author or m["id"].split("/")[0] == author) and (not tag or tag in m["tags"])]
    key = {"createdAt": "createdAt", "lastModified": "lastModified"}.get(sort, sort)
    return sorted(rows, key=lambda m: (m[key], m["id"]), reverse=True)[:limit]

class Hub(BaseHTTPRequestHandler):
    def do_GET(self):
        srv = self.server; srv.requests.append((self.path, self.headers.get("Authorization")))
        u = urllib.parse.urlsplit(self.path); qs = urllib.parse.parse_qs(u.query)
        if srv.mode == "429":
            self.send_response(429); self.send_header("Retry-After", "120"); self.end_headers(); self.wfile.write(b"rate limited"); return
        if u.path != "/api/models":
            self.send_response(404); self.end_headers(); return
        limit = int(qs.get("limit", ["80"])[0]); start = int(qs.get("cursor", ["0"])[0])
        rows = search(qs.get("author", [None])[0], qs.get("filter", [None])[0], qs.get("sort", ["createdAt"])[0], limit)
        size = PAGE if limit >= 1000 else limit; page = rows[start:start + size]
        self.send_response(200); self.send_header("Content-Type", "application/json")
        if start + size < len(rows):
            nxt = dict(qs, cursor=[str(start + size)])
            self.send_header("Link", f'<{u.path}?{urllib.parse.urlencode(nxt, doseq=True)}>; rel="next"')
        self.end_headers(); self.wfile.write(json.dumps(page).encode())
    def log_message(self, *a): pass

def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Hub); srv.mode = "ok"; srv.requests = []
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": .01}, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"

# ---- the connector path's saved results, generated from the same fake Hub in the hf_fs format qcl_catalogue.py parses ----
def hf_entry(m):
    return {"type": "repo", "repo_type": "model", "path": m["id"], "updated_at": m["lastModified"], "downloads": m["downloads"],
            "likes": m["likes"], "gated": m["gated"], "private": m["private"], "task": m["pipeline_tag"]}
def saved_result(ops, path):
    results = []
    for op in ops:
        if op["cmd"] != "search": results.append({"result": {"op": "cat", "content": "x" * 8737}}); continue
        a = op["args"]; author = a[0].removeprefix("hf://models").strip("/") or None
        tag = a[a.index("--tag") + 1] if "--tag" in a else None
        rows = search(author, tag, a[a.index("--sort") + 1], int(a[a.index("--limit") + 1]))
        results.append({"result": {"op": "search", "uri": a[0], "entries": [hf_entry(m) for m in rows]}})
    json.dump({"results": results}, open(path, "w"))

def fresh_root(tmp, name):
    root = os.path.join(tmp, name); os.makedirs(root)
    for f in ("watchlist.json", "discovery.json"): shutil.copy(os.path.join(ROOT, f), root)
    return root
def run(*args, env=None):
    p = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, cwd=ROOT, env={**os.environ, **(env or {})})
    lines = p.stdout.strip().splitlines()
    return p, json.loads(lines[-1]) if lines else None
def outputs(root): return {f: open(os.path.join(root, f), "rb").read() for f in ("watchlist.json", "audit/discovery.json", "docs/QUEUE.md")}

srv, url = serve()
with tempfile.TemporaryDirectory() as tmp:
    # 1. HTTP path
    A = fresh_root(tmp, "http")
    p, rep = run("--http", "--base", url, "--root", A, "--now", NOW, env={"HF_TOKEN": "hf_secret_test_token"})
    assert p.returncode == 0, p.stdout + p.stderr
    r = rep["report"]
    assert r["admitted"] == ["Qwen/Qwen3.8-9B"], r
    assert [x["model"] for x in r["possible_new_families"]] == ["Qwen/Qwen4-Preview"], r
    assert sorted(x["model"] for x in r["unlisted_labs"]) == ["acme-labs/Acme-7B", "acme-labs/Acme-7B-vision"], r
    assert r["source"] == "hub-api" and "git" not in r and not os.path.exists(os.path.join(A, "work")), r
    assert all(auth == "Bearer hf_secret_test_token" for _, auth in srv.requests) and "hf_secret_test_token" not in p.stdout + p.stderr
    paths = [path for path, _ in srv.requests]
    assert sum("cursor=" in x for x in paths) == 3, "a 60-quant and a 45-quant count paginate (25 per page)"   # 2 + 1 extra pages
    assert all("direction=-1" in x and "sort=" in x for x in paths) and any("author=Qwen" in x for x in paths) and any("filter=text-generation" in x for x in paths)
    audit = json.loads(outputs(A)["audit/discovery.json"]); rows = {x["model"]: x for x in audit["counted"]}
    assert rows["Qwen/Qwen3.8-9B"] == {"model": "Qwen/Qwen3.8-9B", "quants": 60, "quants_1k": 12, "released": "2026-09-01", "searched_sorts": 1}, rows
    reasons = {x["model"]: x["reason"] for x in audit["not_admitted"]}
    assert reasons["google/gemma-4-2B-it"] == "fewer than 40 quant repos" and reasons["deepseek-ai/DeepSeek-V4.1-Lite"] == "released more than 6 months ago", reasons
    wl = json.load(open(os.path.join(A, "watchlist.json")))
    new = next(m for m in wl["models"] if m["model"] == "Qwen/Qwen3.8-9B")
    assert new == {"model": "Qwen/Qwen3.8-9B", "slug": "qwen3.8-9b", "status": "queued", "quants": 60, "quants_1k": 12, "released": "2026-09-01"}, new
    assert [m["model"] for m in wl["models"] if m["status"] != "queued"] == [m["model"] for m in json.load(open(os.path.join(ROOT, "watchlist.json")))["models"] if m["status"] != "queued"]
    queue = open(os.path.join(A, "docs/QUEUE.md")).read()
    assert "| 1 | [Qwen/Qwen3.8-9B]" in queue and "- [Qwen/Qwen4-Preview](https://huggingface.co/Qwen/Qwen4-Preview) (80 likes)" in queue, queue

    # 2. connector path, fed the equivalent saved results, round by round, exactly as a session would
    B = fresh_root(tmp, "connector"); rounds = 0; results = []
    while True:
        p, rep = run("--root", B, "--no-git", "--now", NOW, *(["--results", *results] if results else []))
        assert p.returncode == 0, p.stdout + p.stderr
        if rep["do"] == "stop": break
        assert rep["do"] == "hf_fs" and "error" not in rep, rep
        rounds += 1; results = []
        for i, ops in enumerate(rep["op_lists"]):
            assert ops[-6:] == [{"cmd": "cat", "args": ["hf://README.md", "--max-bytes", "8737"]}] * 6, ops
            f = os.path.join(tmp, f"r{rounds}-{i}.json"); saved_result(ops, f); results.append(f)
    assert rounds == 2 and rep["report"]["admitted"] == ["Qwen/Qwen3.8-9B"], (rounds, rep)
    assert outputs(A) == outputs(B), "the HTTP and connector paths must write byte-identical outputs"
    assert outputs(A)["watchlist.json"] != open(os.path.join(ROOT, "watchlist.json"), "rb").read()

    # 3. a wrong number of saved paths is refused, and the state survives for a retry
    p, rep = run("--root", fresh_root(tmp, "short"), "--no-git", "--now", NOW)
    p, rep = run("--root", os.path.join(tmp, "short"), "--no-git", "--now", NOW, "--results", results[0])
    assert p.returncode == 0 and rep["do"] == "hf_fs" and "expected" in rep["error"], rep

    # 4. HTTP 429 with Retry-After: reported, non-zero exit, nothing written
    C = fresh_root(tmp, "limited"); before = open(os.path.join(C, "watchlist.json"), "rb").read(); srv.mode = "429"
    p, rep = run("--http", "--base", url, "--root", C, "--now", NOW)
    assert p.returncode == 2 and "HTTP 429" in rep["error"] and rep["rate_limited"]["retry_after"] == "120" and rep["rate_limited"]["retry_after_seconds"] == 120, (p.returncode, rep)
    assert rep["requests"] == 1, rep
    assert open(os.path.join(C, "watchlist.json"), "rb").read() == before and not os.path.exists(os.path.join(C, "audit")) and not os.path.exists(os.path.join(C, "docs"))
    srv.mode = "ok"

    # 5. a fresh discover.lock (a connector-driven run in progress) skips the HTTP run without writing
    D = fresh_root(tmp, "locked"); open(os.path.join(D, "discover.lock"), "w").write("2026-10-21T02:50:00+00:00")
    p, rep = run("--http", "--base", url, "--root", D, "--now", NOW)
    assert p.returncode == 0 and "skipped" in rep["report"] and not os.path.exists(os.path.join(D, "audit")), rep

    # 6. --commit commits the three output files with the connector path's message; no remote, no push
    E = fresh_root(tmp, "committed")
    g = lambda *c: subprocess.run(["git", *c], cwd=E, capture_output=True, text=True, check=True).stdout.strip()
    g("init", "-q"); g("add", "-A"); g("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "base")
    p, rep = run("--http", "--base", url, "--root", E, "--now", NOW, "--commit", env={"GIT_AUTHOR_NAME": "bot", "GIT_AUTHOR_EMAIL": "bot@example.com",
                                                                                      "GIT_COMMITTER_NAME": "bot", "GIT_COMMITTER_EMAIL": "bot@example.com"})
    assert p.returncode == 0 and rep["report"]["git"] == "committed", rep
    assert g("log", "-1", "--format=%s") == f"Weekly discovery 2026-10-21: 1 admitted, {rep['report']['updated']} queued models recounted"
    assert sorted(g("show", "--format=", "--name-only", "HEAD").splitlines()) == ["audit/discovery.json", "docs/QUEUE.md", "watchlist.json"]
    assert g("status", "--porcelain") == "" and g("remote") == ""
    assert outputs(E) == outputs(A)
srv.shutdown()
print(f"PASS: HTTP and connector discovery identical on {len(MODELS)} fake Hub repos; 429 reported without output; lock skip; --commit")
