"""Card images: pipeline/qcl_images.py pins every content image the pinned cards embed, once and byte-for-byte, against a
loopback Hub and a loopback CDN (two origins, so the token's reach is visible). Nothing here reaches the network, the
repository's models/ or git."""
import base64
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "pipeline"))
import qcl_images  # noqa: E402
import runner_model as M  # noqa: E402

IMAGES, COLLECT = CODE / "pipeline/qcl_images.py", CODE / "pipeline/qcl_collect.py"
MODEL, SLUG = "test/base", "testmodel"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + bytes(range(40))
PNG2 = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + bytes(range(40, 90))
JPG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + bytes(range(60))
BIG = b"\x89PNG\r\n\x1a\n" + bytes(300)
ONE, TWO, THREE = "1" * 40, "main", "3" * 40


def catalogue_path(model):
    return "/api/models?" + urllib.parse.urlencode({"filter": f"base_model:quantized:{model}", "limit": 1000, "full": "true"}) \
        + "&expand[]=sha&expand[]=createdAt&expand[]=lastModified&expand[]=downloads&expand[]=tags"


class Server:
    """One loopback origin: `routes` maps a path to (status, body, headers); `flaky` paths answer 503 that many times first."""
    def __init__(self):
        self.routes, self.flaky, self.requests = {}, {}, []
        me = self
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                me.requests.append((self.path, self.headers.get("Authorization")))
                if me.flaky.get(self.path):
                    me.flaky[self.path] -= 1
                    status, body, headers = 503, b"flaky", {}
                else:
                    status, body, headers = me.respond(self.path)
                self.send_response(status)
                for k, v in headers.items(): self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            def log_message(self, *a): pass
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def respond(self, path):
        return self.routes.get(path, (404, b"not found", {}))

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=1)

    def paths(self, needle=""):
        return [p for p, _ in self.requests if needle in p]


class Hub(Server):
    """The Hub: catalogue, model metadata, raw cards and trees (for qcl_collect) and resolve/ image files."""
    def __init__(self):
        super().__init__()
        self.sha = {MODEL: "b" * 40, "test/one": ONE, "test/two": "2" * 40, "test/three": THREE}
        self.cards = {}
        self.files = {("test/one", "images/kld.png"): PNG, ("test/one", "ppl.png"): PNG2}

    def respond(self, path):
        if path == catalogue_path(MODEL):
            return 200, json.dumps([{"id": rid, "sha": self.sha[rid], "lastModified": "2026-10-20T10:00:00.000Z", "createdAt": "2026-10-01T00:00:00.000Z",
                                     "downloads": 1, "likes": 1, "tags": []} for rid in ("test/one", "test/two", "test/three")]).encode(), {}
        for rid, sha in self.sha.items():
            if path == f"/api/models/{rid}": return 200, json.dumps({"sha": sha, "id": rid}).encode(), {}
            if path == f"/{rid}/raw/{sha}/README.md": return (200, self.cards[rid], {}) if self.cards.get(rid) else (404, b"no readme", {})
            if path == f"/api/models/{rid}/tree/{sha}?recursive=true": return 200, b"[]", {}
            for (frid, fpath), body in self.files.items():
                if frid == rid and path in (f"/{rid}/resolve/{sha}/{fpath}", f"/{rid}/resolve/main/{fpath}"):
                    return 200, body, {"Content-Type": "image/png"}
        return super().respond(path)


def tree_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).rglob("*") if p.is_file()}


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.hub = Hub(); self.addCleanup(self.hub.close)
        self.cdn = Server(); self.addCleanup(self.cdn.close)
        self.cdn.routes = {"/chart.jpg": (200, JPG, {"Content-Type": "image/jpeg"}), "/page.html": (200, b"<html>login</html>", {"Content-Type": "text/html"}),
                           "/big.png": (200, BIG, {"Content-Type": "image/png"})}
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "models" / SLUG
        (self.model / "cache/cards").mkdir(parents=True); (self.model / "snapshots").mkdir()
        (self.model / "model.json").write_text(json.dumps({"model": MODEL, "slug": SLUG, "name": "Test", "others": []}))
        (self.model / "sets.py").write_text("SETS = []\nMETRICS = {}\n")
        (self.root / "watchlist.json").write_text(json.dumps({"models": [{"model": MODEL, "slug": SLUG, "status": "live"}]}))
        self.hub.cards = {MODEL: b"# base\n",
                          "test/one": b"# One\n![kld chart](images/kld.png)\n| Quant | Mean KLD |\n|---|---|\n| Q4 | 0.1250 |\n<img src=\"./ppl.png\" alt=\"ppl chart\">\n",
                          "test/two": f"# Two\n![results]({self.cdn.base}/chart.jpg)\n![badge](https://img.shields.io/badge/x-y-green)\n".encode(),
                          "test/three": b"# Three\nno images\n"}
        self.pin_cards()

    def pin_cards(self):
        """A snapshot index like the connector path writes it: card files relative to the model root, sha main or pinned."""
        repos = {}
        for rid, sha in (("test/one", ONE), ("test/two", TWO), ("test/three", THREE)):
            card = self.hub.cards[rid]
            file = f"cache/cards/{rid.replace('/', '__')}.md"
            (self.model / file).write_bytes(card)
            repos[rid] = {"file": file, "bytes": len(card), "sha256": hashlib.sha256(card).hexdigest(), "sha": sha, "via": "hf-connector"}
        (self.model / "snapshots/index.json").write_text(json.dumps({"model": MODEL, "repos": repos}, indent=1))

    def run_images(self, *extra, code=0):
        p = subprocess.run([sys.executable, str(IMAGES), "--root", str(self.root), "--models", SLUG, "--workers", "2", "--base", self.hub.base, *extra],
                           capture_output=True, text=True, env={**os.environ, "HF_TOKEN": "synthetic-token"})
        self.assertNotIn("synthetic-token", p.stdout + p.stderr)
        self.assertNotIn("login", p.stdout + p.stderr, "no response body reaches the output")
        self.assertEqual(p.returncode, code, p.stdout + p.stderr)
        lines = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1); return lines[0]

    def ledger(self):
        return json.loads((self.model / "cache/images/index.json").read_text())

    def audit(self):
        return json.loads((self.model / "audit/images.json").read_text())

    def url(self, rid, sha, path):
        return f"{self.hub.base}/{rid}/resolve/{sha}/{path}"

    def test_first_run_pins_every_image_and_a_second_run_requests_nothing(self):
        t0 = time.time(); line = self.run_images(); elapsed = time.time() - t0
        self.assertEqual((line["slug"], line["cards"], line["referenced"], line["pinned"], line["new"], line["unresolved"]), (SLUG, 2, 3, 3, 3, 0))
        self.assertIsInstance(line["seconds"], float)
        ledger = self.ledger()
        self.assertEqual(ledger["model"], MODEL)
        expected = {self.url("test/one", ONE, "images/kld.png"): PNG, self.url("test/one", ONE, "ppl.png"): PNG2, f"{self.cdn.base}/chart.jpg": JPG}
        self.assertEqual(sorted(ledger["images"]), sorted(expected))
        for url, body in expected.items():
            rec = ledger["images"][url]
            self.assertEqual((rec["sha256"], rec["bytes"]), (hashlib.sha256(body).hexdigest(), len(body)))
            self.assertEqual(rec["file"], f"cache/images/{rec['sha256']}.{'jpg' if body is JPG else 'png'}")
            self.assertFalse(rec["file"].startswith("/"), "no local path in a public repo")
            self.assertEqual((self.model / rec["file"]).read_bytes(), body, "bytes stored exactly as received")
            self.assertRegex(rec["fetched_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(ledger["images"][self.url("test/one", ONE, "ppl.png")]["repos"], ["test/one"])
        self.assertEqual(ledger["images"][self.url("test/one", ONE, "ppl.png")]["L"], {"test/one": [6]})
        self.assertEqual(ledger["images"][f"{self.cdn.base}/chart.jpg"]["content_type"], "image/jpeg")
        audit = self.audit()
        self.assertEqual((audit["cards"], audit["references"], audit["referenced"], audit["pinned"]), (2, 3, 3, 3))
        self.assertEqual((audit["unresolved"], audit["gone"], audit["refused"]), ([], [], []))
        self.assertEqual((line["gone"], line["refused"]), (0, 0))
        self.assertFalse(self.hub.paths("shields"), "badges are never requested")
        self.assertEqual(sorted(self.hub.paths("resolve")), sorted([f"/test/one/resolve/{ONE}/images/kld.png", f"/test/one/resolve/{ONE}/ppl.png"]))
        self.assertTrue(all(auth == "Bearer synthetic-token" for _, auth in self.hub.requests), "the token goes to the Hub origin")
        self.assertEqual([auth for _, auth in self.cdn.requests], [None], "the token never leaves the Hub origin")
        self.assertLess(elapsed, 20)
        before = tree_hashes(self.root); self.hub.requests.clear(); self.cdn.requests.clear()
        line = self.run_images()
        self.assertEqual((line["pinned"], line["new"], line["unresolved"]), (3, 0, 0))
        self.assertEqual(self.hub.requests + self.cdn.requests, [], "a pinned image is never requested again")
        self.assertEqual(tree_hashes(self.root), before, "an unchanged model must change no file")

    def test_503_then_success_is_retried_and_pinned(self):
        self.hub.flaky[f"/test/one/resolve/{ONE}/images/kld.png"] = 1
        line = self.run_images()
        self.assertEqual((line["pinned"], line["unresolved"]), (3, 0))
        self.assertEqual(self.hub.paths("images/kld.png"), [f"/test/one/resolve/{ONE}/images/kld.png"] * 2)
        self.assertEqual((self.model / self.ledger()["images"][self.url("test/one", ONE, "images/kld.png")]["file"]).read_bytes(), PNG)

    def test_404_is_gone_with_evidence_and_strict_passes(self):
        del self.hub.files[("test/one", "ppl.png")]
        line = self.run_images("--strict", code=0)   # a definitive answer is not a failure of this run
        self.assertEqual((line["referenced"], line["pinned"], line["unresolved"], line["gone"], line["refused"]), (3, 2, 0, 1, 0))
        audit = self.audit()
        self.assertEqual(audit["gone"], [{"url": self.url("test/one", ONE, "ppl.png"), "repo": "test/one", "L": 6, "references": 1,
                                          "status": 404, "attempts": 1, "reason": "HTTP 404"}])
        self.assertEqual((audit["unresolved"], audit["refused"]), ([], []))
        before = tree_hashes(self.root)
        line = self.run_images("--strict", code=0)
        self.assertEqual((line["gone"], line["new"]), (1, 0))
        self.assertEqual(tree_hashes(self.root), before, "a persistent 404 changes no file between runs")
        self.assertFalse(list((self.model / "cache/images").glob("*.tmp")))
        self.cdn.routes["/chart.jpg"] = (403, b"gated", {})   # a gated repository answers 403: gone too, once pinned files are dropped
        for p in (self.model / "cache/images").glob("*"): p.unlink()
        line = self.run_images("--strict", code=0)
        self.assertEqual((line["pinned"], line["gone"], line["unresolved"]), (1, 2, 0))
        self.assertEqual([(g["status"], g["reason"]) for g in self.audit()["gone"]], [(403, "HTTP 403"), (404, "HTTP 404")])

    def test_503_on_every_attempt_is_unresolved_and_strict_fails_loudly(self):
        self.cdn.routes["/chart.jpg"] = (503, b"down", {})
        line = self.run_images("--strict", code=1)
        self.assertEqual((line["referenced"], line["pinned"], line["unresolved"], line["gone"], line["refused"]), (3, 2, 1, 0, 0))
        audit = self.audit()
        self.assertEqual(audit["unresolved"], [{"url": f"{self.cdn.base}/chart.jpg", "repo": "test/two", "L": 2, "references": 1,
                                                "status": None, "attempts": 3, "reason": "gave up after 3 tries"}])
        self.assertEqual((audit["gone"], audit["refused"]), ([], []))
        self.assertEqual(self.cdn.paths("chart.jpg"), ["/chart.jpg"] * 3, "every attempt was made before giving up")
        before = tree_hashes(self.root)
        line = self.run_images()   # without --strict the run reports and exits zero
        self.assertEqual(line["unresolved"], 1)
        self.assertEqual(tree_hashes(self.root), before, "a persistent outage changes no file between runs")
        self.assertFalse(list((self.model / "cache/images").glob("*.tmp")))

    def test_non_image_and_oversize_files_are_refused_with_evidence(self):
        self.hub.cards["test/two"] = f"# Two\n![page]({self.cdn.base}/page.html)\n![big]({self.cdn.base}/big.png)\n".encode()
        self.pin_cards()
        line = self.run_images("--strict", "--max-bytes", "100", code=0)   # a policy refusal is not a failure of this run
        self.assertEqual((line["referenced"], line["pinned"], line["unresolved"], line["gone"], line["refused"]), (4, 2, 0, 0, 2))
        audit = self.audit()
        self.assertEqual((audit["unresolved"], audit["gone"]), ([], []))
        bad = {u["url"]: u for u in audit["refused"]}
        self.assertEqual(bad[f"{self.cdn.base}/page.html"]["reason"], "not an image: content type text/html")
        self.assertEqual(bad[f"{self.cdn.base}/big.png"]["reason"], f"larger than --max-bytes: {len(BIG)} > 100 bytes")
        self.assertEqual((bad[f"{self.cdn.base}/page.html"]["bytes"], bad[f"{self.cdn.base}/page.html"]["content_type"]), (len(b"<html>login</html>"), "text/html"))
        self.assertEqual((bad[f"{self.cdn.base}/big.png"]["bytes"], bad[f"{self.cdn.base}/big.png"]["content_type"]), (len(BIG), "image/png"))
        self.assertTrue(all(u["status"] == 200 and u["attempts"] == 1 and u["repo"] == "test/two" for u in bad.values()))
        stored = {p.read_bytes() for p in (self.model / "cache/images").glob("*") if p.suffix != ".json"}
        self.assertEqual(stored, {PNG, PNG2}, "a refused file is never stored")
        before = tree_hashes(self.root)
        self.run_images("--strict", "--max-bytes", "100", code=0)
        self.assertEqual(tree_hashes(self.root), before, "a persistent refusal changes no file between runs")

    def test_rate_limit_is_recorded_without_spinning(self):
        del self.hub.files[("test/one", "ppl.png")]
        self.hub.routes[f"/test/one/resolve/{ONE}/ppl.png"] = (429, b"slow down", {"Retry-After": "120"})
        t0 = time.time(); line = self.run_images("--workers", "1"); elapsed = time.time() - t0
        self.assertEqual(line["rate_limited"]["retry_after"], "120")
        self.assertGreaterEqual(line["unresolved"], 1); self.assertLess(elapsed, 20, "Retry-After is recorded, not slept on")
        self.assertEqual((line["gone"], line["refused"]), (0, 0), "a 429 is retryable: it stays unresolved")
        audit = self.audit()
        self.assertEqual(audit["rate_limited"]["retry_after"], "120")
        self.assertEqual({(u["status"], u["reason"]) for u in audit["unresolved"] if u["url"].endswith("ppl.png")},
                         {(429, "HTTP 429: the batch stopped; Retry-After recorded, resume a later run")})
        self.assertTrue(all(u["status"] in (429, None) for u in audit["unresolved"]), "the URLs not requested after the 429 are unresolved too")
        self.assertEqual(self.hub.paths("ppl.png"), [f"/test/one/resolve/{ONE}/ppl.png"], "one request, no retry on 429")
        self.run_images("--strict", "--workers", "1", code=1)   # retryable, so a strict run fails until a later run gets through

    def test_data_uri_is_decoded_in_place_and_a_dropped_image_is_pruned(self):
        inline = base64.b64encode(PNG2).decode()
        self.hub.cards["test/three"] = f"# Three\n![inline](data:image/png;base64,{inline})\n".encode()
        self.pin_cards()
        line = self.run_images()
        self.assertEqual((line["cards"], line["referenced"], line["pinned"]), (3, 4, 4))
        self.assertFalse(self.hub.paths("data:"))
        self.hub.cards["test/one"] = b"# One\n![kld chart](images/kld.png)\n| Quant | Mean KLD |\n|---|---|\n| Q4 | 0.1250 |\n"   # ppl.png gone
        self.hub.cards["test/three"] = b"# Three\n"
        self.pin_cards()
        line = self.run_images()
        self.assertEqual((line["referenced"], line["pinned"], line["new"]), (2, 2, 0))
        self.assertEqual(sorted(self.ledger()["images"]), sorted([self.url("test/one", ONE, "images/kld.png"), f"{self.cdn.base}/chart.jpg"]))
        self.assertEqual({p.read_bytes() for p in (self.model / "cache/images").glob("*") if p.suffix != ".json"}, {PNG, JPG})

    def test_mapping_bundle_carries_the_pinned_images_and_numbers_still_need_text(self):
        self.run_images()
        index = json.loads((self.model / "snapshots/index.json").read_text())
        meta = {**index["repos"]["test/one"], "repo": "test/one"}
        raw = (self.model / meta["file"]).read_bytes()
        images = qcl_images.pinned_images(str(self.model), "test/one", meta, raw, prefix=f"models/{SLUG}/", base=self.hub.base)
        self.assertEqual([(i["L"], i["alt"], i["bytes"], i["content_type"]) for i in images], [(2, "kld chart", len(PNG), "image/png"), (6, "", len(PNG2), "image/png")])   # an <img> tag carries no alt
        for i in images:
            self.assertRegex(i["file"], rf"^models/{SLUG}/cache/images/[0-9a-f]{{64}}\.png$")
            self.assertEqual(hashlib.sha256((self.root / i["file"]).read_bytes()).hexdigest(), i["sha256"])
        self.assertEqual(qcl_images.pinned_images(str(self.model), "test/three", {**index["repos"]["test/three"]}, b"# Three\nno images\n", base=self.hub.base), [])
        candidates = {"c1": {"L": 5, "raw": "0.1250", "header": "Mean KLD", "label": "Q4"}}
        task = {"repo": "test/one", "source_sha256": meta["sha256"], "candidates_sha256": M.digest(candidates), "candidates": candidates, "prompt": "p"}
        bundle = M.prepare_mapping(MODEL, meta, raw, task, "A", "v1", images)
        self.assertEqual([i["L"] for i in bundle["images"]], [2, 6])
        self.assertIn("context only", bundle["instructions"])
        self.assertEqual(M.prepare_mapping(MODEL, meta, raw, task, "A", "v1")["images"], [], "a card without pinned images carries an empty list")
        self.assertNotEqual(bundle["fingerprint"], M.prepare_mapping(MODEL, meta, raw, task, "A", "v1")["fingerprint"])
        head = {"repo": "test/one", "source_sha256": meta["sha256"], "candidates_sha256": task["candidates_sha256"]}
        ok = {**head, "entries": [{"role": "own", "name": "Q4", "metrics": {"mean_kld": "c1"}}]}
        self.assertEqual(M.validate_mapping(bundle, ok)["entries"][0]["metrics"], {"mean_kld": "c1"})
        none = {**head, "entries": [], "no_quality_claims": True, "no_claims_reason": "Perplexity appears only in the chart at L6."}
        self.assertTrue(M.validate_mapping(bundle, none)["no_quality_claims"])
        for bad in ({**head, "entries": [{"role": "own", "name": "Q4", "metrics": {"ppl": "c99"}}]},
                    {**head, "entries": [{"role": "own", "name": "Q4", "metrics": {"mean_kld": "c1"}}], "prose": [{"entry": "Q4", "key": "ppl", "L": 6, "raw": "4.5"}]},
                    {**head, "entries": [{"role": "own", "name": "Q4", "metrics": {"mean_kld": "c1"}, "image": 2}]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                M.validate_mapping(bundle, bad)
        for broken in ([{**images[0], "file": "/etc/passwd"}], [{**images[0], "L": 99}], [{**images[0], "sha256": "x"}], [{**images[0], "extra": 1}]):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                M.prepare_mapping(MODEL, meta, raw, task, "A", "v1", broken)

    def test_collect_runs_the_image_step_after_the_cards(self):
        (self.model / "snapshots/index.json").unlink()
        for p in (self.model / "cache/cards").glob("*"): p.unlink()
        def collect(*extra):
            p = subprocess.run([sys.executable, str(COLLECT), "--root", str(self.root), "--all-live", "--workers", "2", "--base", self.hub.base, *extra],
                               capture_output=True, text=True, env={**os.environ, "HF_TOKEN": "synthetic-token"})
            self.assertNotIn("synthetic-token", p.stdout + p.stderr)
            return p.returncode, [json.loads(l) for l in p.stdout.splitlines() if l.strip()], p.stderr
        code, lines, err = collect("--strict-images")
        self.assertEqual(code, 0, err)
        rec = lines[0]
        self.assertEqual((rec["new"], rec["images"]["cards"], rec["images"]["referenced"], rec["images"]["pinned"], rec["images"]["new"], rec["images"]["unresolved"]), (3, 2, 3, 3, 3, 0))
        self.assertEqual(sorted(self.ledger()["images"]), sorted([self.url("test/one", ONE, "images/kld.png"), self.url("test/one", ONE, "ppl.png"), f"{self.cdn.base}/chart.jpg"]))
        path = self.root / "lines.jsonl"; path.write_text(json.dumps(rec))
        self.assertEqual(subprocess.run([sys.executable, str(COLLECT), "--summary", str(path)], capture_output=True, text=True).stdout.strip(),
                         f"{SLUG} +3 img +3; images 3/3 pinned")
        before = tree_hashes(self.root); self.hub.requests.clear(); self.cdn.requests.clear()
        code, lines, _ = collect("--strict-images")
        self.assertEqual(code, 0)
        self.assertEqual(tree_hashes(self.root), before, "an hourly run with no change leaves every file byte-identical")
        self.assertFalse(self.hub.paths("resolve") + self.cdn.paths())
        path.write_text(json.dumps(lines[0]))
        self.assertEqual(subprocess.run([sys.executable, str(COLLECT), "--summary", str(path)], capture_output=True, text=True).stdout.strip(), "no card changed; images 3/3 pinned")
        self.hub.sha["test/two"] = "4" * 40; self.hub.cards["test/two"] = f"# Two\n![results]({self.cdn.base}/chart2.jpg)\n".encode()
        self.cdn.routes["/chart2.jpg"] = (503, b"down", {})   # the CDN is down for chart2 on every attempt
        code, lines, _ = collect("--strict-images")
        self.assertEqual(code, 1, "an unresolved (retryable) image fails a strict run loudly")
        self.assertEqual((lines[0]["changed"], lines[0]["images"]["unresolved"], lines[0]["images"]["gone"], lines[0]["images"]["pinned"]), (1, 1, 0, 2))
        path.write_text(json.dumps(lines[0]))
        self.assertEqual(subprocess.run([sys.executable, str(COLLECT), "--summary", str(path)], capture_output=True, text=True).stdout.strip(),
                         f"{SLUG} ~1 1 img unresolved; images 2/3 pinned")
        code, lines, _ = collect()
        self.assertEqual(code, 0, "without --strict-images the run only reports")
        del self.cdn.routes["/chart2.jpg"]   # now the CDN answers 404: the file is gone, which is evidence, not a failure
        code, lines, _ = collect("--strict-images")
        self.assertEqual(code, 0, "a gone image does not fail a strict run")
        self.assertEqual((lines[0]["images"]["unresolved"], lines[0]["images"]["gone"], lines[0]["images"]["pinned"]), (0, 1, 2))
        self.assertEqual(self.audit()["gone"][0]["status"], 404)
        path.write_text(json.dumps(lines[0]))
        self.assertEqual(subprocess.run([sys.executable, str(COLLECT), "--summary", str(path)], capture_output=True, text=True).stdout.strip(),
                         f"{SLUG} 1 img gone; images 2/3 pinned")
        code, lines, _ = collect("--no-images")
        self.assertEqual(code, 0); self.assertNotIn("images", lines[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
