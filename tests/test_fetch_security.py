"""Loopback integration tests ensure the fetcher never forwards credentials to another origin."""
import importlib.util
import concurrent.futures as cf
import contextlib
import datetime as dt
import email.utils
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qff_fetch_security", ROOT / "pipeline/qcl_fetch.py")
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class FetchSecurityTests(unittest.TestCase):
    def setUp(self):
        self.received = []
        received = self.received
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append((self.server.server_port, self.path, self.headers.get("Authorization")))
                if self.path in self.server.responses:
                    status, body, headers = self.server.responses[self.path]
                    self.send_response(status)
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path in self.server.redirects:
                    self.send_response(302)
                    self.send_header("Location", self.server.redirects[self.path])
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"public card")
            def log_message(self, *args):
                pass
        self.servers = []
        for _ in range(2):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.redirects = {}
            server.responses = {}
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
            thread.start()
            self.servers.append((server, thread))
        self.addCleanup(self.close_servers)
        self.first, self.second = (f"http://127.0.0.1:{server.server_port}" for server, _ in self.servers)

    def close_servers(self):
        for server, thread in self.servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_cross_origin_redirect_fetches_public_bytes_without_fake_token(self):
        self.servers[0][0].redirects["/card"] = self.second + "/cdn-card"
        status, body, _ = fetch.get(self.first + "/card", "synthetic-token", tries=1)
        self.assertEqual((status, body), (200, b"public card"))
        self.assertEqual([record[2] for record in self.received], ["Bearer synthetic-token", None])

    def test_same_origin_redirect_retains_authentication(self):
        self.servers[0][0].redirects["/card"] = self.first + "/canonical-card"
        self.assertEqual(fetch.get(self.first + "/card", "synthetic-token", tries=1)[0], 200)
        self.assertEqual([record[2] for record in self.received], ["Bearer synthetic-token"] * 2)

    def test_cross_origin_chain_never_reattaches_token(self):
        self.servers[0][0].redirects["/card"] = self.second + "/cdn-card"
        self.servers[1][0].redirects["/cdn-card"] = self.first + "/public-return"
        fetch.get(self.first + "/card", "synthetic-token", tries=1)
        self.assertEqual([record[2] for record in self.received], ["Bearer synthetic-token", None, None])

    def test_https_downgrade_is_refused_before_network_io(self):
        request = urllib.request.Request("https://huggingface.co/card", headers={"Authorization": "Bearer synthetic-token"})
        with self.assertRaisesRegex(RuntimeError, "HTTPS-to-HTTP"):
            fetch.SafeRedirectHandler().redirect_request(request, None, 302, "Found", {}, "http://huggingface.co/card")
        self.assertEqual(self.received, [])

    def test_pagination_requires_same_scheme_host_and_port(self):
        source = "https://huggingface.co/api/models?page=1"
        self.assertEqual(fetch.next_page(source, {"Link": '<?page=2>; rel="next"'}),
                         "https://huggingface.co/api/models?page=2")
        for target in ("http://huggingface.co/api/models", "https://other.example/api/models",
                       "https://huggingface.co:444/api/models", "https://user:pass@huggingface.co/api/models"):
            with self.subTest(target=target), self.assertRaises(RuntimeError):
                fetch.next_page(source, {"Link": f'<{target}>; rel="next"'})

    def test_explicit_default_port_is_same_origin(self):
        self.assertEqual(fetch.origin("https://huggingface.co/card"), fetch.origin("https://huggingface.co:443/card"))

    def test_first_429_stops_without_sleeping_or_retrying(self):
        self.servers[0][0].responses["/limited"] = (429, b"slow down", {"Retry-After": "3600"})
        stopped = threading.Event()
        with patch.object(fetch.time, "sleep", side_effect=AssertionError("must not sleep")):
            with self.assertRaises(fetch.RateLimited) as raised:
                fetch.get(self.first + "/limited", "synthetic-token", stop=stopped)
        self.assertTrue(stopped.is_set())
        self.assertEqual(raised.exception.record["retry_after"], "3600")
        self.assertEqual(len(self.received), 1)
        with self.assertRaises(fetch.FetchStopped):
            fetch.get(self.first + "/another", "synthetic-token", stop=stopped)
        self.assertEqual(len(self.received), 1, "stopped worker started another request")

    def test_queued_repository_workers_make_no_requests_after_429(self):
        self.servers[0][0].responses["/test/a/raw/abc/README.md"] = (429, b"slow down", {"Retry-After": "20"})
        stopped = threading.Event()
        with tempfile.TemporaryDirectory() as directory, cf.ThreadPoolExecutor(1) as executor:
            Path(directory, "cards").mkdir()
            Path(directory, "trees").mkdir()
            futures = [executor.submit(fetch.fetch_repo, self.first, {"id": "test/" + name, "sha": "abc"},
                        "synthetic-token", directory, None, stopped) for name in ("a", "b", "c")]
            with self.assertRaises(fetch.RateLimited):
                futures[0].result()
            for future in futures[1:]:
                with self.assertRaises(fetch.FetchStopped):
                    future.result()
        self.assertEqual(len(self.received), 1)

    def test_retry_after_date_and_numeric_formats_and_bounded_5xx_wait(self):
        stamp = 1_700_000_000
        future = email.utils.format_datetime(dt.datetime.fromtimestamp(stamp + 90, tz=dt.timezone.utc), usegmt=True)
        self.assertEqual(fetch.retry_after_seconds(future, now=stamp), 90)
        self.assertEqual(fetch.retry_after_seconds("25"), 25)
        self.assertEqual(fetch.retry_after_seconds("-10"), 0)
        for value in ("not a date", "nan", "inf", ""):
            self.assertIsNone(fetch.retry_after_seconds(value))
        for header, delay in (("86400000", 60), ("not a date", 1)):
            with self.subTest(header=header):
                self.received.clear()
                self.servers[0][0].responses["/unavailable"] = (503, b"unavailable", {"Retry-After": header})
                with patch.object(fetch.time, "sleep") as sleep, self.assertRaisesRegex(RuntimeError, "gave up"):
                    fetch.get(self.first + "/unavailable", None, tries=2)
                sleep.assert_called_once_with(delay)
                self.assertEqual(len(self.received), 2)

    def test_successful_checkpoints_survive_a_rate_limit_and_resume(self):
        server = self.servers[0][0]
        catalogue = [{"id": "test/first", "sha": "abc"}, {"id": "test/second", "sha": "abc"}]
        catalogue_path = "/api/models?" + urllib.parse.urlencode({"filter": "base_model:quantized:test/base", "limit": 1000, "full": "true"}) + "&expand[]=sha&expand[]=createdAt&expand[]=lastModified&expand[]=downloads&expand[]=tags"
        server.responses[catalogue_path] = (200, json.dumps(catalogue).encode(), {})
        server.responses["/api/models/test/base"] = (200, b'{"sha":"abc"}', {})
        for name in ("first", "second"):
            server.responses[f"/api/models/test/{name}/tree/abc?recursive=true"] = (200, b"[]", {})
        limited = "/test/second/raw/abc/README.md"
        server.responses[limited] = (429, b"slow down", {"Retry-After": "0"})
        with tempfile.TemporaryDirectory() as directory:
            argv = ["qcl_fetch.py", "test/base", "--out", directory, "--workers", "1", "--base", self.first]
            with patch.object(fetch.sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, "HTTP 429"):
                    fetch.main()
                index = json.loads(Path(directory, "index.json").read_text())
                self.assertEqual(list(index["repos"]), ["test/first"])
                self.assertEqual(json.loads(Path(directory, "rate_limit.json").read_text())["retry_after"], "0")
                server.responses.pop(limited)
                fetch.main()
                resumed = json.loads(Path(directory, "index.json").read_text())
                self.assertEqual(sorted(resumed["repos"]), ["test/first", "test/second"])
                self.assertFalse(Path(directory, "rate_limit.json").exists())
        self.assertEqual(sum(path == "/test/first/raw/abc/README.md" for _, path, _ in self.received), 1)

    def test_reused_ledger_identity_rejects_without_requests_or_any_writes(self):
        fixtures = [
            {"model": "old/model", "repos": {"old/quant": {"sha": "a" * 40, "bytes": 0, "tree_complete": True}}},
            {"repos": {}}, {"model": None, "repos": {}}, {"model": "", "repos": {}},
            {"model": "invalid", "repos": {}}, {"model": [], "repos": {}}, [], "broken JSON",
        ]
        def state(path):
            return {str(p.relative_to(path)): ("directory" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in path.rglob("*")}
        for fixture in fixtures:
            with self.subTest(fixture=fixture), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                root.joinpath("index.json").write_text(fixture if isinstance(fixture, str) else json.dumps(fixture))
                root.joinpath("catalogue.json").write_text('[{"id":"old/quant"}]')
                before = state(root)
                args = SimpleNamespace(base=self.first, model="new/model", out=directory, include_repo=[], workers=1)
                with patch.object(fetch, "catalogue") as catalogue, self.assertRaises(ValueError):
                    fetch.fetch_model(args, "synthetic-token")
                catalogue.assert_not_called()
                self.assertEqual(state(root), before, "identity rejection changed the source ledger")
                argv = ["qcl_fetch.py", "new/model", "--out", directory, "--base", self.first]
                with patch.object(fetch.sys, "argv", argv), self.assertRaisesRegex(SystemExit, "REFUSING TO FETCH"):
                    fetch.main()
                self.assertEqual(state(root), before, "CLI identity rejection created or changed files")
        self.assertEqual(self.received, [])

    def test_reused_ledger_rejects_conflicting_base_card_or_invalid_repos(self):
        for extra in ({"base": {"repo": "other/model"}}, {"base": None}, {"base": {}}, {"repos": []}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                path = Path(directory, "index.json")
                path.write_text(json.dumps({"model": "new/model", "repos": {}, **extra}))
                before = path.read_bytes()
                args = SimpleNamespace(base=self.first, model="new/model", out=directory, include_repo=[], workers=1)
                with patch.object(fetch, "catalogue") as catalogue, self.assertRaises(ValueError):
                    fetch.fetch_model(args, None)
                catalogue.assert_not_called()
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["index.json"])

    def test_existing_model_config_binds_the_destination_before_creating_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "model.json"
            config.write_text(json.dumps({"slug": "old", "model": "old/model"}))
            before = config.read_bytes()
            args = SimpleNamespace(base=self.first, model="new/model", out=str(root / "cache"), include_repo=[], workers=1)
            with patch.object(fetch, "catalogue") as catalogue, self.assertRaisesRegex(ValueError, "model config identity"):
                fetch.fetch_model(args, None)
            catalogue.assert_not_called()
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["model.json"])


if __name__ == "__main__":
    unittest.main()
