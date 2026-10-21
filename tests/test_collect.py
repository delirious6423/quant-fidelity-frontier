"""Connector-free collection: pipeline/qcl_collect.py against a loopback Hugging Face API, and qcl_next.py --collected.

A tiny fixture (one base model, three quant repos) on a temporary model root. Nothing here reaches the network, the
repository's models/ or git."""
import ast
import contextlib
import hashlib
import http.server
import io
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
COLLECT = CODE / "pipeline/qcl_collect.py"
MODEL = "test/base"


def catalogue_path(model):
    return "/api/models?" + urllib.parse.urlencode({"filter": f"base_model:quantized:{model}", "limit": 1000, "full": "true"}) \
        + "&expand[]=sha&expand[]=createdAt&expand[]=lastModified&expand[]=downloads&expand[]=tags"


class FakeHub:
    """In-memory Hub: a catalogue, one card and one tree per repo at a pinned sha; paths may be rate-limited or denied."""
    def __init__(self):
        self.sha = {MODEL: "b" * 40, "test/one": "1" * 40, "test/two": "2" * 40, "test/three": "3" * 40}
        self.cards = {MODEL: b"# base\n", "test/one": b"| Quant | Mean KLD |\n|---|---|\n| Q4 | 0.125 |\n",
                      "test/two": b"| Quant | PPL |\n|---|---|\n| Q8 | 5.5 |\n", "test/three": b""}
        self.bases = {MODEL: ["test/one", "test/two", "test/three"]}   # base model -> its quant repos
        self.modified = {rid: "2026-10-20T10:00:00.000Z" for rid in self.sha}
        self.downloads, self.likes = 10, 3
        self.limited, self.denied, self.requests = set(), set(), []
        self.limited_once = set()   # paths that answer 429 (Retry-After 1) to their first request only
        hub = self
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hub.requests.append((self.path, self.headers.get("Authorization")))
                status, body, headers = hub.respond(self.path)
                self.send_response(status)
                for k, v in headers.items(): self.send_header(k, v)
                self.end_headers(); self.wfile.write(body)
            def log_message(self, *a): pass
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=1)

    def catalogue(self, base):
        return [{"id": rid, "sha": self.sha[rid], "lastModified": self.modified[rid], "createdAt": "2026-10-01T00:00:00.000Z",
                 "downloads": self.downloads, "likes": self.likes, "tags": ["gguf"]} for rid in self.bases[base]]

    def respond(self, path):
        if path in self.limited: return 429, b"slow down", {"Retry-After": "120"}
        if path in self.limited_once:
            self.limited_once.discard(path); return 429, b"slow down", {"Retry-After": "1"}
        if path in self.denied: return 403, b"private response body", {}
        for base in self.bases:
            if path == catalogue_path(base): return 200, json.dumps(self.catalogue(base)).encode(), {}
        for rid, sha in self.sha.items():
            if path == f"/api/models/{rid}": return 200, json.dumps({"sha": sha, "id": rid}).encode(), {}
            if path == f"/{rid}/raw/{sha}/README.md":
                return (200, self.cards[rid], {}) if self.cards[rid] else (404, b"no readme", {})
            if path == f"/api/models/{rid}/tree/{sha}?recursive=true":
                return 200, json.dumps([{"type": "file", "path": "model.gguf", "size": 1234}]).encode(), {}
        return 404, b"not found", {}


def tree_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).rglob("*") if p.is_file()}


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.hub = FakeHub(); self.addCleanup(self.hub.close)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "models/testmodel"
        self.model.mkdir(parents=True)
        (self.model / "model.json").write_text(json.dumps({"model": MODEL, "slug": "testmodel", "name": "Test", "others": []}))
        (self.model / "sets.py").write_text("SETS = []\nMETRICS = {}\n")
        (self.root / "watchlist.json").write_text(json.dumps({"models": [
            {"model": MODEL, "slug": "testmodel", "status": "live"}, {"model": "test/queued", "slug": "queued", "status": "queued"}]}))
        (self.root / "models/queued").mkdir(); (self.root / "models/queued/model.json").write_text(json.dumps({"model": "test/queued", "slug": "queued"}))

    def run_collect(self, *extra):
        p = subprocess.run([sys.executable, str(COLLECT), "--root", str(self.root), "--all-live", "--workers", "2", "--base", self.hub.base, *extra],
                           capture_output=True, text=True, env={**os.environ, "HF_TOKEN": "synthetic-token"})
        self.assertNotIn("synthetic-token", p.stdout + p.stderr)
        lines = [json.loads(l) for l in p.stdout.splitlines() if l.strip()]
        return p.returncode, lines, p.stderr

    def snapshot(self):
        return json.loads((self.model / "snapshots/index.json").read_text())

    def test_first_run_fetches_and_installs_then_unchanged_run_changes_no_byte(self):
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 1, "only the live model is collected")
        line = lines[0]
        self.assertEqual((line["slug"], line["catalogue"], line["new"], line["changed"], line["unchanged"], line["unresolved"]), ("testmodel", 3, 3, 0, 0, 0))
        self.assertIsInstance(line["seconds"], float)
        snap = self.snapshot()
        self.assertEqual(sorted(snap["repos"]), ["test/one", "test/three", "test/two"])
        for rid, rec in snap["repos"].items():
            self.assertEqual(rec["sha"], self.hub.sha[rid]); self.assertEqual(rec["via"], "hf-api")
            self.assertEqual(rec["sha256"], hashlib.sha256(self.hub.cards[rid]).hexdigest())
            self.assertFalse(rec["file"].startswith("/"), "no local path in a public repo")
            self.assertEqual((self.model / rec["file"]).read_bytes(), self.hub.cards[rid])
        self.assertTrue(snap["repos"]["test/three"]["no_readme"])
        self.assertEqual(snap["base"]["sha"], self.hub.sha[MODEL])
        self.assertTrue(all(auth == "Bearer synthetic-token" for _, auth in self.hub.requests), "the token goes to the requested origin only")
        before = tree_hashes(self.root); self.hub.requests.clear()
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 0)
        self.assertEqual(tree_hashes(self.root), before, "an unchanged model must change no file")
        self.assertEqual((lines[0]["new"], lines[0]["changed"], lines[0]["unchanged"]), (0, 0, 3))
        self.assertFalse([p for p, _ in self.hub.requests if "/raw/" in p and "_base" not in p and f"/{MODEL}/" not in p], "no quant card was requested again")

    def set_captured_at(self, hours_ago):
        path = self.model / "cache/index.json"; index = json.loads(path.read_text())
        index["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours_ago * 3600))
        path.write_text(json.dumps(index, indent=1))

    def test_metadata_only_churn_changes_no_file_until_a_day_has_passed(self):
        self.run_collect()
        self.assertEqual(self.snapshot()["repos"]["test/one"]["downloads"], 10)
        before = tree_hashes(self.root)
        self.hub.downloads, self.hub.likes = 11, 4   # the hourly Hub listing moves; no card or tree changed
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 0)
        self.assertEqual(tree_hashes(self.root), before, "downloads and likes alone must not change a file")
        self.assertEqual((lines[0]["new"], lines[0]["changed"], lines[0]["unchanged"], lines[0]["metadata"]), (0, 0, 3, "deferred"))
        self.set_captured_at(23); before = tree_hashes(self.root)
        self.run_collect()
        self.assertEqual(tree_hashes(self.root), before, "within a day the metadata is still deferred")
        self.set_captured_at(25); before = tree_hashes(self.root)
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 0); self.assertNotIn("metadata", lines[0])
        self.assertEqual((lines[0]["new"], lines[0]["changed"], lines[0]["unchanged"]), (0, 0, 3))
        after = tree_hashes(self.root)
        self.assertEqual({p for p in after if before.get(p) != after[p]},
                         {"models/testmodel/cache/index.json", "models/testmodel/cache/catalogue.json", "models/testmodel/snapshots/index.json"})
        snap = self.snapshot()
        self.assertEqual((snap["repos"]["test/one"]["downloads"], snap["repos"]["test/one"]["sha256"]), (11, hashlib.sha256(self.hub.cards["test/one"]).hexdigest()))
        self.assertEqual(json.loads((self.model / "cache/catalogue.json").read_text())[0]["downloads"], 11)
        before = tree_hashes(self.root)
        self.run_collect()
        self.assertEqual(tree_hashes(self.root), before, "the fresh capture is kept as is")

    def test_changed_card_is_refetched_and_index_updated(self):
        self.run_collect()
        before = tree_hashes(self.root)
        self.hub.sha["test/two"] = "4" * 40; self.hub.cards["test/two"] = b"| Quant | PPL |\n|---|---|\n| Q8 | 5.4 |\n"
        self.hub.modified["test/two"] = "2026-10-21T10:00:00.000Z"; self.hub.downloads = 12   # a real change on the same hour as metadata churn
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 0)
        self.assertEqual((lines[0]["new"], lines[0]["changed"], lines[0]["unchanged"]), (0, 1, 2))
        rec = self.snapshot()["repos"]["test/two"]
        self.assertEqual(rec["sha"], "4" * 40)
        self.assertEqual(rec["sha256"], hashlib.sha256(self.hub.cards["test/two"]).hexdigest())
        self.assertEqual(rec["lastModified"], "2026-10-21T10:00:00.000Z")
        after = tree_hashes(self.root)
        changed = {p for p in set(before) | set(after) if before.get(p) != after.get(p)}
        self.assertEqual(changed, {"models/testmodel/cache/index.json", "models/testmodel/cache/catalogue.json", "models/testmodel/snapshots/index.json",
                                   "models/testmodel/cache/cards/test__two@444444444444.md"})   # the tree is rewritten with the same bytes
        self.assertEqual(self.snapshot()["repos"]["test/one"]["sha256"], hashlib.sha256(self.hub.cards["test/one"]).hexdigest())
        self.assertEqual(self.snapshot()["repos"]["test/one"]["downloads"], 12, "a real change takes the fresh metadata with it")

    def test_rate_limit_is_reported_and_the_run_goes_on(self):
        self.run_collect()
        self.hub.sha["test/three"] = "5" * 40; self.hub.cards["test/three"] = b"new card\n"
        self.hub.limited.add(f"/test/three/raw/{'5' * 40}/README.md")
        (self.root / "models/zzz-other").mkdir()
        (self.root / "models/zzz-other/model.json").write_text(json.dumps({"model": "test/other", "slug": "zzz-other"}))
        (self.root / "models/zzz-other/sets.py").write_text("SETS = []\nMETRICS = {}\n")
        for rid, card in (("test/other", b"# other base\n"), ("other/q1", b"| Q | KLD |\n|---|---|\n| Q4 | 0.1 |\n")):
            self.hub.sha[rid] = "6" * 40; self.hub.cards[rid] = card; self.hub.modified[rid] = "2026-10-20T10:00:00.000Z"
        self.hub.bases["test/other"] = ["other/q1"]
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 0, "a rate limit never fails the run")
        self.assertEqual([l["slug"] for l in lines], ["testmodel", "zzz-other"])
        self.assertEqual(lines[0]["rate_limited"]["retry_after"], "120")
        self.assertEqual(lines[0]["unresolved"], 1)
        self.assertEqual(json.loads((self.model / "cache/rate_limit.json").read_text())["retry_after"], "120")
        self.assertEqual(self.snapshot()["repos"]["test/three"]["sha"], "3" * 40, "the old pin stays until the card is fetched")
        self.assertNotIn("rate_limited", lines[1]); self.assertEqual((lines[1]["catalogue"], lines[1]["new"]), (1, 1))
        self.hub.limited.clear()
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 0)
        self.assertEqual(lines[0]["changed"], 1); self.assertNotIn("rate_limited", lines[0])
        self.assertFalse((self.model / "cache/rate_limit.json").exists())
        self.assertEqual(self.snapshot()["repos"]["test/three"]["sha"], "5" * 40)

    def test_wait_429_resumes_the_model_within_its_budget(self):
        self.hub.limited_once.add(f"/test/two/raw/{'2' * 40}/README.md")
        code, lines, err = self.run_collect("--wait-429", "10")
        self.assertEqual(code, 0)
        self.assertNotIn("rate_limited", lines[0], "the model resumed after the Retry-After and completed")
        self.assertEqual((lines[0]["new"], lines[0]["unresolved"]), (3, 0), "counts add up across the first run and the resumed run")
        self.assertGreaterEqual(lines[0]["waited"], 2.0)
        self.assertIn("waiting 2s before resuming", err)
        self.assertEqual(sorted(self.snapshot()["repos"]), ["test/one", "test/three", "test/two"])
        self.assertFalse((self.model / "cache/rate_limit.json").exists())
        # A budget smaller than the Retry-After is not spent: the limit is reported as before and the model resumes next run.
        self.hub.sha["test/two"] = "7" * 40
        self.hub.limited_once.add(f"/test/two/raw/{'7' * 40}/README.md")
        code, lines, err = self.run_collect("--wait-429", "1")
        self.assertEqual(code, 0)
        self.assertEqual(lines[0]["rate_limited"]["retry_after"], "1"); self.assertNotIn("waited", lines[0])
        self.assertIn("wait budget of 1s is spent", err)

    def test_connector_era_ledger_is_replaced_by_an_http_ledger_bound_to_the_model(self):
        (self.model / "cache/cards").mkdir(parents=True)
        (self.model / "cache/cards/test__one.md").write_bytes(b"old connector card\n")
        (self.model / "cache/index.json").write_text(json.dumps({"repos": {"test/one": {
            "file": "cards/test__one.md", "bytes": 19, "sha256": hashlib.sha256(b"old connector card\n").hexdigest(),
            "size_checked": True, "fetched_at": "2026-10-17T19:15:41Z", "via": "hf-connector"}}}))
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 0, lines)
        self.assertEqual(lines[0]["ledger"], "connector-era ledger replaced")
        self.assertEqual((lines[0]["new"], lines[0]["unresolved"]), (3, 0), "every card is collected over HTTP at its pinned commit")
        ledger = json.loads((self.model / "cache/index.json").read_text())
        self.assertEqual(ledger["model"], MODEL); self.assertEqual(ledger["repos"]["test/one"]["sha"], "1" * 40)
        self.assertEqual(sorted(self.snapshot()["repos"]), ["test/one", "test/three", "test/two"])
        # The replacement happens once: the next run reuses the HTTP ledger and changes nothing.
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 0); self.assertNotIn("ledger", lines[0]); self.assertEqual(lines[0]["unchanged"], 3)

    def test_denied_card_is_counted_without_printing_the_response_body(self):
        self.hub.denied.add(f"/test/one/raw/{'1' * 40}/README.md")
        code, lines, err = self.run_collect()
        self.assertEqual(code, 0)
        self.assertEqual((lines[0]["failed"], lines[0]["unresolved"], lines[0]["new"]), (1, 1, 2))
        self.assertNotIn("private response body", json.dumps(lines) + err)
        self.assertEqual(sorted(self.snapshot()["repos"]), ["test/three", "test/two"])
        self.assertEqual(json.loads((self.model / "cache/not_fetched.json").read_text()), ["test/one"])

    def test_identity_mismatch_is_an_error_for_that_model_only(self):
        (self.model / "cache").mkdir()
        (self.model / "cache/index.json").write_text(json.dumps({"model": "other/model", "repos": {}}))
        code, lines, _ = self.run_collect()
        self.assertEqual(code, 1)
        self.assertIn("does not match", lines[0]["error"])
        self.assertFalse((self.model / "snapshots").exists())

    def test_summary_line(self):
        path = self.root / "lines.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in [
            {"slug": "a", "new": 2, "changed": 1, "unchanged": 5}, {"slug": "b", "new": 0, "changed": 0, "unchanged": 9},
            {"slug": "c", "new": 0, "changed": 0, "unchanged": 1, "rate_limited": {"retry_after": "9"}}, {"slug": "d", "failed": 2},
            {"slug": "e", "images": {"referenced": 5, "pinned": 2, "new": 1, "unresolved": 1, "gone": 1, "refused": 1}}]))
        p = subprocess.run([sys.executable, str(COLLECT), "--summary", str(path)], capture_output=True, text=True)
        self.assertEqual(p.stdout.strip(), "a +2 ~1; c rate-limited; d 2 failed; e img +1 1 img unresolved 1 img gone 1 img refused; images 2/5 pinned")
        path.write_text(json.dumps({"slug": "b", "new": 0, "changed": 0}))
        self.assertEqual(subprocess.run([sys.executable, str(COLLECT), "--summary", str(path)], capture_output=True, text=True).stdout.strip(), "no card changed")


class NewmodelPrecedenceTests(unittest.TestCase):
    """A direct record replaces an older connector shard for the same repo only when the Hub changed after the connector capture."""
    def merged(self, fetched_at, last_modified):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d, "models/m"); (root / "cache/cards").mkdir(parents=True); (root / "cache/trees").mkdir()
            (root / "model.json").write_text(json.dumps({"model": MODEL, "slug": "m"})); (root / "sets.py").write_text("SETS = []\nMETRICS = {}\n")
            old, new = b"old card\n", b"new card\n"
            (root / "cache/cards/test__one.md").write_bytes(old); (root / "cache/cards/test__one@111111111111.md").write_bytes(new)
            (root / "cache/trees/test__one.json").write_text("[]")
            (root / "cache/index_2026a.json").write_text(json.dumps({"repos": {"test/one": {
                "file": "cards/test__one.md", "bytes": len(old), "sha256": hashlib.sha256(old).hexdigest(), "fetched_at": fetched_at}}}))
            (root / "cache/index.json").write_text(json.dumps({"model": MODEL, "repos": {"test/one": {
                "sha": "1" * 40, "file": "models/m/cache/cards/test__one@111111111111.md", "sha256": hashlib.sha256(new).hexdigest(), "bytes": len(new),
                "readme_status": 200, "lastModified": last_modified, "tree_file": "models/m/cache/trees/test__one.json", "tree_complete": True}}}))
            p = subprocess.run([sys.executable, str(CODE / "pipeline/qcl_newmodel.py"), "models/m"], cwd=d, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            return json.loads((root / "snapshots/index.json").read_text())["repos"]["test/one"]

    def test_direct_fetch_wins_when_the_hub_changed_after_the_connector_capture(self):
        rec = self.merged("2026-10-20T14:41:57Z", "2026-10-20T15:00:00.000Z")
        self.assertEqual((rec["via"], rec["sha"], rec["file"]), ("hf-api", "1" * 40, "cache/cards/test__one@111111111111.md"))

    def test_connector_capture_after_the_last_hub_change_is_kept(self):
        rec = self.merged("2026-10-20T15:30:00Z", "2026-10-20T15:00:00.000Z")
        self.assertEqual((rec["via"], rec["sha"], rec["file"]), ("hf-connector", "main", "cache/cards/test__one.md"))


class NextCollectedTests(unittest.TestCase):
    """Replay the real start and a_fetched branches of qcl_next.py with inert stubs (as pipeline/runner_baseline.py does)."""
    def replay(self, args, prepare, environ=None):
        tree = ast.parse((CODE / "pipeline/qcl_next.py").read_text())
        want = {"s is None", "s.get('step') == 'start' and COLLECTED", "s.get('step') == 'start'", "step == 'a_fetched'"}
        body = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in ("hf", "agents", "say"): body.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("COLLECTED", "FETCH_PROMPT", "MAP_PROMPT") for t in node.targets): body.append(node)
            elif isinstance(node, ast.If) and ast.unparse(node.test) in want: body.append(node)
            elif isinstance(node, ast.Assign) and ast.unparse(node) == "step = s['step']": body.append(node)
        self.assertEqual(len(body), 11, "qcl_next.py changed; review this replay")
        calls, pushes = [], []
        def scan(arg):
            calls.append(arg); op = arg.split()[0]
            if op == "plan": return 0, {"delta": ["m1", "m2"], "ingest": None}
            if op == "prepare": return prepare
            if op == "search-ops": return 0, {"op_lists": [[{"cmd": "search", "args": ["hf://models"]}]]}
            return 0, {}
        env = {"args": args, "os": type("os", (), {"environ": environ or {}})(), "sys": sys, "json": json, "math": __import__("math"),
               "scan": scan, "sh": lambda *a, **k: None, "push": lambda msg, paths: pushes.append(msg), "save": lambda s_: None,
               "s": None, "PART": "A", "LOCKF": "scan.lock", "CODE": "/repo", "NEXT": "python3 pipeline/qcl_next.py --part A"}
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
            exec(compile(ast.Module(body=body, type_ignores=[]), "<qcl_next-replay>", "exec"), env)
        return out.getvalue(), env["s"], calls, pushes

    def test_collected_flag_skips_catalogue_calls_and_fetch_workers(self):
        for args, environ in ((["--part", "A", "--collected"], {}), (["--part", "A"], {"QCL_COLLECTED": "1"})):
            with self.subTest(args=args, environ=environ):
                out, s, calls, pushes = self.replay(args, (0, {"queues": 2, "todo": 1}), environ)
                self.assertNotIn('"do": "hf_fs"', out); self.assertNotIn("FETCH.md", out); self.assertNotIn("hf_fs", out)
                printed = json.loads(out.strip().splitlines()[-1])
                self.assertEqual(printed["do"], "agents")
                self.assertTrue(all("MAP.md" in p for p in printed["prompts"]))
                self.assertEqual(len(printed["prompts"]), 4, "one A and one B worker per model with queues")
                self.assertEqual(s["a_shards"], {"m1": [], "m2": []}); self.assertEqual(s["step"], "a_mapped")
                self.assertEqual([c for c in calls if c.startswith(("search-ops", "fetched"))], [])
                self.assertEqual(calls[:4], ["lock", "plan", "prepare m1", "prepare m2"])
                self.assertEqual(pushes, ["scan lock"])

    def test_collected_models_without_queues_are_staged_not_fetched(self):
        out, s, calls, _ = self.replay(["--part", "A", "--collected"], (2, {"queues": 0, "merge": "ok"}))
        self.assertEqual(out, "")
        self.assertEqual(s["a_queues"], {}); self.assertEqual(s["part_a"]["m1"]["merge"], "ok")
        self.assertEqual([c for c in calls if c.startswith("prepare")], ["prepare m1", "prepare m2"])

    def test_connector_path_is_unchanged_without_the_flag(self):
        out, s, calls, pushes = self.replay(["--part", "A"], (0, {"queues": 2}))
        self.assertEqual(json.loads(out.strip().splitlines()[-1])["do"], "hf_fs")
        self.assertEqual(s["step"], "a_catalogue"); self.assertEqual(pushes, [])
        self.assertEqual(calls, ["lock", "plan", "search-ops m1", "search-ops m2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
