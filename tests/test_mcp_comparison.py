"""Independent synthetic regression checks for the MCP evidence/ranking boundary.

No live publisher data is invented: example.org assertions are synthetic fixtures.
"""
import copy
import importlib.util
import json
import os
import tempfile
import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'web'))   # site generators live in web/
spec = importlib.util.spec_from_file_location("qff_mcp_comparison_test_server", ROOT / "qff_mcp/qff_mcp.py")
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


def assertion(value):
    return {"value": value, "evidence": [{"url": "https://example.org/synthetic-protocol#L1"}]}


def point(identifier, owner="quant-a", size=4, value=.1):
    protocol = {k: assertion(v) for k, v in {
        "base_model": "test/base@abc", "reference_checkpoint": "test/base@abc:bf16",
        "dataset": "synthetic-test@abc", "token_selection": "tokens:0-1000",
        "context_length": 512, "implementation": "synthetic-evaluator@abc",
        "settings": {"stride": 512, "dtype": "f32"},
        "metric_definition": {"name": "KLD", "direction": "lower", "unit": "nats", "aggregation": "token mean"},
    }.items()}
    return {"measurement_id": identifier, "repo": owner + "/quant", "variant": identifier,
            "kind": "own", "m": {"kld": {"v": value, "raw": str(value)}},
            "metrics": {"mean_kld": {"v": value, "raw": str(value)}}, "fmt": "GGUF",
            "x": size, "size_gb": size, "protocol": protocol,
            "quantization_owner": owner, "measurement_owner": "evaluator-a",
            "claim_provenance": {"kind": "original"}, "src": "https://example.org/card#L20"}


def fixture(points, legacy=True):
    return {"rows": points, "repos": [{"repo": p["repo"]} for p in points], "sets": [
        {"id": "test", "title": "Synthetic test", "metrics": ["kld"], "comparable": legacy,
         "points": points, "x": {"unit": "GB", "label": "Size (GB)"}},
    ]}


class MCPComparisonTests(unittest.TestCase):
    def setUp(self):
        self.points = [point("a"), point("b", "quant-b", 5, .05)]
        self.data = fixture(self.points)
        self.model = {"model": "test/base", "slug": "test"}
        self.resolver = patch.object(mcp, "_model", return_value=(self.model, self.data))
        self.resolver.start()
        self.addCleanup(self.resolver.stop)

    def test_verified_cross_publisher_permits_frontier_and_retains_owners(self):
        response = mcp.frontier("test", "test", "kld")
        self.assertTrue(response["ranking_permitted"])
        self.assertEqual(response["comparison"]["classification"], "same_protocol")
        self.assertEqual([p["measurement_id"] for p in response["efficient"]], ["a", "b"])
        self.assertEqual(response["efficient"][1]["measurement_owner"], "evaluator-a")
        self.assertEqual(response["efficient"][1]["quantization_owner"], "quant-b")

    def test_stale_legacy_true_does_not_authorize_rankings(self):
        self.points[1]["protocol"].pop("dataset")
        with self.assertRaisesRegex(ValueError, "not comparable"):
            mcp.frontier("test", "test", "kld")
        response = mcp.best_under("test", 10)
        self.assertEqual(response["per_set"], [])
        self.assertEqual(response["excluded_sets"][0]["comparison"]["unknowns"][0]["condition"], "dataset")

    def test_stale_legacy_false_does_not_block_verified_protocol(self):
        self.data["sets"][0]["comparable"] = False
        self.assertEqual(len(mcp.list_sets("test")["sets"]), 1)
        self.assertEqual(len(mcp.best_under("test", 10)["per_set"]), 1)

    def test_legacy_data_without_protocol_fails_closed(self):
        for p in self.points:
            p.pop("protocol")
        self.assertEqual(mcp.list_sets("test")["sets"], [])
        self.assertFalse(mcp.list_sets("test", False)["sets"][0]["ranking_permitted"])
        with self.assertRaises(ValueError):
            mcp.frontier("test", "test")

    def test_same_publisher_different_protocol_rejected(self):
        self.points[1]["quantization_owner"] = "quant-a"
        self.points[1]["protocol"]["context_length"] = assertion(1024)
        result = mcp.compare_measurements("test", "a", "b")
        self.assertFalse(result["ranking_permitted"])
        self.assertTrue(any(d["condition"] == "context_length" for d in result["differences"]))

    def test_contradictory_methodology_rejected(self):
        self.points[1]["protocol"]["dataset"] = [assertion("dataset-a"), assertion("dataset-b")]
        result = mcp.compare_measurements("test", "a", "b")
        self.assertFalse(result["ranking_permitted"])
        self.assertIn("Contradictory", result["differences"][0]["reason"])

    def test_different_base_model_cannot_rank(self):
        self.points[1]["protocol"]["base_model"] = assertion("test/other@def")
        self.points[1]["protocol"]["reference_checkpoint"] = assertion("test/other@def:bf16")
        result = mcp.compare_measurements("test", "a", "b", other_model="other")
        self.assertEqual(result["classification"], "different_base_model")
        self.assertFalse(result["ranking_permitted"])

    def test_copied_claim_requires_original_attribution_and_preserves_claim(self):
        self.points[1]["claim_provenance"] = {"kind": "copied"}
        self.assertFalse(mcp.compare_measurements("test", "a", "b")["ranking_permitted"])
        self.points[1]["claim_provenance"]["original_attribution"] = {
            "verified": True, "evidence": [{"url": "https://example.org/original#L20"}]}
        result = mcp.compare_measurements("test", "a", "b")
        self.assertTrue(result["ranking_permitted"])
        self.assertEqual(result["right"]["claims"]["mean_kld"], "0.05")
        self.assertEqual(result["right"]["claim_provenance"]["kind"], "copied")

    def test_metric_specific_scope_allows_kld_without_ranking_unverified_ppl(self):
        self.data["sets"][0]["metrics"].append("ppl")
        for p in self.points:
            p["m"]["ppl"] = {"v": 3, "raw": "3"}
            p["protocol"]["metrics"] = {"ppl": {"metric_definition": {}}}
        summary = mcp.list_sets("test")["sets"][0]
        self.assertFalse(summary["comparable"])
        self.assertTrue(summary["comparisons"]["kld"]["ranking_permitted"])
        self.assertFalse(summary["comparisons"]["ppl"]["ranking_permitted"])
        self.assertTrue(mcp.frontier("test", "test", "kld")["ranking_permitted"])
        with self.assertRaises(ValueError):
            mcp.frontier("test", "test", "ppl")

    def test_search_and_repo_expose_protocol_provenance_and_classification(self):
        result = mcp.search_quants("test")
        self.assertTrue(result["comparison"]["ranking_permitted"])
        self.assertEqual(result["comparison_scope"], "returned rows")
        self.assertEqual(result["rows"][0]["measurement_id"], "a")
        self.assertIn("protocol", result["rows"][0])
        self.assertIn("comparison", mcp.get_repo("test", "quant-a/quant"))

    def test_classifier_missing_fails_closed(self):
        with patch.object(mcp, "classify_group", None), patch.object(mcp, "classify_pair", None):
            self.assertFalse(mcp.compare_measurements("test", "a", "b")["ranking_permitted"])
            with self.assertRaisesRegex(ValueError, "engine is unavailable"):
                mcp.frontier("test", "test", "kld")

    def test_absent_metric_is_not_mistaken_for_verified_measurement(self):
        with self.assertRaisesRegex(ValueError, "no published 'ppl' claim"):
            mcp.compare_measurements("test", "a", "b", "ppl")

    def test_unknown_id_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "unknown measurement_id"):
            mcp.compare_measurements("test", "a", "absent")

    def test_best_under_prefix_does_not_compare_different_resolved_metric_series(self):
        sets = []
        for locale in ("english", "korean"):
            series = "kld:" + locale
            points = [point(locale + "-a"), point(locale + "-b", size=5, value=.05)]
            for p in points:
                p["m"] = {series: p["m"]["kld"]}
                p["protocol"]["metrics"] = {series: {"dataset": assertion("dataset-" + locale)}}
            group = fixture(points)["sets"][0]
            group.update(id=locale, metrics=[series])
            sets.append(group)
        self.data["sets"] = sets
        result = mcp.best_under("test", 10, metric="kld")
        self.assertEqual(len(result["per_set"]), 2)
        self.assertTrue(all(entry["ranking_permitted"] for entry in result["per_set"]))
        self.assertEqual({entry["metric"] for entry in result["per_set"]}, {"kld:english", "kld:korean"})
        self.assertEqual(result["comparison"]["classification"], "different_or_unverified_protocol")
        self.assertFalse(result["comparison"]["ranking_permitted"])

    def test_best_under_prefix_applies_the_actual_named_metric_overlay(self):
        for second_dataset, comparable in (("english", True), ("different-corpus", False)):
            with self.subTest(second_dataset=second_dataset):
                sets = []
                for index, dataset in enumerate(("english", second_dataset)):
                    points = [point(str(index) + "-a"), point(str(index) + "-b", size=5, value=.05)]
                    for p in points:
                        p["m"] = {"kld:english": p["m"]["kld"]}
                        p["protocol"]["metrics"] = {"kld:english": {"dataset": assertion(dataset)}}
                    group = fixture(points)["sets"][0]
                    group.update(id=str(index), metrics=["kld:english"])
                    sets.append(group)
                self.data["sets"] = sets
                result = mcp.best_under("test", 10, metric="kld")
                self.assertEqual(len(result["per_set"]), 2)
                self.assertTrue(all(entry["ranking_permitted"] for entry in result["per_set"]))
                self.assertEqual(result["comparison"]["metric"], "kld:english")
                self.assertEqual(result["comparison"]["ranking_permitted"], comparable)

    def test_metric_direction_comes_from_verified_definition(self):
        for p in self.points:
            p["protocol"]["metric_definition"]["value"]["direction"] = "higher"
        result = mcp.best_under("test", 10)
        self.assertEqual(result["per_set"][0]["best"]["measurement_id"], "a")


class MCPComparisonTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_stdio_exposes_evidence_and_refuses_stale_rankings(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        with tempfile.TemporaryDirectory(prefix="qff-mcp-evidence-") as folder:
            data = fixture([point("a"), point("b", "quant-b", 5, .05)])
            data['model'] = 'test/base'
            bad = copy.deepcopy(data["sets"][0])
            bad["id"] = "stale-true"
            bad["points"][1]["protocol"].pop("reference_checkpoint")
            data["sets"].append(bad)
            pathlib.Path(folder, "models.json").write_text(json.dumps({"models": [
                {"model": "test/base", "slug": "test", "data": "data.json"}]}))
            pathlib.Path(folder, "data.json").write_text(json.dumps(data))
            params = StdioServerParameters(command=sys.executable,
                args=[str(ROOT / "qff_mcp/qff_mcp.py")], env={**os.environ, "QFF_DATA": folder})
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as client:
                    await client.initialize()
                    tools = {tool.name for tool in (await client.list_tools()).tools}
                    self.assertIn("compare_measurements", tools)
                    result = await client.call_tool("compare_measurements", {
                        "model": "test", "left_id": "a", "right_id": "b"})
                    self.assertFalse(getattr(result, "isError", False))
                    payload = json.loads(result.content[0].text)
                    self.assertTrue(payload["ranking_permitted"])
                    self.assertTrue(payload["evidence"])
                    result = await client.call_tool("frontier", {"model": "test", "set_id": "stale-true"})
                    self.assertTrue(bool(getattr(result, "is_error", None) or getattr(result, "isError", False)))


class MCPPublishedDataTests(unittest.TestCase):
    """Local assembled capture; no network, no synthetic evidence assignments."""
    def setUp(self):
        self.site = pathlib.Path(os.environ.get("QFF_DATA", ROOT / "_site"))
        if not (self.site / "models.json").is_file():
            self.skipTest("assemble.py must run before local published-data checks")
        self.manifest = json.loads((self.site / "models.json").read_text())["models"]
        self.qwen = next((m for m in self.manifest if m["slug"] == "qwen3.8-27b"), None)
        if not self.qwen:
            self.skipTest("Qwen3.8-27B absent from assembled data")
        self.data = json.loads((self.site / self.qwen["data"]).read_text())
        self.atomic = next(s for s in self.data["sets"] if s["id"] == "a-atomicchat-gguf")
        if "comparisons" not in self.atomic:
            self.skipTest("assembled capture predates the comparison metadata build")
        self.addCleanup(patch.stopall)
        patch.object(mcp, "SRC", str(self.site)).start()
        patch.object(mcp, "_cache", {}).start()

    def test_atomic_evaluator_compares_distinct_quant_publishers(self):
        P = [p for p in self.atomic["points"] if not p.get("is_ref") and p.get("m", {}).get("kld", {}).get("v") is not None]
        left, right = next((a, b) for a in P for b in P if a["quantization_owner"] != b["quantization_owner"])
        self.assertEqual(left["measurement_owner"], right["measurement_owner"])
        self.assertTrue(self.atomic["comparisons"]["kld"]["ranking_permitted"])
        pair = mcp.compare_measurements(self.qwen["model"], left["measurement_id"], right["measurement_id"])
        self.assertEqual(pair["classification"], "same_protocol")
        self.assertTrue(pair["ranking_permitted"])
        self.assertTrue(pair["evidence"])
        frontier = mcp.frontier(self.qwen["model"], self.atomic["id"], "kld")
        self.assertTrue(frontier["ranking_permitted"])
        self.assertTrue(frontier["efficient"])
        repo = mcp.get_repo(self.qwen["model"], "AtomicChat/Qwen3.8-27B-GGUF")
        self.assertTrue(repo["comparisons"]["mean_kld"]["ranking_permitted"])
        self.assertTrue(repo["comparisons"]["top1_pct"]["ranking_permitted"])
        rows = {r["measurement_id"]: r for r in self.data["rows"]}
        for p in frontier["efficient"]:
            source = rows[p["measurement_id"]]
            self.assertIn(p["claims"]["kld"], [v["raw"] for v in source["metrics"].values()])

    def test_real_cross_model_pair_never_ranks_fidelity(self):
        other = next(m for m in self.manifest if m["model"] != self.qwen["model"])
        other_data = json.loads((self.site / other["data"]).read_text())
        def kld_point(data):
            return next(p for s in data["sets"] for p in s["points"]
                        if p.get("measurement_id") and p.get("m", {}).get("kld", {}).get("v") is not None)
        left, right = kld_point(self.data), kld_point(other_data)
        pair = mcp.compare_measurements(self.qwen["model"], left["measurement_id"], right["measurement_id"], other_model=other["model"])
        self.assertEqual(pair["classification"], "different_base_model")
        self.assertFalse(pair["ranking_permitted"])
        self.assertTrue(any(d["condition"] == "base_model" for d in pair["differences"]))


if __name__ == "__main__":
    unittest.main()
