#!/usr/bin/env python3
"""Regression coverage for large native-quantized models and multiple card panels."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]


class ModelPanels(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", category=ResourceWarning)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "snapshots").mkdir()
        (self.root / "snapshots/index.json").write_text(json.dumps({"repos": {}}))
        (self.root / "snapshots/hf_sizes_summary.json").write_text("{}")
        (self.root / "sets.py").write_text("SETS = []\nMETRICS = {}\n")

    def module(self, config=None):
        (self.root / "model.json").write_text(json.dumps(config or {}))
        spec = importlib.util.spec_from_file_location("qff_build_panels", CODE / "web" / "build.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"QCL_ROOT": str(self.root)}):
            spec.loader.exec_module(module)
        return module

    def test_scientific_notation_preserves_scale(self):
        build = self.module()
        self.assertEqual(build.num("2.29e-7"), 2.29e-7)
        self.assertEqual(build.num("−3.5E+2"), -350)
        self.assertEqual(build.err("0.03 ± 1.2e-4"), 1.2e-4)

    def test_model_size_bound_and_gib_conversion(self):
        default = self.module()
        self.assertTrue(default.size_plausible(400))
        self.assertFalse(default.size_plausible(default.to_gb(400, "GiB")))
        large = self.module({"max_size_gb": 2000})
        self.assertTrue(large.size_plausible(large.to_gb(492, "GiB")))
        self.assertFalse(large.size_plausible(2001))
        self.assertFalse(large.size_plausible(0))

    def test_native_reference_is_model_specific(self):
        method = {"reference": "original model with native MXFP4 routed experts"}
        self.assertEqual(self.module().ref_class(method), "bf16")
        native = self.module({"native_reference_patterns": [r"\bnative\b", r"\boriginal (model|weights)\b"]})
        self.assertEqual(native.ref_class(method), "native")
        self.assertEqual(native.ref_class({"reference": "Q8_0 (native source)"}), "q8")
        self.assertEqual(native.ref_class({"reference": "no native reference"}), "other")

    def test_entry_method_controls_row_reference(self):
        build = self.module({"native_reference_patterns": [r"\bnative\b"]})
        build.IDX["repos"]["publisher/model"] = {"sha": "a" * 40, "lastModified": "2026-10-17"}
        rows, _ = build.build_rows([{
            "repo": "publisher/model", "fmt": "EXL3", "methodology": {"reference": "BF16"},
            "variants": [{"name": "2.7bpw", "m": {"mean_kld": "0.0243"},
                          "methodology": {"reference": "native weights"}, "panel": "held-out"}],
        }])
        self.assertEqual(rows[0]["ref_class"], "native")
        self.assertEqual(rows[0]["panel"], "held-out")

    def rows(self, **overrides):
        return [{"repo": "publisher/model", "measured_by": "publisher/model", "kind": "own",
                 "variant": f"Q{i}", "metrics": {"mean_kld": {"v": 0.01 * i, "raw": str(0.01 * i)}},
                 "size_gb": 100 * i, "size_raw": f"{100 * i} GB", "basis": "disk",
                 "src": "https://example.test/card", "src_label": "L1", **overrides}
                for i in (2, 3)]

    def test_overlapping_quant_ids_survive_separate_panels(self):
        build = self.module()
        method = {"reference": "native", "dataset": "held-out", "ctx": "2048", "tool": "runner"}
        rows = self.rows(panel="token weighted", methodology=method)
        rows += self.rows(panel="domain weighted", methodology=method)
        sets, _ = build.build_sets(rows, [{"repo": "publisher/model", "fmt": "EXL3", "methodology": method}])
        comparable = [s for s in sets if s["id"] != "all"]
        self.assertTrue(all(not s["comparable"] for s in comparable))
        self.assertEqual(len(comparable), 2)
        self.assertEqual({s["title"].split(" · ")[-1] for s in comparable}, {"token weighted", "domain weighted"})
        self.assertTrue(all(len(s["points"]) == 2 for s in comparable))

    def test_context_and_tool_split_and_unknown_sizes_are_not_ranked(self):
        build = self.module()
        rows = []
        for context, tool in (("2048", "runner A"), ("4096", "runner A"), ("2048", "runner B")):
            rows += self.rows(methodology={"reference": "native", "dataset": "panel", "ctx": context, "tool": tool})
        repos = [{"repo": "publisher/model", "methodology": {}}]
        self.assertEqual(len(build.auto_sets(rows, repos, set())), 3)
        self.assertEqual(build.auto_sets(self.rows(basis=None), repos, set()), [])

    def test_named_corpus_series_preserve_domains_values_and_ui_definitions(self):
        build = self.module({"named_metric_series": True})
        keys = ["kld:neutral_text", "top1:neutral_text", "kld:code", "top1:code",
                "kld:agentic", "top1:agentic", "kld:fineweb-edu", "top1:fineweb-edu",
                "ppl:fineweb-edu", "kld:eval_mix", "kld:thinking", "kld:wikitext-2",
                "kld:english", "top1:english", "ppl:wikitext"]
        values = {key: {"v": i / 10, "raw": f"published {i}/10"} for i, key in enumerate(keys, 1)}
        rows = self.rows(metrics=values)
        repos = [{"repo": "publisher/model", "fmt": "EXL3", "methodology": {}}]
        sets, definitions = build.build_sets(rows, repos)
        comparable = [st for st in sets if st["id"] != "all"]
        self.assertTrue(all(not st["comparable"] for st in comparable))
        self.assertEqual(len(comparable), 1)
        self.assertEqual(set(comparable[0]["metrics"]), set(keys))
        for st in sets:
            self.assertTrue(set(st["metrics"]) <= set(definitions))
            for point in st["points"]:
                if st["id"] == "all":   # the overview shows one value per family, taken verbatim from a named series
                    for fam, value in point["m"].items():
                        self.assertIn(fam, ("kld", "top1", "ppl"))
                        self.assertTrue(value["series"].startswith(fam + ":"))
                        self.assertEqual({k: value[k] for k in ("v", "raw")}, values[value["series"]])
                    continue
                self.assertFalse({"kld", "top1", "ppl"} & set(point["m"]))
                for key, value in point["m"].items():
                    self.assertEqual(value, values[key])
        for key in keys:
            family, corpus = key.split(":", 1)
            definition = definitions[key]
            self.assertEqual(definition["corpus"], corpus)
            self.assertIn(corpus.replace("_", " "), definition["label"])
            self.assertEqual(definition["unit"], {"kld": "nats", "top1": "%", "ppl": ""}[family])
            self.assertEqual(definition["better"], "higher" if family == "top1" else "lower")
            self.assertEqual(definition["log"], family == "kld")

    def test_named_domains_cannot_supply_each_others_second_point(self):
        build = self.module({"named_metric_series": True})
        rows = self.rows()
        rows[0]["metrics"] = {"kld:english": {"v": 0.01, "raw": "0.01"}}
        rows[1]["metrics"] = {"kld:code": {"v": 0.02, "raw": "0.02"}}
        self.assertEqual(build.auto_sets(rows, [{"repo": "publisher/model"}], set()), [])
        self.assertIsNone(build.fam_key(rows[0], "kld"))

    def test_named_metric_series_is_opt_in(self):
        build = self.module()
        rows = self.rows(metrics={"kld:english": {"v": 0.01, "raw": "0.01"},
                                  "kld:code": {"v": 0.02, "raw": "0.02"}})
        sets = build.auto_sets(rows, [{"repo": "publisher/model"}], set())
        self.assertEqual(sets[0]["metrics"], ["kld"])
        self.assertEqual(sets[0]["points"][0]["m"], {"kld": {"v": 0.01, "raw": "0.01"}})


if __name__ == "__main__":
    unittest.main()
