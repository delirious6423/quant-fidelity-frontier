"""Exact-source freshness, mapping writes, and one-time migration regressions."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE / "gates"), str(CODE / "pipeline")]
import current
import gatecache
import qcl_pin_reviews
from test_mapping_panels import MAP, source_functions


class SourcePins(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qff-source-pins-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for folder in ("snapshots", "audit", "verified", "cache/cards"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self.repo = "publisher/reviewed-GGUF"
        self.entry = {"repo": self.repo, "sha": "main", "fmt": "GGUF", "variants": [
            {"name": "Q4", "m": {"mean_kld": "0.1250"}, "L": 1}]}
        self.card = self.root / "cache/cards/reviewed.md"
        self.card.write_text("| Q4 | 0.1250 |\n" + "Reviewed source context. " * 4)
        self.digest = hashlib.sha256(self.card.read_bytes()).hexdigest()
        self.idx = {self.repo: {"sha": "main", "sha256": self.digest, "file": "cache/cards/reviewed.md", "tags": []},
                    "publisher/unreviewed-GGUF": {"sha": "main", "sha256": self.digest, "file": "cache/cards/reviewed.md", "tags": []}}
        self.save_index()
        self.save_entry()
        self.registry = qcl_pin_reviews.baseline_registry(self.root)
        (self.root / "audit/review_pins.json").write_text(json.dumps(self.registry))

    def save_index(self):
        (self.root / "snapshots/index.json").write_text(json.dumps({"repos": self.idx}))

    def save_entry(self):
        (self.root / "verified/part1.py").write_text("R = " + repr([self.entry]) + "\n")

    def reviewed(self):
        return next(row for row in current.load(self.root) if row["repo"] == self.repo)

    def test_legacy_mutable_review_is_kept_only_with_its_exact_registry_pin(self):
        self.assertEqual(self.reviewed()["variants"], self.entry["variants"])
        self.assertEqual(self.reviewed()["_review_source_sha256"], self.digest)
        (self.root / "audit/review_pins.json").unlink()
        self.assertTrue(self.reviewed()["auto"])
        self.assertEqual(self.reviewed()["variants"], [])

    def test_source_change_with_same_branch_withdraws_claims(self):
        self.idx[self.repo]["sha256"] = "f" * 64
        self.save_index()
        self.assertTrue(self.reviewed()["auto"])
        self.assertEqual(self.reviewed()["flags"], [current.STALE_FLAG])

    def test_modified_entry_cannot_inherit_a_legacy_review(self):
        for key, value in (("variants", [{"name": "Q4", "m": {"mean_kld": "99.999"}, "L": 1}]),
                           ("methodology", {"reference": "a different reference"})):
            with self.subTest(field=key):
                original = copy.deepcopy(self.entry)
                self.entry[key] = value
                self.save_entry()
                self.assertTrue(self.reviewed()["auto"])
                self.entry = original
                self.save_entry()

    def test_internal_pin_cannot_be_injected_into_a_claim_file(self):
        (self.root / "audit/review_pins.json").unlink()
        self.entry["_review_source_sha256"] = self.digest
        self.save_entry()
        self.assertTrue(self.reviewed()["auto"])

    def test_explicit_pin_tracks_content_even_across_revision_changes(self):
        self.entry["source_sha256"] = self.digest
        self.save_entry()
        self.idx[self.repo]["sha"] = "abcdef12" + "3" * 32
        self.save_index()
        self.assertEqual(self.reviewed()["variants"], self.entry["variants"])
        self.idx[self.repo]["sha256"] = "f" * 64
        self.save_index()
        self.assertTrue(self.reviewed()["auto"])

    def test_unpinned_legacy_revision_requires_a_full_matching_immutable_prefix(self):
        index = {self.repo: {"sha": "abcdef12" + "3" * 32}}
        self.assertTrue(current.fresh({"repo": self.repo, "sha": "abcdef12"}, index))
        for revision in ("", "main", "abc", "abcdef12" + "4" * 32):
            with self.subTest(revision=revision):
                self.assertFalse(current.fresh({"repo": self.repo, "sha": revision}, index))

    def test_source_pin_checks_bytes_and_refreshes_the_line_cache(self):
        namespace = source_functions("pipeline/qcl_map.py", {"source_pin", "A_lines"}, {
            "A": SimpleNamespace(IDX=self.idx), "ROOT": str(self.root), "os": os,
            "hashlib": hashlib, "_lc": {self.repo: ["obsolete cached text"]}})
        self.assertEqual(namespace["source_pin"](self.repo), self.digest)
        self.assertEqual(namespace["A_lines"](self.repo), self.card.read_text().split("\n"))
        self.card.write_bytes(self.card.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            namespace["source_pin"](self.repo)

    def mapped(self):
        mapping = {"repo": self.repo, "entries": [{"role": "own", "name": "Q4", "metrics": {"mean_kld": "c1"}}]}
        return MAP["apply"](mapping, {"c1": {"raw": "0.1250", "L": 1, "label": "Q4", "header": "KLD"}},
                            source_sha256=self.digest)

    def writer(self):
        return source_functions("pipeline/qcl_watch.py", {"write"}, {
            "ROOT": str(self.root), "OUT": str(self.root / "verified/part8_machine.py"), "json": json,
            "os": os, "re": re, "hashlib": hashlib, "F": SimpleNamespace(fmt_of=lambda *args: "GGUF")})["write"]

    def test_mapping_and_writer_preserve_exact_source_pin(self):
        applied = self.mapped()
        self.assertEqual(applied["source_sha256"], self.digest)
        self.writer()({self.repo: applied}, [])
        namespace = {}
        exec((self.root / "verified/part8_machine.py").read_text(), namespace)
        self.assertEqual(namespace["R"][0]["source_sha256"], self.digest)
        self.assertEqual(namespace["R"][0]["variants"][0]["m"]["mean_kld"], "0.1250")

    def test_writer_rejects_missing_stale_or_changed_source(self):
        write = self.writer()
        for pin in (None, "f" * 64):
            applied = self.mapped()
            applied["source_sha256"] = pin
            with self.subTest(pin=pin), self.assertRaisesRegex(ValueError, "missing or stale"):
                write({self.repo: applied}, [])
        self.card.write_bytes(self.card.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            write({self.repo: self.mapped()}, [])
        self.assertFalse((self.root / "verified/part8_machine.py").exists())

    def test_auto_regeneration_preserves_reviewed_coverage_then_lists_changed_card(self):
        def regenerate():
            result = subprocess.run([sys.executable, str(CODE / "pipeline/gen_auto.py")],
                                    env={**os.environ, "QCL_ROOT": str(self.root)}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            namespace = {}
            exec((self.root / "verified/part9_auto.py").read_text(), namespace)
            return {entry["repo"] for entry in namespace["R"]}
        before = self.reviewed()
        self.assertEqual(regenerate(), {"publisher/unreviewed-GGUF"})
        self.assertEqual(self.reviewed(), before)
        self.idx[self.repo]["sha256"] = "f" * 64
        self.save_index()
        self.assertEqual(regenerate(), set(self.idx))
        self.assertEqual(self.reviewed()["variants"], [])

    def test_baseline_migration_rejects_source_mismatch_and_repinning(self):
        self.card.write_bytes(self.card.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "source pin mismatch"):
            qcl_pin_reviews.baseline_registry(self.root)
        self.idx[self.repo]["sha256"] = hashlib.sha256(self.card.read_bytes()).hexdigest()
        self.save_index()
        with self.assertRaisesRegex(ValueError, "cannot be expanded or repinned"):
            qcl_pin_reviews.migrate(self.root)
        self.assertEqual(json.loads((self.root / "audit/review_pins.json").read_text()), self.registry)

    def test_baseline_migration_installs_nothing_if_any_gate_fails(self):
        registry = self.root / "audit/review_pins.json"
        registry.unlink()
        result = SimpleNamespace(returncode=1, stdout="counterfactual gate failure", stderr="")
        with patch.object(gatecache, "ROOT", str(self.root)), patch.object(qcl_pin_reviews.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(ValueError, "Baseline gate .* failed"):
                qcl_pin_reviews.migrate(self.root)
        self.assertFalse(registry.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
