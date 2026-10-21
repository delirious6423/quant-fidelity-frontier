"""Counterfactual tests for gate cache reuse and pinned source integrity.

Mutations happen only in throwaway copies. No source claim is edited in place.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gates"))
import gatecache
import current

FLASH = REPO / "models/qwen3.8-flash-next"
sys.path.insert(0, str(REPO / "tests"))
import frozen


class GateIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qff-gate-integrity-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "model"
        shutil.copytree(FLASH, self.root, ignore=shutil.ignore_patterns("site", "site_dev", ".gatecache.json", "__pycache__"))
        frozen.freeze(self.root)

    def run_gate(self, gate):
        return subprocess.run([sys.executable, str(REPO / "gates" / gate)], cwd=REPO,
                              env={**os.environ, "QCL_ROOT": str(self.root)}, capture_output=True, text=True)

    def index(self):
        return json.loads((self.root / "snapshots/index.json").read_text())

    def test_identical_source_is_current_when_mutable_revision_becomes_a_commit(self):
        # The live data pins image claims directly (source_sha256). Rebuild the legacy shape in the throwaway copy:
        # an entry without source_sha256, pinned through the review registry by its full fingerprint.
        part = self.root / "verified/part3_images.py"
        namespace = {}; exec(part.read_text(), namespace)
        raw_entry = namespace["R"][0]
        source_pin = raw_entry.pop("source_sha256")
        part.write_text(part.read_text().replace(f"source_sha256='{source_pin}', ", "", 1))
        pins = json.loads((self.root / "audit/review_pins.json").read_text())
        pins["entries"][current.entry_fingerprint(raw_entry)] = {"repo": raw_entry["repo"], "source_sha256": source_pin}
        (self.root / "audit/review_pins.json").write_text(json.dumps(pins))
        entry = next(r for r in current.load(self.root) if r.get("_review_source_sha256") and r.get("variants"))
        index = self.index()
        index["repos"][entry["repo"]]["sha"] = "1234567890abcdef1234567890abcdef12345678"
        (self.root / "snapshots/index.json").write_text(json.dumps(index))
        result = self.run_gate("verify.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(any(r.get("variants") == entry["variants"] for r in current.load(self.root) if r["repo"] == entry["repo"]))

    def cache_scope(self, relative=False):
        root = os.path.relpath(self.root) if relative else str(self.root)
        return patch.multiple(gatecache, ROOT=root, CACHE=str(self.root / "audit/.gatecache.json"), GATES=("verify.py",))

    def test_relative_and_absolute_roots_have_the_same_fingerprint(self):
        with self.cache_scope(relative=True):
            relative = gatecache.inputs_hash("verify.py")
        with self.cache_scope():
            self.assertEqual(relative, gatecache.inputs_hash("verify.py"))

    def test_relative_root_hashes_claims_cards_indexes_exceptions_and_images(self):
        card = self.root / self.index()["repos"]["AesSedai/Qwen3.8-Flash-Next-GGUF"]["file"]
        files = [self.root / "verified/part1.py", card, self.root / "snapshots/index.json",
                 self.root / "audit/column_exceptions.json", self.root / "cache/manifest.json",
                 self.root / "snapshots/images/turboderp_qb_kld.png"]
        with self.cache_scope(relative=True):
            baseline = gatecache.inputs_hash("verify.py")
            for file in files:
                with self.subTest(file=file.relative_to(self.root)):
                    original = file.read_bytes()
                    file.write_bytes(original + b"\nchanged\n")
                    self.assertNotEqual(baseline, gatecache.inputs_hash("verify.py"))
                    file.write_bytes(original)
            card.unlink()
            self.assertNotEqual(baseline, gatecache.inputs_hash("verify.py"))

    def test_gate_helpers_invalidate_all_gate_results(self):
        code = Path(self.temporary.name) / "code"
        shutil.copytree(REPO / "gates", code / "gates", ignore=shutil.ignore_patterns("__pycache__"))
        (code / "pipeline").mkdir()
        shutil.copyfile(REPO / "pipeline/autoextract.py", code / "pipeline/autoextract.py")
        with self.cache_scope(), patch.object(gatecache, "CODE", str(code)):
            baseline = gatecache.inputs_hash("verify.py")
            for name in ("current.py", "gatecache.py", "verify_columns.py"):
                with self.subTest(helper=name):
                    file = code / "gates" / name
                    original = file.read_bytes()
                    file.write_bytes(original + b"\n# changed gate dependency\n")
                    self.assertNotEqual(baseline, gatecache.inputs_hash("verify.py"))
                    file.write_bytes(original)

    def test_warm_relative_root_cache_cannot_mask_invalid_claim(self):
        with self.cache_scope(relative=True):
            first = gatecache.run_all()["verify.py"]
            self.assertEqual(first[0], 0, first[1])
            self.assertTrue(gatecache.run_all()["verify.py"][2], "unchanged inputs should reuse a passing result")
            file = self.root / "verified/part1.py"
            old = file.read_text()
            self.assertIn('mean_kld="0.020218 ± 0.000145"', old)
            file.write_text(old.replace('mean_kld="0.020218 ± 0.000145"', 'mean_kld="0.1234567 ± 0.000145"', 1))
            result = gatecache.run_all()["verify.py"]
            self.assertNotEqual(result[0], 0)
            self.assertFalse(result[2])
            self.assertIn("NOT FOUND", result[1])
            self.assertNotIn("verify.py", json.loads(Path(gatecache.CACHE).read_text()))

    def test_inputs_changed_during_gate_are_neither_published_nor_cached(self):
        file = self.root / "verified/part1.py"
        def changed_during_run(*args, **kwargs):
            file.write_bytes(file.read_bytes() + b"\n# concurrent mapping update\n")
            return SimpleNamespace(returncode=0, stdout="old inputs passed\n", stderr="")
        with self.cache_scope(), patch.object(gatecache.subprocess, "run", side_effect=changed_during_run):
            result = gatecache.run_all()["verify.py"]
            self.assertNotEqual(result[0], 0)
            self.assertIn("inputs changed", result[1])
            self.assertNotIn("verify.py", json.loads(Path(gatecache.CACHE).read_text()))

    def test_changed_pinned_card_fails_even_when_all_claim_values_remain(self):
        card = self.root / self.index()["repos"]["AesSedai/Qwen3.8-Flash-Next-GGUF"]["file"]
        card.write_bytes(card.read_bytes() + b"\nchanged uncited content\n")
        result = self.run_gate("verify.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SOURCE source SHA-256 differs", result.stdout)

    def test_card_without_a_content_pin_fails_closed(self):
        index = self.index()
        index["repos"]["AesSedai/Qwen3.8-Flash-Next-GGUF"].pop("sha256")
        (self.root / "snapshots/index.json").write_text(json.dumps(index))
        result = self.run_gate("verify.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no valid pinned SHA-256", result.stdout)

    def test_missing_required_image_fails_closed(self):
        for name in ("turboderp_qb_kld.png", "turboderp_qb_ppl.png"):
            with self.subTest(image=name):
                file = self.root / "snapshots/images" / name
                original = file.read_bytes()
                file.unlink()
                result = self.run_gate("verify_images.py")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("required chart image is missing", result.stdout)
                file.write_bytes(original)

    def test_changed_pinned_image_fails_even_if_pixels_are_unchanged(self):
        file = self.root / "snapshots/images/turboderp_qb_kld.png"
        file.write_bytes(file.read_bytes() + b"appended bytes")
        result = self.run_gate("verify_images.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("image SHA-256 differs", result.stdout)

    def test_withdrawn_image_claims_do_not_require_unused_images(self):
        index = self.index()
        index["repos"]["turboderp/Qwen3.8-Flash-Next-exl3"]["sha"] = "new-source-revision"
        index["repos"]["turboderp/Qwen3.8-Flash-Next-exl3"]["sha256"] = "0" * 64   # the card changed: its pinned claims are withdrawn
        (self.root / "snapshots/index.json").write_text(json.dumps(index))
        shutil.rmtree(self.root / "snapshots/images")
        result = self.run_gate("verify_images.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no active image-sourced claims", result.stdout)

    def test_valid_current_image_claims_still_verify(self):
        result = self.run_gate("verify_images.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("image claims checked: 19, failures: 0", result.stdout)

    def test_hf_branch_size_unit_cannot_change_without_converting_value(self):
        file = self.root / "verified/part3_images.py"
        original = file.read_text()
        self.assertIn("'unit': 'GB'", original)
        file.write_text(original.replace("'unit': 'GB'", "'unit': 'GiB'", 1))
        result = self.run_gate("verify.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("branch size or UNIT mismatch", result.stdout)

    def test_hf_listing_size_unit_cannot_change_without_converting_value(self):
        file = self.root / "verified/part2b.py"
        original = file.read_text()
        self.assertIn('size=132.73, unit="GB"', original)
        file.write_text(original.replace('size=132.73, unit="GB"', 'size=132.73, unit="GiB"', 1))
        result = self.run_gate("verify.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HF listing (value/UNIT mismatch)", result.stdout)

    def test_negative_claim_cannot_drop_its_sign(self):
        file = self.root / "verified/part4_5.py"
        original = file.read_text()
        self.assertIn('"ppl_delta_pct":"−0.16 %"', original)
        file.write_text(original.replace('"ppl_delta_pct":"−0.16 %"', '"ppl_delta_pct":"0.16 %"', 1))
        result = self.run_gate("verify.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SIGN", result.stdout)

    def test_extra_image_metric_cannot_skip_validation(self):
        file = self.root / "verified/part3_images.py"
        file.write_text(file.read_text() + "\nR[0]['variants'][0]['m']['top1'] = '99.999'\n")
        result = self.run_gate("verify_images.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported image-sourced metric shape", result.stdout)

    def test_extra_image_reference_cannot_skip_validation(self):
        file = self.root / "verified/part3_images.py"
        file.write_text(file.read_text() + "\nR[0]['baseline2'] = {'ppl': '99.999', 'src': 'chart_image', 'L': 28}\n")
        result = self.run_gate("verify_images.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported image-sourced reference shape", result.stdout)

    def test_nonfinite_or_outside_image_coordinates_fail_closed(self):
        file = self.root / "verified/part3_images.py"
        original = file.read_text()
        for value in ("100000", "-1", "float('nan')", "float('inf')"):
            with self.subTest(coordinate=value):
                file.write_text(original + f"\nR[0]['variants'][0]['px']['x'] = {value}\nR[0]['variants'][0]['bpw'] = 236.50440917107585\n")
                result = self.run_gate("verify_images.py")
                self.assertNotEqual(result.returncode, 0)
                if "float(" in value:
                    # Directly pinned entries are not fingerprinted, so the image coordinate gate itself rejects
                    # NaN/Infinity (legacy registry-pinned entries are rejected earlier by their canonical fingerprint).
                    self.assertTrue("FAIL invalid image claim source" in result.stdout
                                    or "coordinates must be finite and inside" in result.stdout, result.stdout)
                else:
                    self.assertIn("coordinates must be finite and inside", result.stdout)

    def test_ppl_score_requires_its_own_marker(self):
        file = self.root / "verified/part3_images.py"
        file.write_text(file.read_text() + "\nR[0]['variants'][0]['m']['ppl'] = '1.43'\nR[0]['variants'][0]['px']['ppl_y'] = 897.875\n")
        result = self.run_gate("verify_images.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no ppl marker", result.stdout)

    def test_changed_external_docs_fail_even_when_claim_values_remain(self):
        file = self.root / "snapshots/unsloth.ai__docs__qwen3.8-next.md"
        file.write_bytes(file.read_bytes() + b"\nchanged uncited source content\n")
        result = self.run_gate("verify.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("external unsloth_docs: SOURCE source SHA-256 differs", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
