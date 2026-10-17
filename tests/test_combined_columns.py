#!/usr/bin/env python3
"""Focused counterfactuals for documented corpus-column KL/top-1 pairs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
INTRO = (
    "Each cell shows the mean KL\n"
    "lower bound in nats, then top-1 agreement with the original."
)
ROWS = [
    ("Original model, second run (noise floor)", ["0.0158", "0.0100", "0.0055"], ["96.1%", "97.7%", "98.6%"]),
    ("NVIDIA's NVFP4 checkpoint", ["0.0352", "0.0197", "0.0089"], ["94.1%", "96.8%", "98.3%"]),
    ("This pack, as served", ["0.0545", "0.0321", "0.0101"], ["92.9%", "96.0%", "98.2%"]),
    ("Previous 2.0 bpw pack", ["0.3613", "0.2147", "0.0766"], ["80.4%", "88.5%", "94.7%"]),
]


class CombinedColumnsTest(unittest.TestCase):
    def gate(self, *, intro=INTRO, mutate=None, delimiter=" · ", trailing_intro=False):
        lines = ["# Model", "", "## Fidelity", "", intro if not trailing_intro else "No ordering stated.", "",
                 "| Model | Neutral text | Code | Agentic |", "|---|---:|---:|---:|"]
        variants = []
        for label, kl, top1 in ROWS:
            lines.append("| " + label + " | " + " | ".join(
                "**" + k + delimiter + t + "**" for k, t in zip(kl, top1)) + " |")
            metrics = {}
            for corpus, k, t in zip(("neutral_text", "code", "agentic"), kl, top1):
                metrics["kld:" + corpus] = k
                metrics["top1:" + corpus] = t
            # INTRO spans lines; join/split before counting the final row line.
            variants.append(dict(name=label, row_label=label, L=len("\n".join(lines).splitlines()), m=metrics))
        if trailing_intro: lines += ["", INTRO]
        if mutate: mutate(variants)
        with tempfile.TemporaryDirectory(prefix="qcl-combined-columns-") as tmp:
            root = Path(tmp)
            (root / "snapshots").mkdir()
            (root / "verified").mkdir()
            (root / "snapshots/card.md").write_text("\n".join(lines) + "\n")
            (root / "snapshots/index.json").write_text(json.dumps({"repos": {"test/model": {"file": "snapshots/card.md"}}}))
            (root / "verified/part1.py").write_text("R = " + repr([dict(repo="test/model", variants=variants)]))
            env = dict(os.environ, QCL_ROOT=str(root))
            return subprocess.run([sys.executable, str(REPO_ROOT / "gates/verify_columns.py")],
                                  env=env, capture_output=True, text=True)

    def test_all_four_published_row_shapes_pass(self):
        result = self.gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("table-cell bindings checked: 24", result.stdout)

    def test_same_cell_component_swap_fails(self):
        def swap(entries):
            metrics = entries[2]["m"]
            metrics["kld:neutral_text"], metrics["top1:neutral_text"] = (
                metrics["top1:neutral_text"], metrics["kld:neutral_text"])
        result = self.gate(mutate=swap)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MISMATCH", result.stdout)

    def test_corpus_key_must_match_header(self):
        def alias(entries):
            metrics = entries[2]["m"]
            metrics["kld:neutral"] = metrics.pop("kld:neutral_text")
        self.assertNotEqual(self.gate(mutate=alias).returncode, 0)

    def test_corpus_column_swap_fails(self):
        def swap(entries):
            metrics = entries[2]["m"]
            metrics["kld:code"], metrics["kld:agentic"] = metrics["kld:agentic"], metrics["kld:code"]
        self.assertNotEqual(self.gate(mutate=swap).returncode, 0)

    def test_undocumented_order_fails(self):
        self.assertNotEqual(self.gate(intro="Each cell contains two fidelity measurements.").returncode, 0)

    def test_reversed_documented_order_fails(self):
        self.assertNotEqual(self.gate(intro="Each cell shows top-1 agreement, then mean KL lower bound in nats.").returncode, 0)

    def test_different_cell_delimiter_fails(self):
        self.assertNotEqual(self.gate(delimiter=" / ").returncode, 0)

    def test_order_description_must_precede_table(self):
        self.assertNotEqual(self.gate(trailing_intro=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
