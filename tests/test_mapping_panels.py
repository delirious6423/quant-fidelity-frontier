#!/usr/bin/env python3
"""Mapping regressions using source functions without loading a model's snapshot index.

Run: python3 tests/test_mapping_panels.py
"""
import ast
import contextlib
import copy
import glob
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import types
import unittest
import warnings
from unittest.mock import patch


CODE = Path(__file__).resolve().parents[1]


def source_functions(relative, names, namespace):
    """Execute the real functions; skip module-level model reads and CLI actions."""
    path = CODE / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            assigned = {t.id for t in node.targets if isinstance(t, ast.Name)} & set(names)
            if assigned:
                selected.append(node)
                found.update(assigned)
    assert found == set(names), (relative, set(names) - found)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


AX = types.ModuleType("autoextract")
AX.__dict__["re"] = re
source_functions("pipeline/autoextract.py", {"NUM", "numtoks", "clean", "gt_family"}, AX.__dict__)
MAP = source_functions("pipeline/qcl_map.py", {"clean_raw", "clean_label", "apply", "exact_in", "QUAL", "APPROX"}, {"re": re, "A": AX})
WATCH = source_functions("pipeline/qcl_watch.py", {"agree", "panel_key", "write"}, {"re": re, "json": json})


def mapping(entries, **extra):
    return dict(repo="publisher/test-GGUF", entries=entries, method={"reference": "native MXFP4"}, **extra)


def entry(metrics, **extra):
    return dict(role="own", name="Q4", metrics=metrics, **extra)


def applied(metrics, panel="Current", kind="variants", **extra):
    out = dict(repo="publisher/test-GGUF", variants=[], ladder=[], cross=[], flags=[], methodology={})
    out[kind] = [dict(name="Q4", panel=panel, m=metrics, **extra)]
    return out


class MappingPanels(unittest.TestCase):
    def test_size_only_source_label_and_explicit_measured_survive(self):
        cands = {"c1": dict(raw="214 GB", L=8, label="this pack, 1.59 bpw", header="Size GB")}
        e = dict(role="cross", name="Q4", metrics={}, size="c1", unit="GB", basis="total", measured=True)
        row = MAP["apply"](mapping([e]), cands)["cross"][0]
        self.assertEqual(row["row_label"], ["this pack, 1.59 bpw"])
        self.assertIs(row["measured"], True)

    def agree(self, a, b):
        with patch.dict(sys.modules, {"autoextract": AX}):
            return WATCH["agree"](a, b)

    def test_inline_code_in_source_row_label_is_preserved(self):
        self.assertEqual(MAP["clean_label"]("Call-or-answer (tokens after `</think>`)"), "Call-or-answer (tokens after </think>)")

    def test_candidates_copy_published_strings_without_rounding(self):
        cands = {
            "c1": dict(raw=" **0.020218 ± 0.000145** ", L=41, label="**Q4**"),
            "c2": dict(raw="−0.1270%", L=42, label="Q4"),
            "c3": dict(raw="`99.0100%`", L=42, label="Q4"),
        }
        result = MAP["apply"](mapping([entry({"mean_kld": "c1", "ppl_delta_pct": "c2", "top1_pct": "c3", "ppl": "invented"})]), cands)
        row = result["variants"][0]
        self.assertEqual(row["m"], {"mean_kld": "0.020218 ± 0.000145", "ppl_delta_pct": "−0.1270%", "top1_pct": "99.0100%"})
        self.assertEqual((row["L"], row["L_extra"]), (41, [42]))
        self.assertEqual(row["row_label"], ["Q4"])

    def test_entry_panel_and_methodology_survive_application_and_agreement(self):
        method = {"reference": "native MXFP4", "dataset": "domain-equal", "ctx": "4096", "L": [12]}
        cands = {"c1": dict(raw="0.012300", L=14, label="Q4")}
        for role, kind in (("own", "variants"), ("ladder", "ladder"), ("cross", "cross"), ("quoted", "cross")):
            with self.subTest(role=role):
                e = dict(role=role, name="Q4", metrics={"mean_kld": "c1"}, panel="Current full-vocabulary", methodology=method)
                result = MAP["apply"](mapping([e]), cands)
                agreed, held = self.agree(result, result)
                self.assertEqual(held, [])
                self.assertEqual(agreed[kind][0]["panel"], e["panel"])
                self.assertEqual(agreed[kind][0]["methodology"], method)
                if role == "quoted":
                    self.assertIs(agreed[kind][0]["measured"], False)

    def test_identical_values_in_different_panels_do_not_agree(self):
        result, held = self.agree(applied({"mean_kld": "0.0123"}), applied({"mean_kld": "0.0123"}, panel="Legacy top-k"))
        self.assertEqual(result["variants"], [])
        self.assertEqual(held, [("publisher/test-GGUF", "Q4", "mean_kld", "0.0123")])

    def test_panel_titles_compare_without_heading_marks_or_case(self):
        result, held = self.agree(applied({"mean_kld": "0.0123"}, panel="## Benchmarks"), applied({"mean_kld": "0.0123"}, panel="benchmarks"))
        self.assertEqual((len(result["variants"]), held), (1, []))
        self.assertEqual(result["variants"][0]["panel"], "## Benchmarks")

    def test_a_run_naming_no_panel_agrees_and_the_named_panel_is_kept(self):
        for a_panel, b_panel in (("Legacy top-k", None), (None, "Legacy top-k")):
            with self.subTest(a=a_panel, b=b_panel):
                result, held = self.agree(applied({"mean_kld": "0.0123"}, panel=a_panel), applied({"mean_kld": "0.0123"}, panel=b_panel))
                self.assertEqual(held, [])
                self.assertEqual(result["variants"][0]["panel"], "Legacy top-k")

    def test_unpanelled_entry_whose_values_split_across_named_panels_is_held(self):
        a = applied({"mean_kld": "0.0123", "top1_pct": "99.1%"}, panel=None)
        b = applied({"mean_kld": "0.0123"}, panel="Legacy top-k")
        b["variants"].append(dict(name="Q4", panel="Current", m={"top1_pct": "99.1%"}))
        result, held = self.agree(a, b)
        self.assertEqual(result["variants"], [])
        self.assertEqual(len(held), 2)

    def test_identical_values_from_different_domains_do_not_agree(self):
        for prefix in ("kld", "top1", "ppl", "kl_mnats"):
            with self.subTest(prefix=prefix):
                result, held = self.agree(applied({prefix + ":code": "0.0123"}), applied({prefix + ":english": "0.0123"}))
                self.assertEqual(result["variants"], [])
                self.assertEqual(len(held), 1)

    def test_same_panel_and_domain_agree_despite_different_variant_names(self):
        a = applied({"kld:code": "0.012300", "top1:english": "99.10%"})
        b = applied({"kld:code": "0.012300", "top1:english": "99.10%"})
        b["variants"][0]["name"] = "Q4_K_M"
        result, held = self.agree(a, b)
        self.assertEqual(result["variants"], a["variants"])
        self.assertEqual(held, [])
        b["variants"][0]["m"]["kld:code"] = "0.0123"
        result, held = self.agree(a, b)
        self.assertEqual(result["variants"][0]["m"], {"top1:english": "99.10%"})
        self.assertEqual(len(held), 1)

    def test_nats_millinats_and_domain_aggregates_are_not_interchangeable(self):
        for left, right in (("mean_kld", "mean_kl_mnats"), ("mean_kl_mnats", "mean_kld"),
                            ("kld:code", "kl_mnats:code"), ("mean_kld", "kld:code")):
            with self.subTest(left=left, right=right):
                result, held = self.agree(applied({left: "0.0123"}), applied({right: "0.0123"}))
                self.assertEqual(result["variants"], [])
                self.assertEqual(len(held), 1)

    def test_quoted_values_cannot_confirm_measured_cross_values(self):
        measured = applied({"mean_kld": "0.0123"}, kind="cross")
        quoted = applied({"mean_kld": "0.0123"}, kind="cross", measured=False)
        for a, b in ((measured, quoted), (quoted, measured)):
            with self.subTest(first_measured=a["cross"][0].get("measured", True)):
                result, held = self.agree(a, b)
                self.assertEqual(result["cross"], [])
                self.assertEqual(len(held), 1)

    def test_prose_values_attach_only_to_the_named_panel_and_exact_source(self):
        m = mapping([entry({}, panel="Legacy"), entry({}, panel="Current")])
        m["prose"] = [
            dict(entry="Q4", panel="Current", key="mean_kld", L=2, raw="0.000123 ± 0.000001"),
            dict(entry="Q4", panel="Current", key="top1_pct", L=2, raw="99.9999%"),
            dict(entry="Q4", key="ppl", L=2, raw="0.000123"),
        ]
        with patch.dict(MAP, {"A_lines": lambda _: ["# Current", "KL 0.000123 ± 0.000001 measured."]}):
            result = MAP["apply"](m, {})
        self.assertEqual(result["variants"][0]["m"], {})
        self.assertEqual(result["variants"][1]["m"], {"mean_kld": "0.000123 ± 0.000001"})
        self.assertEqual(result["variants"][1]["L"], 2)

    def test_either_run_can_mark_modified_weights(self):
        for first, second in ((True, False), (False, True), (None, True), (True, None)):
            with self.subTest(first=first, second=second):
                a = applied({"mean_kld": "0.0123"})
                b = applied({"mean_kld": "0.0123"})
                a["weights_modified"], b["weights_modified"] = first, second
                result, held = self.agree(a, b)
                self.assertIs(result["weights_modified"], True)
                self.assertEqual(held, [])

    def test_modified_weights_survive_apply_agree_and_generated_verified_file(self):
        cands = {"c1": dict(raw="0.0123", L=14, label="Q4")}
        with tempfile.TemporaryDirectory(prefix="qff-mapping-panels-") as temp:
            root = Path(temp)
            (root / "snapshots").mkdir()
            (root / "snapshots/index.json").write_text(json.dumps({"repos": {"publisher/test-GGUF": {"sha": "a" * 40}}}))
            namespace = dict(WATCH, ROOT=str(root), OUT=str(root / "machine.py"),
                             F=types.SimpleNamespace(fmt_of=lambda *_: "GGUF"))
            # Functions retain their source namespace, so load write with this temporary destination.
            source_functions("pipeline/qcl_watch.py", {"write"}, namespace)
            for modified in (True, False, None):
                with self.subTest(modified=modified):
                    extra = {} if modified is None else {"weights_modified": modified}
                    a = MAP["apply"](mapping([entry({"mean_kld": "c1"})], **extra), cands)
                    agreed, held = self.agree(a, a)
                    self.assertEqual(held, [])
                    self.assertIs(agreed["weights_modified"], modified)
                    base = [{"repo": a["repo"], "weights_modified": True}]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ResourceWarning)
                        namespace["write"]({a["repo"]: agreed}, base)
                    generated = {}
                    exec((root / "machine.py").read_text(), generated)
                    written = generated["R"][0]
                    self.assertIs(written["weights_modified"], True if modified is None else modified)
                    self.assertEqual(written["sha"], "a" * 40)
                    self.assertEqual(written["variants"][0]["m"], {"mean_kld": "0.0123"})


class ResolverOverlay(unittest.TestCase):
    def fixture(self, root):
        repo = "publisher/test-GGUF"
        name = repo.replace("/", "__")
        for path in ("snapshots", "maps/cands", "maps/resolved", "audit"):
            (root / path).mkdir(parents=True)
        source = b"# Current\n| Q4 | 0.012300 |\n"
        (root / "card.md").write_bytes(source)
        source_hash = hashlib.sha256(source).hexdigest()
        (root / "snapshots/index.json").write_text(json.dumps({"repos": {repo: {"file": "card.md", "sha256": source_hash}}}))
        cands = {"c1": {"raw": "0.012300", "L": 2, "label": "Q4"}}
        (root / f"maps/cands/{name}.cands.json").write_text(json.dumps(cands))
        map_path = root / f"maps/resolved/{name}.map.json"
        map_path.write_text(json.dumps(mapping([entry({"mean_kld": "c1"}, panel="Current")])) + "\n")
        pin = {"source_sha256": source_hash, "mapping_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
               "resolved_by": "source review", "reason": "panel headings reconciled"}
        manifest = root / "maps/resolved/manifest.json"
        manifest.write_text(json.dumps({"repos": {repo: pin}}))
        runs = {"A": {repo: applied({"mean_kld": "0.012300"})}, "B": {repo: applied({"mean_kld": "0.0999"})}}
        writes = []
        namespace = {
            "ROOT": str(root), "glob": glob, "hashlib": hashlib, "json": json, "os": os, "re": re, "time": time,
            "Q": types.SimpleNamespace(apply=MAP["apply"]),
            "applied_from": lambda directory: copy.deepcopy(runs[directory]),
            "load_verified": lambda: [], "write": lambda entries, _: writes.append(copy.deepcopy(entries)),
            "failing": lambda _: set(),
            "build_failing": lambda _: set(),
        }
        source_functions("pipeline/qcl_watch.py", {"agree", "panel_key", "resolved_from", "merge"}, namespace)
        return repo, map_path, manifest, pin, namespace, writes

    def merge(self, namespace, root):
        output = io.StringIO()
        with patch.dict(sys.modules, {"autoextract": AX}), contextlib.redirect_stdout(output), warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            namespace["merge"](["A", "B"])
        return json.loads((root / "audit/review_queue.json").read_text()), output.getvalue()

    def test_current_pins_apply_after_agreement_and_preserve_disagreement_provenance(self):
        with tempfile.TemporaryDirectory(prefix="qff-resolver-") as temp:
            root = Path(temp)
            repo, path, manifest, pin, namespace, writes = self.fixture(root)
            original_map = path.read_bytes()
            audit, output = self.merge(namespace, root)
            self.assertEqual(writes[-1][repo]["variants"][0]["m"], {"mean_kld": "0.012300"})
            self.assertEqual(audit["disagreements"], [])
            self.assertEqual(audit["resolved_repos"], 1)
            resolved = audit["resolved_disagreements"]
            self.assertEqual(len(resolved), 1)
            self.assertEqual({key: resolved[0][key] for key in ("repo", "entry", "key", "raw")},
                             dict(repo=repo, entry="Q4", key="mean_kld", raw="0.012300"))
            self.assertEqual(resolved[0]["resolver"], {**pin, "mapping_file": str(path.relative_to(root))})
            self.assertIn("resolved 1 repos", output)
            self.assertEqual(path.read_bytes(), original_map)

    def test_stale_source_or_mapping_and_missing_pins_leave_disagreements_held(self):
        for stale in ("source", "mapping", "missing_source_pin", "missing_manifest", "missing_map"):
            with self.subTest(stale=stale), tempfile.TemporaryDirectory(prefix="qff-resolver-") as temp:
                root = Path(temp)
                repo, path, manifest, pin, namespace, writes = self.fixture(root)
                if stale == "source":
                    index = root / "snapshots/index.json"
                    data = json.loads(index.read_text())
                    data["repos"][repo]["sha256"] = hashlib.sha256(b"new source").hexdigest()
                    index.write_text(json.dumps(data))
                elif stale == "mapping":
                    path.write_bytes(path.read_bytes() + b" ")
                elif stale == "missing_source_pin":
                    del pin["source_sha256"]
                    manifest.write_text(json.dumps({"repos": {repo: pin}}))
                elif stale == "missing_manifest": manifest.unlink()
                else: path.unlink()
                audit, output = self.merge(namespace, root)
                self.assertEqual(writes[-1][repo]["variants"], [])
                self.assertEqual(audit["resolved_disagreements"], [])
                self.assertEqual(audit["resolved_repos"], 0)
                self.assertEqual(audit["disagreements"], [dict(repo=repo, entry="Q4", key="mean_kld", raw="0.012300")])
                self.assertIn("resolved 0 repos", output)

    def test_resolved_entries_still_pass_through_quarantine(self):
        with tempfile.TemporaryDirectory(prefix="qff-resolver-") as temp:
            root = Path(temp)
            repo, _, _, _, namespace, writes = self.fixture(root)
            checks = iter([{(repo, "variants", "Q4")}, set()])
            namespace["failing"] = lambda _: next(checks)
            audit, _ = self.merge(namespace, root)
            self.assertEqual(writes[0][repo]["variants"][0]["m"], {"mean_kld": "0.012300"})
            self.assertEqual(writes[-1][repo]["variants"], [])
            self.assertEqual(audit["quarantined"], [dict(repo=repo, kind="variants", entry="Q4", reason="failed a gate")])
            self.assertEqual(len(audit["resolved_disagreements"]), 1)

    def test_resolver_cannot_add_repo_without_independent_agreement_inputs(self):
        with tempfile.TemporaryDirectory(prefix="qff-resolver-") as temp:
            _, _, _, _, namespace, _ = self.fixture(Path(temp))
            self.assertEqual(namespace["resolved_from"]([]), ({}, {}))



class TieBreaker(unittest.TestCase):
    """Run C publishes a held run-A value only when C filed the same string in the same role and family, for the
    card bytes it was pinned to; values A and B agreed on are never changed."""
    def fixture(self, root, c_value="0.012300", pinned=True):
        repo = "publisher/test-GGUF"
        for path in ("snapshots", "maps/C", "audit"): (root / path).mkdir(parents=True)
        source_hash = hashlib.sha256(b"card").hexdigest()
        (root / "snapshots/index.json").write_text(json.dumps({"repos": {repo: {"file": "card.md", "sha256": source_hash}}}))
        (root / "maps/mapped_c.json").write_text(json.dumps({repo: source_hash if pinned else "0" * 64}))
        runs = {"A": {repo: applied({"mean_kld": "0.012300", "top1_pct": "99.1%"})},
                "B": {repo: applied({"top1_pct": "99.1%", "mean_kld": "0.0999"})},
                "C": {repo: applied({"mean_kld": c_value})}}
        writes = []
        namespace = {
            "ROOT": str(root), "glob": glob, "hashlib": hashlib, "json": json, "os": os, "re": re, "time": time,
            "Q": types.SimpleNamespace(apply=MAP["apply"]),
            "applied_from": lambda directory: copy.deepcopy(runs[os.path.basename(directory)]),
            "load_verified": lambda: [], "write": lambda entries, _: writes.append(copy.deepcopy(entries)),
            "failing": lambda _: set(), "build_failing": lambda _: set(),
        }
        source_functions("pipeline/qcl_watch.py", {"agree", "panel_key", "resolved_from", "merge"}, namespace)
        return repo, namespace, writes

    def merge(self, namespace, root):
        output = io.StringIO()
        with patch.dict(sys.modules, {"autoextract": AX}), contextlib.redirect_stdout(output), warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            namespace["merge"](["A", "B"])
        return json.loads((root / "audit/review_queue.json").read_text()), output.getvalue()

    def test_c_confirming_a_held_value_publishes_it(self):
        with tempfile.TemporaryDirectory(prefix="qff-tb-") as temp:
            root = Path(temp); repo, ns, writes = self.fixture(root)
            audit, output = self.merge(ns, root)
            self.assertEqual(writes[-1][repo]["variants"][0]["m"], {"top1_pct": "99.1%", "mean_kld": "0.012300"})
            self.assertEqual(audit["disagreements"], [])
            self.assertEqual(audit["tiebroken"], [dict(repo=repo, entry="Q4", key="mean_kld", raw="0.012300")])
            self.assertIn("tie-broken 1 values", output)

    def test_c_disagreeing_or_stale_keeps_the_value_held(self):
        for kw in ({"c_value": "0.0999"}, {"pinned": False}):
            with self.subTest(**kw), tempfile.TemporaryDirectory(prefix="qff-tb-") as temp:
                root = Path(temp); repo, ns, writes = self.fixture(root, **kw)
                audit, _ = self.merge(ns, root)
                self.assertEqual(writes[-1][repo]["variants"][0]["m"], {"top1_pct": "99.1%"})
                self.assertEqual(audit["disagreements"], [dict(repo=repo, entry="Q4", key="mean_kld", raw="0.012300")])
                self.assertEqual(audit["tiebroken"], [])

    def test_gate_failure_after_tiebreak_drops_only_the_tiebroken_values(self):
        with tempfile.TemporaryDirectory(prefix="qff-tb-") as temp:
            root = Path(temp); repo, ns, writes = self.fixture(root)
            calls = []
            def failing(entries):   # the entry fails while it holds the tie-broken KLD
                calls.append(1)
                v = entries[repo]["variants"]
                return {(repo, "variants", "Q4")} if v and "mean_kld" in v[0]["m"] else set()
            ns["failing"] = failing
            audit, output = self.merge(ns, root)
            self.assertEqual(writes[-1][repo]["variants"][0]["m"], {"top1_pct": "99.1%"})
            self.assertEqual(audit["quarantined"], [])
            self.assertEqual(audit["tiebroken"], [])
            self.assertEqual(audit["disagreements"], [dict(repo=repo, entry="Q4", key="mean_kld", raw="0.012300")])

if __name__ == "__main__":
    unittest.main(verbosity=2)
