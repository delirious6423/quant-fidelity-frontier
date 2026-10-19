#!/usr/bin/env python3
"""Verified cross-publisher views retain strict complete-link compatibility."""
import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comparison import classify_group
from verified_protocol_sets import add_verified_cross_publisher_sets
from test_comparison import measurement, assertion, SOURCE


def point(identity, owner):
    result = measurement(evaluator=owner)
    result.update(measurement_id=identity, qid="same-quant", basis="download", x=10,
                  repo="quantizer/model", src="https://example.org/" + identity,
                  kind="own", is_ref=False)
    return result


def view(identity, record, **axis):
    return {"id": identity, "title": identity, "reference": "Published reference description",
            "data": "Published dataset description", "refs": {}, "metrics": ["kld"],
            "x": {"field": "size_gb", "unit": "GB", "label": "Download size", **axis},
            "points": [record]}


class VerifiedProtocolSetTests(unittest.TestCase):
    def test_two_publishers_same_verified_protocol_form_comparison(self):
        a, b = point("a", "Alice/model"), point("b", "Bob/model")
        sets = [view("a-source", a), view("b-source", b)]
        before = copy.deepcopy(sets)
        result = add_verified_cross_publisher_sets(sets)
        self.assertEqual(sets[:2], before)
        self.assertEqual(len(result), 1)
        group = result[0]
        self.assertTrue(group["id"].startswith("protocol-"))
        self.assertTrue(group["title"].startswith("Verified protocol · "))
        self.assertEqual(group["metrics"], ["kld"])
        self.assertEqual(group["refs"], {})
        self.assertEqual(len({p["qid"] for p in group["points"]}), 2)
        self.assertEqual(group["reference"], before[0]["reference"])
        for original, added in zip((a, b), group["points"]):
            self.assertEqual({k: v for k, v in added.items() if k != "qid"}, {k: v for k, v in original.items() if k != "qid"})
        self.assertTrue(classify_group(group["points"], "kld")["ranking_permitted"])

    def test_explicit_unknown_evaluator_cannot_fall_back_to_claim_publisher(self):
        a, b = point("a", "Alice/model"), point("b", "Bob/model")
        b["measurement_owner"] = None
        b["measured_by"] = "Copier/model"
        b["claim_provenance"] = {"kind":"copied", "original_attribution":{"verified":True,"evidence":[SOURCE]}}
        self.assertEqual(add_verified_cross_publisher_sets([view("a",a),view("b",b)]), [])

    def test_same_measurement_publisher_different_cards_not_cross_publisher(self):
        sets = [view("a", point("a", "Alice/model-a")), view("b", point("b", "Alice/model-b"))]
        self.assertEqual(add_verified_cross_publisher_sets(sets), [])

    def test_quantization_publishers_do_not_substitute_for_measurement_publishers(self):
        a, b = point("a", "Evaluator/model"), point("b", "Evaluator/model")
        a["quantization_owner"], b["quantization_owner"] = "quant-a", "quant-b"
        self.assertEqual(add_verified_cross_publisher_sets([view("a", a), view("b", b)]), [])

    def test_unknown_and_conflicting_protocols_are_not_grouped(self):
        for mode in ("missing", "mismatch", "contradictory"):
            a, b = point("a", "Alice/model"), point("b", "Bob/model")
            if mode == "missing":
                b["protocol"].pop("dataset")
            elif mode == "mismatch":
                b["protocol"]["dataset"] = assertion("different")
            else:
                b["protocol"]["dataset"] = [assertion("a"), assertion("b")]
            with self.subTest(mode=mode):
                self.assertEqual(add_verified_cross_publisher_sets([view("a", a), view("b", b)]), [])

    def test_complete_link_cover_never_uses_transitive_compatibility(self):
        records = [point(name, name + "/model") for name in ("A", "B", "C")]
        equivalents = [{"condition": "implementation", "values": list(pair), "evidence": [SOURCE]} for pair in (("A", "B"), ("B", "C"))]
        for name, record in zip(("A", "B", "C"), records):
            record["protocol"]["implementation"] = assertion(name)
            record["protocol"]["equivalences"] = equivalents
        groups = add_verified_cross_publisher_sets([view(name, record) for name, record in zip(("A", "B", "C"), records)])
        self.assertEqual({frozenset(p["measurement_id"] for p in group["points"]) for group in groups}, {frozenset(("A", "B")), frozenset(("B", "C"))})
        self.assertTrue(all(classify_group(group["points"], "kld")["ranking_permitted"] for group in groups))

    def test_size_axis_unit_field_and_basis_each_separate_groups(self):
        for boundary in ("unit", "field", "basis"):
            a, b = point("a", "Alice/model"), point("b", "Bob/model")
            left, right = view("a", a), view("b", b)
            if boundary == "basis":
                b["basis"] = "weights"
            else:
                right["x"][boundary] = "GiB" if boundary == "unit" else "bpw"
            with self.subTest(boundary=boundary):
                self.assertEqual(add_verified_cross_publisher_sets([left, right]), [])

    def test_null_or_missing_size_basis_does_not_establish_compatibility(self):
        for basis in (None, "", "unknown", "Not stated", "UNSPECIFIED", "not documented", " n/a "):
            a, b = point("a", "Alice/model"), point("b", "Bob/model")
            a["basis"] = b["basis"] = basis
            self.assertEqual(add_verified_cross_publisher_sets([view("a", a), view("b", b)]), [])

    def test_metrics_are_evaluated_separately(self):
        a, b = point("a", "Alice/model"), point("b", "Bob/model")
        for record in (a, b):
            record["m"]["ppl"] = {"v": 4}
        b["protocol"]["metrics"] = {"ppl": {"dataset": assertion("different")}}
        source = [view("a", a), view("b", b)]
        for item in source:
            item["metrics"].append("ppl")
        groups = add_verified_cross_publisher_sets(source)
        self.assertEqual([group["metrics"] for group in groups], [["kld"]])

    def test_references_and_unverified_copied_claims_excluded(self):
        for mode in ("reference", "copied", "quoted-without-provenance"):
            a, b = point("a", "Alice/model"), point("b", "Bob/model")
            if mode == "reference":
                b["is_ref"] = True
            elif mode == "copied":
                b["claim_provenance"] = {"kind": "copied"}
            else:
                b["kind"] = "quoted"
                b.pop("claim_provenance", None)
            with self.subTest(mode=mode):
                self.assertEqual(add_verified_cross_publisher_sets([view("a", a), view("b", b)]), [])

    def test_verified_original_attribution_can_admit_copied_measurement(self):
        a, b = point("a", "Alice/model"), point("b", "Bob/model")
        b["claim_provenance"] = {"kind": "copied", "original_attribution": {"verified": True, "evidence": [SOURCE]}}
        self.assertEqual(len(add_verified_cross_publisher_sets([view("a", a), view("b", b)])), 1)

    def test_duplicate_measurements_deduplicated_and_conflicts_excluded(self):
        a, b = point("a", "Alice/model"), point("b", "Bob/model")
        group = add_verified_cross_publisher_sets([view("a", a), view("b", b), view("a-copy", copy.deepcopy(a))])[0]
        self.assertEqual(len(group["points"]), 2)
        bad_copy = copy.deepcopy(a)
        bad_copy["m"]["kld"]["v"] = 8
        self.assertEqual(add_verified_cross_publisher_sets([view("a", a), view("b", b), view("bad", bad_copy)]), [])

    def test_overview_does_not_supply_evidence_and_generation_is_idempotent(self):
        a, b = point("a", "Alice/model"), point("b", "Bob/model")
        self.assertEqual(add_verified_cross_publisher_sets([view("all", a), view("b", b)]), [])
        sets = [view("a", a), view("b", b)]
        self.assertEqual(len(add_verified_cross_publisher_sets(sets)), 1)
        self.assertEqual(add_verified_cross_publisher_sets(sets), [])

    def test_group_ids_deterministic_under_source_reordering(self):
        source = [view("a", point("a", "Alice/model")), view("b", point("b", "Bob/model"))]
        first = add_verified_cross_publisher_sets(copy.deepcopy(source))
        second = add_verified_cross_publisher_sets(copy.deepcopy(source[::-1]))
        self.assertEqual(first, second)

    def test_current_qwen_reviewed_protocols_do_not_create_false_cross_group(self):
        registry = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protocol_evidence", "qwen3.8-27b.json")
        with open(registry) as handle:
            bindings = json.load(handle)["bindings"]
        sources = []
        for i, binding in enumerate(bindings):
            record = point(str(i), binding["measurement_repo"])
            record["protocol"] = binding["protocol"]
            sources.append(view(str(i), record))
        self.assertEqual(add_verified_cross_publisher_sets(sources), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
