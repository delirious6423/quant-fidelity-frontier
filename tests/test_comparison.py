#!/usr/bin/env python3
"""Regression cases for the evidence-only comparison contract.

Run with ``python -m unittest discover -s tests -p test_comparison.py -v``.
Synthetic citations below are fixtures, not claims about measured models.
"""
import copy
import json
import os
import random
import sys
import unittest
from unittest.mock import patch
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web'))   # site generators live in web/
from comparison import CONDITIONS, classify_pair, classify_group, classify_by_metric, summarize_comparisons


SOURCE = {"url": "https://example.org/experiment/1", "quote": "Synthetic protocol fixture"}


def assertion(value):
    return {"value": value, "evidence": [dict(SOURCE)]}


def measurement(quant_owner="publisher-a", evaluator="evaluator-a"):
    values = {
        "base_model": "Qwen/Qwen3.8-27B@revision-1",
        "reference_checkpoint": "reference-bf16-sha256-fixture",
        "dataset": {"id": "fixture", "revision": "rev-1", "split": "test"},
        "token_selection": {"token_ids_sha256": "fixture-hash", "positions": "all non-padding"},
        "metric_definition": {"name": "KLD(reference || quant)", "direction": "lower", "unit": "nats", "aggregation": "mean over evaluated tokens"},
        "context_length": 4096,
        "implementation": "evaluator@commit-1",
        "settings": {"batch": 1, "attention": "full", "dtype": "float32"},
    }
    return {"quantization_owner": quant_owner, "measurement_owner": evaluator,
            "m": {"kld": {"v": 0.1}}, "protocol": {k: assertion(v) for k, v in values.items()}}


class ComparisonTests(unittest.TestCase):
    def assert_class(self, left, right, expected, metric="kld"):
        result = classify_pair(left, right, metric)
        self.assertEqual(result["classification"], expected, result)
        self.assertEqual(result["ranking_permitted"], expected == "same_protocol")
        return result

    def test_empty_and_placeholder_identities_are_unknown(self):
        for value in ({}, [], False, True, "unknown", "Not stated", "   ", float("nan")):
            with self.subTest(value=value):
                a, b = measurement(), measurement()
                a["protocol"]["dataset"] = assertion(value)
                b["protocol"]["dataset"] = assertion(value)
                self.assert_class(a, b, "different_or_unverified_protocol")

    def test_unknown_optimization_is_not_comparable(self):
        for value in ({"optimization": "unknown"}, {}, "unverified"):
            a, b = measurement(), measurement()
            for record in (a, b):
                record["protocol"]["metric_definition"]["value"]["direction"] = value
            self.assert_class(a, b, "different_or_unverified_protocol")

    def test_invalid_context_length_is_unknown(self):
        for value in (0, -1, "unknown", "4096", False):
            a, b = measurement(), measurement()
            for record in (a, b):
                record["protocol"]["context_length"] = assertion(value)
            self.assert_class(a, b, "different_or_unverified_protocol")

    def test_same_publisher_same_protocol(self):
        result = self.assert_class(measurement(), measurement(), "same_protocol")
        self.assertEqual(result["verified_matching_conditions"], list(CONDITIONS))
        self.assertTrue(result["evidence"])

    def test_different_publishers_same_verified_protocol(self):
        self.assert_class(measurement(), measurement("publisher-b", "evaluator-b"), "same_protocol")

    def test_one_evaluator_multiple_quantization_publishers(self):
        records = [measurement(owner, "AtomicChat-fixture") for owner in ("a", "b", "c")]
        result = classify_group(records, "kld")
        self.assertTrue(result["ranking_permitted"])
        self.assertEqual(result["comparable_pair_count"], 3)

    def test_same_evaluator_is_never_enough(self):
        a, b = measurement(), measurement()
        del a["protocol"]["dataset"]
        del b["protocol"]["dataset"]
        result = self.assert_class(a, b, "different_or_unverified_protocol")
        self.assertEqual(len(result["unknowns"]), 2)
        self.assertFalse(result["differences"])

    def test_same_publisher_different_protocols(self):
        a, b = measurement(), measurement()
        b["protocol"]["context_length"] = assertion(2048)
        result = self.assert_class(a, b, "different_or_unverified_protocol")
        self.assertEqual(result["differences"][0]["condition"], "context_length")
        self.assertFalse(result["unknowns"])

    def test_different_publishers_unknown_protocols(self):
        self.assert_class({"quantization_owner": "a"}, {"quantization_owner": "b"}, "different_or_unverified_protocol")

    def test_different_base_models(self):
        a, b = measurement(), measurement()
        b["protocol"]["base_model"] = assertion("other/model@revision")
        self.assert_class(a, b, "different_base_model")

    def test_different_reference_distributions(self):
        a, b = measurement(), measurement()
        b["protocol"]["reference_checkpoint"] = assertion("other-reference")
        self.assert_class(a, b, "different_base_model")

    def test_missing_reference_or_dataset_is_unknown_not_mismatch(self):
        for field in ("reference_checkpoint", "dataset"):
            with self.subTest(field=field):
                a, b = measurement(), measurement()
                del b["protocol"][field]
                result = self.assert_class(a, b, "different_or_unverified_protocol")
                self.assertFalse(result["differences"])
                self.assertIn(field, [x["condition"] for x in result["unknowns"]])

    def test_assertion_without_evidence_is_unknown(self):
        a, b = measurement(), measurement()
        b["protocol"]["dataset"]["evidence"] = []
        result = self.assert_class(a, b, "different_or_unverified_protocol")
        self.assertIn("lacks source evidence", result["unknowns"][0]["reason"])

    def test_evidence_requires_resolvable_url(self):
        for bad in ({"url": "javascript:alert(1)"}, {"path": "card.md"}, {"url": ""}):
            a, b = measurement(), measurement()
            b["protocol"]["dataset"]["evidence"] = [bad]
            self.assert_class(a, b, "different_or_unverified_protocol")

    def test_contradictory_metadata_never_selects_convenient_assertion(self):
        a, b = measurement(), measurement()
        b["protocol"]["dataset"] = [b["protocol"]["dataset"], assertion("contradictory dataset")]
        result = self.assert_class(a, b, "different_or_unverified_protocol")
        self.assertIn("Contradictory", result["differences"][0]["reason"])

    def test_contradictory_model_is_not_confirmed_cross_model_identity(self):
        a, b = measurement(), measurement()
        b["protocol"]["base_model"] = [assertion("model-a"), assertion("model-b")]
        self.assert_class(a, b, "different_or_unverified_protocol")

    def test_duplicate_assertions_are_not_contradictions(self):
        a, b = measurement(), measurement()
        b["protocol"]["dataset"] = [b["protocol"]["dataset"], copy.deepcopy(b["protocol"]["dataset"])]
        self.assert_class(a, b, "same_protocol")

    def test_copied_or_quoted_claims_need_original_attribution(self):
        for kind in ("copied", "quoted"):
            a, b = measurement(), measurement()
            b["claim_provenance"] = {"kind": kind}
            result = self.assert_class(a, b, "different_or_unverified_protocol")
            self.assertEqual(result["unknowns"][0]["condition"], "claim_provenance")
            b["claim_provenance"]["original_attribution"] = {"verified": True, "evidence": [SOURCE]}
            self.assert_class(a, b, "same_protocol")

    def test_unverified_attribution_cannot_unlock_copied_claim(self):
        a, b = measurement(), measurement()
        b["claim_provenance"] = {"kind": "quoted", "original_attribution": {"verified": False, "evidence": [SOURCE]}}
        self.assert_class(a, b, "different_or_unverified_protocol")

    def test_direct_equivalence_is_bound_to_its_endpoint_scopes(self):
        a, b = measurement(), measurement()
        a['protocol']['reference_checkpoint'] = dict(assertion('ref-a'), identity_scope='run-one')
        b['protocol']['reference_checkpoint'] = dict(assertion('ref-b'), identity_scope='run-two')
        entry = {'condition':'reference_checkpoint', 'values':['ref-a','ref-b'], 'evidence':[SOURCE]}
        a['protocol']['equivalences'] = [entry]
        self.assert_class(a,b,'different_or_unverified_protocol')
        entry['identity_scopes'] = ['run-one','run-two']
        self.assert_class(a,b,'same_protocol')
        b['protocol']['reference_checkpoint']['identity_scope']='unrelated-run'
        self.assert_class(a,b,'different_or_unverified_protocol')

    def test_evidenced_direct_implementation_equivalence(self):
        a, b = measurement(), measurement()
        b["protocol"]["implementation"] = assertion("evaluator@commit-2")
        self.assert_class(a, b, "different_or_unverified_protocol")
        a["protocol"]["equivalences"] = [{"condition": "implementation", "values": ["evaluator@commit-2", "evaluator@commit-1"], "evidence": [SOURCE]}]
        self.assert_class(a, b, "same_protocol")

    def test_equivalence_without_evidence_does_not_unlock(self):
        a, b = measurement(), measurement()
        b["protocol"]["implementation"] = assertion("other")
        a["protocol"]["equivalences"] = [{"condition": "implementation", "values": ["other", "evaluator@commit-1"]}]
        self.assert_class(a, b, "different_or_unverified_protocol")

    def test_equivalence_is_not_transitive_and_group_is_all_pairs(self):
        records = [measurement() for _ in range(3)]
        equivalences = [{"condition": "implementation", "values": list(pair), "evidence": [SOURCE]} for pair in (("A", "B"), ("B", "C"))]
        for record, implementation in zip(records, ("A", "B", "C")):
            record["protocol"]["implementation"] = assertion(implementation)
            record["protocol"]["equivalences"] = equivalences
        self.assert_class(records[0], records[1], "same_protocol")
        self.assert_class(records[1], records[2], "same_protocol")
        self.assert_class(records[0], records[2], "different_or_unverified_protocol")
        result = classify_group(records, "kld")
        self.assertFalse(result["ranking_permitted"])
        self.assertEqual((result["pair_count"], result["comparable_pair_count"]), (3, 2))

    def test_empty_and_singleton_do_not_establish_comparison(self):
        for records in ([], [measurement()]):
            result = classify_group(records, "kld")
            self.assertFalse(result["ranking_permitted"])
            self.assertEqual(result["pair_count"], 0)

    def test_metric_definitions_not_inferred_from_metric_alias(self):
        a, b = measurement(), measurement()
        del b["protocol"]["metric_definition"]
        self.assert_class(a, b, "different_or_unverified_protocol")

    def test_incomplete_metric_definition_is_unknown(self):
        a, b = measurement(), measurement()
        del b["protocol"]["metric_definition"]["value"]["aggregation"]
        self.assert_class(a, b, "different_or_unverified_protocol")

    def test_metric_direction_units_and_aggregation_must_match(self):
        for field, value in (("direction", "higher"), ("unit", "bits"), ("aggregation", "median")):
            a, b = measurement(), measurement()
            b["protocol"]["metric_definition"]["value"][field] = value
            self.assert_class(a, b, "different_or_unverified_protocol")

    def test_metric_overrides_are_resolved_per_metric(self):
        a, b = measurement(), measurement()
        for record in (a, b):
            record["m"]["ppl"] = {"v": 5}
        b["protocol"]["metrics"] = {"ppl": {"dataset": assertion("other")}}
        self.assert_class(a, b, "same_protocol", metric="kld")
        self.assert_class(a, b, "different_or_unverified_protocol", metric="ppl")
        results = classify_by_metric([a, b], ["kld", "ppl"])
        self.assertTrue(results["kld"]["ranking_permitted"])
        self.assertFalse(results["ppl"]["ranking_permitted"])

    def test_per_metric_population_excludes_missing_claims(self):
        a, b, c = measurement(), measurement(), {"m": {"kld": {"v": None}}}
        result = classify_by_metric([a, b, c], ["kld"])["kld"]
        self.assertTrue(result["ranking_permitted"])
        self.assertEqual(result["pair_count"], 1)

    def test_group_failure_examples_are_bounded_but_all_pairs_checked(self):
        records = [measurement() for _ in range(6)]
        for index, record in enumerate(records):
            record["protocol"]["context_length"] = assertion(index + 1)
        result = classify_group(records, "kld")
        self.assertEqual(result["pair_count"], 15)
        self.assertEqual(result["comparable_pair_count"], 0)
        self.assertEqual(len(result["failure_examples"]), 5)

    def test_input_records_are_not_mutated(self):
        a, b = measurement(), measurement()
        original = copy.deepcopy([a, b])
        classify_group([a, b], "kld")
        self.assertEqual([a, b], original)

    def test_same_scoped_reference_can_establish_relative_identity(self):
        a, b = measurement(), measurement()
        for record in (a, b):
            record["protocol"]["reference_checkpoint"]["identity_scope"] = "experiment-a"
        self.assert_class(a, b, "same_protocol")

    def test_distinct_experiment_references_are_unknown_not_different_models(self):
        a, b = measurement(), measurement()
        a["protocol"]["reference_checkpoint"]["identity_scope"] = "experiment-a"
        b["protocol"]["reference_checkpoint"] = assertion("other-scoped-reference")
        b["protocol"]["reference_checkpoint"]["identity_scope"] = "experiment-b"
        result = self.assert_class(a, b, "different_or_unverified_protocol")
        self.assertFalse(result["differences"])
        self.assertIn("cross-experiment", result["unknowns"][0]["reason"])

    def test_scoped_versus_absolute_identity_never_infers_mismatch(self):
        a, b = measurement(), measurement()
        a["protocol"]["reference_checkpoint"]["identity_scope"] = "experiment-a"
        b["protocol"]["reference_checkpoint"] = assertion("absolute-checkpoint-sha256")
        self.assert_class(a, b, "different_or_unverified_protocol")

    def test_metric_summary_requires_every_metric_to_be_eligible(self):
        a, b = measurement(), measurement()
        a["m"]["ppl"] = b["m"]["ppl"] = {"v": 3}
        b["protocol"]["metrics"] = {"ppl": {"context_length": assertion(512)}}
        summary = summarize_comparisons(classify_by_metric([a, b], ["kld", "ppl"]))
        self.assertFalse(summary["ranking_permitted"])
        self.assertEqual(summary["comparable_metrics"], ["kld"])
        self.assertEqual(summary["differences"][0]["metric"], "ppl")

    def test_metric_summary_can_permit_all_verified_metrics(self):
        summary = summarize_comparisons(classify_by_metric([measurement(), measurement()], ["kld"]))
        self.assertTrue(summary["ranking_permitted"])
        self.assertEqual(summary["classification"], "same_protocol")
        self.assertFalse(summarize_comparisons({})["ranking_permitted"])

    def test_malformed_nonempty_issues_are_not_silently_dropped(self):
        a, b = measurement(), measurement()
        b["protocol"]["issues"] = "failed source validation"
        self.assert_class(a, b, "different_or_unverified_protocol")

    def test_invalid_metric_overlay_fails_closed(self):
        a, b = measurement(), measurement()
        b["protocol"]["metrics"] = {"kld": "not a protocol object"}
        self.assert_class(a, b, "different_or_unverified_protocol")

    def test_weighted_profiles_match_naive_all_pairs_on_randomized_groups(self):
        """Compare every output field governed by group aggregation to a naive oracle."""
        def unique(values):
            return list({json.dumps(value, sort_keys=True): value for value in values}.values())
        rng = random.Random(90210)
        profiles = [measurement() for _ in range(7)]
        profiles[1]["protocol"].pop("dataset")
        profiles[2]["protocol"]["dataset"] = assertion("other-dataset")
        profiles[3]["protocol"]["base_model"] = assertion("other-model")
        profiles[4]["protocol"]["settings"] = [assertion({"batch": 1}), assertion({"batch": 2})]
        profiles[5]["claim_provenance"] = {"kind": "quoted"}
        profiles[6]["protocol"]["reference_checkpoint"]["identity_scope"] = "experiment-b"
        for trial in range(80):
            records = [copy.deepcopy(rng.choice(profiles)) for _ in range(rng.randint(2, 16))]
            before = copy.deepcopy(records)
            naive = [(i, j, classify_pair(left, right, "kld")) for (i, left), (j, right) in combinations(enumerate(records), 2)]
            result = classify_group(records, "kld")
            with self.subTest(trial=trial):
                permitted = sum(pair["ranking_permitted"] for _, _, pair in naive)
                expected_code = "same_protocol" if permitted == len(naive) else ("different_base_model" if any(pair["classification"] == "different_base_model" for _, _, pair in naive) else "different_or_unverified_protocol")
                self.assertEqual(result["classification"], expected_code)
                self.assertEqual(result["ranking_permitted"], permitted == len(naive))
                self.assertEqual(result["pair_count"], len(naive))
                self.assertEqual(result["comparable_pair_count"], permitted)
                for field in ("evidence", "differences", "unknowns", "limitations"):
                    self.assertEqual(result[field], unique([item for _, _, pair in naive for item in pair[field]]))
                self.assertEqual(result["verified_matching_conditions"], [condition for condition in CONDITIONS if all(condition in pair["verified_matching_conditions"] for _, _, pair in naive)])
                expected_failures = [{"pair": [i, j], "classification": pair["classification"], "differences": pair["differences"], "unknowns": pair["unknowns"]} for i, j, pair in naive if not pair["ranking_permitted"]][:5]
                self.assertEqual(result["failure_examples"], expected_failures)
                self.assertEqual(records, before)

    def test_duplicate_profiles_evaluated_once_per_directed_profile_pair(self):
        import comparison
        a, b, c = measurement(), measurement(), measurement()
        b["protocol"].pop("dataset")
        c["protocol"]["context_length"] = assertion(512)
        records = [copy.deepcopy(record) for record in (a, b, c) * 7]
        with patch("comparison.classify_pair", wraps=comparison.classify_pair) as call:
            result = classify_group(records, "kld")
            self.assertEqual(call.call_count, 9)  # 3 within-profile and 6 directed cross-profile checks.
            self.assertEqual(result["pair_count"], 210)
            self.assertEqual(result["comparable_pair_count"], 42)  # Two complete profiles; the missing-dataset profile fails closed.

    def test_source_validation_issue_overrides_apparent_matches(self):
        a, b = measurement(), measurement()
        b["protocol"]["issues"] = [{"condition": "dataset", "reason": "Pinned source content digest does not match"}]
        result = self.assert_class(a, b, "different_or_unverified_protocol")
        self.assertIn("digest", result["differences"][0]["reason"])

    def test_shared_and_metric_scoped_issues_both_preserved(self):
        a, b = measurement(), measurement()
        b["protocol"]["issues"] = [{"reason": "Shared source issue"}]
        b["protocol"]["metrics"] = {"kld": {"issues": [{"reason": "Metric source issue"}]}}
        result = self.assert_class(a, b, "different_or_unverified_protocol")
        self.assertEqual(len(result["differences"]), 2)

    def test_limitations_are_visible_without_negating_verified_relative_identity(self):
        a, b = measurement(), measurement()
        a["protocol"]["limitations"] = ["Absolute reference file hash was not published"]
        result = self.assert_class(a, b, "same_protocol")
        self.assertEqual(result["limitations"], a["protocol"]["limitations"])
        group = classify_group([a, b], "kld")
        self.assertEqual(group["limitations"], a["protocol"]["limitations"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
