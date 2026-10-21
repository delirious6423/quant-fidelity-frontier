"""Offline request schema, independent evidence and semantic-review regressions."""
import copy
import hashlib
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import runner_model as M


class ModelBoundary(unittest.TestCase):
    def setUp(self):
        self.raw = b"# Quant\n| Quant | Mean KLD |\n| --- | --- |\n| Q4 | 0.1250 |\nMeasured on corpus A.\nPerplexity is about 4.5.\n![results](chart.png)\n"
        self.source = {"repo": "publisher/model", "sha": "a" * 40,
                       "sha256": hashlib.sha256(self.raw).hexdigest()}
        self.candidates = {"c1": {"L": 4, "raw": "0.1250", "header": "Mean KLD", "label": "Q4"}}
        self.task = {"repo": self.source["repo"], "source_sha256": self.source["sha256"],
                     "candidates_sha256": M.digest(self.candidates), "candidates": self.candidates,
                     "prompt": "Trusted generated schema and digest"}
        self.bundle = M.prepare_mapping("base/model", self.source, self.raw, self.task, "A", "v1")
        self.response = {k: self.task[k] for k in ("repo", "source_sha256", "candidates_sha256")}
        self.response["entries"] = [{"role": "own", "name": "Q4", "metrics": {"mean_kld": "c1"}}]
        self.claim = {"repo": self.source["repo"], "kind": "own", "entry": "Q4",
                      "key": "mean_kld", "raw": "0.1250", "lines": [4],
                      "panel": "main", "unit": None, "methodology": {"dataset": "corpus A"}}

    def review(self, claim=None, **versions):
        return M.prepare_review("base/model", self.source, self.raw, claim or self.claim,
                                mapping_version=versions.get("mapping_version", "v1"),
                                extractor_version=versions.get("extractor_version", "v1"),
                                policy_version=versions.get("policy_version", M.REVIEW_POLICY))

    def test_complete_blind_source_in_each_run(self):
        for run in ("A", "B", "C"):
            bundle = M.prepare_mapping("base/model", self.source, self.raw, self.task, run, "v1")
            self.assertEqual(bundle["source"]["text"].encode(), self.raw)
            self.assertEqual(bundle["source"]["lines"][3]["text"], "| Q4 | 0.1250 |")
            self.assertIn("untrusted", bundle["instructions"])
            self.assertNotIn("response", bundle)
            self.assertEqual(M.validate_mapping(bundle, self.response), self.response)

    def test_response_detached_and_sources_pinned(self):
        validated = M.validate_mapping(self.bundle, self.response)
        validated["entries"].clear()
        self.assertEqual(len(self.response["entries"]), 1)
        for field in ("repo", "source_sha256", "candidates_sha256"):
            bad = copy.deepcopy(self.response)
            bad[field] = "incorrect"
            with self.assertRaises(ValueError):
                M.validate_mapping(self.bundle, bad)

    def test_json_duplicates_and_nonfinite_numbers_fail_closed(self):
        for raw in ('{"verdict":"ok","verdict":"wrong"}', '{"entries":[{"role":"own","role":"quoted"}]}',
                    '{"number":NaN}'):
            with self.assertRaises(ValueError):
                M.parse_response(raw)
        self.assertEqual(M.parse_response('{"verdict":"ok"}'), {"verdict": "ok"})

    def test_changed_source_or_candidates_cannot_prepare(self):
        with self.assertRaises(ValueError):
            M.prepare_mapping("base/model", self.source, self.raw + b"x", self.task, "A", "v1")
        self.task["candidates"]["c1"]["raw"] = "999"
        with self.assertRaises(ValueError):
            M.prepare_mapping("base/model", self.source, self.raw, self.task, "A", "v1")

    def test_mutated_bundle_rejected(self):
        self.bundle["source"]["text"] = "changed"
        with self.assertRaises(ValueError):
            M.validate_mapping(self.bundle, self.response)

    def test_invalid_schema_and_candidate_ids(self):
        bads = []
        for entry in ({"role": "invented"}, {"metrics": {"mean_kld": 0.125}},
                      {"metrics": {"mean_kld": "c99"}}, {"size": "c99"},
                      {"metrics": {"invalid\nmetric": "c1"}}, {"unit": "MB"},
                      {"role": "quoted", "measured": True}, {"L": 4}):
            bad = copy.deepcopy(self.response)
            bad["entries"][0].update(entry)
            bads.append(bad)
        bads += [{**self.response, "commands": ["sh"]}, {**self.response, "entries": None}]
        for bad in bads:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                M.validate_mapping(self.bundle, bad)

    def test_existing_prose_exception_preserves_qualifier(self):
        self.response["prose"] = [{"entry": "Q4", "key": "ppl", "L": 6, "raw": "about 4.5"}]
        M.validate_mapping(self.bundle, self.response)
        for raw in ("4.5", "4", "99"):
            self.response["prose"][0]["raw"] = raw
            with self.assertRaises(ValueError):
                M.validate_mapping(self.bundle, self.response)

    def test_existing_extended_metric_and_methodology_vocabulary(self):
        self.response["entries"][0]["metrics"] = {
            "ppl_delta_pct:wikitext-2": "c1", "component:engram_mean_cosine": "c1",
            "kld_999": "c1", "bench_floor:AI2D": "c1"}
        self.response["method"] = {"routing": "as stated", "batch": "as stated",
                                   "statistic": "as stated", "L": [5]}
        M.validate_mapping(self.bundle, self.response)

    def test_prose_cannot_replace_candidate_or_unknown_entry(self):
        self.response["prose"] = [{"entry": "Q4", "key": "ppl", "L": 4, "raw": "0.1250"}]
        with self.assertRaises(ValueError):
            M.validate_mapping(self.bundle, self.response)
        self.response["prose"] = [{"entry": "invented", "key": "ppl", "L": 6, "raw": "about 4.5"}]
        with self.assertRaises(ValueError):
            M.validate_mapping(self.bundle, self.response)

    def test_meaning_changes_invalidate_identical_number(self):
        old = self.review()["fingerprint"]
        for changes in ({"panel": "different"}, {"kind": "quoted"}, {"key": "ppl"},
                        {"unit": "mnats"}, {"methodology": {"dataset": "corpus B"}},
                        {"lines": [5]}, {"candidate_id": "c2"}):
            newer = self.review({**self.claim, **changes})
            self.assertNotEqual(old, newer["fingerprint"])
            self.assertEqual(newer["claim"]["raw"], "0.1250")
        for version in ("mapping_version", "extractor_version", "policy_version"):
            self.assertNotEqual(old, self.review(**{version: "v2"})["fingerprint"])
        self.source["sha"] = "b" * 40
        self.assertNotEqual(old, self.review()["fingerprint"])

    def test_partial_review_is_independent_and_duplicates_block(self):
        bundle = self.review()
        verdict = {"id": bundle["fingerprint"], "verdict": "ok", "reason": "Mean KLD table [L4]."}
        self.assertEqual(M.validate_review(bundle, verdict), verdict)
        for bad in ([], [verdict, verdict], {**verdict, "id": "q0"},
                    {**verdict, "verdict": "unsure"}, {**verdict, "reason": ""}):
            with self.assertRaises(ValueError):
                M.validate_review(bundle, bad)
        self.assertEqual(M.validate_review(bundle, {**verdict, "verdict": "wrong"})["verdict"], "wrong")
        self.assertEqual(self.review({**self.claim, "id": "q1"})["fingerprint"], bundle["fingerprint"])

    def test_sample_count_and_completed_evidence_reuse(self):
        bundles = [self.review({**self.claim, "entry": f"Q{i}"}) for i in range(31)]
        sample = M.sample_reviews(bundles, [], "commit")
        self.assertEqual(len(sample), 20)
        self.assertEqual(sample, M.sample_reviews(bundles, [], "commit"))
        self.assertEqual(M.sample_reviews(bundles, [b["fingerprint"] for b in bundles], "commit"), [])
        with self.assertRaises(ValueError):
            M.sample_reviews([bundles[0], bundles[0]], [], "commit")


if __name__ == "__main__":
    unittest.main()
