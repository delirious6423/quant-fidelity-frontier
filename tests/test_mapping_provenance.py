"""A changed card must be reviewed again before old mapping files can publish it."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "pipeline"))
import mapping_provenance as MP


class MappingProvenance(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qff-mapping-provenance-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = "publisher/model-GGUF"
        self.name = self.repo.replace("/", "__")
        for folder in ("audit", "snapshots", "cache/cards", "maps/A", "maps/B", "maps/C", "maps/cands", "verified"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self.card = self.root / "cache/cards/card.md"
        self.card.write_text("| Quant | KLD |\n| --- | --- |\n| Q4 | 0.1250 |\nDataset: Old corpus.\n")
        self.pin = self.update_index()
        self.save("audit/triage.json", {"wave1": [self.repo], "wave2": []})
        self.save("maps/mapped.json", {self.repo: self.pin})
        self.mapping = {"repo": self.repo, "method": {"dataset": "Old corpus"}, "entries": [
            {"role": "own", "name": "Q4", "metrics": {"mean_kld": "c1"}}]}
        self.candidates = {"c1": {"L": 3, "raw": "0.1250", "label": "Q4", "header": "KLD"}}
        self.save(f"maps/cands/{self.name}.cands.json", self.candidates)
        for run in ("A", "B"):
            self.save(f"maps/{run}/{self.name}.map.json", self.mapping)

    def save(self, path, value):
        (self.root / path).write_text(json.dumps(value))

    def update_index(self):
        pin = hashlib.sha256(self.card.read_bytes()).hexdigest()
        self.index = {self.repo: {"file": "cache/cards/card.md", "sha": "main", "sha256": pin}}
        self.save("snapshots/index.json", {"repos": self.index})
        return pin

    def change_source(self):
        self.card.write_text(self.card.read_text().replace("Old corpus", "Changed corpus"))
        return self.update_index()

    def run_script(self, name, *args):
        result = subprocess.run([sys.executable, str(CODE / "pipeline" / name), *map(str, args)],
                                env={**os.environ, "QCL_ROOT": str(self.root)}, capture_output=True, text=True)
        return result

    def record(self):
        result = self.run_script("qcl_mapset.py", "--record")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def applied(self, run="A"):
        script = ("import json,sys;sys.path.insert(0,sys.argv[1]);import qcl_watch as W;"
                  "print(json.dumps(W.applied_from(sys.argv[2])))")
        result = subprocess.run([sys.executable, "-c", script, str(CODE / "pipeline"), str(self.root / "maps" / run)],
                                env={**os.environ, "QCL_ROOT": str(self.root)}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def new_maps(self, pin, runs=("A", "B")):
        candidate_pin = MP.candidates_digest(self.candidates)
        self.save(f"maps/cands/{self.name}.cands.json", {
            "repo": self.repo, "source_sha256": pin, "candidates_sha256": candidate_pin, "candidates": self.candidates})
        for run in runs:
            self.save(f"maps/{run}/{self.name}.map.json", {**self.mapping, "source_sha256": pin, "candidates_sha256": candidate_pin})

    def test_unchanged_legacy_maps_keep_the_existing_pin_and_values(self):
        self.assertEqual(self.record()["todo"], 0)
        applied = self.applied()[self.repo]
        self.assertEqual(applied["source_sha256"], self.pin)
        self.assertEqual(applied["variants"][0]["m"], {"mean_kld": "0.1250"})

    def test_changed_card_cannot_be_repinned_by_old_maps_before_queueing(self):
        changed_pin = self.change_source()
        self.assertNotEqual(changed_pin, self.pin)
        self.assertEqual(self.record()["todo"], 1)
        self.assertEqual(json.loads((self.root / "maps/mapped.json").read_text())[self.repo], self.pin)
        self.assertEqual(json.loads((self.root / "audit/mapset.json").read_text())["todo"], [self.repo])
        self.assertEqual(self.applied(), {})

    def test_preparing_new_candidates_does_not_make_an_old_worker_map_current(self):
        changed_pin = self.change_source()
        self.new_maps(changed_pin, runs=("A",))
        self.assertEqual(self.record()["todo"], 1)
        self.assertEqual(self.applied("B"), {})

    def test_two_new_source_pinned_maps_can_advance_the_recorded_pin(self):
        changed_pin = self.change_source()
        self.new_maps(changed_pin)
        self.assertEqual(self.record()["todo"], 0)
        self.assertEqual(json.loads((self.root / "maps/mapped.json").read_text())[self.repo], changed_pin)
        self.assertEqual(self.applied()[self.repo]["source_sha256"], changed_pin)

    def test_explicit_old_pin_never_falls_back_to_a_current_legacy_record(self):
        self.new_maps("a" * 64)
        self.assertEqual(self.applied(), {})

    def test_candidate_task_identity_and_source_must_match_the_mapping(self):
        candidate_pin = MP.candidates_digest(self.candidates)
        mapping = {**self.mapping, "source_sha256": self.pin, "candidates_sha256": candidate_pin}
        original = {"repo": self.repo, "source_sha256": self.pin, "candidates_sha256": candidate_pin, "candidates": self.candidates}
        for change in ({"repo": "other/model"}, {"source_sha256": "f" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                MP.mapping_source(mapping, {**original, **change}, self.index, path=self.root / f"maps/A/{self.name}.map.json")
        with self.assertRaises(ValueError):
            MP.mapping_source(mapping, self.candidates, self.index, path=self.root / f"maps/A/{self.name}.map.json")

    def test_candidate_ids_cannot_change_under_a_previously_completed_map(self):
        self.new_maps(self.pin)
        document = json.loads((self.root / f"maps/cands/{self.name}.cands.json").read_text())
        document["candidates"]["c1"]["raw"] = "0.2500"
        for update_pin in (False, True):
            with self.subTest(update_candidate_pin=update_pin):
                if update_pin: document["candidates_sha256"] = MP.candidates_digest(document["candidates"])
                self.save(f"maps/cands/{self.name}.cands.json", document)
                self.assertEqual(self.applied(), {})

    def test_baseline_records_only_current_reviews_and_never_advances_a_pin(self):
        (self.root / "maps/mapped.json").unlink()
        review = {"repo": self.repo, "sha": "main", "source_sha256": self.pin, "variants": []}
        claim_file = self.root / "verified/part1.py"
        for records in ([], [{**review, "auto": True}], [{**review, "source_sha256": "f" * 64}]):
            with self.subTest(records=records):
                claim_file.write_text("R = " + repr(records) + "\n")
                result = self.run_script("qcl_mapset.py", "--baseline")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(json.loads(result.stdout)["todo"], 1)
                self.assertEqual(json.loads((self.root / "maps/mapped.json").read_text()), {})
        claim_file.write_text("R = " + repr([review]) + "\n")
        # A hand-reviewed baseline with no retained maps can establish coverage.
        for run in ("A", "B"):
            (self.root / f"maps/{run}/{self.name}.map.json").unlink()
        self.assertEqual(self.run_script("qcl_mapset.py", "--baseline").returncode, 0)
        self.assertEqual(json.loads((self.root / "maps/mapped.json").read_text())[self.repo], self.pin)
        self.change_source()
        self.assertEqual(self.run_script("qcl_mapset.py", "--baseline").returncode, 0)
        self.assertEqual(json.loads((self.root / "maps/mapped.json").read_text())[self.repo], self.pin)

    def test_backup_filenames_cannot_reactivate_old_maps_after_a_fresh_pair(self):
        for run in ("A", "B"):
            self.save(f"maps/{run}/zz-stale.map.json", self.mapping)
            self.save(f"maps/{run}/zz-stale.cands.json", self.candidates)
        changed_pin = self.change_source()
        self.mapping["method"]["dataset"] = "Changed corpus"
        self.new_maps(changed_pin)
        self.assertEqual(self.record()["todo"], 0)
        for run in ("A", "B"):
            self.assertEqual(self.applied(run)[self.repo]["methodology"]["dataset"], "Changed corpus")
        result = self.run_script("qcl_map.py", "apply", self.root / "maps/A/zz-stale.map.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("filename does not match", result.stderr)

    def test_noncanonical_fresh_maps_cannot_advance_the_recorded_pin(self):
        changed_pin = self.change_source()
        self.new_maps(changed_pin)
        for run in ("A", "B"):
            path = self.root / f"maps/{run}/{self.name}.map.json"
            path.rename(path.with_name("zz-fresh.map.json"))
            self.save(f"maps/{run}/zz-fresh.cands.json", json.loads((self.root / f"maps/cands/{self.name}.cands.json").read_text()))
        self.assertEqual(self.record()["todo"], 1)
        self.assertEqual(json.loads((self.root / "maps/mapped.json").read_text())[self.repo], self.pin)
        self.assertEqual(self.applied(), {})

    def test_baseline_does_not_authorize_retained_maps_without_an_original_pin(self):
        (self.root / "maps/mapped.json").unlink()
        changed_pin = self.change_source()
        review = {"repo": self.repo, "sha": "main", "source_sha256": changed_pin, "variants": []}
        (self.root / "verified/part1.py").write_text("R = " + repr([review]) + "\n")
        result = self.run_script("qcl_mapset.py", "--baseline")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["todo"], 1)
        self.assertNotIn(self.repo, json.loads((self.root / "maps/mapped.json").read_text()))
        self.assertEqual(self.applied(), {})

    def test_task_cli_emits_a_source_pinned_envelope_and_mapcheck_reads_its_ids(self):
        result = self.run_script("qcl_map.py", "tasks", self.root / "tasks", self.repo)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        candidate_path = self.root / f"tasks/{self.name}.cands.json"
        candidates = json.loads(candidate_path.read_text())
        self.assertEqual(candidates["source_sha256"], self.pin)
        self.assertIn(f"Source SHA-256: {self.pin}", (self.root / f"tasks/{self.name}.prompt.txt").read_text())
        self.save(f"maps/cands/{self.name}.cands.json", candidates)
        self.save(f"maps/A/{self.name}.map.json", {**self.mapping, "source_sha256": self.pin,
                                                  "candidates_sha256": candidates["candidates_sha256"]})
        queue = self.root / "queue.txt"; queue.write_text(self.name + "\n")
        result = self.run_script("qcl_mapcheck.py", queue, self.root / "maps/A")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        bad = copy.deepcopy(self.mapping); bad.update(source_sha256=self.pin, candidates_sha256=candidates["candidates_sha256"])
        bad["entries"][0]["metrics"]["mean_kld"] = "c9999"
        self.save(f"maps/A/{self.name}.map.json", bad)
        self.assertNotEqual(self.run_script("qcl_mapcheck.py", queue, self.root / "maps/A").returncode, 0)

    def test_empty_card_generated_maps_are_pinned_at_task_creation(self):
        self.card.write_text("There are no published numerical measurements in this card.\n")
        current_pin = self.update_index()
        self.save("audit/mapset.json", {"todo": [self.repo]})
        result = self.run_script("qcl_queue.py", self.root / "work")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["no_numbers"], 1)
        self.assertEqual(self.record()["todo"], 0)
        for run in ("A", "B"):
            self.assertEqual(self.applied(run)[self.repo]["source_sha256"], current_pin)

    def test_tiebreaker_task_can_upgrade_legacy_candidates_without_repining_ab(self):
        # Generate the exact current candidate vocabulary, retaining the legacy
        # flat A/B file as real deployed datasets do.
        result = self.run_script("qcl_map.py", "tasks", self.root / "tasks", self.repo)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        generated = json.loads((self.root / f"tasks/{self.name}.cands.json").read_text())
        self.save(f"maps/cands/{self.name}.cands.json", generated["candidates"])
        self.save("audit/review_queue.json", {"disagreements": [{"repo": self.repo}]})
        result = self.run_script("qcl_tiebreak.py", "queue", self.root / "work")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["queued"], 1)
        self.save(f"maps/C/{self.name}.map.json", {**self.mapping, "source_sha256": self.pin,
                                                  "candidates_sha256": generated["candidates_sha256"]})
        (self.root / "work/claimed/C00.txt").write_text(self.name + "\n")
        result = self.run_script("qcl_mapcheck.py", self.root / "work/claimed/C00.txt", self.root / "maps/C")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.run_script("qcl_tiebreak.py", "record", self.root / "work")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((self.root / "maps/mapped_c.json").read_text())[self.repo], self.pin)
        self.assertEqual(self.applied("C")[self.repo]["source_sha256"], self.pin)
        self.assertEqual(self.applied("A")[self.repo]["source_sha256"], self.pin)


if __name__ == "__main__":
    unittest.main(verbosity=2)
