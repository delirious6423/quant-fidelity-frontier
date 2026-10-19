"""Adversarial bindings: a real pinned number can still be the wrong claim."""
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import build_benchmarks


class BenchmarkSourceBindings(unittest.TestCase):
    def assert_rejected(self, slug, mutate, reason, source_index=0):
        root = ROOT / 'models' / slug
        original = json.loads((root / 'benchmarks.json').read_text())
        index = json.loads((root / 'snapshots/index.json').read_text())
        expected_repo = original['sources'][source_index]['repo']
        baseline = build_benchmarks(root, {})
        source = original['sources'][source_index]
        if hashlib.sha256((root / index['repos'][expected_repo]['file']).read_bytes()).hexdigest() != source['card_sha256']:
            self.skipTest('fixture source changed; changed-card withdrawal is covered separately')
        self.assertIn(expected_repo, [s['repo'] for s in baseline['sources']])
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            (target / 'snapshots').mkdir()
            (target / 'snapshots/index.json').write_text(json.dumps(index))
            for source in original['sources']:
                path = index['repos'][source['repo']]['file']
                (target / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / path, target / path)
            manifest = copy.deepcopy(original)
            mutate(manifest['sources'][source_index])
            (target / 'benchmarks.json').write_text(json.dumps(manifest))
            result = build_benchmarks(target, {})
        self.assertEqual([s['repo'] for s in result['sources']],
                         [s['repo'] for s in baseline['sources'] if s['repo'] != expected_repo])
        rejected = [s for s in result['withheld'] if s['repo'] == expected_repo]
        self.assertEqual(len(result['withheld']), len(baseline['withheld']) + 1)
        self.assertEqual(len(rejected), 1)
        self.assertIn(reason, rejected[0]['reason'])

    def test_adjacent_task_score_on_same_pinned_row_is_rejected(self):
        # Both values occur on the same immutable row. Numeric substring
        # validation alone accepts AI2D's 84.6 as GSM8K chat's score.
        self.assert_rejected('glm-5.3-flash', lambda s:
                             s['metrics'][0]['points'][0].update(raw='84.6', score=84.6),
                             'benchmark column')

    def test_renamed_task_and_scale_are_rejected(self):
        for change in ({'name': 'AI2D vision'}, {'name': 'Unpublished task'}, {'unit': 'percent'}):
            with self.subTest(change=change):
                self.assert_rejected('glm-5.3-flash', lambda s: s['metrics'][0].update(change),
                                     'benchmark')

    def test_fabricated_variant_repository_or_reference_is_rejected(self):
        for change in ({'variant': 'fabricated-model'}, {'repo': 'other/model'},
                       {'reference': True}, {'reference': 'false'}, {'reference': 0}):
            with self.subTest(change=change):
                self.assert_rejected('glm-5.3-flash', lambda s:
                                     s['metrics'][0]['points'][0].update(change),
                                     'row variant or repository')

    def test_supported_but_wrong_size_unit_is_rejected(self):
        for slug, unit in (('glm-5.3-flash', 'GB'), ('qwen3.8-27b', 'GiB')):
            with self.subTest(slug=slug):
                self.assert_rejected(slug, lambda s: s['metrics'][0]['points'][0].update(size_unit=unit),
                                     'size value or unit')

    def test_bpw_cannot_be_substituted_for_full_file_size(self):
        self.assert_rejected('qwen3.8-27b', lambda s:
                             s['metrics'][0]['points'][0].update(size_raw='1.7500', size_value=1.75),
                             'size value or unit')

    def test_uncertainty_cannot_be_substituted_for_the_score(self):
        def mutate(source):
            metric = next(m for m in source['metrics'] if m['name'] == 'ARC Challenge')
            metric['points'][0].update(raw='1.8246', score=1.8246)
        self.assert_rejected('qwen3.8-27b', mutate, 'benchmark column')

    def test_acc_norm_cannot_be_substituted_for_acc(self):
        def mutate(source):
            metric = next(m for m in source['metrics'] if m['name'] == 'PIQA acc')
            metric['points'][0].update(raw='0.8613', score=0.8613)
        self.assert_rejected('deepseek-v4.1-flash', mutate, 'benchmark column')

    def test_transposed_size_requires_the_matching_variant_column(self):
        # Official and quantized sizes share one cited line, but not a column.
        self.assert_rejected('deepseek-v4-flash-0731', lambda s:
                             s['metrics'][0]['points'][0].update(size_raw='167.08', size_value=167.08),
                             'size value or unit')

    def test_dual_unit_size_does_not_allow_cross_pairing(self):
        self.assert_rejected('deepseek-v4.1-flash', lambda s:
                             s['metrics'][0]['points'][0].update(size_unit='GiB'),
                             'size value or unit')

    def test_score_and_size_must_link_the_same_full_artifact(self):
        def mutate(source):
            first, second = source['metrics'][0]['points'][:2]
            # These builds share a HumanEval score, so value equality cannot
            # identify the artifact. Every size field remains source-correct.
            first.update({key: value for key, value in second.items() if key.startswith('size_')})
        self.assert_rejected('qwen3.8-27b', mutate, 'different artifacts', source_index=1)

    def test_cross_repository_artifact_attribution_cannot_be_erased(self):
        def mutate(source):
            point = next(p for p in source['metrics'][0]['points'] if p['variant'] == 'GSQ-RCO-BuffedMod')
            point.pop('repo')
        self.assert_rejected('qwen3.8-27b', mutate, 'row variant or repository', source_index=1)

    def test_another_row_with_the_same_score_does_not_validate_the_variant(self):
        def mutate(source):
            first, second = source['metrics'][0]['points'][4:6]
            first.update(line=second['line'], quote=second['quote'])
        self.assert_rejected('qwen3.8-27b', mutate, 'row variant or repository')

    def test_existing_source_order_and_all_results_are_preserved(self):
        for path in sorted((ROOT / 'models').glob('*/benchmarks.json')):
            with self.subTest(model=path.parent.name):
                manifest = json.loads(path.read_text())
                index = json.loads((path.parent / 'snapshots/index.json').read_text())['repos']
                unchanged = [s for s in manifest['sources'] if hashlib.sha256(
                    (path.parent / index[s['repo']]['file']).read_bytes()).hexdigest() == s['card_sha256']]
                result = build_benchmarks(path.parent, {})
                self.assertEqual(len(result['withheld']), len(manifest['sources']) - len(unchanged))
                self.assertEqual([s['repo'] for s in result['sources']],
                                 [s['repo'] for s in unchanged])
                for original, source in zip(unchanged, result['sources']):
                    self.assertEqual([m['name'] for m in source['metrics']],
                                     [m['name'] for m in original['metrics']])
                    for before, after in zip(original['metrics'], source['metrics']):
                        self.assertEqual([(p['variant'], p['raw'], p['score']) for p in after['points']],
                                         [(p['variant'], p['raw'], p['score']) for p in before['points']])


if __name__ == '__main__':
    unittest.main()
