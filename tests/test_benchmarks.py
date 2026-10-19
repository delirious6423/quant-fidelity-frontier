"""Pilot provenance and fail-closed eligibility regressions."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import build_pilot, render_pilot


class BenchmarkPilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT / 'models/glm-5.3-flash/model.json').read_text())
        cls.rows = json.loads((ROOT / 'models/glm-5.3-flash/site/data.json').read_text())['rows']

    def test_real_source_values(self):
        data = build_pilot(self.rows, self.cfg)
        self.assertEqual([len(m['points']) for m in data['metrics']], [2, 1, 1, 1])
        a, b = data['metrics'][0]['points']
        self.assertEqual((a['raw'], a['size'], a['size_raw'], a['line']), ('98.4', 205.08, '191.0 GiB', 76))
        self.assertEqual((b['raw'], b['size'], b['size_raw'], b['line']), ('98.8', 328.35, '305.8 GiB', 75))
        self.assertTrue(b['reference'])
        raw = (ROOT / 'models/glm-5.3-flash/cache/cards/mbehr90__GLM-5.3-Flash-nvfp4.md').read_text().splitlines()
        for m in data['metrics']:
            for p in m['points']:
                self.assertIn(p['raw'], raw[p['line'] - 1])
                self.assertIn(p['size_raw'], raw[p['line'] - 1])

    def test_reject_unsafe_points(self):
        own = next(r for r in self.rows if r['measured_by'] == 'mbehr90/GLM-5.3-Flash-nvfp4' and r['kind'] == 'own')
        for field, value in [('basis', 'vram'), ('basis', None), ('size_gb', None), ('size_gb', float('nan')), ('size_gb', 0), ('kind', 'quoted'), ('copied_from', 'elsewhere'), ('weights_modified', True), ('measured', False), ('panel', 'different setup'), ('measured_by', 'another/publisher')]:
            r = copy.deepcopy(own); r[field] = value
            with self.subTest(field=field, value=value):
                self.assertFalse(any(m['points'] for m in build_pilot([r], self.cfg)['metrics']))

    def test_missing_and_nonfinite_scores(self):
        rows = copy.deepcopy(self.rows)
        for r in rows:
            r['metrics'].pop('bench:GSM8K chat', None)
            if 'bench:ARC-C' in r['metrics']:
                r['metrics']['bench:ARC-C']['v'] = float('inf')
        data = build_pilot(rows, self.cfg)
        self.assertEqual(data['metrics'][0]['points'], [])
        self.assertEqual(data['metrics'][2]['points'], [])

    def test_other_models_unchanged(self):
        for cfg in ({}, {'slug': 'qwen3.8-27b', 'benchmark_pilot': True}, dict(self.cfg, benchmark_pilot=False)):
            self.assertIsNone(build_pilot(self.rows, cfg))
        self.assertEqual(render_pilot(None), '')

    def test_fallback_and_escaping(self):
        data = build_pilot(self.rows, self.cfg)
        data['metrics'][0]['points'][0]['variant'] = '</script><script>alert(1)</script>'
        page = render_pilot(data)
        self.assertNotIn('</script><script>alert', page)
        self.assertIn('score units', page)
        self.assertEqual(page.count('<tr>'), 6)


if __name__ == '__main__':
    unittest.main()
