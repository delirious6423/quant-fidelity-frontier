"""Reviewed benchmark discovery/search with independent source and transport checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qff_benchmark_server', ROOT / 'qff_mcp/qff_mcp.py')
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


def benchmark_source(owner, score='90.00', unit='as published', name='GPQA-Diamond'):
    repo = owner + '/evaluation'
    source = f'https://example.org/{repo}/blob/rev-abc/README.md'
    return dict(repo=repo, label=owner + ' evaluation', context='Synthetic fixture; 10 tasks.',
                notes=['Independent source report.'], revision='rev-abc', card_sha256='a' * 64,
                metrics=[dict(name=name, unit=unit, points=[dict(
                    repo='quant-owner/build', variant='Q4', score=float(score), raw=score,
                    size=5.0, size_raw='5.00 GB', reference=False, src=source + '#L10',
                    size_src=source + '#L9', line=10, size_line=9, flags=['Synthetic uncertainty'])])])


def write_fixture(folder):
    """A local, synthetic manifest; no publisher claims are changed."""
    folder = Path(folder)
    models = [dict(model='test/' + slug, slug=slug, data=slug + '/data.json') for slug in ('alpha', 'beta')]
    sources = [benchmark_source('eval-a'), benchmark_source('eval-b', '91.00'),
               benchmark_source('eval-a', '0.90', 'fraction'),
               benchmark_source('eval-a', '70.00', name='GPQA-Diamond Lite')]
    reference = copy.deepcopy(sources[0]['metrics'][0]['points'][0])
    reference.update(repo='test/alpha', variant='BF16', reference=True, size=10., size_raw='10 GB',
                     score=95., raw='95.000', line=11, src=reference['src'].replace('#L10', '#L11'))
    sources[0]['metrics'][0]['points'].append(reference)
    payloads = [sources, [benchmark_source('eval-c', '80.00')]]
    for model, source_list in zip(models, payloads):
        path = folder / model['data']
        path.parent.mkdir()
        path.write_text(json.dumps(dict(model=model['model'], captured_at='2026-10-19T00:00:00Z',
            rows=[], repos=[], sets=[], benchmarks=dict(sources=source_list,
                withheld=[dict(repo='changed/card', reason='card changed since review')]))))
    (folder / 'models.json').write_text(json.dumps(dict(models=models)))
    return models


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = write_fixture(self.tmp.name)
        for name, value in (('SRC', self.tmp.name), ('_cache', {}), ('_loaded_at', None)):
            mocked = patch.object(mcp, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_discovery_keeps_exact_names_and_units_with_source_and_model_counts(self):
        result = mcp.list_benchmarks(query='GPQA')
        self.assertEqual(result['matches'], 3)
        groups = {(b['name'], b['unit']): b for b in result['benchmarks']}
        common = groups['GPQA-Diamond', 'as published']
        self.assertEqual(common['models'], ['test/alpha', 'test/beta'])
        self.assertEqual((common['model_count'], common['source_count'], common['point_count']), (2, 3, 4))
        self.assertEqual(common['reference_count'], 1)
        self.assertEqual(len(mcp.list_benchmarks(min_models=2)['benchmarks']), 1)
        self.assertEqual(len(result['withheld_sources']), 2)
        self.assertFalse(result['ranking_permitted'])

    def test_ambiguous_units_and_nonexact_benchmark_names_are_errors(self):
        with self.assertRaisesRegex(ValueError, 'multiple units'):
            mcp.search_benchmarks('GPQA-Diamond', min_score=.8)
        with self.assertRaisesRegex(ValueError, 'exact name'):
            mcp.search_benchmarks('GPQA')
        with self.assertRaisesRegex(ValueError, 'not'):
            mcp.search_benchmarks('GPQA-Diamond', unit='percent')
        self.assertEqual(mcp.search_benchmarks('GPQA-Diamond', unit='fraction')['matches'], 1)

    def test_independent_reports_and_raw_evidence_are_retained(self):
        result = mcp.search_benchmarks('GPQA-Diamond', unit='as published', model='alpha')
        points = result['points']
        self.assertEqual([p['raw'] for p in points], ['90.00', '95.000', '91.00'])
        self.assertEqual(len({p['id'] for p in points}), 3)
        duplicate_builds = [p for p in points if p['repo'] == 'quant-owner/build']
        self.assertEqual(len(duplicate_builds), 2)
        self.assertEqual({p['source_repo'] for p in duplicate_builds}, {'eval-a/evaluation', 'eval-b/evaluation'})
        self.assertEqual(points[0]['size_raw'], '5.00 GB')
        for key in ('src', 'size_src', 'line', 'size_line', 'flags', 'context', 'notes', 'revision', 'card_sha256', 'captured_at'):
            self.assertTrue(points[0][key], key)
        self.assertFalse(result['ranking_permitted'])
        self.assertNotIn('rank', points[0])

    def test_all_filters_apply_and_publisher_is_the_reporting_source(self):
        result = mcp.search_benchmarks('GPQA-Diamond', unit='as published', model='alpha',
            query='q4 alpha', publisher='EVAL-B', max_size_gb=5, min_score=90.5, include_reference=False)
        self.assertEqual(result['matches'], 1)
        self.assertEqual(result['points'][0]['source_repo'], 'eval-b/evaluation')
        self.assertEqual(result['points'][0]['raw'], '91.00')
        self.assertEqual(mcp.search_benchmarks('GPQA-Diamond', unit='as published', publisher='quant-owner')['matches'], 0)
        self.assertEqual(mcp.search_benchmarks('GPQA-Diamond', model='beta')['matches'], 1)
        self.assertEqual(mcp.search_benchmarks('GPQA-Diamond', unit='as published', max_size_gb=0)['points'], [])

    def test_pagination_returns_every_report_once_and_is_bounded(self):
        first = mcp.search_benchmarks('GPQA-Diamond', unit='as published', limit=2)
        second = mcp.search_benchmarks('GPQA-Diamond', unit='as published', limit=2, offset=first['next_offset'])
        all_points = mcp.search_benchmarks('GPQA-Diamond', unit='as published')['points']
        self.assertEqual(first['points'] + second['points'], all_points)
        self.assertEqual(first['matches'], second['matches'])
        self.assertIsNone(second['next_offset'])
        self.assertEqual(mcp.list_benchmarks(limit=1)['next_offset'], 1)
        self.assertEqual(mcp.list_benchmarks(offset=999)['benchmarks'], [])
        for kwargs in ({'limit': 0}, {'limit': 201}, {'limit': True}, {'offset': -1}, {'offset': .5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mcp.list_benchmarks(**kwargs)

    def test_nonfinite_filters_and_payload_values_are_rejected(self):
        for key in ('min_score', 'max_size_gb'):
            for value in (float('nan'), float('inf'), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    mcp.search_benchmarks('GPQA-Diamond', unit='as published', **{key: value})
        with self.assertRaises(ValueError):
            mcp.search_benchmarks('GPQA-Diamond', unit='as published', max_size_gb=-1)
        data = mcp._model('alpha')[1]
        data['benchmarks']['sources'][0]['metrics'][0]['points'][0]['score'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            mcp.list_benchmarks()

    def test_refresh_updates_benchmark_scores_even_if_manifest_is_unchanged(self):
        with patch.object(mcp.time, 'monotonic', return_value=0):
            self.assertEqual(mcp.search_benchmarks('GPQA-Diamond', model='beta')['points'][0]['raw'], '80.00')
        path = Path(self.tmp.name, 'beta/data.json')
        data = json.loads(path.read_text())
        data['benchmarks']['sources'][0]['metrics'][0]['points'][0].update(score=81., raw='81.00')
        path.write_text(json.dumps(data))
        with patch.object(mcp.time, 'monotonic', return_value=300):
            self.assertEqual(mcp.search_benchmarks('GPQA-Diamond', model='beta')['points'][0]['raw'], '81.00')

    def test_older_payload_without_benchmarks_remains_supported(self):
        mcp._model('alpha')[1].pop('benchmarks')
        self.assertEqual(mcp.list_benchmarks(model='alpha')['benchmarks'], [])
        self.assertEqual(mcp.search_quants('alpha')['rows'], [])


class BenchmarkStdioTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_matches_all_reviewed_site_reports(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        site = Path(os.environ.get('QFF_DATA', ROOT / '_site'))
        if not (site / 'models.json').is_file():
            self.skipTest('assemble.py must run before published-data transport checks')
        manifest = json.loads((site / 'models.json').read_text())
        expected = {}
        for model in manifest['models']:
            data = json.loads((site / model['data']).read_text())
            for source in (data.get('benchmarks') or {}).get('sources', []):
                for metric in source['metrics']:
                    for point in metric['points']:
                        expected.setdefault((metric['name'], metric['unit']), []).append(
                            (model['model'], source['repo'], point['variant'], point['raw'], point['size_raw'],
                             point['score'], point['size'], point['src'], point['size_src'], source['revision'], source['card_sha256']))
        self.assertTrue(expected, 'The assembled site should contain reviewed task scores')
        params = StdioServerParameters(command=sys.executable, args=[str(ROOT / 'qff_mcp/qff_mcp.py')],
                                       env={**os.environ, 'QFF_DATA': str(site)})
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()
                tools = {tool.name for tool in (await client.list_tools()).tools}
                self.assertTrue({'list_benchmarks', 'search_benchmarks'} <= tools)
                async def call(name, args):
                    result = await client.call_tool(name, args)
                    self.assertFalse(bool(getattr(result, 'is_error', False) or getattr(result, 'isError', False)), str(result))
                    return json.loads(result.content[0].text)
                discovery = await call('list_benchmarks', {'limit': 200})
                self.assertEqual(discovery['matches'], len(expected))
                self.assertEqual({(b['name'], b['unit']) for b in discovery['benchmarks']}, set(expected))
                for (name, unit), want in expected.items():
                    with self.subTest(benchmark=name, unit=unit):
                        result = await call('search_benchmarks', dict(benchmark=name, unit=unit, limit=200))
                        got = [(p['model'], p['source_repo'], p['variant'], p['raw'], p['size_raw'],
                                p['score'], p['size_gb'], p['src'], p['size_src'], p['revision'], p['card_sha256'])
                               for p in result['points']]
                        self.assertEqual(got, want)
                        self.assertEqual(result['matches'], len(want))
                        self.assertFalse(result['ranking_permitted'])
                error = await client.call_tool('search_benchmarks', {'benchmark': 'not-a-benchmark'})
                self.assertTrue(bool(getattr(error, 'is_error', False) or getattr(error, 'isError', False)))
                print(f'Benchmark stdio: {len(expected)} exact benchmark/unit groups; '
                      f'{sum(len(points) for points in expected.values())} source reports match the assembled data.')


if __name__ == '__main__':
    unittest.main()
