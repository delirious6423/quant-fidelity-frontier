"""Raw-source provenance and fail-closed benchmark regressions."""
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks import build_benchmarks, render_benchmarks, _evidence
EXPECTED = {'deepseek-v4-flash-0731': (2,11), 'deepseek-v4.1-flash': (1,14), 'glm-5.3-flash': (1,5), 'qwen3.8-27b': (2,68), 'qwen3.8-flash-next': (1,15)}
class BenchmarkEvidence(unittest.TestCase):
    def payload(self,slug):
        root=ROOT/'models'/slug
        return build_benchmarks(root,json.loads((root/'model.json').read_text()))
    def test_all_models_have_verified_sources(self):
        for slug,(ns,np) in EXPECTED.items():
            with self.subTest(slug=slug):
                d=self.payload(slug)
                root=ROOT/'models'/slug
                manifest=json.loads((root/'benchmarks.json').read_text())['sources']
                self.assertEqual(len(manifest),ns)
                self.assertEqual(sum(len(m['points']) for s in manifest for m in s['metrics']),np)
                idx=json.loads((root/'snapshots/index.json').read_text())['repos']
                unchanged=[s for s in manifest if hashlib.sha256((root/idx[s['repo']]['file']).read_bytes()).hexdigest()==s['card_sha256']]
                self.assertEqual(len(d['sources']),len(unchanged))
                self.assertEqual(len(d['withheld']),len(manifest)-len(unchanged))
                self.assertEqual(sum(len(m['points']) for s in d['sources'] for m in s['metrics']),sum(len(m['points']) for s in unchanged for m in s['metrics']))
    def test_manifest_raw_quotes_and_hashes(self):
        for slug in EXPECTED:
            root=ROOT/'models'/slug
            idx=json.loads((root/'snapshots/index.json').read_text())['repos']
            for s in json.loads((root/'benchmarks.json').read_text())['sources']:
                raw=(root/idx[s['repo']]['file']).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=s['card_sha256']:continue  # changed sources are withdrawn, not build blockers
                lines=raw.decode().splitlines()
                for m in s['metrics']:
                    for p in m['points']:
                        self.assertEqual(lines[p['line']-1],p['quote'])
                        self.assertEqual(lines[p['size_line']-1],p['size_quote'])
                        self.assertIn(p['raw'],p['quote'])
                        self.assertIn(p['size_raw'],p['size_quote'])
    def temporary_glm(self,folder):
        src=ROOT/'models/glm-5.3-flash'; dst=Path(folder)
        manifest=json.loads((src/'benchmarks.json').read_text())
        idx=json.loads((src/'snapshots/index.json').read_text())
        for s in manifest['sources']:
            path=idx['repos'][s['repo']]['file']; target=dst/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/path,target)
        (dst/'snapshots').mkdir(exist_ok=True)
        (dst/'snapshots/index.json').write_text(json.dumps(idx))
        (dst/'benchmarks.json').write_text(json.dumps(manifest))
        return dst,manifest,idx
    def test_changed_card_withdraws_source(self):
        with tempfile.TemporaryDirectory() as folder:
            dst,m,idx=self.temporary_glm(folder)
            p=dst/idx['repos'][m['sources'][0]['repo']]['file'];p.write_bytes(p.read_bytes()+b'\nchanged')
            d=build_benchmarks(dst,{})
            self.assertEqual(d['sources'],[])
            self.assertIn('changed',d['withheld'][0]['reason'])
            self.assertIn('No reviewed benchmark',render_benchmarks(d))
    def test_invalid_evidence_cannot_plot(self):
        mutations=[('score',float('nan')),('score',float('inf')),('score',99),('raw','8.4'),('size_value',0),('size_unit','VRAM'),('line',0),('size_line',99999),('quote','wrong cell')]
        for field,value in mutations:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                dst,m,_=self.temporary_glm(folder);m['sources'][0]['metrics'][0]['points'][0][field]=value
                (dst/'benchmarks.json').write_text(json.dumps(m))
                self.assertEqual(build_benchmarks(dst,{})['sources'],[])
    def test_scales_and_size_conversion_preserved(self):
        if not self.payload('glm-5.3-flash')['sources'] or not self.payload('deepseek-v4.1-flash')['sources']:self.skipTest('fixture source changed; withdrawal covered separately')
        glm=self.payload('glm-5.3-flash')['sources'][0]['metrics'][0]
        self.assertEqual([(p['raw'],p['size']) for p in glm['points']],[('98.4',205.08),('98.8',328.35)])
        ds=self.payload('deepseek-v4.1-flash')['sources'][0]['metrics'][0]
        self.assertEqual(ds['unit'],'fraction')
        self.assertEqual([p['score'] for p in ds['points']],[.9553,.9522])
        self.assertEqual([p['size'] for p in ds['points']],[510.29,510.51])
    def test_static_chart_and_escaping(self):
        data=self.payload('qwen3.8-27b')
        if not data['sources'] or not data['sources'][0]['repo'].startswith('eaddario/'):self.skipTest('fixture source changed; withdrawal covered separately')
        data['sources'][0]['metrics'][0]['points'][0]['variant']='</script><script>alert(1)</script>'
        page=render_benchmarks(data)
        self.assertNotIn('</script><script>alert',page)
        self.assertIn('<div id="bm-plot" class="bm-plot"><svg',page)
        self.assertEqual(page.count('class="bm-dot"'),8)
        self.assertIn('Score L127',page)
        self.assertIn('Size L111',page)
    def test_unsigned_number_cannot_match_negative_source(self):
        point=dict(line=1,quote='| -0.5 |',raw='0.5',score=.5)
        with self.assertRaises(ValueError):_evidence([point['quote']],point,'score')

    def test_no_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(build_benchmarks(folder,{}))
        self.assertEqual(render_benchmarks(None),'')
if __name__=='__main__':unittest.main()
