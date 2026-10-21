"""Browser projection must retain every interactive claim and visible citation."""
import copy
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0, str(ROOT / 'web'))   # site generators live in web/
from web_payload import chart_payload


class BrowserPayloadTests(unittest.TestCase):
    def test_projection_is_nonmutating_and_only_omits_unused_diagnostics(self):
        paths=list((ROOT/'_site').glob('*/data.json'))
        models=0
        for path in paths:
            data=json.loads(path.read_text())
            if 'sets' not in data:
                continue
            models+=1
            original=copy.deepcopy(data['sets'])
            projected=chart_payload(data['sets'],{})
            self.assertEqual(data['sets'],original)
            for before,after in zip(original,projected['sets']):
                self.assertEqual(set(before)-set(after),{'comparison'})
                for key in set(before)-{'points','comparison','comparisons'}:
                    self.assertEqual(before[key],after[key])
                self.assertEqual(len(before['points']),len(after['points']))
                for point,visible in zip(before['points'],after['points']):
                    self.assertEqual({k:v for k,v in point.items() if k!='protocol'},visible)
                for metric,comparison in before['comparisons'].items():
                    self.assertEqual({k:v for k,v in comparison.items() if k!='failure_examples'},after['comparisons'][metric])
            html=(path.parent/'index.html').read_text()
            embedded=json.loads(re.search(r'<script type="application/json" id="exdata">(.*?)</script>',html,re.S)[1])
            self.assertEqual(embedded['sets'],projected['sets'])
        self.assertGreater(models,0,'build and assemble all live models first')


if __name__=='__main__':
    unittest.main()
