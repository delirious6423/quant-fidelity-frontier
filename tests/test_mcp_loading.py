"""Hostile manifest paths, bounded reads, and long-running MCP freshness."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('qff_loading_test',ROOT/'qff_mcp/qff_mcp.py')
mcp=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


class LoadingTests(unittest.TestCase):
    def setUp(self):
        mcp._cache.clear()
        mcp._loaded_at=None
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.root.joinpath('sample').mkdir()
        self.manifest={'models':[{'model':'org/sample','slug':'sample','data':'sample/data.json'}]}
        self.data={'model':'org/sample','sets':[],'rows':[],'repos':[],'generation':1}
        self.write()
        source=patch.object(mcp,'SRC',str(self.root));source.start();self.addCleanup(source.stop)

    def write(self):
        self.root.joinpath('models.json').write_text(json.dumps(self.manifest))
        self.root.joinpath('sample/data.json').write_text(json.dumps(self.data))

    def test_refresh_invalidates_model_even_if_manifest_counts_do_not_change(self):
        with patch.object(mcp.time,'monotonic',return_value=0):
            self.assertEqual(mcp._model('sample')[1]['generation'],1)
        self.data['generation']=2;self.write()
        with patch.object(mcp.time,'monotonic',return_value=299):
            self.assertEqual(mcp._model('sample')[1]['generation'],1)
        with patch.object(mcp.time,'monotonic',return_value=300):
            self.assertEqual(mcp._model('sample')[1]['generation'],2)

    def test_failed_refresh_raises_and_retries_without_marking_cache_fresh(self):
        with patch.object(mcp.time,'monotonic',return_value=0):
            mcp._model('sample')
        self.root.joinpath('models.json').write_text('{broken')
        with patch.object(mcp.time,'monotonic',return_value=300):
            with self.assertRaises(ValueError):mcp._model('sample')
            self.assertEqual(mcp._loaded_at,0)
            self.data['generation']=2;self.write()
            self.assertEqual(mcp._model('sample')[1]['generation'],2)

    def test_path_traversal_and_urls_rejected(self):
        for path in ('../secret.json','/etc/secret.json','a/../secret.json','a\\secret.json',
                     'https://other.test/data.json','//other.test/data.json','%2e%2e/data.json',
                     'sample/data.json?x=1','sample/data.json#x','sample//data.json','./models.json'):
            with self.subTest(path=path),self.assertRaises(ValueError):mcp._read(path)

    def test_symlink_outside_root_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            target=Path(outside)/'data.json';target.write_text('{}')
            self.root.joinpath('escape.json').symlink_to(target)
            with self.assertRaisesRegex(ValueError,'outside'):mcp._read('escape.json')

    def test_manifest_path_is_validated_before_any_model_read(self):
        self.manifest['models'][0]['data']='../secret.json';self.write()
        with self.assertRaises(ValueError):mcp._manifest()
        self.assertNotIn('m',mcp._cache)

    def test_model_identity_is_validated(self):
        self.data['model']='org/unrelated';self.write()
        with self.assertRaisesRegex(ValueError,'identity'):mcp._model('sample')
        self.assertNotIn('sample/data.json',mcp._cache)

    def test_size_bound(self):
        with patch.object(mcp,'MAX_MODEL_BYTES',20):
            with self.assertRaisesRegex(ValueError,'byte limit'):mcp._read('sample/data.json')

    def test_redirects_cannot_leave_source_or_traverse(self):
        with patch.object(mcp,'SRC','https://example.org/qff/'):
            handler=mcp._SourceRedirects();request=Request('https://example.org/qff/models.json')
            for url in ('https://other.test/data.json','http://example.org/qff/data.json',
                        'https://example.org/other/data.json','https://example.org/qff/../data.json',
                        'https://example.org/qff/%2e%2e/data.json'):
                with self.subTest(url=url),self.assertRaises(ValueError):
                    handler.redirect_request(request,None,302,'Found',{},url)
            result=handler.redirect_request(request,None,302,'Found',{},'https://example.org/qff/sample/data.json')
            self.assertEqual(result.full_url,'https://example.org/qff/sample/data.json')


if __name__=='__main__':unittest.main()
