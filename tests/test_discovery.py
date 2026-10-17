"""Validate assembled discovery surfaces and lossless HF export against built data."""
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
root = Path(__file__).resolve().parents[1]
site = root / '_site'
base = 'https://delirious6423.github.io/quant-fidelity-frontier/'
urls = [n.text for n in ET.parse(site / 'sitemap.xml').iter('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
assert len(urls) == len(set(urls))
for url in urls:
    assert url.startswith(base)
    page = site / url.removeprefix(base) / 'index.html'
    assert page.is_file(), url
    if json.loads((root / "site.json").read_text()).get("analytics", {}).get("website_id"):
        assert "data-website-id=" in page.read_text(), url
    assert f'rel="canonical" href="{url}"' in page.read_text(), url
assert len([u for u in urls if '/guides/' in u]) == 4
assert 'LLM quantization comparison' in (site / 'index.html').read_text()
manifest = json.loads((site / 'models.json').read_text())
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, str(root / 'pipeline/export_hf_dataset.py'), '--site', str(site), '--out', tmp], check=True)
    exported = [json.loads(line) for line in (Path(tmp) / 'claims.jsonl').read_text().splitlines()]
    original = []
    for model in manifest['models']:
        data = json.loads((site / model['data']).read_text())
        assert json.loads((Path(tmp) / (model['slug'] + '.json')).read_text()) == data
        original.extend(data['rows'])
        assert base + model['data'] in (site / 'llms.txt').read_text()
    assert all(row['source_revision_pinned'] == bool(re.fullmatch(r'[0-9a-fA-F]{40}', row['sha'] or '')) for row in exported)
    assert [json.loads(row['claim_json']) for row in exported] == original
    assert all(row['src'] == src['src'] and row['sha'] == src['sha'] for row, src in zip(exported, original))
print(f'PASS: {len(urls)} canonical pages; {len(exported)} claims preserved exactly; all model agent links resolve')
