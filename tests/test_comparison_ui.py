"""Browser regression contracts for comparison permissions, ownership, and disclosure.

Run `python tests/test_comparison_ui.py` after building Qwen3.8-27B. The browser
uses an isolated source-injected fixture to test permissions independently of
which real cards currently have sufficiently documented methodology.
"""
import asyncio
import copy
import json
import os
import re
import tempfile
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/qwen3.8-27b/site/index.html'
LABELS = {
    'same_protocol': 'Same Protocol — Comparable',
    'different_or_unverified_protocol': 'Different or Unverified Protocol — Not Directly Comparable',
    'different_base_model': 'Different Base Model — Fidelity Not Directly Comparable',
}


def result(classification, permitted=False):
    return dict(classification=classification, label=LABELS[classification],
                explanation='Source evidence establishes the conditions shown below.',
                ranking_permitted=permitted,
                evidence=[{'url': 'https://example.org/protocol', 'quote': 'Shared evaluation protocol.'}],
                verified_matching_conditions=['Base model', 'Dataset'],
                differences=[] if permitted else [{'condition': 'Implementation', 'explanation': 'Confirmed mismatch.'}],
                unknowns=[] if permitted else [{'condition': 'Reference checkpoint', 'explanation': 'Not documented.'}])


def fixture():
    source = MODEL.read_text()
    D = json.loads(re.search(r'<script type="application/json" id="exdata">(.*?)</script>', source, re.S).group(1))
    template = next(s for s in D['sets'] if s['id'] != 'all' and len(s['points']) > 1)
    template = copy.deepcopy(template)
    template['metrics'] = ['kld', 'ppl']
    template['points'] = template['points'][:2]
    for i, point in enumerate(template['points']):
        point.update(qid='fixture-' + str(i), variant='Fixture quant ' + str(i), x=10 + i,
                     pub=['Quant One', 'Quant Two'][i], quantization_owner=['Quant One', 'Quant Two'][i],
                     measurement_owner='Shared Evaluator', measured_by='Shared Evaluator',
                     measurement_id='measurement-' + str(i), compatible_sets=['verified'], claim_provenance={'kind': 'quoted', 'original_attribution': {'verified': True}},
                     repo=['quant-one/checkpoint', 'quant-two/checkpoint'][i], is_ref=False,
                     m={'kld': {'v': .1 + .1*i, 'raw': str(.1 + .1*i)}, 'ppl': {'v': 2+i, 'raw': str(2+i)}})
    sets = []
    for id_, classification, allowed in [('all', 'different_or_unverified_protocol', False),
                                         ('verified', 'same_protocol', True),
                                         ('unknown', 'different_or_unverified_protocol', False),
                                         ('different-model', 'different_base_model', False),
                                         ('legacy-only', 'different_or_unverified_protocol', False)]:
        item = copy.deepcopy(template)
        item.update(id=id_, title=id_, auto=id_ in ('unknown', 'legacy-only'), comparable=True)
        for point in item['points']:
            point['qid'] += '@overview' if id_ == 'all' else '|destination'
            if id_ == 'unknown':
                point['measurement_owner'] = None
        # kld is verified only in the explicit set. ppl remains unverified even there.
        if id_ != 'legacy-only':
            item['comparison'] = result(classification, allowed)
            item['comparisons'] = {'kld': result(classification, allowed), 'ppl': result('different_or_unverified_protocol')}
        else:
            item.pop('comparison', None)
            item.pop('comparisons', None)
        sets.append(item)
    D['sets'] = sets
    source = re.sub(r'(<script type="application/json" id="exdata">).*?(</script>)', lambda m: m[1] + json.dumps(D) + m[2], source, flags=re.S)
    source = re.sub(r'<script>(.*?)</script>', lambda m: '<script>' + (ROOT / 'web/site_script.js').read_text() + '\n' + (ROOT / 'web/site_explorer.js').read_text() + '</script>' if 'var dataEl' in m[1] else m[0], source, flags=re.S)
    source = source.replace('<p class="ex-meta" id="exmeta"></p>', '')
    if 'id="exmeta"' not in source: source = source.replace('<p class="ex-scope" id="exscope"></p>', '<p class="ex-scope" id="exscope"></p><div class="ex-meta" id="exmeta"></div>')
    return source.replace('</head>', '<style>' + (ROOT / 'web/site_style.css').read_text() + '</style></head>')


async def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'comparison-ui.html'
        path.write_text(fixture())
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            for width in (320, 390, 1366):
                context = await browser.new_context(viewport={'width': width, 'height': 900})
                await context.route(re.compile(r'^https?://'), lambda route: route.abort())
                page = await context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                await page.goto(path.as_uri(), wait_until='domcontentloaded')
                assert await page.input_value('#exset') == 'all'
                assert 'Mixed protocols' in await page.locator('#exscope').inner_text()
                assert await page.locator('#exsvg circle.mk').count() == 2
                assert await page.locator('#exset option').count() == 5
                assert await page.locator('#exset optgroup').evaluate_all("nodes => nodes.find(n => n.label.includes('Overview')).children.length") == 1
                assert await page.locator('#exmeta details').count() == 1
                assert not await page.locator('#exmeta details').evaluate('e => e.open')
                assert not await page.locator('#exfront').is_visible()
                assert await page.locator('#extable th[data-key=kld]').count() == 0
                assert await page.locator('#extable th[aria-disabled=true]').count() == 2
                if await page.locator('#quant-results').count() and not await page.locator('#quant-results').evaluate('e => e.open'):
                    await page.locator('#quant-results > summary').click()
                await page.locator('#extable th[data-key=_x]').click()
                assert await page.locator('#extable th[data-key=_x]').get_attribute('aria-sort') == 'descending'
                await page.locator('#extable th[data-key=_x]').click()
                assert await page.locator('#extable th[data-key=_x]').get_attribute('aria-sort') == 'ascending'
                await page.locator('#exsvg circle.mk').first.press('Enter')
                assert await page.locator('#exdetail .comparison-jump').count() == 1
                assert 'Claim provenance: quoted' in await page.locator('#exdetail').inner_text()
                selected = await page.locator('#exdetail h3').inner_text()
                await page.locator('#exdetail .comparison-jump').click()
                assert await page.input_value('#exset') == 'verified'
                assert await page.locator('#extable tr.sel').count() == 1
                await page.locator('#exsvg circle.mk').first.press('Enter')
                assert await page.locator('#exdetail h3').inner_text() == selected
                await page.keyboard.press('Escape')
                await page.select_option('#exset', 'all')
                await page.locator('#exmeta summary').click()
                assert 'Confirmed mismatch' in await page.locator('#exmeta').inner_text()
                assert 'Not documented' in await page.locator('#exmeta').inner_text()
                assert await page.locator('#exmeta a').get_attribute('href') == 'https://example.org/protocol'
                await page.locator('#exmeta .comparison-jump').click()
                assert await page.input_value('#exset') == 'verified'
                assert LABELS['same_protocol'] in await page.locator('#exscope').inner_text()
                assert await page.locator('#extable th[data-key=kld]').count() == 1
                assert await page.locator('#extable th[data-key=ppl]').count() == 0
                await page.locator('.ex-options > summary').click()
                assert await page.locator('#exfront').is_visible()
                await page.check('#exfront')
                assert await page.locator('#exsvg polyline.frontier').count() == 1 or await page.locator('#extable .chip.f').count() >= 1
                await page.locator('#exsvg circle.mk').first.press('Enter')
                detail = await page.locator('#exdetail').inner_text()
                assert 'Quant publisher: Quant' in detail and 'Measurement publisher: Shared Evaluator' in detail
                await page.keyboard.press('Escape')
                await page.locator('#exmetric button').nth(1).click()
                assert not await page.locator('#exfront').is_visible()
                assert await page.locator('#exsvg polyline.frontier').count() == 0
                assert 'Beaten by' not in await page.locator('#extable thead').inner_text()
                assert 'sorted by GB' in await page.locator('#extcap').inner_text()
                for id_ in ('unknown', 'different-model', 'legacy-only'):
                    await page.select_option('#exset', id_)
                    assert (LABELS['different_base_model'] if id_ == 'different-model' else LABELS['different_or_unverified_protocol']) in await page.locator('#exscope').inner_text()
                    assert await page.locator('#exsvg circle.mk').count() == 2
                    assert not await page.locator('#exfront').is_visible()
                    assert await page.locator('#extable th[data-key=kld]').count() == 0
                    if id_ == 'unknown':
                        await page.locator('#exsvg circle.mk').first.press('Enter')
                        assert 'Measurement publisher: original evaluator unverified' in await page.locator('#exdetail').inner_text()
                        await page.keyboard.press('Escape')
                assert 'compatibility' in await page.locator('#exscope').inner_text() or 'ranking' in await page.locator('#exscope').inner_text()
                assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
                assert not errors, errors
                print(f'PASS: comparison UI at {width}px: strict per-metric permission, fail closed, ownership, evidence, all points retained, no overflow')
                await context.close()
            await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
