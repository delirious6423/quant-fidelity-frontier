"""Explorer regressions for exact shared views, zero values and keyboard controls.

Uses production browser code and real built model pages. The small controlled
fixture isolates boundary values and protocol permissions; no card data changes.
"""
import asyncio
import copy
import json
import re
import tempfile
import threading
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.async_api import TimeoutError as PlaywrightTimeout, async_playwright
from test_comparison_ui import fixture as comparison_fixture

ROOT = Path(__file__).resolve().parents[1]


def production_script(source):
    return re.sub(
        r'<script>(.*?)</script>',
        lambda m: '<script>' + (ROOT / 'web/assets/site_script.js').read_text() + '\n' +
        (ROOT / 'web/assets/site_explorer.js').read_text() + '</script>' if 'var dataEl' in m[1] else m[0],
        source, flags=re.S)


def fixture():
    source = comparison_fixture()
    data = json.loads(re.search(r'<script type="application/json" id="exdata">(.*?)</script>', source, re.S)[1])
    for measurement in data['sets']:
        first, second = measurement['points']
        first.update(x=10.111111, fmt='GGUF')
        second.update(x=20.222222, fmt='EXL3')
        third = copy.deepcopy(second)
        third.update(qid='third&quant=3#' + measurement['id'], measurement_id='measurement-2',
                     variant='Third quant', x=30.333333, fmt='GGUF',
                     m={'kld': {'v': .3, 'raw': '0.3'}, 'ppl': {'v': 4, 'raw': '4'}})
        reference = copy.deepcopy(third)
        reference.update(qid='reference-' + measurement['id'], measurement_id='measurement-ref',
                         variant='Zero reference', x=40.444444, is_ref=True,
                         m={'kld': {'v': 0, 'raw': '0'}, 'ppl': {'v': 1, 'raw': '1'}})
        measurement['points'].extend([third, reference])
    return re.sub(r'(<script type="application/json" id="exdata">).*?(</script>)',
                  lambda m: m[1] + json.dumps(data) + m[2], source, flags=re.S)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


async def copy_view(page):
    await page.evaluate('navigator.clipboard.writeText = async (text) => { window.copiedView = text; }')
    # Keyboard activation also works while the mobile selected-quant sheet is open.
    await page.locator('#exlink').press('Enter')
    return await page.evaluate('window.copiedView')


async def zero_values(page, base):
    await page.goto(base + 'fixture.html?set=verified')
    assert await page.locator('#exsvg circle.mk').count() == 3
    assert '1 zero or negative' in await page.locator('#exchartnote').inner_text()
    await page.locator('.ex-options > summary').click()
    await page.click('#exlin')
    assert await page.locator('#exsvg circle.mk').count() == 4, 'linear KLD must retain published zero values'
    assert 'cannot appear' not in await page.locator('#exchartnote').inner_text()
    await page.locator('#exyr').evaluate('(e) => { e.value = 0; e.dispatchEvent(new Event("input")); }')
    assert await page.locator('#exsvg circle.mk').count() == 1, 'a zero quality cap must retain the zero reference'
    assert '0' in await page.locator('#exyr').get_attribute('aria-valuetext')
    await page.click('#exlog')
    assert await page.locator('#exsvg circle.mk').count() == 0
    assert await page.locator('#exyr').input_value() == '0'
    assert 'Linear' in await page.locator('#exsvg').inner_text(), 'empty log charts must explain a zero threshold'
    await page.click('#exrclear')
    assert await page.locator('#exsvg circle.mk').count() == 3
    assert 'NaN' not in await page.locator('#exsvg').inner_html()


async def shared_state(page, base):
    await page.goto(base + 'fixture.html?set=verified&max=20.222222&ylim=0.2&scale=linear')
    await page.locator('#quant-picker > summary').click()
    checkbox = page.locator('#expick .pitem input').first
    hidden_id = await checkbox.evaluate('(e) => e.closest("label").dataset.qid')
    await checkbox.uncheck()
    before = await page.locator('#extable tbody tr[data-qid]').evaluate_all('(rs) => rs.map(r => r.dataset.qid)')
    assert len(before) == 1
    shared = await copy_view(page)
    query = parse_qs(urlparse(shared).query)
    assert float(query['max'][0]) == 20.222222, 'display rounding must not change the size cap'
    assert float(query['ylim'][0]) == .2, 'the quality limit must survive sharing'
    assert query['scale'] == ['linear']
    assert query['hide'] == [hidden_id]
    await page.goto(shared)
    after = await page.locator('#extable tbody tr[data-qid]').evaluate_all('(rs) => rs.map(r => r.dataset.qid)')
    assert after == before, 'opening the copied URL must reproduce the exact filtered population'
    assert await page.locator('#exlin').get_attribute('aria-pressed') == 'true'
    assert await page.locator('#exyr').get_attribute('aria-valuetext') == '≤ 0.2'

    # Thresholds selected by a slider also round-trip without display rounding.
    await page.goto(base + 'fixture.html?set=verified')
    await page.locator('.ex-options > summary').click()
    await page.locator('#exxr').evaluate('(e) => { e.value = 500; e.dispatchEvent(new Event("input")); }')
    assert await page.locator('#exsvg circle.mk').count() == 2
    shared = await copy_view(page)
    assert float(parse_qs(urlparse(shared).query)['max'][0]) == 10.111111 + .5 * (30.333333 - 10.111111)
    await page.goto(shared)
    assert await page.locator('#exsvg circle.mk').count() == 2, 'an inclusive boundary must retain the same quant'

    # Empty/mostly hidden selections use the shorter allowlist representation.
    await page.locator('#quant-picker > summary').click()
    await page.click('#exnone')
    empty_link = await copy_view(page)
    query = parse_qs(urlparse(empty_link).query)
    assert query['selection'] == ['only'] and 'hide' not in query and 'pick' not in query
    await page.goto(empty_link)
    assert await page.locator('#exsvg circle.mk').count() == 0
    await page.locator('#quant-picker > summary').click()
    selected = page.locator('#expick .pitem input').nth(2)
    selected_id = await selected.evaluate('(e) => e.closest("label").dataset.qid')
    await selected.check()
    sparse_link = await copy_view(page)
    assert parse_qs(urlparse(sparse_link).query)['pick'] == [selected_id]
    await page.goto(sparse_link)
    assert await page.locator('#expick .pitem input:checked').count() == 1
    assert await page.locator('#expick .pitem input:checked').evaluate('(e) => e.closest("label").dataset.qid') == selected_id

    # Presentation choices and the selected source point are part of the view.
    await page.goto(base + 'fixture.html?set=verified')
    await page.locator('.ex-options > summary').click()
    await page.check('#exlabels')
    await page.check('#exfront')
    await page.locator('#quant-results > summary').click()
    await page.locator('#extable th[data-key=_x]').click()
    await page.locator('#extable th[data-key=_x]').click()
    point = page.locator('#exsvg circle.mk').first
    point_id = await point.get_attribute('data-qid')
    await point.press('Enter')
    presentation = await copy_view(page)
    query = parse_qs(urlparse(presentation).query)
    assert query['labels'] == ['1'] and query['frontier'] == ['1']
    assert query['sort'] == ['_x'] and query['dir'] == ['desc']
    assert query['point'] == [point_id]
    await page.goto(presentation)
    assert await page.locator('#exlabels').is_checked()
    assert await page.locator('#exfront').is_checked()
    assert await page.locator('#extable th[data-key=_x]').get_attribute('aria-sort') == 'descending'
    assert await page.locator('#exdetail').is_visible()
    assert await page.locator('#extable tr.sel').get_attribute('data-qid') == point_id

    # A copied labels-off view wins over the recipient's saved labels preference.
    await page.goto(base + 'fixture.html?set=verified&labels=0')
    compact_link = await copy_view(page)
    query = parse_qs(urlparse(compact_link).query)
    assert 'hide' not in query and 'pick' not in query and 'selection' not in query
    await page.evaluate('localStorage.setItem("qcl.labels", "1")')
    await page.goto(compact_link)
    assert not await page.locator('#exlabels').is_checked()
    await page.goto(base + 'fixture.html')
    assert await page.locator('#exlabels').is_checked(), 'an ordinary visit keeps the existing saved preference'

    # Malformed limits and forged identities never create undefined state.
    await page.goto(base + 'fixture.html?set=verified&max=Infinity&ylim=bad&hide=missing&sort=unknown&scale=bad')
    assert await page.locator('#exsvg circle.mk').count() == 3
    assert await page.locator('#exlog').get_attribute('aria-pressed') == 'true'
    await page.goto(base + 'fixture.html?set=verified&max=0')
    assert await page.locator('#exsvg circle.mk').count() == 0


async def keyboard_controls(page, base):
    await page.goto(base + 'fixture.html?set=verified')
    metrics = page.locator('#exmetric [role=radio]')
    assert await metrics.evaluate_all('(bs) => bs.filter(b => b.tabIndex === 0).length') == 1
    await metrics.first.focus()
    await page.keyboard.press('ArrowRight')
    assert await metrics.nth(1).get_attribute('aria-checked') == 'true'
    assert await page.evaluate('document.activeElement.getAttribute("aria-checked")') == 'true'
    await page.keyboard.press('Home')
    assert await metrics.first.get_attribute('aria-checked') == 'true'
    await page.keyboard.press('ArrowLeft')
    assert await metrics.nth(1).get_attribute('aria-checked') == 'true'
    await page.keyboard.press('End')
    assert await metrics.nth(1).get_attribute('aria-checked') == 'true'
    await page.keyboard.press('Home')
    await page.locator('#quant-picker > summary').click()
    chip = page.locator('#exfmts .fchip').first
    await chip.focus()
    text = await chip.inner_text()
    await page.keyboard.press('Enter')
    assert await page.evaluate('document.activeElement.textContent') == text, 'format filtering must preserve keyboard focus'
    await page.keyboard.press('Enter')
    checkbox = page.locator('#expick .pitem input').first
    id_ = await checkbox.get_attribute('id')
    await checkbox.focus()
    await page.keyboard.press('Space')
    assert await page.evaluate('document.activeElement.id') == id_, 'picker updates must preserve checkbox focus'
    await page.keyboard.press('Space')
    assert await checkbox.is_checked()
    await page.locator('#quant-results > summary').click()
    header = page.locator('#extable th[data-key=_x]')
    await header.focus()
    await page.keyboard.press('Enter')
    assert await page.evaluate('document.activeElement.dataset.key') == '_x', 'sort updates must preserve header focus'
    assert await header.get_attribute('aria-sort') == 'ascending'
    await page.keyboard.press('Enter')
    assert await header.get_attribute('aria-sort') == 'descending'
    legend = page.locator('#exlegend .lg').first
    await legend.focus()
    label = await legend.text_content()
    await page.keyboard.press('Enter')
    assert await page.evaluate('document.activeElement.textContent') == label, 'legend updates must preserve button focus'
    await page.keyboard.press('Enter')
    assert await page.locator('#exsvg circle.mk').count() == 3


async def real_zero_claims(page, base):
    checked = 0
    for slug in ('qwen3.8-27b', 'deepseek-v4.1-flash', 'deepseek-v4-flash-0731'):
        await page.goto(base + slug + '.html?set=all')
        data = json.loads(await page.locator('#exdata').text_content())
        overview = next(s for s in data['sets'] if s['id'] == 'all')
        expected = [p for p in overview['points'] if p['m'].get('kld', {}).get('v') is not None]
        zero = sum(p['m']['kld']['v'] == 0 for p in expected)
        assert zero > 0
        before = await page.locator('#exsvg circle.mk').count()
        await page.locator('.ex-options > summary').click()
        await page.click('#exlin')
        assert await page.locator('#exsvg circle.mk').count() == len(expected), slug
        assert len(expected) - before == zero, slug
        checked += zero
    print(f'PASS real source claims: {checked} published zero KLD values remain visible on linear across 3 models')


async def sizeless_claims(page, base):
    checked = 0
    for slug in ('qwen3.8-27b', 'deepseek-v4.1-flash', 'deepseek-v4-flash-0731'):
        await page.goto(base + slug + '.html?set=all')
        data = json.loads(await page.locator('#exdata').text_content())
        overview = next(s for s in data['sets'] if s['id'] == 'all')
        unsized = overview.get('unsized', [])
        assert unsized, slug
        k = overview['metrics'][0]
        with_metric = [p for p in unsized if p['m'].get(k, {}).get('v') is not None and (not data['metrics'][k].get('log') or p['m'][k]['v'] > 0)]
        assert await page.locator('#exsvg circle.umk').count() == len(with_metric), slug
        assert await page.locator('#exunsized tbody tr').count() == len(unsized), slug
        raws = await page.locator('#exunsized tbody td.n').all_text_contents()
        assert all(p['m'][m]['raw'] in raws for p in unsized for m in p['m']), slug   # every value as published
        await page.locator('.ex-options > summary').click()
        await page.locator('#expresets button').first.click()   # a size limit: an unknown size cannot meet it
        assert await page.locator('#exsvg circle.umk').count() == 0, slug
        assert await page.locator('#exunsized tbody tr').count() == 0, slug
        checked += len(unsized)
    await page.goto(base + 'deepseek-v4.1-flash.html?set=all')   # no size limit
    await page.locator('#exsvg circle.umk').first.click()   # a size-less dot pins its tip; the document click handler leaves it
    assert await page.locator('#extip').is_visible(), 'size-less dot stays pinned'
    assert 'not published' in await page.locator('#extip').text_content()
    await page.keyboard.press('Escape')
    await page.locator('#exsvg circle.mk').first.focus()   # dots can overlap, so open one from the keyboard
    await page.keyboard.press('Enter')
    box = await page.locator('#exdetail .quant-close').bounding_box()
    assert box and box['height'] <= 40 and box['width'] >= box['height'], 'close button keeps its label on one line'
    print(f'PASS size-less claims: {checked} beside the chart and in the table across 3 models, hidden under a size limit')


async def negative_threshold(page, base):
    await page.goto(base + 'deepseek-v4.1-flash.html?set=a-atomicchat-nvfp4-nvidia&m=ppl_delta&ylim=-0.13')
    data=json.loads(await page.locator('#exdata').text_content())
    group=next(s for s in data['sets'] if s['id']=='a-atomicchat-nvfp4-nvidia')
    expected=[p for p in group['points'] if p['m'].get('ppl_delta',{}).get('v') is not None and p['m']['ppl_delta']['v']<=-0.13]
    assert expected and any(p['m']['ppl_delta']['v']==-0.13 for p in expected)
    assert await page.locator('#exsvg circle.mk').count()==len(expected)
    print('PASS exact negative threshold retains the real -0.13% claim')


async def main(base):
    errors = []
    failures = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for width in (390, 1366):
            for check in (zero_values, shared_state, keyboard_controls):
                page = await browser.new_page(viewport={'width': width, 'height': 900})
                await page.route(re.compile(r'^https?://(?!127\.0\.0\.1:)'), lambda route: route.abort())
                page.on('pageerror', lambda error: errors.append(str(error)))
                try:
                    try:
                        await check(page, base)
                    except PlaywrightTimeout:   # a loaded CI runner can stall a click once (CI run 210); a second timeout is real
                        print(f'RETRY {width}px: {check.__name__} timed out once', flush=True)
                        await page.close()
                        page = await browser.new_page(viewport={'width': width, 'height': 900})
                        await page.route(re.compile(r'^https?://(?!127\.0\.0\.1:)'), lambda route: route.abort())
                        page.on('pageerror', lambda error: errors.append(str(error)))
                        await check(page, base)
                    print(f'PASS {width}px: {check.__name__}')
                except Exception as error:
                    traceback.print_exc()
                    failures.append(f'{width}px {check.__name__}: {error}')
                await page.close()
        page = await browser.new_page(viewport={'width': 390, 'height': 900})
        await page.route(re.compile(r'^https?://(?!127\.0\.0\.1:)'), lambda route: route.abort())
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            await real_zero_claims(page, base)
            await sizeless_claims(page, base)
            await negative_threshold(page, base)
        except Exception as error:
            traceback.print_exc()
            failures.append(f'real_zero_claims: {error}')
        await browser.close()
    assert not errors, errors
    assert not failures, '\n'.join(failures)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, 'fixture.html').write_text(fixture())
        for slug in ('qwen3.8-27b', 'deepseek-v4.1-flash', 'deepseek-v4-flash-0731'):
            source = (ROOT / 'models' / slug / 'site/index.html').read_text()
            Path(tmp, slug + '.html').write_text(production_script(source))
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=tmp))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            asyncio.run(main(f'http://127.0.0.1:{server.server_port}/'))
        finally:
            server.shutdown()
