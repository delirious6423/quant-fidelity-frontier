"""Browser contracts for compact model pages, source evidence, and unchanged set selection.

Run after building all live models and assembling _site. Uses the GitHub Pages
project prefix and blocks external requests so results do not depend on analytics.
"""
import asyncio
import json
import re
import tempfile
import threading
import traceback
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / '_site'


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


async def check_model(browser, base, model):
    context = await browser.new_context(viewport={'width': 390, 'height': 844})
    page = await context.new_page()
    page.set_default_timeout(5000)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    await page.route(re.compile(r'^https?://(?!127\.0\.0\.1:)'), lambda route: route.abort())
    url = base + model['slug'] + '/'

    async def visit(suffix=''):
        # A fresh document is required to exercise initial hash selection.
        await page.goto('about:blank')
        await page.goto(url + suffix, wait_until='domcontentloaded')
        await page.locator('#exsvg circle.mk').first.wait_for()

    await visit()
    assert await page.input_value('#exset') == 'all', model['slug']
    assert not await page.locator('#exlabels').is_checked()
    source = json.loads(await page.locator('#exdata').text_content())
    sets = source['sets']
    choices = [s for s in sets if s['comparable'] and len(s['points']) >= 2]
    assert len(choices) >= 2
    first, second = choices[:2]
    # Default overview is retained. Compact controls must make the actual plot
    # reachable without several screens of introductory text.
    for width in (320, 390, 1366):
        await page.set_viewport_size({'width': width, 'height': 844})
        await page.evaluate('window.scrollTo(0, 0)')
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (model['slug'], width)
        top = await page.locator('#exsvg').evaluate('(el) => el.getBoundingClientRect().top + scrollY')
        if width < 500:
            assert top < 900, (model['slug'], width, top)
        for selector in ('#coverage', '.ex-options'):
            disclosure = page.locator(selector)
            assert not await disclosure.evaluate('(el) => el.open')
            await disclosure.locator('summary').click()
            assert await disclosure.evaluate('(el) => el.open')
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (model['slug'], width, selector)
            await disclosure.locator('summary').click()

    # Labels keep their source titles, and repeated titles become unambiguous.
    counts = Counter(s['title'] for s in sets)
    labels = await page.locator('#exset option').evaluate_all('(nodes) => Object.fromEntries(nodes.map(n => [n.value, n.textContent]))')
    for item in sets:
        assert item['title'] in labels[item['id']]
        if counts[item['title']] > 1:
            assert all(labels[item['id']] != labels[other['id']] for other in sets if other['id'] != item['id'])

    # Searching a publisher narrows choices without silently changing the chart.
    publisher = first['title'].split(' · ')[0]
    await page.fill('#exsetsearch', publisher)
    assert await page.input_value('#exset') == 'all'
    visible = await page.locator('#exset option').evaluate_all('(nodes) => nodes.filter(n => !n.hidden).map(n => ({id:n.value,label:n.textContent}))')
    assert 1 < len(visible) < len(sets), (model['slug'], publisher)
    assert all(o['id'] == 'all' or publisher.lower() in o['label'].lower() for o in visible)
    await page.fill('#exsetsearch', '')
    assert await page.locator('#exset option').evaluate_all('(nodes) => nodes.every(n => !n.hidden)')

    await page.set_viewport_size({'width': 390, 'height': 844})
    # Overview allows evidence inspection but never a supposedly comparable shortlist.
    await page.locator('#exsvg circle.mk').first.press('Enter')
    assert await page.locator('#exdetail').is_visible()
    assert await page.locator('#exshortadd').count() == 0
    await page.keyboard.press('Escape')
    assert not await page.locator('#exdetail').is_visible()

    await page.select_option('#exset', first['id'])
    point = page.locator('#exsvg circle.mk').first
    qid = await point.get_attribute('data-qid')
    actual = next(p for p in first['points'] if p['qid'] == qid)
    await point.press('Enter')
    assert await page.locator('#exdetail h3').inner_text() == actual['variant']
    shown_labels = await page.locator('#exsvg .plab').evaluate_all('(nodes) => nodes.map(n => n.dataset.qid)')
    assert all(label == qid for label in shown_labels)
    assert await page.locator('#exdetail a').get_attribute('href') == actual['src']
    assert first['reference'] in await page.locator('#exdetail').inner_text()
    assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    await page.click('#exshortadd')
    assert await page.locator('.quant-shortlist li').count() == 1
    await page.click('[data-short-remove]')
    assert await page.locator('.quant-shortlist li').count() == 0
    await page.click('#exshortadd')
    await page.keyboard.press('Escape')
    await page.select_option('#exset', second['id'])
    assert await page.locator('.quant-shortlist li').count() == 0
    assert not await page.locator('#exdetail').is_visible()

    # Existing priority: query > legacy hash > valid saved choice > fresh all.
    await page.evaluate('(id) => localStorage.setItem("qff.set", id)', first['id'])
    await visit()
    assert await page.input_value('#exset') == first['id']
    await visit('#set-' + second['id'])
    assert await page.input_value('#exset') == second['id']
    await visit('?set=all#set-' + second['id'])
    assert await page.input_value('#exset') == 'all'
    await page.evaluate('localStorage.setItem("qff.set", "missing-set")')
    await visit('?set=missing-set#set-missing-set')
    assert await page.input_value('#exset') == 'all'
    assert not errors, (model['slug'], errors)
    print(f"PASS: {model['slug']}: 320/390/1366px, disclosures, search, source evidence, shortlist, initial-set precedence")
    await context.close()


async def main(base):
    models = json.loads((SITE / 'models.json').read_text())['models']
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        # Two concurrent pages bound runtime without multiplying memory by all models.
        semaphore = asyncio.Semaphore(2)
        async def bounded(model):
            async with semaphore:
                await check_model(browser, base, model)
        results = await asyncio.gather(*(bounded(model) for model in models), return_exceptions=True)
        await browser.close()
        failures = [(models[i]['slug'], ''.join(traceback.format_exception(result))) for i, result in enumerate(results) if isinstance(result, BaseException)]
        assert not failures, failures
    print(f'PASS: all {len(models)} live model pages')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / 'quant-fidelity-frontier').symlink_to(SITE, target_is_directory=True)
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=folder))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            asyncio.run(main(f'http://127.0.0.1:{server.server_port}/quant-fidelity-frontier/'))
        finally:
            server.shutdown()
