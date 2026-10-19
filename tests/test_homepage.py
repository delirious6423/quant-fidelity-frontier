"""Homepage data integrity, responsive layouts, and quick-start → explorer contract."""
import asyncio
import json
import re
import tempfile
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from playwright.async_api import async_playwright
ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / '_site'
TEMP = tempfile.TemporaryDirectory()
(Path(TEMP.name) / 'quant-fidelity-frontier').symlink_to(SITE, target_is_directory=True)
class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args): pass
SERVER = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=TEMP.name))
threading.Thread(target=SERVER.serve_forever, daemon=True).start()
HOME = f'http://127.0.0.1:{SERVER.server_port}/quant-fidelity-frontier/'

async def main():
    manifest = json.loads((SITE / 'models.json').read_text())['models']
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for width in (360, 390, 768, 1366):
            page = await browser.new_page(viewport={'width':width, 'height':900})
            await page.route(re.compile(r'^https?://(?!127\.0\.0\.1:)'), lambda route: route.abort())
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            await page.goto(HOME)
            assert await page.locator('.model-card').count() == len(manifest)
            assert await page.locator('#quick').is_visible()
            assert await page.input_value('#home-model') == 'qwen3.8-27b'
            assert await page.input_value('#home-set') == 'a-atomicchat-gguf'
            assert await page.locator('.preview strong').inner_text() == 'Qwen3.8-27B'
            assert await page.locator('.preview circle').count() == 27
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
            # Expanded provenance and all model options must fit narrow screens too.
            await page.locator('summary').first.click()
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
            for model in manifest:
                await page.select_option('#home-model', model['slug'])
                source = json.loads((SITE / model['data']).read_text())
                catalog = json.loads(await page.locator('#home-data').text_content())
                home = next(m for m in catalog if m['slug'] == model['slug'])
                for item in home['sets']:
                    actual = next(s for s in source['sets'] if s['id'] == item['id'])
                    assert actual['comparable'] and actual['x']['unit'] == 'GB'
                    await page.select_option('#home-set', item['id'])
                    assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                if not home['sets']: continue
                await page.select_option('#home-set', home['sets'][0]['id'])
                available = await page.locator('#home-format option').evaluate_all('(nodes) => nodes.map(n => n.value)')
                await page.select_option('#home-format', available[-1])
                selected_set = await page.input_value('#home-set')
                selected_basis = await page.input_value('#home-basis')
                selected_format = await page.input_value('#home-format')
                await page.click('#start')
                query = parse_qs(urlparse(page.url).query)
                assert query['set'] == [selected_set] and query['fmt'] == [selected_format] and query['basis'] == [selected_basis]
                await page.reload()
                assert '/quant-fidelity-frontier/' in urlparse(page.url).path
                assert await page.input_value('#exset') == selected_set
                assert await page.locator('#exfmts button[aria-pressed="true"]').count() == 1
                assert await page.locator('#exsvg circle.mk').count() > 0
                # Shared links must preserve the exact family selection.
                await page.evaluate("navigator.clipboard.writeText = async function (s) { window.copiedView = s; }")
                await page.click('#exlink')
                shared = await page.evaluate('window.copiedView')
                assert parse_qs(urlparse(shared).query)['fmt'] == [selected_format]
                await page.goto(HOME)
            assert not errors, errors
            print(f'PASS: {width}px layout; every model/set; format + basis deep links and share round-trip')
            await page.close()
        page = await browser.new_page(java_script_enabled=False)
        await page.goto(HOME)
        assert await page.locator('.model-card h3 a').count() == len(manifest)
        assert not await page.locator('#quick').is_visible()
        print('PASS: model navigation remains available without JavaScript')
        await browser.close()

try:
    asyncio.run(main())
finally:
    SERVER.shutdown()
    TEMP.cleanup()
