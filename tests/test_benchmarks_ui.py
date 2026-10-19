"""Real-page desktop/mobile pilot checks; run after the gated GLM build."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for width in (320, 390, 1366):
            page = await browser.new_page(viewport={'width': width, 'height': 1000})
            await page.route('https://**/*', lambda route: route.abort())
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            url = (ROOT / 'models/glm-5.3-flash/site/index.html').as_uri()
            await page.goto(url)
            assert await page.locator('#chart').is_visible()
            await page.locator('#tab-benchmarks').click()
            await page.go_back()
            assert await page.locator('#chart').is_visible()
            assert not await page.locator('#benchmarks').is_visible()
            await page.go_forward()
            assert await page.locator('#benchmarks').is_visible()
            await page.goto(url + '#benchmarks')
            await page.locator('#bm-plot svg').wait_for()
            assert await page.locator('#bm-plot .bm-dot').count() == 2
            assert await page.locator('#bm-metric option').count() == 4
            assert not await page.locator('#chart').is_visible()
            assert await page.locator('#benchmarks').is_visible()
            await page.locator('#tab-fidelity').click()
            assert await page.locator('#chart').is_visible()
            assert not await page.locator('#benchmarks').is_visible()
            assert await page.locator('#exset').input_value() == 'all'
            assert not await page.locator('#quant-picker').evaluate('(e) => e.open')
            assert not await page.locator('#quant-results').evaluate('(e) => e.open')
            out = ROOT / 'test-results'; out.mkdir(exist_ok=True)
            await page.screenshot(path=str(out / f'glm-tabs-fidelity-{width}.png'))
            await page.locator('#quant-picker > summary').click()
            assert await page.locator('#exq').is_visible()
            await page.locator('#quant-results > summary').click()
            assert await page.locator('#extable').is_visible()
            await page.locator('#tab-fidelity').press('ArrowRight')
            assert await page.locator('#benchmarks').is_visible()
            await page.go_back()
            assert await page.locator('#chart').is_visible()
            await page.go_forward()
            assert await page.locator('#benchmarks').is_visible()

            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert '98.4' in await page.locator('#bm-detail').inner_text()
            await page.screenshot(path=str(out / f'glm-tabs-benchmarks-{width}.png'))
            await page.select_option('#bm-pick', '1')
            assert '98.8' in await page.locator('#bm-detail').inner_text()
            assert '328.35' in await page.locator('#bm-detail').inner_text()
            await page.locator('#tab-fidelity').click()
            await page.locator('#tab-benchmarks').click()
            assert await page.locator('#bm-pick').input_value() == '1'
            await page.check('#bm-zero')
            assert 'zero baseline' in await page.locator('#bm-axis').inner_text()
            await page.fill('#bm-limit', '200')
            assert '0 within file-size limit' in await page.locator('#bm-status').inner_text()
            await page.select_option('#bm-metric', '1')
            assert await page.locator('#bm-plot .bm-dot').count() == 1
            assert '90.4' in await page.locator('#bm-detail').inner_text()
            assert 'no pairwise comparison' in await page.locator('#bm-status').inner_text()
            await page.select_option('#bm-metric', '0')
            await page.uncheck('#bm-zero')
            await page.fill('#bm-limit', '')
            await page.locator('#bm-plot a').nth(1).click()
            assert '98.8' in await page.locator('#bm-detail').inner_text()
            assert await page.evaluate("document.activeElement === document.querySelectorAll('#bm-plot a')[1]")
            assert not errors, errors
            out = ROOT / 'test-results'; out.mkdir(exist_ok=True)
            await page.locator('#benchmarks').screenshot(path=str(out / f'glm-benchmarks-{width}.png'))
            print(f'PASS {width}px: 2/1-point charts, selection, axes, size limit, no overflow or JS errors')
            await page.close()
        await browser.close()


asyncio.run(main())
