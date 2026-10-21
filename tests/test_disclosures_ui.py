"""Collapsed evidence sections retain direct navigation and filter state."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
ROOT = Path(__file__).resolve().parents[1]
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for width in (320, 390, 1366):
            page = await browser.new_page(viewport={'width': width, 'height': 900})
            await page.route('https://**/*', lambda route: route.abort())
            errors=[]
            page.on('pageerror', lambda e: errors.append(str(e)))
            url=(ROOT/'models/glm-5.3-flash/site/index.html').as_uri()
            await page.goto(url)
            for ident in ('ledger-details','publisher-details','no-claims-details'):
                assert not await page.locator('#'+ident).evaluate('(e) => e.open')
            await page.locator('#ledger').scroll_into_view_if_needed()
            out=ROOT/'docs/history/previews'; out.mkdir(exist_ok=True)
            await page.screenshot(path=str(out/f'collapsed-evidence-{width}.png'))
            await page.locator('#ledger-details > summary').click()
            await page.fill('#q', 'mbehr90')
            assert not (await page.locator('#count').inner_text()).startswith('0 ')
            await page.locator('#ledger-details > summary').click()
            await page.locator('.toc a[href="#ledger"]').click()
            assert await page.locator('#q').is_visible()
            assert await page.locator('#q').input_value() == 'mbehr90'
            await page.goto(url+'#publishers')
            assert await page.locator('#cq').is_visible()
            await page.fill('#cq','mbehr90')
            assert not (await page.locator('#ccount').inner_text()).startswith('0 ')
            card_id=await page.locator('#cards > *').first.get_attribute('id')
            await page.goto(url+'#'+card_id)
            assert await page.locator('#publisher-details').evaluate('(e) => e.open')
            assert await page.locator('[id="'+card_id+'"]').is_visible()
            await page.goto(url+'#ledger')
            assert await page.locator('#q').is_visible()
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            await page.goto(url+'#no-claims')
            assert await page.locator('#ncq').is_visible()
            await page.select_option('#ncs', 'read in full: no quality claims')
            assert not (await page.locator('#nccount').inner_text()).startswith('0 ')
            await page.locator('#no-claims-details > summary').click()
            await page.locator('#method a[href="#no-claims"]').click()
            assert await page.locator('#ncs').input_value() == 'read in full: no quality claims'
            assert await page.locator('#ncq').is_visible()
            assert not errors, errors
            print(f'PASS {width}px: collapsed defaults, direct section/card links, search state, no overflow or JS errors')
            await page.close()
        page=await browser.new_page(java_script_enabled=False)
        await page.route('https://**/*', lambda route: route.abort())
        await page.goto(url)
        await page.locator('#ledger-details > summary').click()
        await page.locator('#publisher-details > summary').click()
        await page.locator('#no-claims-details > summary').click()
        assert await page.locator('#nctable').is_visible()
        assert await page.locator('#t').is_visible()
        assert await page.locator('#cards').is_visible()
        print('PASS no JavaScript: all three sections open natively')
        await browser.close()
asyncio.run(main())
