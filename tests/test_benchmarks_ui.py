"""Every model, every reviewed benchmark and source; desktop/mobile behavior."""
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parents[1]
SLUGS=sorted(p.parent.name for p in (ROOT/'models').glob('*/benchmarks.json'))
async def main():
    async with async_playwright() as p:
        browser=await p.chromium.launch()
        for slug in SLUGS:
            for width in (320,390,1366):
                page=await browser.new_page(viewport={'width':width,'height':1000})
                await page.route('https://**/*',lambda route:route.abort())
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                url=(ROOT/'models'/slug/'site/index.html').as_uri()
                await page.goto(url+'#benchmarks')
                d=await page.locator('#bm-data').text_content();d=json.loads(d)
                if not d['sources']:
                    assert 'No reviewed benchmark' in await page.locator('#bm-plot').inner_text()
                    assert not await page.locator('#bm-controls').is_visible()
                    await page.close()
                    print(f'PASS {slug} {width}px: changed sources withheld, readable empty state')
                    continue
                assert await page.locator('#chart').is_visible()
                assert await page.locator('#benchmarks').is_visible()
                assert await page.evaluate("document.querySelector('#chart').nextElementSibling.id === 'benchmarks'")
                assert not await page.locator('#quant-picker').evaluate('(e)=>e.open')
                assert not await page.locator('#quant-results').evaluate('(e)=>e.open')
                initial_set=await page.locator('#exset').input_value()
                checked=0
                for si,source in enumerate(d['sources']):
                    await page.select_option('#bm-source',str(si))
                    assert await page.locator('#bm-metric option').count()==len(source['metrics'])
                    for mi,metric in enumerate(source['metrics']):
                        await page.select_option('#bm-metric',str(mi))
                        assert await page.locator('#bm-plot .bm-dot').count()==len(metric['points'])
                        assert await page.locator('#bm-classification').inner_text()==metric['comparison']['label']
                        assert metric['comparison']['ranking_permitted'] is False
                        point=metric['points'][-1]
                        await page.select_option('#bm-pick',str(len(metric['points'])-1))
                        assert point['raw'] in await page.locator('#bm-detail').inner_text()
                        assert await page.locator('#bm-detail a').nth(0).get_attribute('href')==point['src']
                        assert await page.locator('#bm-detail a').nth(1).get_attribute('href')==point['size_src']
                        await page.check('#bm-zero')
                        assert 'zero baseline' in await page.locator('#bm-axis').inner_text()
                        await page.uncheck('#bm-zero')
                        await page.fill('#bm-limit','0')
                        assert '0 within file-size limit' in await page.locator('#bm-status').inner_text()
                        await page.fill('#bm-limit','')
                        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                        checked+=1
                await page.select_option('#bm-source','0');await page.select_option('#bm-metric','0')
                await page.locator('#bm-plot a').last.click()
                assert await page.evaluate("document.activeElement === document.querySelector('#bm-plot a:last-of-type')")
                assert await page.locator('#exset').input_value()==initial_set
                assert not errors,errors
                out=ROOT/'test-results';out.mkdir(exist_ok=True)
                await page.locator('#benchmarks').screenshot(path=str(out/f'{slug}-benchmarks-{width}.png'))
                print(f'PASS {slug} {width}px: {checked} source/benchmark views, sources, controls, no overflow/errors')
                await page.close()
            page=await browser.new_page(java_script_enabled=False,viewport={'width':390,'height':1000})
            await page.route('https://**/*',lambda route:route.abort())
            await page.goto(url+'#benchmarks')
            if d['sources']:assert await page.locator('#bm-plot > svg').is_visible()
            else:assert 'No reviewed benchmark' in await page.locator('#bm-plot').inner_text()
            assert await page.locator('#bm-evidence').evaluate('(e)=>e.open')
            assert await page.locator('#bm-evidence tbody tr').count()==sum(len(m['points'])for s in d['sources']for m in s['metrics'])
            await page.close();print(f'PASS {slug}: no-JavaScript chart + all evidence rows')
        await browser.close()
asyncio.run(main())
