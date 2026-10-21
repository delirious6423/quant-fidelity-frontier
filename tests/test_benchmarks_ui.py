"""Every model, every reviewed benchmark and source; desktop/mobile behavior."""
import asyncio,json,copy,re,sys
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0, str(ROOT / 'web'))   # site generators live in web/
from benchmarks import group_benchmarks


async def bench_value(page, name):
    return await page.evaluate("(n) => [...document.querySelectorAll('#bm-metric option')].find(o => o.textContent.startsWith(n + ' · ')).value", name)

SLUGS=sorted(p.parent.name for p in (ROOT/'models').glob('*/benchmarks.json'))
async def check_combined_sources(browser):
    # Test-only second report uses the same artifact/coordinates: neither may disappear.
    html=(ROOT/'models/qwen3.8-27b/site/index.html').read_text()
    data=json.loads(re.search(r'<script id="bm-data" type="application/json">(.*?)</script>',html).group(1))
    metric=copy.deepcopy(next(m for m in data['sources'][0]['metrics'] if m['name']=='GPQA-Diamond'))
    metric['points']=metric['points'][:1]
    extra=copy.deepcopy(data['sources'][0]);extra['label']='Fixture second evaluation';extra['metrics']=[metric]
    data['sources'].append(extra);data['groups']=group_benchmarks(data['sources'])
    html=re.sub(r'(<script id="bm-data" type="application/json">).*?(</script>)',lambda m:m[1]+json.dumps(data).replace('<','\\u003c')+m[2],html)
    for width in (320,390,1366):
        page=await browser.new_page(viewport={'width':width,'height':1000})
        await page.route('https://**/*',lambda route:route.abort())
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        await page.set_content(html)
        # No source choice is needed to find GPQA or EvalPlus tasks.
        await page.select_option('#bm-metric',value=await bench_value(page,'GPQA-Diamond'))
        assert await page.locator('#bm-source').input_value()=='all'
        assert await page.locator('#bm-plot .bm-dot').count()==9
        assert await page.locator('#bm-sources .bm-source-label').count()==2
        assert await page.locator('#bm-plot .bm-dot').evaluate_all('(dots)=>new Set(dots.map(d=>d.style.fill)).size')==2
        assert 'Not Directly Comparable' in await page.locator('#bm-classification').inner_text()
        await page.select_option('#bm-pick','8')
        assert 'Fixture second evaluation' in await page.locator('#bm-detail').inner_text()
        await page.select_option('#bm-source',str(len(data['sources'])-1))
        assert await page.locator('#bm-plot .bm-dot').count()==1
        await page.select_option('#bm-source','0')
        assert await page.locator('#bm-plot .bm-dot').count()==8
        await page.select_option('#bm-metric',value=await bench_value(page,'HumanEval'))
        assert not await page.locator('#bm-source-choice').is_visible()
        assert await page.locator('#bm-plot .bm-dot').count()==5
        await page.select_option('#bm-metric',value=await bench_value(page,'GPQA-Diamond'))
        assert await page.locator('#bm-source').input_value()=='all'
        assert await page.locator('#bm-plot .bm-dot').count()==9
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors,errors
        await page.close()
        print(f'PASS combined-source fixture {width}px: one-step GPQA, all sources, overlapping results retained, source filter, task switching')

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
                assert await page.locator('#bm-controls > label').first.get_attribute('for') == 'bm-metric'
                assert await page.locator('#bm-metric option').count() == len(d['groups'])
                choices = await page.locator('#bm-metric option').evaluate_all('(options) => options.map(o => ({value:o.value,label:o.textContent}))')
                for group in d['groups']:
                    total = sum(len(d['sources'][e['source']]['metrics'][e['metric']]['points']) for e in group['evaluations'])
                    tail = f" · {total} result" + ('' if total == 1 else 's')
                    # the selector says how many results each benchmark has
                    choice = next(o for o in choices if o['label'] == group['name'] + tail or o['label'] == group['name'] + ' · ' + group['unit'] + tail)
                    await page.select_option('#bm-metric', choice['value'])
                    expected = sum(len(d['sources'][e['source']]['metrics'][e['metric']]['points']) for e in group['evaluations'])
                    assert await page.locator('#bm-plot .bm-dot').count() == expected
                    if len(group['evaluations']) > 1:
                        assert await page.locator('#bm-source').input_value() == 'all'
                    else:
                        assert not await page.locator('#bm-source-choice').is_visible()
                    for evaluation in group['evaluations']:
                        si, mi = evaluation['source'], evaluation['metric']
                        source, metric = d['sources'][si], d['sources'][si]['metrics'][mi]
                        if len(group['evaluations']) > 1:
                            await page.select_option('#bm-source', str(si))
                        assert await page.locator('#bm-plot .bm-dot').count()==len(metric['points'])
                        assert await page.locator('#bm-classification').inner_text()==metric['comparison']['label']
                        point=metric['points'][-1]
                        await page.select_option('#bm-pick',str(len(metric['points'])-1))
                        assert point['raw'] in await page.locator('#bm-detail').inner_text()
                        assert source['label'] in await page.locator('#bm-detail').inner_text()
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
                await page.select_option('#bm-metric', choices[0]['value'])
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
        await check_combined_sources(browser)
        await browser.close()
asyncio.run(main())
