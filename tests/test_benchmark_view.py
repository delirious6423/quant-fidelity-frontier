"""Cross-model provenance, exact task matching, navigation and chart interactions."""
import asyncio
import copy
import json
import re
import sys
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'web'))   # site generators live in web/
from benchmark_view import collect_shared, static_plot
SITE = ROOT/'_site'


class SharedData(unittest.TestCase):
    def test_reviewed_shared_coverage_and_lossless_points(self):
        data = json.loads((SITE/'benchmarks/data.json').read_text())
        models=json.loads((SITE/'models.json').read_text())['models']
        expected={}
        for m in models:
            original=json.loads((SITE/m['data']).read_text())
            for source in (original.get('benchmarks') or {}).get('sources',[]):
                for metric in source['metrics']:
                    expected.setdefault((metric['name'],metric['unit']),set()).add(m['slug'])
        expected={k:v for k,v in expected.items() if len(v)>=2}
        self.assertEqual({(v['name'],v['unit']):set(v['models']) for v in data['views']},expected)
        self.assertEqual(len(data['models']),len(models))
        total=sum(len(v['points']) for v in data['views'])
        self.assertEqual(len({p['id'] for v in data['views'] for p in v['points']}),total)
        for v in data['views']:
            self.assertGreaterEqual(len(v['models']),2)
            for p in v['points']:
                original = json.loads((SITE/p['slug']/'data.json').read_text())['benchmarks']
                _, si, mi, pi = p['id'].split(':')
                source = original['sources'][int(si)]
                metric = source['metrics'][int(mi)]
                self.assertEqual((metric['name'],metric['unit']),(v['name'],v['unit']))
                self.assertTrue(all(p[k]==value for k,value in metric['points'][int(pi)].items()))
                self.assertEqual(p['context'],source['context'])
                self.assertEqual(p['card_sha256'],source['card_sha256'])

    def test_variants_units_missing_sizes_and_repeated_reports(self):
        point = dict(size=10,score=.8,variant='build',raw='0.8',src='score',size_src='size')
        source = dict(label='source',context='setup',notes=[],card_sha256='hash',revision='revision',
                      metrics=[dict(name='GPQA-Diamond',unit='fraction',points=[point])])
        models = [dict(slug=s,model=s+'/model',data=s+'/data.json') for s in ['a','b']]
        with tempfile.TemporaryDirectory() as tmp:
            for model in models:
                target=Path(tmp)/model['data'];target.parent.mkdir()
                target.write_text(json.dumps(dict(benchmarks=dict(sources=[copy.deepcopy(source)]))))
            self.assertEqual(len(collect_shared(tmp,models)['views'][0]['points']),2)
            target=Path(tmp)/models[1]['data'];data=json.loads(target.read_text())
            for field,value in [('name','GPQA'),('unit','percent')]:
                changed=copy.deepcopy(data);changed['benchmarks']['sources'][0]['metrics'][0][field]=value
                target.write_text(json.dumps(changed));self.assertEqual(collect_shared(tmp,models)['views'],[])
            for value in [None,0,float('nan')]:
                changed=copy.deepcopy(data);changed['benchmarks']['sources'][0]['metrics'][0]['points'][0]['size']=value
                target.write_text(json.dumps(changed));self.assertEqual(collect_shared(tmp,models)['views'],[])
            changed=copy.deepcopy(data);changed['benchmarks']['sources'].append(copy.deepcopy(source))
            target.write_text(json.dumps(changed));view=collect_shared(tmp,models)['views'][0]
            self.assertEqual(len(view['points']),3)  # same build, separate reports retained
        self.assertIn('No benchmark',static_plot(None,{}))


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self,*args): pass


async def browser_checks(base):
    data=json.loads((SITE/'benchmarks/data.json').read_text())
    default=data['views'][0] if data['views'] else None
    total=sum(len(v['points']) for v in data['views'])
    async with async_playwright() as p:
        browser=await p.chromium.launch()
        for width in (320,390,768,1366):
            page=await browser.new_page(viewport={'width':width,'height':844})
            await page.route(re.compile(r'^https?://(?!127\.0\.0\.1:)'),lambda route:route.abort())
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            await page.goto(base+'benchmarks/')
            if not default:
                assert 'No benchmark' in await page.locator('#shared-plot').inner_text()
                assert not await page.locator('#shared-controls').is_visible()
                await page.close();continue
            assert await page.locator('#shared-benchmark').input_value()=='0'
            assert await page.locator('.shared-dot').count()==len(default['points'])
            assert not await page.locator('#shared-options').evaluate('(e)=>e.open')
            assert await page.locator('#shared-options > summary').inner_text()=='Filters & chart options'
            assert not await page.locator('#shared-size').is_visible()
            assert await page.locator('.shared-dot').evaluate_all('(ds)=>new Set(ds.map(d=>d.style.fill)).size')==len(default['models'])
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            if width<500:
                assert await page.locator('#shared-plot').evaluate('(e)=>e.getBoundingClientRect().top')<550
            await page.locator('#shared-options > summary').click()
            assert await page.locator('#shared-ranges input[type=range]').count()==2
            assert await page.locator('#shared-ranges input[type=number]').count()==0
            for vi,view in enumerate(data['views']):
                await page.select_option('#shared-benchmark',str(vi))
                assert await page.locator('.shared-dot').count()==len(view['points'])
                assert await page.locator('#shared-size-output').inner_text()=='any'
                assert await page.locator('#shared-score-output').inner_text()=='any'
                assert not await page.locator('#shared-reset').is_visible()
                assert await page.locator('#shared-models input:checked').count()==len(view['models'])
                for point in view['points']:
                    await page.select_option('#shared-pick',point['id'])
                    assert point['model_name'] in await page.locator('#shared-detail h2').inner_text()
                    assert point['raw'] in await page.locator('.point-score').inner_text()
                    assert await page.locator('#shared-detail a').nth(0).get_attribute('href')==point['src']
                    assert await page.locator('#shared-detail a').nth(1).get_attribute('href')==point['size_src']
                for box in await page.locator('#shared-models input').all():await box.uncheck()
                assert await page.locator('.shared-dot').count()==0
                assert 'No results match' in await page.locator('#shared-plot').inner_text()
                assert not await page.locator('#shared-inspect').is_visible()
                for box in await page.locator('#shared-models input').all():await box.check()
                # Same normalized sliders and higher-is-better threshold as the model explorer.
                for axis,t in [('size',500),('score',400)]:
                    await page.locator('#shared-'+axis).evaluate('(e,t)=>{e.value=t;e.dispatchEvent(new Event("input"))}',t)
                cap=min(p['size'] for p in view['points'])+.5*(max(p['size'] for p in view['points'])-min(p['size'] for p in view['points']))
                floor=min(p['score'] for p in view['points'])+.4*(max(p['score'] for p in view['points'])-min(p['score'] for p in view['points']))
                expected=[p for p in view['points'] if p['size']<=cap+1e-9 and p['score']>=floor-1e-9]
                assert await page.locator('.shared-dot').count()==len(expected)
                assert '≤' in await page.locator('#shared-size-output').inner_text()
                assert '≥' in await page.locator('#shared-score-output').inner_text()
                assert await page.locator('#shared-zero').is_disabled()
                assert float(await page.locator('#shared-plot svg').get_attribute('data-x-max'))==cap
                assert float(await page.locator('#shared-plot svg').get_attribute('data-y-min'))==floor
                assert 'limits active' in await page.locator('#shared-options > summary').inner_text()
                # Bounds do not shift when a model/reference is hidden.
                outputs=await page.locator('#shared-ranges output').all_text_contents()
                box=page.locator('#shared-models input').first;await box.uncheck();await box.check()
                assert await page.locator('#shared-ranges output').all_text_contents()==outputs
                await page.click('#shared-reset')
                assert await page.locator('.shared-dot').count()==len(view['points'])
            # Exact inclusive thresholds supplied in a share link retain repeated reports.
            await page.select_option('#shared-benchmark','0')
            point=default['points'][len(default['points'])//2]
            from urllib.parse import urlencode
            exact=base+'benchmarks/?'+urlencode({'benchmark':default['name'],'max':point['size'],'score_min':point['score']})
            await page.goto(exact)
            expected=[p for p in default['points'] if p['size']<=point['size'] and p['score']>=point['score']]
            assert await page.locator('.shared-dot').count()==len(expected)
            assert (await page.locator('#shared-options > summary').inner_text()).endswith('limits active')
            await page.locator('#shared-options > summary').click()
            await page.evaluate('navigator.clipboard.writeText=async (text)=>{window.sharedLink=text;}')
            await page.click('#shared-copy');shared=await page.evaluate('window.sharedLink')
            query=parse_qs(urlparse(shared).query)
            assert float(query['max'][0])==point['size'] and float(query['score_min'][0])==point['score']
            await page.goto(shared)
            assert await page.locator('.shared-dot').count()==len(expected)
            await page.locator('#shared-options > summary').click()
            await page.click('#shared-reset')
            handle=page.locator('#shared-size');await handle.focus()
            await page.keyboard.press('ArrowLeft')
            assert await page.locator('#shared-size-output').inner_text()!='any'
            await page.click('#shared-reset')
            await page.locator('#shared-score').focus();await page.keyboard.press('ArrowRight')
            assert await page.locator('#shared-score-output').inner_text()!='any'
            await page.click('#shared-reset')
            await handle.scroll_into_view_if_needed();track=await handle.bounding_box()
            await page.mouse.move(track['x']+track['width']-8,track['y']+22)
            await page.mouse.down();await page.mouse.move(track['x']+track['width']*.5,track['y']+22,steps=8);await page.mouse.up()
            assert await page.locator('#shared-size-output').inner_text()!='any'
            await page.click('#shared-reset')
            await page.uncheck('#shared-refs')
            assert await page.locator('.shared-dot').count()==sum(not p['reference'] for p in default['points'])
            await page.check('#shared-log');await page.uncheck('#shared-zero')
            assert 'log scale' in await page.locator('#shared-plot').inner_text()
            # Responsive repaints retain keyboard access to the same report.
            marker=page.locator('#shared-plot a').first;await marker.focus()
            id_=await marker.get_attribute('data-point');old=await marker.element_handle()
            await page.set_viewport_size({'width':min(width+5,1100),'height':844})
            await page.wait_for_function('(marker)=>!marker.isConnected',arg=old)
            assert await page.evaluate('document.activeElement.getAttribute("data-point")')==id_
            await page.keyboard.press('Enter');assert await page.locator('#shared-pick').input_value()==id_
            await page.locator('.page-picker > summary').click()
            assert await page.locator('.page-menu a').count()==len(data['models'])
            providers=await page.locator('.page-provider').all_text_contents();assert providers==sorted(providers,key=str.casefold)
            await page.locator('.page-picker a').first.focus()
            await page.keyboard.press('Escape');assert not await page.locator('.page-picker').evaluate('(e)=>e.open')
            assert await page.locator('.page-picker > summary').evaluate('(e)=>e===document.activeElement')
            for query,expected_count in [('max=Infinity&score_min=bad',len(default['points'])),('max=0&scale=log',0)]:
                await page.goto(base+'benchmarks/?'+query)
                assert await page.locator('.shared-dot').count()==expected_count
                assert 'NaN' not in await page.locator('#shared-plot svg').inner_html()
            await page.locator('#shared-options > summary').click();await page.click('#shared-reset')
            assert await page.locator('.shared-dot').count()==len(default['points'])
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            assert not errors,errors
            await page.close()
            print(f'PASS {width}px: KLD-style Max size/Min score, exact thresholds, all reports, reset, sharing, pointer/keyboard and navigation')
        if default:
            touch=await browser.new_context(viewport={'width':390,'height':844},has_touch=True,is_mobile=True)
            page=await touch.new_page();await page.goto(base+'benchmarks/')
            await page.locator('#shared-options > summary').tap()
            client=await touch.new_cdp_session(page)
            for axis,initial,target in [('size',.98,.5),('score',.02,.6)]:
                handle=page.locator('#shared-'+axis);await handle.scroll_into_view_if_needed();track=await handle.bounding_box()
                for kind,fraction in [('touchStart',initial),('touchMove',target),('touchEnd',0)]:
                    await client.send('Input.dispatchTouchEvent',{'type':kind,'touchPoints':[] if kind=='touchEnd' else [{'x':track['x']+track['width']*fraction,'y':track['y']+22}]})
                assert await page.locator('#shared-'+axis+'-output').inner_text()!='any'
            await page.evaluate('window.scrollTo(0,document.querySelector("#shared-options").getBoundingClientRect().top+scrollY-8)')
            assert await page.locator('#shared-plot').evaluate('(e)=>e.getBoundingClientRect().bottom<=innerHeight')
            assert await page.locator('#shared-size').evaluate('(e)=>e.getBoundingClientRect().top>=0')
            assert await page.locator('#shared-score').evaluate('(e)=>e.getBoundingClientRect().top>=0')
            await page.locator('#shared-reset').tap()
            assert await page.locator('.shared-dot').count()==len(default['points'])
            await touch.close();print('PASS phone touch: size cap, score floor and clear limits')
        page=await browser.new_page(java_script_enabled=False,viewport={'width':390,'height':844})
        await page.goto(base+'benchmarks/')
        assert await page.locator('.shared-dot').count()==(len(default['points']) if default else 0)
        assert await page.locator('#shared-evidence').evaluate('(e)=>e.open')
        assert await page.locator('#shared-evidence tbody tr').count()==total
        assert not await page.locator('#shared-controls').is_visible()
        await page.close()
        page=await browser.new_page()
        await page.route(re.compile(r'^https?://(?!127\.0\.0\.1:)'),lambda route:route.abort())
        await page.goto(base);await page.locator('.benchmark-cta').click();assert page.url==base+'benchmarks/'
        for m in data['models']:
            await page.goto(base+m['slug']+'/');await page.locator('.msw > summary').click()
            assert await page.locator('.msw-panel > :first-child').inner_text()=='All-model benchmarks'
            assert await page.locator('.msw-benchmarks').get_attribute('href')=='../benchmarks/'
        await browser.close()
        print(f'PASS shared-page links on homepage and all {len(data["models"])} model selectors; no-JavaScript chart and {total} evidence rows')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SharedData)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():sys.exit(1)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp)/'quant-fidelity-frontier').symlink_to(SITE,target_is_directory=True)
        server=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=tmp))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:asyncio.run(browser_checks(f'http://127.0.0.1:{server.server_port}/quant-fidelity-frontier/'))
        finally:server.shutdown()
