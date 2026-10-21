"""Interaction tests for the explorer and page. Exit 1 on any failure."""
import asyncio, sys, json, os, re
from playwright.async_api import async_playwright
import os
SRC = "file://" + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("QCL_SITE", "models/qwen3.8-flash-next/site"), "index.html")
def ranking_allowed(s, metric=None):
    return s['id'] != 'all' and bool(s.get('metrics')) and s.get('comparisons', {}).get(metric or s['metrics'][0], {}).get('ranking_permitted') is True

fails, oks = [], []
def check(c, name):
    (oks if c else fails).append(name)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for name, vp in [("desk", {"width": 1366, "height": 900}), ("phone", {"width": 390, "height": 844})]:
            ctx = await b.new_context(viewport=vp, has_touch=(name == "phone"))
            # the page is tested offline: web fonts and analytics are external and optional, and a network that never answers
            # would otherwise stall page load (cloud sessions and some CI runners)
            await ctx.route(lambda u: not u.startswith(("file:", "data:", "blob:", "about:")), lambda r: r.abort())
            ctx.set_default_navigation_timeout(120000); ctx.set_default_timeout(60000)   # big pages on a busy machine: slow is not broken
            pg = await ctx.new_page(); errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: m.type == "error" and "ERR_TUNNEL" not in m.text and "ERR_FAILED" not in m.text and errs.append(m.text))
            await pg.goto(SRC); await pg.wait_for_timeout(500)
            # header: one-line model menu, grouped by org; opens and closes; links open in a new tab
            check(await pg.locator(".topbar .msw > summary").count() == 1, f"{name}: header has one model menu")
            await pg.click(".msw > summary"); await pg.wait_for_timeout(100)
            check(await pg.evaluate("document.querySelector('.msw').open") and await pg.locator(".msw-cur").count() == 1, f"{name}: model menu opens and marks the current model")
            hs = await pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth"); check(hs <= 0, f"{name}: open model menu fits the screen ({hs}px)")
            await pg.keyboard.press("Escape"); await pg.wait_for_timeout(50)
            check(not await pg.evaluate("document.querySelector('.msw').open"), f"{name}: Escape closes the model menu")
            sw = await pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            check(sw <= 0, f"{name}: no horizontal page scroll (got {sw}px)")
            ids = await pg.evaluate("[...document.querySelectorAll('[id]')].map(e=>e.id)")
            dup = sorted({i for i in ids if ids.count(i) > 1}); check(not dup, f"{name}: unique DOM ids {dup[:5]}")
            # expectations derived from this model's data, so the suite works for any model size
            _DX = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("QCL_SITE", "models/qwen3.8-flash-next/site"), "data.json")))
            _ALL = next(x for x in _DX["sets"] if x["id"] == "all"); NALL = len(_ALL["points"])
            _gx = [q["x"] for q in _ALL["points"] if q.get("x") is not None and _ALL["metrics"][0] in q["m"]]   # the overview opens on its first metric
            NTIER = sum(1 for t_ in (8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256) if _gx and min(_gx) < t_ < max(_gx)) if (_ALL.get("x") or {}).get("unit") == "GB" else 0
            NBASIS = len({q.get("basis") or "unknown" for q in _ALL["points"]})
            import collections as _c
            _vs = _c.Counter(); [_vs.update({q["variant"] for q in x["points"]}) for x in _DX["sets"] if x["id"] != "all"]
            FIND, FIND_N = (_vs.most_common(1)[0] if _vs else ("", 0))
            gh = await pg.evaluate("[...document.querySelectorAll('.toc a.ext')].map(a=>a.href)")
            check(gh == ["https://github.com/delirious6423/quant-fidelity-frontier"], f"{name}: header links back to the GitHub repo {gh}")
            dflt = await pg.evaluate("[document.getElementById('exset').value, document.getElementById('exset').options[0].value, document.getElementById('exscope').textContent]")
            check(dflt[0] == "all" and dflt[1] == "all" and "Mixed protocols" in dflt[2] and "Not Directly Comparable" in dflt[2], f"{name}: overview of every claim is the default, listed first and labelled not comparable {dflt[:2]}")
            check(await pg.locator("#exsvg circle.mk").count() == sum(1 for q in _ALL["points"] if q["m"].get(_ALL["metrics"][0], {}).get("v", 0) > 0), f"{name}: overview chart draws every claim ({NALL} points)")
            owners = await pg.evaluate("[...document.querySelectorAll('#exset optgroup')].slice(1).map(g=>[...g.querySelectorAll('option')].map(o=>o.textContent.split(' · ')[0].toLowerCase()))")
            check(all(g == sorted(g) for g in owners), f"{name}: sets listed alphabetically by owner within each group")
            await pg.locator(".ex-options > summary").click()
            _first = None
            for _v in await pg.evaluate("[...document.querySelectorAll('#exset optgroup')].slice(1).flatMap(g=>[...g.querySelectorAll('option')].map(o=>o.value))"):
                await pg.select_option("#exset", _v); await pg.wait_for_timeout(40)
                candidate = next(x for x in _DX["sets"] if x["id"] == _v)
                nonrefs = [q for q in candidate["points"] if not q.get("is_ref") and q["m"].get(candidate["metrics"][0], {}).get("v", 0) > 0]
                if ranking_allowed(candidate) and len(nonrefs) >= 2 and await pg.locator("#exlegend .lg:not([hidden]):not(.more)").count() >= 2: _first = _v; break
            if _first is None:   # small models may have no multi-publisher set: use the largest comparable set
                _first = max((x for x in _DX["sets"] if x["id"] != "all"), key=lambda x: (ranking_allowed(x), len(x["points"])))["id"]
            await pg.select_option("#exset", _first); await pg.wait_for_timeout(150)
            opts = await pg.evaluate("[...document.querySelectorAll('#exset option')].map(o=>o.value)")
            check(len(opts) == len(set(opts)) and len(opts) == len(_DX["sets"]), f"{name}: one unique option per set ({len(opts)} of {len(_DX['sets'])})")
            # chart fits container (no clipping)
            fit = await pg.evaluate("(()=>{const s=document.querySelector('#exsvg svg'),h=document.getElementById('exsvg');return s.getBoundingClientRect().right<=h.getBoundingClientRect().right+1})()")
            check(fit, f"{name}: chart fits its box")
            n0 = await pg.locator("#exsvg circle.mk").count()
            await pg.locator('#quant-picker > summary').click()
            await pg.locator('#quant-results > summary').click()
            # picker filter narrows chart and list
            first_set = next(x for x in _DX["sets"] if x["id"] == _first)
            first_metric = first_set["metrics"][0]
            terms = sorted({t for q in first_set["points"] for t in re.findall(r"[A-Za-z0-9_.-]+", q["variant"] + " " + q["repo"])}, key=len, reverse=True)
            for _t in terms:
                await pg.fill("#exq", _t); await pg.wait_for_timeout(150)
                n1 = await pg.locator("#exsvg circle.mk").count(); li = await pg.locator("#expick .pitem").count()
                if 0 < n1 < n0: break
            check(0 < n1 < n0 and li == sum(1 for q in first_set["points"] if _t.lower() in (q["variant"] + " " + q["repo"] + " " + (q.get("fmt") or "")).lower()), f"{name}: picker filter narrows chart ({n0}->{n1}, list {li})")
            await pg.click("#exnone"); await pg.wait_for_timeout(100)
            check(await pg.locator("#exsvg circle.mk").count() == 0, f"{name}: untick matches clears them")
            await pg.click("#exreset"); await pg.wait_for_timeout(100)
            check(await pg.locator("#exsvg circle.mk").count() == n0 and await pg.input_value("#exq") == "", f"{name}: reset restores")
            # format chips
            fch = pg.locator("#exfmts .fchip")
            if await fch.count() > 1:
                await fch.first.click(); await pg.wait_for_timeout(100)
                n2 = await pg.locator("#exsvg circle.mk").count(); check(0 < n2 < n0, f"{name}: format chip filters ({n2})")
                await fch.first.click()
            # legend toggles a repo
            await pg.select_option("#exset", _first); await pg.wait_for_timeout(150)
            lgs = pg.locator("#exlegend .lg:not(.more)"); before = await pg.locator("#exsvg circle.mk").count()
            if await lgs.count() >= 2:
                await lgs.first.click(); await pg.wait_for_timeout(100)
                check(await pg.locator("#exsvg circle.mk").count() < before, f"{name}: legend hides a repo")
                await pg.locator("#exlegend .lg:not(.more)").first.click()
            # tap/click a point pins tooltip with source link and selects row
            _ti = await pg.evaluate("(()=>{const cs=[...document.querySelectorAll('#exsvg circle.mk')];for(let i=0;i<cs.length;i++){const r=cs[i].getBoundingClientRect();if(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===cs[i])return i}return 0})()")
            await pg.locator("#exsvg circle.mk").nth(_ti).scroll_into_view_if_needed()
            _ti = await pg.evaluate("(()=>{const cs=[...document.querySelectorAll('#exsvg circle.mk')];for(let i=0;i<cs.length;i++){const r=cs[i].getBoundingClientRect();if(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===cs[i])return i}return 0})()")
            await pg.locator("#exsvg circle.mk").nth(_ti).click(); await pg.wait_for_timeout(100)   # topmost marker: exact duplicates can overlap
            tipvis = await pg.evaluate("!document.getElementById('extip').hidden && !!document.querySelector('#extip a[href]')")
            check(tipvis, f"{name}: point click pins tooltip with source link")
            check(await pg.locator("#extable tr.sel").count() == 1, f"{name}: pinned point selects its table row")
            tb = await pg.evaluate("(()=>{const t=document.getElementById('extip').getBoundingClientRect();return t.left>=0&&t.right<=innerWidth+1})()")
            check(tb, f"{name}: tooltip stays on screen")
            await pg.keyboard.press("Escape"); await pg.wait_for_timeout(50)
            check(await pg.evaluate("document.getElementById('extip').hidden"), f"{name}: Escape closes tooltip")
            # frontier
            if ranking_allowed(first_set):
                await pg.check("#exfront"); await pg.wait_for_timeout(100)
                frontier_marks = await pg.locator("#extable .chip.f").filter(has_text=re.compile(r"^frontier$")).count()
                check(frontier_marks >= 1 and await pg.locator("#exsvg polyline.frontier").count() == int(frontier_marks > 1), f"{name}: frontier drawn and marked ({frontier_marks} efficient points)")
                await pg.uncheck("#exfront")
            else:
                check(not await pg.locator("#exfront").is_visible() and await pg.locator("#extable th[data-key='" + first_metric + "']").count() == 0, f"{name}: unverified real claims disable frontier and metric rankings")
            # table sort by size
            if await pg.locator("#extable th[data-key=_x]").get_attribute("aria-sort") != "ascending":
                await pg.locator("#extable th[data-key=_x]").click()
            await pg.wait_for_timeout(50)
            xs = await pg.evaluate("[...document.querySelectorAll('#extable tbody tr')].map(r=>parseFloat(r.cells[1].textContent))")
            check(xs == sorted(xs), f"{name}: table sorts by size")
            _D = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("QCL_SITE", "models/qwen3.8-flash-next/site"), "data.json")))
            _sets = [x for x in _D["sets"] if x["id"] != "all"]
            _nk = lambda x: sum(1 for q in x["points"] if "kld" in q["m"])
            BIG = "turboderp" if any(x["id"] == "turboderp" for x in _sets) else max(_sets, key=lambda x: (sum(1 for q in x["points"] if q["m"].get("kld", {}).get("v", 0) > 0 and not q.get("is_ref")), _nk(x)))["id"]
            verified_kld = [x for x in _sets if x['metrics'][0] == 'kld' and ranking_allowed(x, 'kld') and len(x['points']) >= 2]
            if verified_kld: BIG = max(verified_kld, key=lambda x: len(x['points']))['id']
            REF = next((x['id'] for x in _sets if ranking_allowed(x) and any(q.get('is_ref') for q in x['points'])), None)
            import re as _re
            def _qs():   # two words that occur together in one ledger row: a publisher and a word of its variant name
                if any("unsloth" in r["repo"].lower() and "q4" in str(r.get("variant")).lower() for r in _D["rows"]): return "unsloth Q4"
                for r in _D["rows"]:
                    w = next((t for t in _re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", str(r.get("variant") or "")) if t.lower() not in r["repo"].lower()), None)
                    if w: return r["repo"].split("/")[0] + " " + w
                return "Q4"
            QS = _qs()
            # limits: size slider, metric slider, efficient-only, Beaten-by consistency, independent Pareto check
            await pg.select_option("#exset", BIG); await pg.wait_for_timeout(100)
            await pg.evaluate("(()=>{const r=document.getElementById('exxr');r.value=500;r.dispatchEvent(new Event('input'))})()"); await pg.wait_for_timeout(100)
            xmax = await pg.evaluate("parseFloat(document.getElementById('exxo').textContent.replace('≤',''))")
            xs = await pg.evaluate("[...document.querySelectorAll('#extable tbody tr')].map(r=>parseFloat(r.cells[1].textContent))")
            # the table lists every quant under the limit; the chart plots those of them that report the open metric
            _B = [x for x in _sets if x["id"] == BIG][0]; _m0 = _B["metrics"][0]; _tol = 0.5 if xmax >= 100 else 0.05   # limits show rounded
            check(xs and max(xs) <= xmax + _tol and len(xs) < len(_B["points"]), f"{name}: size limit filters table ({len(xs)} rows of {len(_B['points'])}, max {max(xs) if xs else None} <= {xmax})")
            _rq = set(await pg.evaluate("[...document.querySelectorAll('#extable tbody tr[data-qid]')].map(r=>r.dataset.qid)"))
            _nc = sum(1 for q in _B["points"] if q["qid"] in _rq and _m0 in q["m"])
            check(await pg.locator("#exsvg circle.mk").count() == _nc, f"{name}: size limit filters chart too ({_nc} with {_m0})")
            await pg.click("#exrclear"); await pg.wait_for_timeout(100)
            await pg.evaluate("(()=>{const r=document.getElementById('exyr');r.value=400;r.dispatchEvent(new Event('input'))})()"); await pg.wait_for_timeout(100)
            ylim = await pg.evaluate("parseFloat(document.getElementById('exyo').textContent.replace('≤',''))")
            ys = await pg.evaluate("[...document.querySelectorAll('#extable tbody tr')].map(r=>{const t=r.querySelector('td.cur').textContent;const v=parseFloat(t);return /mnat/i.test(t)?v/1000:v})")   # millinat cells show the card's unit
            check(ys and max(ys) <= ylim * 1.0001 and len(ys) < _nk([x for x in _sets if x["id"] == BIG][0]), f"{name}: KLD limit filters ({len(ys)} rows, max {max(ys) if ys else None} <= {ylim})")
            await pg.click("#exrclear"); await pg.wait_for_timeout(100)
            if ranking_allowed(_B, 'kld'):
                await pg.check("#exonly"); await pg.wait_for_timeout(100)
                ui_eff = sorted(await pg.evaluate("[...document.querySelectorAll('#extable tbody tr')].map(r=>r.dataset.qid)"))
                beats = await pg.evaluate("[...document.querySelectorAll('#extable tbody td.beat')].map(c=>c.textContent)")
                check(beats and all(b.startswith('none') for b in beats), f"{name}: efficient-only rows are all unbeaten")
                S = [x for x in json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("QCL_SITE", "models/qwen3.8-flash-next/site"), "data.json")))["sets"] if x["id"] == BIG][0]
                P = [q for q in S["points"] if "kld" in q["m"] and not q.get("is_ref")]
                ref_eff = sorted(q["qid"] for q in P if not any((o["x"] <= q["x"] and o["m"]["kld"]["v"] < q["m"]["kld"]["v"]) or (o["x"] < q["x"] and o["m"]["kld"]["v"] <= q["m"]["kld"]["v"]) for o in P if o is not q))
                check(ui_eff == ref_eff, f"{name}: efficient set matches independent Pareto computation ({len(ui_eff)} vs {len(ref_eff)})")
                await pg.uncheck("#exonly"); await pg.check("#exfront"); await pg.wait_for_timeout(100)
                cons = await pg.evaluate("[...document.querySelectorAll('#extable tbody tr')].every(r=>!!r.querySelector('.chip.f') === r.querySelector('td.beat').textContent.startsWith('none') || (r.querySelectorAll('.chip.f').length===2))")
                fr = sorted(await pg.evaluate("[...document.querySelectorAll('#extable tbody tr')].filter(r=>[...r.querySelectorAll('.chip.f')].some(c=>c.textContent==='frontier')).map(r=>r.dataset.qid)"))
                check(fr == ref_eff, f"{name}: frontier marks == unbeaten quants")
                await pg.uncheck("#exfront")
            else:
                check(not await pg.locator('#exonly').is_visible() and not await pg.locator('#exfront').is_visible(), f"{name}: legacy sets with unverified methodology fail closed for efficient-only and frontiers")
                check(await pg.locator('#extable .beat').count() == 0, f"{name}: unverified real claims have no dominance ranking")
            # F13: references never ranked; F20: no frontier controls on the non-comparable set; F10: inferred rows not cross-linked
            if REF:
                await pg.select_option("#exset", REF); await pg.wait_for_timeout(100)
                await pg.check("#exonly"); await pg.wait_for_timeout(100)
                refq = {q["qid"] for x in _sets if x["id"] == REF for q in x["points"] if q.get("is_ref")}
                shown = await pg.evaluate("[...document.querySelectorAll('#extable tbody tr')].map(r=>r.dataset.qid)")
                check(shown and not (refq & set(shown)), f"{name}: reference build never listed as efficient ({len(shown)} rows)")
                await pg.uncheck("#exonly")
            await pg.select_option("#exset", "all"); await pg.wait_for_timeout(100)
            hid = not await pg.locator("#exfront").is_visible() and not await pg.locator("#exonly").is_visible()
            check(hid, f"{name}: frontier controls hidden on the not-comparable set")
            if BIG == "turboderp":
              await pg.select_option("#exset", BIG); await pg.wait_for_timeout(100)
              ud = await pg.evaluate("[...document.querySelectorAll('#extable tbody tr')].filter(r=>/UD-/.test(r.cells[0].textContent)).map(r=>[r.textContent.includes('repo not named on card'), r.querySelectorAll('.jump').length])")
              check(ud and all(a and n == 0 for a, n in ud), f"{name}: turboderp UD rows labelled and not cross-linked {ud[:2]}")
            # borrowed controls: size presets, size type, log/linear, shared view links, chart-first order
            await pg.select_option("#exset", "all"); await pg.wait_for_timeout(100)
            pre = pg.locator("#expresets .fchip"); np_ = await pre.count()
            check(np_ >= min(3, NTIER), f"{name}: size-budget presets offered ({np_}, {NTIER} tiers in range)")
            if np_ >= 2:
                lab = await pre.nth(1).text_content(); tier = float(lab.replace("≤", "").replace("GB", ""))
                await pre.nth(1).click(); await pg.wait_for_timeout(100)
                xs = await pg.evaluate("[...document.querySelectorAll('#extable tbody tr')].map(r=>parseFloat(r.cells[1].textContent)).filter(x=>!isNaN(x))")
                check(xs and max(xs) <= tier + 0.01, f"{name}: preset {lab} caps size ({max(xs) if xs else None})")
                await pg.click("#exrclear"); await pg.wait_for_timeout(50)
            bopts = await pg.evaluate("[...document.querySelectorAll('#exbasis option')].map(o=>o.value).filter(Boolean)")
            check(NBASIS < 2 or (len(bopts) >= 2 and not await pg.evaluate("document.getElementById('exbasiswrap').hidden")), f"{name}: size-type selector on the mixed set ({len(bopts)} types)")
            if len(bopts) >= 2:   # with one size type the selector is hidden by design
              await pg.select_option("#exbasis", bopts[0]); await pg.wait_for_timeout(100)
              only = await pg.evaluate("document.getElementById('excount').textContent")
              data_n = sum(1 for q in [x for x in json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ.get("QCL_SITE", "models/qwen3.8-flash-next/site"), "data.json")))["sets"] if x["id"] == "all"][0]["points"] if (q["basis"] or "unknown") == bopts[0])
              check(only.startswith(f"{data_n} of"), f"{name}: size type '{bopts[0]}' shows exactly its {data_n} points ({only})")
              await pg.click("#exrclear"); await pg.wait_for_timeout(50)
            if await pg.locator("#exlin").is_visible():   # the log/linear toggle shows only for log-scale metrics (KLD)
                await pg.click("#exlin"); await pg.wait_for_timeout(100)
                ax = await pg.evaluate("[...document.querySelectorAll('#exsvg text.axis')].map(t=>t.textContent).join(' ')")
                check(", log" not in ax, f"{name}: linear scale toggle")
                await pg.click("#exlog"); await pg.wait_for_timeout(50)
            order = await pg.evaluate("[...document.querySelectorAll('main > section')].map(s=>s.id)")
            check(order.index("chart") < order.index("head-to-head"), f"{name}: explorer comes before head-to-head")
            SH = BIG
            _xs = sorted(q["x"] for x in _sets if x["id"] == SH for q in x["points"] if q.get("x")); CAP = 40 if SH == "gyro" else int(_xs[len(_xs) // 2]) + 1
            pg2 = await ctx.new_page(); await pg2.goto(SRC + f"?set={SH}&max={CAP}&eff=1#chart"); await pg2.wait_for_timeout(400)
            st = await pg2.evaluate("[document.getElementById('exset').value, document.getElementById('exonly').checked, [...document.querySelectorAll('#extable tbody tr')].map(r=>parseFloat(r.cells[1].textContent))]")
            check(st[0] == SH and st[1] and st[2] and max(st[2]) <= CAP, f"{name}: shared view link restores set, size cap and efficient-only {st[:2]}")
            await pg2.close()
            # find across sets -> jump
            _ft = "UD-Q4_K_XL" if _vs.get("UD-Q4_K_XL", 0) >= 2 else FIND   # a quant listed in at least two sets
            if _ft and FIND_N >= 2:
                await pg.fill("#exfind", _ft)
                try: await pg.wait_for_function("document.querySelectorAll('#exfindres button[data-set]').length >= 2", timeout=5000)   # slow CI runners
                except Exception: pass
                btns = pg.locator("#exfindres button[data-set]"); nb = await btns.count()
                check(nb >= 2, f"{name}: find lists sets containing {_ft} ({nb})")
                if nb >= 2:
                    target = await btns.nth(1).get_attribute("data-set"); await btns.nth(1).click()
                    try: await pg.wait_for_function("!document.getElementById('extip').hidden", timeout=5000)
                    except Exception: pass
                    check(await pg.input_value("#exset") == target and not await pg.evaluate("document.getElementById('extip').hidden"), f"{name}: find jumps to set and pins the quant")
            # every set renders without error and with unique checkbox ids
            bad = []
            for sid in opts:
                await pg.select_option("#exset", sid); await pg.wait_for_timeout(30)
                if await pg.locator("#exsvg svg").count() != 1: bad.append(sid)
            check(not bad, f"{name}: every set renders {bad[:3]}")
            await pg.select_option("#exset", "all"); await pg.wait_for_timeout(100)
            lh = await pg.evaluate("document.getElementById('expick').getBoundingClientRect().height")
            check(lh >= (300 if name == "desk" else 200), f"{name}: quant list has room in the 'all' set ({lh:.0f}px)")
            more = pg.locator("#exfmts .fchip.more"); nchips = await pg.locator("#exfmts .fchip").count()
            check(await more.count() == 1 or nchips <= 9, f"{name}: extra format chips collapsed ({nchips} chips)")
            await pg.click("#exexpand"); await pg.wait_for_timeout(150)
            pw = await pg.evaluate("document.querySelector('.ex-picker').getBoundingClientRect().width / document.querySelector('.ex-body').getBoundingClientRect().width")
            check(pw > 0.95, f"{name}: expand makes list full width ({pw:.2f})")
            await pg.click("#exexpand"); await pg.wait_for_timeout(100)
            format_term = next(q.get("fmt") for q in _ALL["points"] if q.get("fmt") and q["m"].get(_ALL["metrics"][0], {}).get("v", 0) > 0)
            await pg.fill("#exq", format_term); await pg.wait_for_timeout(100)
            check(await pg.locator("#exsvg circle.mk").count() > 0, f"{name}: 'all' view filterable by format text")
            # Ledger and publisher notes are collapsed until requested.
            check(not await pg.locator('#ledger-details').evaluate('(e) => e.open'), f"{name}: ledger starts collapsed")
            check(not await pg.locator('#publisher-details').evaluate('(e) => e.open'), f"{name}: publisher notes start collapsed")
            await pg.locator('#ledger-details > summary').click()
            await pg.locator('#publisher-details > summary').click()
            # ledger + notes filters
            await pg.fill("#q", QS); await pg.wait_for_timeout(100)
            c = await pg.text_content("#count"); check(c and not c.startswith("0 ") and "of" in c, f"{name}: ledger multi-word search ({c})")
            await pg.click("#freset")
            note_texts = await pg.locator("#cards > *").all_text_contents()
            note_term = next(t for t in re.findall(r"[A-Za-z]{4,}", note_texts[0]) if 0 < sum(t.lower() in x.lower() for x in note_texts) < len(note_texts))
            await pg.fill("#cq", note_term); await pg.wait_for_timeout(100)
            cc = await pg.text_content("#ccount"); check(cc and cc.startswith(str(sum(note_term.lower() in x.lower() for x in note_texts)) + " of"), f"{name}: notes search ({cc})")
            check(not await pg.locator('#no-claims-details').evaluate('(e) => e.open'), f"{name}: repos without extracted claims start collapsed")
            await pg.locator('#no-claims-details > summary').click()
            await pg.select_option("#ncs", "read in full: no quality claims"); await pg.wait_for_timeout(50)
            ncc = await pg.text_content("#nccount"); check(ncc and not ncc.startswith("0 "), f"{name}: no-claims status filter ({ncc})")
            sw = await pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            check(sw <= 0, f"{name}: still no horizontal scroll after interaction ({sw}px)")
            check(not errs, f"{name}: no JS errors {errs[:3]}")
            await ctx.close()
        await b.close()
asyncio.run(main())
for n in oks: print("[PASS]", n)
for n in fails: print("[FAIL]", n)
sys.exit(1 if fails else 0)
