"""Source-derived, progressively enhanced homepage. No measurement values are authored here."""
import html
from collections import Counter
import json
import math
import re
from pathlib import Path
from urllib.parse import urlencode


def family(value):
    value = str(value or '').strip()
    match = re.match(r'^[A-Za-z0-9_+.\-]+', value)
    key = match[0].upper() if match else value
    return 'other' if not key or key == 'OTHER' else key


def build_homepage(out, models, base, analytics, support):
    out = Path(out)
    esc = lambda s: html.escape(str(s), quote=True)
    catalog, cards = [], {}
    preview = None
    for model in models:
        data = json.loads((out / model['data']).read_text())
        sets = []
        for source in data['sets']:
            if source['id'] == 'all' or source.get('x', {}).get('unit') != 'GB':
                continue
            metric = 'kld' if 'kld' in source['metrics'] else source['metrics'][0]
            points = [p for p in source['points'] if p.get('x') is not None and metric in p.get('m', {})]
            if not points:
                continue
            sets.append(dict(id=source['id'], title=source['title'], metric=metric,
                             size_label=source['x']['label'], reference=source.get('reference', 'Not stated'),
                             comparison=source.get('comparisons', {}).get(metric, {}),
                             points=[dict(fmt=family(p.get('fmt')), basis=p.get('basis') or 'unknown', x=p['x']) for p in points]))
            # A real, homogeneous KLD example, with every point linked to its source.
            if metric == 'kld' and len(points) >= 3 and len({p.get('basis') for p in points}) == 1:
                rank = (model['slug'] == 'qwen3.8-27b', len(points))
                previous = (preview[0]['slug'] == 'qwen3.8-27b', len(preview[2])) if preview else (False, 0)
                if rank > previous:
                    preview = (model, source, points)
        entry = dict(slug=model['slug'], name=model['model'].split('/')[-1], sets=sets)
        catalog.append(entry)
        quality = {r['measured_by'] for r in data['rows'] if r.get('metrics') and r['kind'] == 'own' and not r.get('copied_from')}
        quality &= {r['repo'] for r in data['repos']}
        format_counts = Counter(family(r.get('fmt')) for r in data['repos'])
        formats = sorted(format_counts, key=lambda f: (-format_counts[f], f))
        pills = ''.join(f'<span class="pill">{esc(f)}</span>' for f in formats[:4])
        if len(formats) > 4:
            pills += f'<span class="pill">+{len(formats)-4} more</span>'
        target = model['slug'] + '/'
        if sets:
            target += '?' + urlencode({'set': sets[0]['id'], 'm': sets[0]['metric']}) + '#chart'
        captured = (model.get('captured_at') or 'Unknown')[:10]
        checked = model.get('checked_at')
        scan = f'Checked for new repos: {checked[:16].replace("T", " ")} UTC' if checked else 'New-repo scan time unavailable'
        cards[model['slug']] = f'''<article class="model-card"><h4><a href="{esc(target)}">{esc(entry['name'])}</a></h4>
<p class="org">{esc(model['model'])}</p><p class="coverage"><strong>{model['repos']:,}</strong> repos indexed · <strong>{len(quality):,}</strong> with own quality claims</p>
<div class="card-actions"><div class="pills">{pills}</div><a class="explore" href="{esc(target)}">Explore quants <span aria-hidden="true">→</span></a></div>
<details><summary>Coverage &amp; freshness</summary><p>{model['rows']:,} claim rows · {model['sets']} measurement sets<br>Cards captured: {esc(captured)}<br>{esc(scan)}</p></details></article>'''
    provider_names = {'deepseek-ai': 'DeepSeek', 'Qwen': 'Qwen', 'zai-org': 'Z.ai'}
    def provider(m):
        owner = m['model'].split('/')[0]
        return provider_names.get(owner, owner)
    groups = {}
    for model in sorted(models, key=lambda m: (provider(m).casefold(), -m['repos'], m['model'].casefold())):
        groups.setdefault(provider(model), []).append(cards[model['slug']])
    grouped_cards = ''.join(f'<section class="provider-group" aria-label="{esc(name)} models"><h3>{esc(name)}</h3><div class="model-grid">{"".join(items)}</div></section>' for name, items in groups.items())
    chart = ''
    if preview:
        model, source, points = preview
        x0, x1 = min(p['x'] for p in points), max(p['x'] for p in points)
        omitted = sum(p['m']['kld']['v'] <= 0 for p in points)
        points = [p for p in points if p['m']['kld']['v'] > 0]
        low = math.log10(min(p['m']['kld']['v'] for p in points)) - .12
        high = math.log10(max(p['m']['kld']['v'] for p in points)) + .12
        def Y(value):
            return 185 - 155 * (math.log10(value) - low) / (high - low)
        ticks = ''
        for exponent in range(math.ceil(low), math.floor(high) + 1):
            value = 10 ** exponent
            y = Y(value)
            ticks += f'<path class="gridline" d="M58 {y:.2f}H338"/><text x="49" y="{y+3:.2f}" text-anchor="end">{value:g}</text>'
        zero_note = f' · {omitted} zero/non-positive values omitted from log scale' if omitted else ''
        dots = ''
        for p in points:
            x = 65 + 265 * (p['x']-x0)/(x1-x0 or 1)
            y = Y(p['m']['kld']['v'])
            title = f"{p['variant']}: {p['x']:g} GB, KLD {p['m']['kld']['raw']}"
            dots += f'<a href="{esc(p["src"])}" aria-label="{esc(title)}"><circle cx="{x:.2f}" cy="{y:.2f}" r="4"><title>{esc(title)}</title></circle></a>'
        href = model['slug'] + '/?' + urlencode({'set':source['id'], 'm':'kld'}) + '#chart'
        chart = f'''<figure class="preview"><figcaption><span class="eyebrow">A REAL COMPARISON</span><strong>{esc(model['model'].split('/')[-1])}</strong></figcaption>
<svg viewBox="0 0 360 225" role="img" aria-label="Mean KLD versus size within {esc(source['title'])}; logarithmic KLD axis, linear size axis"><path d="M58 24V185H338"/>{ticks}<text x="65" y="203">{x0:g}</text><text x="330" y="203" text-anchor="end">{x1:g}</text><text x="195" y="221" text-anchor="middle">Size (GB)</text><text transform="translate(14 106) rotate(-90)" text-anchor="middle">Mean KLD · log ↓</text>{dots}</svg>
<p>{esc(source['title'])} · Log scale{esc(zero_note)}<br>{esc(source['x']['label'])}<br>{esc(source.get('comparisons', {}).get('kld', {}).get('label', 'Protocol unverified'))}</p><a href="{esc(href)}">Explore this comparison →</a></figure>'''
    default_slug = preview[0]['slug'] if preview else (catalog[0]['slug'] if catalog else '')
    for entry in catalog:
        if preview and entry['slug'] == default_slug:
            entry['sets'].sort(key=lambda item: item['id'] != preview[1]['id'])
    options = ''.join(f'<option {"selected" if m["slug"] == default_slug else ""} value="{esc(m["slug"])}">{esc(m["name"])}</option>' for m in catalog)
    payload = json.dumps(catalog).replace('<', '\\u003c')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM quantization comparison: KLD vs size · QFF</title>{analytics}<meta name="description" content="Compare published LLM quantization fidelity: KLD, perplexity and model size, with source-linked claims and explicit measurement boundaries."><link rel="canonical" href="{base}"><link rel="stylesheet" href="home.css"><script defer src="home.js"></script></head><body>
<a class="skip" href="#models">Skip to models</a><div class="shell"><header><a class="brand" href="./"><b>QFF</b><span>Quant Fidelity Frontier</span></a><nav aria-label="Main"><a href="#models">Models</a><a href="guides/">Guides</a><a href="#agents">For agents</a><a href="https://github.com/delirious6423/quant-fidelity-frontier">GitHub ↗</a></nav></header>
<main><section class="hero" aria-labelledby="intro"><div><p class="eyebrow">LLM QUANTIZATION, WITH THE EVIDENCE</p><h1 id="intro">Compare quant size.<br>Understand the trade-offs.</h1><p class="lede">Explore publisher-reported fidelity, with every claim linked to its source.</p><p class="trust">Claims, not independent benchmarks. Direct ranking requires a verified compatible protocol.</p><a class="text-link" href="guides/">How to compare fairly →</a></div>{chart}</section>
<section class="quick" id="quick" hidden aria-labelledby="quick-title"><h2 id="quick-title">Start a comparison</h2><form id="start-form"><div class="controls"><label>Model<select id="home-model">{options}</select></label><label>Format<select id="home-format"></select></label><label>Size limit<select id="home-limit"><option value="">Any</option>{''.join(f'<option value="{n}">≤ {n} GB</option>' for n in (16,24,32,48,64,96,128,192,256))}</select></label><button type="submit" id="start">Explore →</button></div><div class="context-controls"><label>Measurement set<select id="home-set"></select></label><label>Size definition<select id="home-basis"></select></label></div><p id="selection-note" role="status"></p><p class="caution">File size is not total memory usage. Context, runtime buffers and offloading matter.</p></form></section>
<section id="models" aria-labelledby="browse"><div class="section-heading"><h2 id="browse">Browse models</h2><span>{len(models)} models indexed</span></div>{grouped_cards}</section>
<section class="resources" id="agents"><div><h2>Understand the evidence</h2><p>Read the measurement boundaries, size definitions and source citations before choosing a quant.</p><a href="guides/">Read the comparison guides →</a></div><div><h2>For agents &amp; developers</h2><p>Use the same source-linked claims in your own tools.</p><a href="llms.txt">Agent reading guide</a> · <a href="models.json">Model index JSON</a> · <a href="https://github.com/delirious6423/quant-fidelity-frontier/tree/main/qff_mcp">MCP server</a></div></section>
<footer><span>QFF · Publisher claims, linked to their sources.</span><span><a href="https://github.com/delirious6423/quant-fidelity-frontier">Source &amp; data</a>{support}</span></footer></main></div><script id="home-data" type="application/json">{payload}</script></body></html>'''
    (out / 'index.html').write_text(page)
    for name in ('home.css', 'home.js'):
        (out / name).write_text((Path(__file__).parent / 'web' / name).read_text())
