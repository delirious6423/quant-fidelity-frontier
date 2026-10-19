"""Standalone shared-task chart, derived solely from reviewed model payloads."""
import html
import json
import math
from pathlib import Path
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

COLORS = ('#087880', '#ad65bd', '#bf7428', '#4078b8', '#b45977', '#548537')


def collect_shared(out, models):
    """Keep exact task variants and score scales; never average source reports."""
    groups = {}
    for model in models:
        data = json.loads((Path(out) / model['data']).read_text())
        for si, source in enumerate((data.get('benchmarks') or {}).get('sources', [])):
            for mi, metric in enumerate(source['metrics']):
                key = (metric['name'], metric['unit'])
                group = groups.setdefault(key, dict(name=key[0], unit=key[1], points=[]))
                for pi, point in enumerate(metric['points']):
                    # Do not turn missing or invalid values into chart coordinates.
                    if not all(isinstance(point.get(k), (float, int)) and not isinstance(point[k], bool)
                               and math.isfinite(point[k]) for k in ('size', 'score')) or point['size'] <= 0:
                        continue
                    group['points'].append(dict(point, id=f'{model["slug"]}:{si}:{mi}:{pi}',
                        model=model['model'], slug=model['slug'], model_name=model['model'].split('/')[-1],
                        source_label=source['label'], context=source['context'], notes=source['notes'],
                        card_sha256=source['card_sha256'], revision=source['revision'], captured_at=data.get('captured_at')))
    views = []
    for group in groups.values():
        slugs = sorted({p['slug'] for p in group['points']})
        if len(slugs) < 2:
            continue
        views.append(dict(group, models=slugs))
    views.sort(key=lambda g: (g['name'] != 'GPQA-Diamond', g['name'].casefold(), g['unit']))
    return dict(views=views, models=[dict(slug=m['slug'], name=m['model'].split('/')[-1], model=m['model']) for m in models])


def static_plot(view, palette):
    if not view:
        return '<p>No benchmark currently has reviewed results with file sizes for two models.</p>'
    points = view['points']
    lo, hi = min(p['score'] for p in points), max(p['score'] for p in points)
    pad = max((hi-lo)*.12, .001 if view['unit'] == 'fraction' else 1)
    lo = min(0,lo-pad); hi += pad
    xmax = max(p['size'] for p in points)*1.08
    x = lambda v: 65 + 640*v/xmax
    y = lambda v: 310 - 265*(v-lo)/(hi-lo)
    esc = html.escape
    svg = ['<svg viewBox="0 0 740 370" role="img" aria-label="Published score versus file size across models">']
    for i in range(5):
        score = lo+(hi-lo)*i/4; size = xmax*i/4
        svg.append(f'<path class="gridline" d="M65 {y(score):.2f}H705"/><text x="57" y="{y(score)+4:.2f}" text-anchor="end">{score:.4g}</text><text x="{x(size):.2f}" y="332" text-anchor="middle">{size:.0f}</text>')
    for p in points:
        title = f'{p["model_name"]} · {p["variant"]}: {p["raw"]} at {p["size"]:g} GB · {p["source_label"]}'
        xx, yy = x(p['size']), y(p['score'])
        shape = f'<path d="M{xx:.2f} {yy-6:.2f}l6 6l-6 6l-6 -6Z"' if p['reference'] else f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="5"'
        svg.append(f'<a href="{esc(p["src"], quote=True)}" aria-label="{esc(title, quote=True)}">{shape} class="shared-dot" style="fill:{palette[p["slug"]]}"><title>{esc(title)}</title></'+('path' if p['reference'] else 'circle')+'></a>')
    svg.append('<text x="385" y="360" text-anchor="middle">File size (GB)</text><text x="65" y="20">Published score ↑</text></svg>')
    return ''.join(svg)


def build_benchmark_view(out, models, base, analytics='', support=''):
    out = Path(out)
    target = out/'benchmarks'; target.mkdir(parents=True, exist_ok=True)
    payload = collect_shared(out, models)
    esc = lambda s: html.escape(str(s), quote=True)
    palette = {m['slug']: COLORS[i % len(COLORS)] for i, m in enumerate(sorted(payload['models'], key=lambda m:m['name']))}
    payload['colors'] = palette
    first = payload['views'][0] if payload['views'] else None
    table = ''.join(f'<tr><td>{esc(v["name"])}</td><td>{esc(p["model_name"])}</td><td>{esc(p["variant"])}</td><td>{p["size"]:g}</td><td>{esc(p["raw"])} ({esc(v["unit"])})</td><td>{esc(p["source_label"])} · <a href="{esc(p["src"])}">Score</a> · <a href="{esc(p["size_src"])}">Size</a></td></tr>' for v in payload['views'] for p in v['points'])
    providers = {'deepseek-ai': 'DeepSeek', 'Qwen': 'Qwen', 'zai-org': 'Z.ai'}
    groups = {}
    for m in sorted(models, key=lambda m:(providers.get(m['model'].split('/')[0],m['model'].split('/')[0]).casefold(),-m.get('repos',0),m['model'].casefold())):
        owner = m['model'].split('/')[0]
        groups.setdefault(providers.get(owner,owner),[]).append(m)
    picker = ''.join(f'<p class="page-provider">{esc(provider)}</p>'+''.join(f'<a href="../{esc(m["slug"])}/">{esc(m["model"].split("/")[-1])}</a>' for m in members) for provider,members in groups.items())
    legend = ''.join(f'<span class="model-label" style="--model-color:{palette[slug]}">{esc(next(m["name"] for m in payload["models"] if m["slug"]==slug))}</span>' for slug in (first['models'] if first else []))
    ranges = ''.join(f'''<fieldset class="axis-range" id="shared-{axis}-range"><legend>{label}</legend><div class="range-values"><label>Min<input id="shared-{axis}-min" type="number" step="any" aria-label="Minimum {aria}"></label><span>–</span><label>Max<input id="shared-{axis}-max" type="number" step="any" aria-label="Maximum {aria}"></label></div><div class="range-track"><span></span><input id="shared-{axis}-low" type="range" aria-label="Minimum {aria} slider"><input id="shared-{axis}-high" type="range" aria-label="Maximum {aria} slider"></div></fieldset>''' for axis,label,aria in [('size','Size (GB)','size in GB'),('score','Score','score')])
    encoded = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    styles = (Path(__file__).parent/'web/benchmark_view.css').read_text()
    script = (Path(__file__).parent/'web/benchmark_view.js').read_text()
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compare model benchmarks by file size · QFF</title><meta name="description" content="Explore shared benchmark scores across LLM models and quantized builds, with size in GB and publisher source citations."><link rel="canonical" href="{base}benchmarks/">{analytics}<style>{styles}</style></head><body>
<a class="skip" href="#comparison">Skip to chart</a><div class="shell"><header><a class="brand" href="../"><b>QFF</b><span>Quant Fidelity Frontier</span></a><details class="page-picker"><summary>All-model benchmarks ▾</summary><div class="page-menu"><span aria-current="page">All-model benchmarks</span>{picker}</div></details><nav aria-label="Main"><a href="../#models">Models</a><a href="./" aria-current="page">Benchmarks</a><a href="../guides/">Guides</a></nav></header>
<main><div class="page-heading"><h1>Model benchmarks</h1><p>Score vs. file size across models.</p></div>
<section id="comparison" aria-label="Cross-model benchmark comparison"><div class="controls" id="shared-controls" hidden><label>Benchmark<select id="shared-benchmark"></select></label></div>
<div class="chart-heading"><p id="shared-status" role="status">{esc(first['name']) if first else 'No shared benchmarks yet'}</p><details class="chart-options" id="shared-options" hidden><summary>Chart options</summary><div class="option-panel"><label><input type="checkbox" id="shared-log"> Log size axis</label><label><input type="checkbox" id="shared-zero"> Start score axis at zero</label><label><input type="checkbox" id="shared-refs" checked> Include reference builds</label><button id="shared-copy" type="button">Copy view link</button></div></details></div>
<div id="shared-models" class="models" aria-label="Show or hide models">{legend}</div><p class="scope" id="shared-scope">Published evaluations may use different test settings.</p><div id="shared-plot" class="plot">{static_plot(first,palette)}</div><p class="shape-key">● Quantized build <span>◆ Reference build</span></p>
<div id="shared-ranges" class="range-panel" hidden><div class="range-heading"><h2>Zoom range</h2><button id="shared-reset" type="button">Reset range</button></div><div class="range-grid">{ranges}</div></div>
<div id="shared-inspect" hidden><label>Inspect result<select id="shared-pick"></select></label><div id="shared-detail" class="point-detail" aria-live="polite"></div></div>
<details id="shared-notes"><summary>Evaluation details &amp; coverage</summary><p>Only exact benchmark variants with the same declared score units and reviewed file sizes enter a shared view. Each source report remains a separate point. Missing results are unknown. File size is not total VRAM usage.</p><p>QFF transcribes publisher claims; it does not independently reproduce these evaluations. Common benchmark names do not verify identical prompting, sample selection or scoring settings. No cross-source ranking or averaged score is inferred. Source links may point to mutable cards; captured hashes are retained in the data.</p><div id="shared-contexts"></div></details>
<details open id="shared-evidence"><summary>Source-linked results</summary><div class="tablebox"><table><thead><tr><th>Benchmark</th><th>Model</th><th>Build</th><th>File GB</th><th>Published score</th><th>Source</th></tr></thead><tbody>{table}</tbody></table></div></details></section>
<footer><span><a href="../">QFF</a> · Publisher claims, linked to their sources.</span><span><a href="data.json">Chart data</a>{support}</span></footer></main></div><script id="shared-data" type="application/json">{encoded}</script><script>{script}</script></body></html>'''
    (target/'index.html').write_text(page)
    (target/'data.json').write_text(json.dumps(payload, separators=(',', ':')))
    ns = 'http://www.sitemaps.org/schemas/sitemap/0.9'
    tree = ET.parse(out/'sitemap.xml')
    node = ET.SubElement(tree.getroot(), '{'+ns+'}url')
    ET.SubElement(node, '{'+ns+'}loc').text = base+'benchmarks/'
    tree.write(out/'sitemap.xml', encoding='utf-8', xml_declaration=True)
    return payload
