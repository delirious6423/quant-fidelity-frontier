"""Standalone shared-task chart, derived solely from reviewed model payloads."""
import html
import json
import math
import re
from pathlib import Path
from urllib.parse import urlencode, urlsplit
import xml.etree.ElementTree as ET
from bench_tasks import canonical

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


BASIS = {"text_weights": "text weights", "disk": "download", "hf_listing_total": "download (HF listing)", "vram": "GPU weights", "resident": "resident",
         "ram_in_use": "RAM in use", "trunk_only": "trunk only", "hf_branch_total": "download (HF branch listing)", "excl_engram": "download (excl. Engram)",
         "decoder_only": "decoder only", "engram_only": "Engram only", "routed_experts": "routed experts only"}


def _card_repo(src, fallback):
    """owner/name of the card a score is read from (the measuring card, which may differ from the tested repo)."""
    parts = urlsplit(src or '').path.strip('/').split('/')
    return '/'.join(parts[:2]) if len(parts) >= 4 and parts[2] == 'blob' else fallback


_CARDS = {}


def _card_lines(slug, repo):
    """Lines of the captured card for a repo of a live model (cached); [] when unavailable."""
    key = (slug, repo)
    if key not in _CARDS:
        lines = []
        try:
            root = Path(__file__).parent.parent / 'models' / slug
            meta = json.loads((root / 'snapshots/index.json').read_text())['repos'][repo]
            lines = (root / meta['file']).read_text(encoding='utf-8', errors='replace').splitlines()
        except (OSError, KeyError, ValueError):
            pass
        _CARDS[key] = lines
    return _CARDS[key]


def _score_line(lines, raw, near, name=''):
    """The card line (1-based) on which the score is printed, nearest to the ledger row's line; None if it is not found."""
    import re
    nums = re.findall(r"\d*\.?\d+", str(raw).replace(',', ''))
    if not nums:
        return None
    rx = re.compile(r"(?<![\d.])" + re.escape(nums[0]) + r"(?!\d)")
    hits = [i + 1 for i, line in enumerate(lines) if rx.search(line.replace(',', ''))]
    # a printed score such as "6/8" or "96.28%" is matched whole before falling back to its first number
    whole = str(raw).strip()
    exact = [i + 1 for i, line in enumerate(lines) if whole and (
        re.search(r"(?<![\w./])" + re.escape(whole) + r"(?![\w./]|\.\d)", line))]
    pool = exact or hits
    # several lines can print the same figure: prefer one that also names the benchmark, then the one nearest the ledger row
    key = re.sub(r"[^a-z0-9]", "", str(name).lower().split(" (")[0].split("/")[0])
    named = [n for n in pool if key and key in re.sub(r"[^a-z0-9]", "", lines[n - 1].lower())]
    pool = named or pool
    return min(pool, key=lambda n: (abs(n - near), n)) if pool else None


def collect_extracted(out, models):
    """Every task score a card reports, grouped by canonical task. Values come from the gate-checked ledger rows (copied
    verbatim from card text); the raw benchmark name, publisher and exact card line stay on each entry. Not individually
    reviewed for comparable protocols, sizes or score scales, so there is no chart or ranking here."""
    tasks = {}
    for model in models:
        data = json.loads((Path(out) / model['data']).read_text())
        for r in data.get('rows', []):
            if r.get('measured') is False or r.get('kind') == 'quoted':
                continue
            for key, m in r.get('metrics', {}).items():
                if not key.startswith('bench:') or m.get('v') is None:
                    continue
                task = canonical(key[6:])
                if task in (None, 'unclassified'):
                    continue
                card = _card_repo(r.get('src'), r['repo'])
                src = r.get('src') or ''
                near = r.get('L') or 1
                found = _score_line(_card_lines(model['slug'], card), m.get('raw'), near, key[6:]) if src else None
                if found:
                    src = src.split('#')[0] + f'#L{found}'
                    # a bare count (e.g. 8 correct of 20) in a table that also prints n/m totals is not a percentage score
                    line_text = _card_lines(model['slug'], card)[found - 1]
                    if re.fullmatch(r'\d+', str(m.get('raw')).strip()) and re.search(r'\b\d+\s*/\s*\d+\b', line_text):
                        continue
                tasks.setdefault(task, []).append(dict(
                    model=model['model'].split('/')[-1], slug=model['slug'], repo=r['repo'], card=card, build=str(r.get('variant')),
                    measured_by=r.get('measured_by'), metric=key[6:], raw=str(m.get('raw')), size=r.get('size_raw'), basis=BASIS.get(r.get('basis'), r.get('basis') or ''),
                    modified=bool(r.get('weights_modified')), src=src, kind=r.get('kind')))
    order = sorted(tasks, key=lambda t: (-len({e['slug'] for e in tasks[t]}), -len(tasks[t]), t))
    return [dict(task=t, models=sorted({e['model'] for e in tasks[t]}), count=len(tasks[t]),
                 entries=sorted(tasks[t], key=lambda e: (e['model'].casefold(), e['metric'].casefold(), e['repo'].casefold(), e['build'].casefold()))) for t in order]


def render_extracted(groups):
    esc = lambda s: html.escape(str(s), quote=True)
    if not groups:
        return ''
    total = sum(g['count'] for g in groups)
    rows_html = []
    for g in groups:
        body = ''.join(
            f'<tr><td>{esc(e["model"])}</td><td>{esc(e["build"])}{" <em>(modified weights)</em>" if e["modified"] else ""}</td><td>{esc(e["card"])}</td>'
            f'<td>{esc(e["metric"])}</td><td class="n">{esc(e["raw"])}</td><td>{esc(e["size"] or "—")}{" · " + esc(e["basis"]) if e["size"] and e["basis"] else ""}</td>'
            f'<td><a href="{esc(e["src"])}">card line</a></td></tr>' for e in g['entries'])
        rows_html.append(f'<details class="task" data-task="{esc(g["task"]).lower()}"><summary><b>{esc(g["task"])}</b> · {len(g["models"])} model{"s" if len(g["models"]) != 1 else ""} · {g["count"]} score{"s" if g["count"] != 1 else ""}</summary>'
                         f'<p class="dim">Models: {esc(", ".join(g["models"]))}</p><div class="tablebox"><table><thead><tr><th>Model</th><th>Build</th><th>Reported on card</th><th>Benchmark, as published</th><th>Score, as published</th><th>Size, as published</th><th>Source</th></tr></thead><tbody>{body}</tbody></table></div></details>')
    return (f'<section id="all-reported" aria-labelledby="all-reported-title"><h2 id="all-reported-title">Every reported task score</h2>'
            f'<p class="scope">{len(groups)} benchmarks, {total} scores read from {len({e["slug"] for g in groups for e in g["entries"]})} models&#8217; cards. Unlike the chart above, these are not individually reviewed: '
            'the benchmark name, score and size are shown exactly as the card publishes them, grouped by task name, and the settings (sample size, shots, thinking, score scale) may differ between rows. '
            'No ranking or averaging is made. Each row links to its card line.</p>'
            '<label class="ex-find">Find a benchmark <input id="all-find" type="search" placeholder="e.g. AIME, LiveCodeBench, GPQA" autocomplete="off"></label><p id="all-find-status" class="dim" role="status"></p>'
            + ''.join(rows_html) +
            '<script>(function(){var q=document.getElementById("all-find"),st=document.getElementById("all-find-status"),d=[].slice.call(document.querySelectorAll("#all-reported details.task"));'
            'function go(){var v=q.value.trim().toLowerCase(),n=0;d.forEach(function(x){var m=!v||x.getAttribute("data-task").indexOf(v)>=0;x.hidden=!m;if(m)n++;if(v&&m&&n<=3)x.open=true;else if(!v)x.open=false});'
            'st.textContent=v?n+" benchmark"+(n===1?"":"s")+" match":"";}q.addEventListener("input",go);'
            'if(location.hash){var t=decodeURIComponent(location.hash.slice(1)).toLowerCase();d.forEach(function(x){if(x.getAttribute("data-task")===t){x.open=true;x.scrollIntoView()}})}})();</script></section>')


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
    extracted = collect_extracted(out, models)
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
    ranges = '<div class="ex-range" id="shared-ranges" role="group" aria-label="Size and score limits" hidden><label class="rg" for="shared-size"><span>Max size</span><input id="shared-size" type="range" min="0" max="1000" step="1" value="1000"><output id="shared-size-output" for="shared-size">any</output></label><label class="rg" for="shared-score"><span>Min score</span><input id="shared-score" type="range" min="0" max="1000" step="1" value="0"><output id="shared-score-output" for="shared-score">any</output></label><button id="shared-reset" type="button" hidden>Clear limits</button></div>'
    encoded = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    styles = (Path(__file__).parent/'assets/benchmark_view.css').read_text()
    script = (Path(__file__).parent/'assets/benchmark_view.js').read_text()
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compare model benchmarks by file size · QFF</title><meta name="description" content="Explore shared benchmark scores across LLM models and quantized builds, with size in GB and publisher source citations."><link rel="canonical" href="{base}benchmarks/">{analytics}<style>{styles}</style></head><body>
<a class="skip" href="#comparison">Skip to chart</a><div class="shell"><header><a class="brand" href="../"><b>QFF</b><span>Quant Fidelity Frontier</span></a><details class="page-picker"><summary>All-model benchmarks ▾</summary><div class="page-menu"><span aria-current="page">All-model benchmarks</span>{picker}</div></details><nav aria-label="Main"><a href="../#models">Models</a><a href="./" aria-current="page">Benchmarks</a><a href="../guides/">Guides</a></nav></header>
<main><div class="page-heading"><h1>Model benchmarks</h1><p>Score vs. file size across models.</p></div>
<section id="comparison" aria-label="Cross-model benchmark comparison"><div class="controls" id="shared-controls" hidden><label>Benchmark<select id="shared-benchmark"></select></label></div>
<div class="chart-heading"><p id="shared-status" role="status">{esc(first['name']) if first else 'No shared benchmarks yet'}</p></div>
<details class="chart-options" id="shared-options" hidden><summary>Filters &amp; chart options</summary>{ranges}<div class="option-panel"><label><input type="checkbox" id="shared-log"> Log size axis</label><label><input type="checkbox" id="shared-zero"> Start score axis at zero</label><label><input type="checkbox" id="shared-refs" checked> Include reference builds</label><button id="shared-copy" type="button">Copy view link</button></div></details>
<div id="shared-models" class="models" aria-label="Show or hide models">{legend}</div><p class="scope" id="shared-scope">Published evaluations may use different test settings.</p><div id="shared-plot" class="plot">{static_plot(first,palette)}</div><p class="shape-key">● Quantized build <span>◆ Reference build</span></p>
<div id="shared-inspect" hidden><label>Inspect result<select id="shared-pick"></select></label><div id="shared-detail" class="point-detail" aria-live="polite"></div></div>
<details id="shared-notes"><summary>Evaluation details &amp; coverage</summary><p>Only exact benchmark variants with the same declared score units and reviewed file sizes enter a shared view. Each source report remains a separate point. Missing results are unknown. File size is not total VRAM usage.</p><p>QFF transcribes publisher claims; it does not independently reproduce these evaluations. Common benchmark names do not verify identical prompting, sample selection or scoring settings. No cross-source ranking or averaged score is inferred. Source links may point to mutable cards; captured hashes are retained in the data.</p><div id="shared-contexts"></div></details>
<details open id="shared-evidence"><summary>Source-linked results</summary><div class="tablebox"><table><thead><tr><th>Benchmark</th><th>Model</th><th>Build</th><th>File GB</th><th>Published score</th><th>Source</th></tr></thead><tbody>{table}</tbody></table></div></details></section>
<footer><span><a href="../">QFF</a> · Publisher claims, linked to their sources.</span><span><a href="data.json">Chart data</a>{support}</span></footer></main></div><script id="shared-data" type="application/json">{encoded}</script><script>{script}</script></body></html>'''
    page = page.replace('<footer>', render_extracted(extracted) + '<footer>', 1)
    (target/'index.html').write_text(page)
    payload['extracted'] = [dict(task=g['task'], models=g['models'], count=g['count'], entries=g['entries']) for g in extracted]   # data file only; not inlined in the page
    (target/'data.json').write_text(json.dumps(payload, separators=(',', ':')))
    ns = 'http://www.sitemaps.org/schemas/sitemap/0.9'
    tree = ET.parse(out/'sitemap.xml')
    node = ET.SubElement(tree.getroot(), '{'+ns+'}url')
    ET.SubElement(node, '{'+ns+'}loc').text = base+'benchmarks/'
    tree.write(out/'sitemap.xml', encoding='utf-8', xml_declaration=True)
    return payload
