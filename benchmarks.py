"""Conservative, opt-in benchmark pilot built solely from existing gated rows."""
import html
import json
import math
from comparison import classify_group

BENCHMARKS = ('GSM8K chat', 'GSM8K lm-eval', 'ARC-C', 'AI2D vision')


def build_pilot(rows, cfg):
    if not cfg.get('benchmark_pilot') or cfg.get('slug') != 'glm-5.3-flash':
        return None
    # Deliberately one source table; do not infer comparability across cards.
    source = 'mbehr90/GLM-5.3-Flash-nvfp4'
    eligible = [r for r in rows if r['measured_by'] == source
                and r.get('panel') == 'Measured on 4x H100 80 GB'
                and r['kind'] in ('own', 'cross') and not r.get('copied_from')
                and r.get('measured') is not False and not r.get('weights_modified')
                and (r['repo'] == source or (r['repo'] == cfg['model'] and r.get('is_ref')))
                and r.get('basis') == 'disk'
                and isinstance(r.get('size_gb'), (int, float))
                and math.isfinite(r['size_gb']) and r['size_gb'] > 0]
    metrics = []
    for name in BENCHMARKS:
        points = []
        for r in eligible:
            v = r['metrics'].get('bench:' + name, {})
            if not isinstance(v.get('v'), (int, float)) or not math.isfinite(v['v']):
                continue
            points.append(dict(variant=r['variant'], repo=r['repo'], size=r['size_gb'],
                               size_raw=r['size_raw'], score=v['v'], raw=v['raw'],
                               reference=bool(r.get('is_ref')), src=r['src'],
                               flags=r.get('flags', []), line=r['L']))
        measured = [r for r in eligible if r.get('metrics', {}).get('bench:' + name, {}).get('v') is not None]
        metrics.append(dict(name=name, points=points, comparison=classify_group(measured, metric='bench:' + name)))
    return dict(source=source, metrics=metrics)


def static_chart(payload):
    """Render the default chart without requiring a script-enabled viewer."""
    points = payload['metrics'][0]['points']
    if not points:
        return '<p>No eligible benchmark results with known file size.</p>'
    lo = min(p['score'] for p in points) - 1
    hi = max(p['score'] for p in points) + 1
    xmax = max(p['size'] for p in points) * 1.12
    x = lambda size: 18 + 76 * size / xmax
    y = lambda score: 260 - 210 * (score - lo) / (hi - lo)
    svg = ['<svg width="100%" height="320" role="img" aria-label="GSM8K chat published score versus file size in GB"><title>Published GSM8K chat scores; linear zoomed axis</title>']
    svg.append('<text x="18%" y="20">Published score ↑</text>')
    for i in range(4):
        score = lo + (hi - lo) * i / 3
        yy = y(score)
        svg.append(f'<line x1="18%" x2="94%" y1="{yy}" y2="{yy}" stroke="var(--line)"/><text x="15%" y="{yy+4}" text-anchor="end">{score:.1f}</text>')
        size = xmax * i / 3
        svg.append(f'<text x="{x(size)}%" y="283" text-anchor="middle">{size:.0f}</text>')
    for p in points:
        xx, yy = x(p['size']), y(p['score'])
        label = html.escape(str(p['raw']))
        shape = (f'<rect x="-5" y="-5" width="10" height="10" transform="rotate(45)" class="bm-dot"/>' if p['reference'] else '<circle r="6" class="bm-dot"/>')
        svg.append(f'<svg x="{xx}%" y="{yy}" width="1" height="1" overflow="visible">{shape}</svg><text x="{xx}%" y="{yy-17}" text-anchor="middle" class="bm-label">{label}</text>')
    svg.append('<text x="56%" y="312" text-anchor="middle">File size (GB)</text></svg>')
    return ''.join(svg)


def render_pilot(payload):
    if payload is None:
        return ''
    E = html.escape
    # A useful source-linked fallback remains visible without JavaScript.
    table = ''.join(f'<tr><td>{E(m["name"])}</td><td>{E(p["variant"])}</td>'
                    f'<td>{p["size"]:g}</td><td>{E(str(p["raw"]))}</td>'
                    f'<td><a href="{E(p["src"], quote=True)}">L{p["line"]}</a></td></tr>'
                    for m in payload['metrics'] for p in m['points'])
    data = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    markup = '''<section id="benchmarks" class="bm-pilot" aria-labelledby="bm-title">
<p class="eyebrow">GLM pilot · Publisher-reported evidence</p>
<h2 id="bm-title">Benchmark score vs file size</h2>
<p id="bm-classification" class="dim">__CLASSIFICATION__</p>
<p class="note">One publisher's table, separate benchmarks. Scores are shown exactly as published; the card does not state their unit or full evaluation settings. No frontier or cross-publisher ranking is inferred.</p>
<details id="bm-methodology"><summary>Methodology &amp; comparison evidence</summary><p id="bm-reason">__REASON__</p><p>The source table omits exact dataset/token selection, implementation, scoring units and other evaluation settings. A shared publisher alone cannot verify compatibility. <a href="#bm-evidence">Read the source-linked values below</a>.</p></details>
<div class="bm-controls" id="bm-controls" hidden>
<label for="bm-metric">Benchmark<select id="bm-metric"></select></label>
<label class="bm-check"><input id="bm-zero" type="checkbox"> Start score axis at zero</label>
<label for="bm-limit">File-size limit (GB)<input id="bm-limit" type="number" min="0" step="any" placeholder="Any size"></label>
</div>
<p id="bm-status" class="dim" role="status"></p>
<p id="bm-axis" class="dim" hidden></p>
<div id="bm-plot" class="bm-plot">__STATIC_CHART__</div>
<p class="bm-legend" id="bm-legend">● Quantized build &nbsp; ◆ Source FP8 reference · Higher score ↑</p>
<div id="bm-selection" hidden><label for="bm-pick">Inspect result<select id="bm-pick"></select></label>
<div id="bm-detail" class="bm-detail" aria-live="polite"></div></div>
<p class="note">File size is not total VRAM usage. The source FP8 run used CPU offload; throughput is not comparable. Prompting, sample counts and score units are not specified in this table. Source links open the current model card, which may have changed since capture.</p>
<details open id="bm-evidence"><summary>Source-linked benchmark values</summary>
<div class="tablebox"><table class="disagree"><thead><tr><th>Benchmark</th><th>Build</th><th>File GB</th><th>Published score</th><th>Source</th></tr></thead><tbody>''' + table + '''</tbody></table></div></details>
<script id="bm-data" type="application/json">''' + data + '''</script></section>'''

    comparison = payload['metrics'][0]['comparison']
    return (markup.replace('__STATIC_CHART__', static_chart(payload))
            .replace('__CLASSIFICATION__', E(comparison['label']))
            .replace('__REASON__', E(comparison['explanation'])))
