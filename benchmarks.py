"""Conservative, opt-in benchmark pilot built solely from existing gated rows."""
import html
import json
import math

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
        metrics.append(dict(name=name, points=points))
    return dict(source=source, metrics=metrics)


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
    return '''<section id="benchmarks" class="bm-pilot" aria-labelledby="bm-title">
<p class="eyebrow">GLM pilot · Publisher-reported evidence</p>
<h2 id="bm-title">Benchmark score vs file size</h2>
<p class="note">One publisher's table, separate benchmarks. Scores are shown exactly as published; the card does not state their unit or full evaluation settings. No frontier or cross-publisher ranking is inferred.</p>
<div class="bm-controls" id="bm-controls" hidden>
<label for="bm-metric">Benchmark<select id="bm-metric"></select></label>
<label class="bm-check"><input id="bm-zero" type="checkbox"> Start score axis at zero</label>
<label for="bm-limit">File-size limit (GB)<input id="bm-limit" type="number" min="0" step="any" placeholder="Any size"></label>
</div>
<p id="bm-status" class="dim" role="status"></p>
<p id="bm-axis" class="dim" hidden></p>
<div id="bm-plot" class="bm-plot" hidden></div>
<p class="bm-legend" id="bm-legend" hidden>● Quantized build &nbsp; ◆ Source FP8 reference · Higher score ↑</p>
<div id="bm-selection" hidden><label for="bm-pick">Inspect result<select id="bm-pick"></select></label>
<div id="bm-detail" class="bm-detail" aria-live="polite"></div></div>
<p class="note">File size is not total VRAM usage. The source FP8 run used CPU offload; throughput is not comparable. Prompting, sample counts and score units are not specified in this table. Source links open the current model card, which may have changed since capture.</p>
<details open id="bm-evidence"><summary>Source-linked benchmark values</summary>
<div class="tablebox"><table class="disagree"><thead><tr><th>Benchmark</th><th>Build</th><th>File GB</th><th>Published score</th><th>Source</th></tr></thead><tbody>''' + table + '''</tbody></table></div></details>
<script id="bm-data" type="application/json">''' + data + '''</script></section>'''
