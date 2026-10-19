"""Reviewed task scores with independent score and full-file-size provenance.

Manifests are pinned to captured card bytes. A changed card withdraws its whole
source until reviewed; absent sizes/scores are never synthesized.
"""
import hashlib
import html
import json
import math
import os
import re
from pathlib import Path
from comparison import classify_group


def _number(raw):
    value = str(raw).strip().rstrip('%').strip()
    if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)', value):
        raise ValueError('not a scalar number')
    return float(value)


def _evidence(lines, point, prefix):
    line = point['size_line' if prefix == 'size' else 'line']
    quote = point['size_quote' if prefix == 'size' else 'quote']
    raw = str(point['size_raw' if prefix == 'size' else 'raw'])
    if not isinstance(line, int) or line < 1 or line > len(lines) or lines[line - 1] != quote:
        raise ValueError('source quote or line changed')
    # Numeric boundaries prevent a reviewed 8.0 silently matching 88.0.
    if not re.search(r'(?<![+\-−–\d.])' + re.escape(raw) + r'(?![\d.])', quote):
        raise ValueError('published value absent from cited line')
    value = point['size_value' if prefix == 'size' else 'score']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('nonfinite value')
    if _number(raw) != value:
        raise ValueError('numeric value differs from source text')


def build_benchmarks(root, cfg):
    root = Path(root)
    manifest = root / 'benchmarks.json'
    if not manifest.exists():
        return None
    index = json.loads((root / 'snapshots/index.json').read_text())['repos']
    sources, withheld = [], []
    for source in json.loads(manifest.read_text())['sources']:
        repo = source['repo']
        try:
            meta = index[repo]
            card_path = (root / meta['file']).resolve()
            if not card_path.is_relative_to(root.resolve()):
                raise ValueError('card outside model directory')
            card = card_path.read_bytes()
            if hashlib.sha256(card).hexdigest() != source['card_sha256']:
                raise ValueError('card changed since benchmark review')
            lines = card.decode('utf-8').splitlines()
            metrics = []
            for metric in source['metrics']:
                if metric['unit'] not in ('percent', 'fraction', 'as published') or metric['direction'] != 'higher':
                    raise ValueError('unsupported score units or direction')
                points = []
                for point in metric['points']:
                    _evidence(lines, point, 'size')
                    _evidence(lines, point, 'score')
                    if point['size_unit'] not in ('GB', 'GiB') or point['size_value'] <= 0:
                        raise ValueError('invalid full-file size')
                    size = round(point['size_value'] * (1.073741824 if point['size_unit'] == 'GiB' else 1), 2)
                    if metric['unit'] == 'fraction' and not 0 <= point['score'] <= 1:
                        raise ValueError('fraction outside 0–1')
                    if metric['unit'] == 'percent' and not 0 <= point['score'] <= 100:
                        raise ValueError('percentage outside 0–100')
                    url = f'https://huggingface.co/{repo}/blob/{meta["sha"]}/README.md'
                    points.append(dict(variant=point['variant'], repo=point.get('repo', repo), size=size,
                                       size_raw=f'{point["size_raw"]} {point["size_unit"]}', score=point['score'], raw=point['raw'],
                                       reference=bool(point.get('reference')), src=url + f'#L{point["line"]}',
                                       size_src=url + f'#L{point["size_line"]}', line=point['line'], size_line=point['size_line'],
                                       flags=point.get('flags', [])))
                if points:
                    metrics.append(dict(name=metric['name'], unit=metric['unit'], points=points,
                                        comparison=classify_group(points, metric='bench:' + metric['name'])))
            if metrics:
                sources.append(dict(repo=repo, label=source['label'], context=source['context'], notes=source.get('notes', []),
                                    metrics=metrics, revision=meta['sha'], card_sha256=source['card_sha256']))
        except (KeyError, ValueError, OSError, TypeError) as exc:
            withheld.append(dict(repo=repo, reason=str(exc)))
    return dict(model=cfg.get('name', cfg.get('model', 'Model')), sources=sources, withheld=withheld)


def static_chart(payload):
    if not payload['sources']:
        return '<p>No reviewed benchmark scores with verified file sizes are currently available.</p>'
    metric = payload['sources'][0]['metrics'][0]
    points = metric['points']
    lo, hi = min(p['score'] for p in points), max(p['score'] for p in points)
    pad = max((hi - lo) * .2, .001 if metric['unit'] == 'fraction' else 1)
    lo -= pad; hi += pad
    xmax = max(p['size'] for p in points) * 1.12
    x = lambda size: 18 + 76 * size / xmax
    y = lambda score: 260 - 210 * (score - lo) / (hi - lo)
    E = html.escape
    svg = [f'<svg width="100%" height="320" role="img" aria-label="{E(metric["name"])} published score versus file size in GB"><title>{E(metric["name"])}: publisher-reported scores; zoomed linear axis</title>']
    unit_label = ' (%)' if metric['unit'] == 'percent' else ' (0–1)' if metric['unit'] == 'fraction' else ''
    svg.append(f'<text x="18%" y="20">Published score{unit_label} ↑</text>')
    for i in range(4):
        score = lo + (hi - lo) * i / 3; yy = y(score); size = xmax * i / 3
        svg.append(f'<line x1="18%" x2="94%" y1="{yy}" y2="{yy}" stroke="var(--line)"/><text x="15%" y="{yy+4}" text-anchor="end">{score:.3g}</text>')
        svg.append(f'<text x="{x(size)}%" y="283" text-anchor="middle">{size:.0f}</text>')
    for i, p in enumerate(points):
        xx, yy = x(p['size']), y(p['score'])
        shape = '<rect x="-5" y="-5" width="10" height="10" transform="rotate(45)" class="bm-dot"/>' if p['reference'] else '<circle r="6" class="bm-dot"/>'
        svg.append(f'<svg x="{xx}%" y="{yy}" width="1" height="1" overflow="visible">{shape}<title>{E(p["variant"])}: {E(str(p["raw"]))}</title></svg>')
        if len(points) <= 4 or i == 0:
            svg.append(f'<text x="{xx}%" y="{yy-17}" text-anchor="middle" class="bm-label">{E(str(p["raw"]))}</text>')
    svg.append('<text x="56%" y="312" text-anchor="middle">File size (GB)</text></svg>')
    return ''.join(svg)


def render_benchmarks(payload):
    if payload is None:
        return ''
    E = html.escape
    table = ''.join(f'<tr><td>{E(s["label"])}</td><td>{E(m["name"])}</td><td>{E(p["variant"])}</td>'
                    f'<td>{p["size"]:g}</td><td>{E(str(p["raw"]))} ({E(m["unit"])})</td>'
                    f'<td><a href="{E(p["src"], quote=True)}">Score L{p["line"]}</a> · <a href="{E(p["size_src"], quote=True)}">Size L{p["size_line"]}</a></td></tr>'
                    for s in payload['sources'] for m in s['metrics'] for p in m['points'])
    data = json.dumps(payload, separators=(',', ':')).replace('<', '\\u003c')
    context = E(payload['sources'][0]['context']) if payload['sources'] else ''
    notes = ''.join('<p><strong>' + E(s['label']) + '</strong></p><p>' + E(s['context']) + '</p><ul>' + ''.join('<li>' + E(n) + '</li>' for n in s['notes']) + '</ul>' for s in payload['sources'])
    comparison = payload['sources'][0]['metrics'][0]['comparison'] if payload['sources'] else None
    classification = E(comparison['label']) if comparison else ''
    reason = E(comparison['explanation']) if comparison else ''
    withheld = f'<p class="dim">{len(payload["withheld"])} source table(s) awaiting review after a source change.</p>' if payload['withheld'] else ''
    return '''<section id="benchmarks" class="bm-pilot" aria-labelledby="bm-title">
<h2 id="bm-title">Benchmark score vs file size</h2>
<p id="bm-classification" class="dim">''' + classification + '''</p>
<p class="note">Published scores, grouped by evaluation source.</p>
<div class="bm-controls" id="bm-controls" hidden>
<label for="bm-source">Evaluation source<select id="bm-source"></select></label>
<label for="bm-metric">Benchmark<select id="bm-metric"></select></label>
<label class="bm-check"><input id="bm-zero" type="checkbox"> Start score axis at zero</label>
<label for="bm-limit">File-size limit (GB)<input id="bm-limit" type="number" min="0" step="any" placeholder="Any size"></label>
</div><p id="bm-context" class="dim" hidden>''' + context + '''</p><p id="bm-status" class="dim" role="status"></p>
<p id="bm-axis" class="dim" hidden></p><div id="bm-plot" class="bm-plot">''' + static_chart(payload) + '''</div>
<p class="bm-legend" id="bm-legend">● Quantized build &nbsp; ◆ Reference · Higher score ↑</p>
<div id="bm-selection" hidden><label for="bm-pick">Inspect result<select id="bm-pick"></select></label><div id="bm-detail" class="bm-detail" aria-live="polite"></div></div>
<details id="bm-notes"><summary>Evaluation &amp; size notes</summary><p id="bm-reason">''' + reason + '''</p><p>File size is not total VRAM usage. Scores retain their published scale; no cross-source ranking is inferred. Source links may open a mutable model card; captured card hashes are recorded in the data.</p><div id="bm-source-notes">''' + notes + '''</div></details>''' + withheld + '''
<details open id="bm-evidence"><summary>Source-linked benchmark values</summary><div class="tablebox"><table class="disagree"><thead><tr><th>Evaluation</th><th>Benchmark</th><th>Build</th><th>File GB</th><th>Published score</th><th>Sources</th></tr></thead><tbody>''' + table + '''</tbody></table></div></details>
<script id="bm-data" type="application/json">''' + data + '''</script></section>'''
