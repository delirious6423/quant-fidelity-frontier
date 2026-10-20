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
from urllib.parse import urljoin, urlsplit
from comparison import classify_group


# Reviewed source vocabulary, not numeric claims. Display names sometimes expand
# abbreviated headers or row labels; keep those aliases scoped to their source.
# In particular, an acc/acc_norm pair is two different metrics in one cell.
_SOURCE_BINDINGS = {
    'intel-ai/DeepSeek-V4-Flash-0731-MXFP4-Mixed-CT-AutoRound': {
        # What the size column is, from the card (reviewed vocabulary, not a number); none of these cards publishes resident memory.
        'size_basis': ('full checkpoint size on disk (all layers and the MTP module)', False),
        'unit': 'fraction',
        'metrics': {
            'GSM8K strict (5-shot)': ('GSM8K (strict)', None),
            'MMLU (0-shot)': ('MMLU', None),
            'PIQA (0-shot)': ('PIQA', None),
            'HellaSwag (0-shot)': ('HellaSwag', None),
            'AVG (four-task average)': ('AVG', None),
        },
        'rows': {
            'Official DeepSeek-V4-Flash-0731 (native FP4+FP8)':
                ('Official native FP4/FP8', 'deepseek-ai/DeepSeek-V4-Flash-0731', True, 'Official'),
            'MXFP4-Mixed-CT (this checkpoint)':
                ('MXFP4/MXFP8 AutoRound', None, False, 'This checkpoint'),
        },
    },
    'intel-ai/DeepSeek-V4.1-Flash-MXFP4-Mixed-CT-AutoRound': {
        # What the size column is, from the card (reviewed vocabulary, not a number); none of these cards publishes resident memory.
        'size_basis': ('complete checkpoint size on disk, including the Engram (n-gram) table and MTP', True),
        'unit': 'fraction',
        'metrics': {
            'GSM8K strict (5-shot)': ('GSM8K (5-shot, strict / flexible)', 0),
            'GSM8K flexible (5-shot)': ('GSM8K (5-shot, strict / flexible)', 1),
            'MMLU': ('MMLU', None),
            'PIQA acc': ('PIQA (acc / acc_norm)', 0),
            'PIQA acc_norm': ('PIQA (acc / acc_norm)', 1),
            'HellaSwag acc': ('HellaSwag (acc / acc_norm)', 0),
            'HellaSwag acc_norm': ('HellaSwag (acc / acc_norm)', 1),
        },
        'rows': {
            'deepseek-ai/DeepSeek-V4.1-Flash (official, W4A8 MoE)':
                ('Official native mixed precision (W4A8 MoE)', 'deepseek-ai/DeepSeek-V4.1-Flash', True, 'Official'),
            'This checkpoint (W4A4 MoE, OCP MXFP8)':
                ('MXFP4/MXFP8 AutoRound (W4A4 MoE)', None, False, 'This checkpoint'),
        },
    },
    'mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed': {
        # What the size column is, from the card (reviewed vocabulary, not a number); none of these cards publishes resident memory.
        'size_basis': ("file size of the repository checkpoint (the card's Size column)", False),
        'unit': 'fraction',
        'metrics': {'MMLU-Pro (600-question sample)': ('mmlu_pro (n=600)', None)},
        'rows': {
            'DeepSeek-V4-Flash-0731-2.4bit-mixed (this repo)':
                ('MLX mixed 2.44 bits/weight', None, False, None),
        },
    },
    'mbehr90/GLM-5.3-Flash-nvfp4': {
        # What the size column is, from the card (reviewed vocabulary, not a number); none of these cards publishes resident memory.
        'size_basis': ('file size on disk (publisher-reported GiB, converted to GB)', False),
        'unit': 'as published',
        'metrics': {name: (name, None) for name in ('GSM8K chat', 'GSM8K lm-eval', 'ARC-C', 'AI2D vision')},
        'rows': {
            'zai-org/GLM-5.3-Flash (source, FP8)':
                ('zai-org/GLM-5.3-Flash (source FP8)', 'zai-org/GLM-5.3-Flash', True, None),
        },
    },
    'eaddario/Qwen3.8-27B-GGUF': {
        # What the size column is, from the card (reviewed vocabulary, not a number); none of these cards publishes resident memory.
        'size_basis': ("GGUF file size (the card's Size (GB) column)", False),
        'unit': 'as published',
        'metrics': {
            **{name: (name, None) for name in ('HellaSwag', 'ARC Challenge', 'GPQA-Diamond', 'Truthful QA', 'WinoGrande')},
            # The card's section heading and dataset link identify Redux; its
            # table header has the documented spelling error "Redox".
            'MMLU-Redux': ('MMLU-Redox', None),
        },
    },
    'tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF': {
        # What the size column is, from the card (reviewed vocabulary, not a number); none of these cards publishes resident memory.
        'size_basis': ("GGUF file size (the card's Size (GB) column)", False),
        'unit': 'fraction',
        'metrics': {name: (name, None) for name in ('HumanEval', 'HumanEval+', 'MBPP', 'MBPP+')},
    },
    'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF': {
        # What the size column is, from the card (reviewed vocabulary, not a number); none of these cards publishes resident memory.
        'size_basis': ('full text-model download, including the weight and n-gram shards', True),
        'unit': 'as published',
        'metrics': {
            'AIME25': ('AIME25↑', None),
            'GPQA-Diamond': ('GPQA-D↑', None),
            'LiveCodeBench v6': ('LCB v6↑', None),
        },
        'rows': {'BF16': ('BF16 reference', 'Qwen/Qwen3.8-Flash-Next', True, None)},
    },
}
_SCALAR = r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)'
_LINK = re.compile(r'\[([^\]]+)\]\(([^\s)]+)\)')


def _plain(cell):
    """Remove presentation markup without erasing meaningful name characters."""
    cell = _LINK.sub(lambda match: match[1], cell)
    return ' '.join(cell.replace('**', '').replace('`', '').strip().split())


def _cells(line):
    line = line.strip()
    if not line.startswith('|') or not line.endswith('|'):
        raise ValueError('evidence is not a complete table row')
    return [cell.strip() for cell in re.split(r'(?<!\\)\|', line[1:-1])]


def _table_row(lines, line):
    """Bind a cited row to the header of its own contiguous Markdown table."""
    start = line - 1
    while start > 0 and lines[start - 1].strip().startswith('|'):
        start -= 1
    if line - 1 < start + 2:
        raise ValueError('evidence is not a table data row')
    headers, separator, cells = _cells(lines[start]), _cells(lines[start + 1]), _cells(lines[line - 1])
    if len(headers) != len(cells) or len(headers) != len(separator) or not all(
        re.fullmatch(r':?-{3,}:?', cell) for cell in separator
    ):
        raise ValueError('unsupported or ambiguous source table')
    return [_plain(header) for header in headers], cells


def _column(headers, name):
    columns = [i for i, header in enumerate(headers) if header == name]
    if len(columns) != 1:
        raise ValueError('benchmark or size column is absent or ambiguous')
    return columns[0]


def _artifact(cell, repo, revision):
    """The full linked artifact, including its revision, identifies a row."""
    links = _LINK.findall(cell)
    if len(links) > 1:
        raise ValueError('ambiguous row artifact')
    if not links:
        return None
    url = urljoin(f'https://huggingface.co/{repo}/blob/{revision}/README.md', links[0][1])
    parsed = urlsplit(url)
    parts = parsed.path.strip('/').split('/')
    if parsed.scheme != 'https' or parsed.netloc != 'huggingface.co' or len(parts) < 2:
        raise ValueError('unsupported row artifact')
    return url, '/'.join(parts[:2])


def _bound_evidence(lines, source, metric, point, revision):
    """Validate the metric cell, row identity, and same-artifact size cell.

    A pinned line proves only that bytes existed. These bindings additionally
    prove which column and build those bytes describe. Unknown source layouts
    fail closed until their vocabulary is reviewed; no nearest-name matching.
    """
    repo = source['repo']
    schema = _SOURCE_BINDINGS.get(repo)
    if schema is None or metric['name'] not in schema['metrics'] or metric['unit'] != schema['unit']:
        raise ValueError('unreviewed benchmark name or score scale')
    headers, row = _table_row(lines, point['line'])
    header, part = schema['metrics'][metric['name']]
    cell = _plain(row[_column(headers, header)])
    if part is not None:
        pair = cell.split('/')
        if len(pair) != 2:
            raise ValueError('paired benchmark cell changed')
        cell = pair[part].strip()
    score = re.fullmatch(r'(' + _SCALAR + r'%?)(?:\s*±\s*' + _SCALAR + r')?', cell)
    if not score or score[1] != str(point['raw']):
        raise ValueError('score differs from its benchmark column')

    label = _plain(row[0])
    variant, build_repo, reference, size_column = schema.get('rows', {}).get(
        label, (label, None, False, None))
    artifact = _artifact(row[0], repo, revision)
    build_repo = build_repo or (artifact[1] if artifact else repo)
    if (point['variant'] != variant or point.get('repo', repo) != build_repo
            or point.get('reference', False) is not reference):
        raise ValueError('benchmark row variant or repository differs from source')
    if artifact and artifact[1] != build_repo:
        raise ValueError('benchmark row artifact differs from repository')

    size_headers, size_row = _table_row(lines, point['size_line'])
    if size_column is not None:
        # Intel publishes sizes in a transposed summary: select the reviewed
        # Official/This checkpoint column, never either number on the line.
        if _plain(size_row[0]) not in ('Size', 'Size on disk'):
            raise ValueError('size evidence is not a full-checkpoint size row')
        column = _column(size_headers, size_column)
    else:
        size_artifact = _artifact(size_row[0], repo, revision)
        if artifact != size_artifact or (artifact is None and label != _plain(size_row[0])):
            raise ValueError('score and size describe different artifacts')
        candidates = [i for i, name in enumerate(size_headers)
                      if name.lower() in ('size', 'size (gb)', 'size (gib)', 'gb', 'gib')]
        if len(candidates) != 1:
            raise ValueError('full-file size column is absent or ambiguous')
        column = candidates[0]
    size_cell = _plain(size_row[column])
    units = re.findall(r'\b(GB|GiB)\b', size_headers[column])
    # A unit can be supplied by the header, or accompany its own number in a
    # dual-unit cell. Text after a comma describes shards or source notes.
    measurements = size_cell.split(',', 1)[0].split('/')
    matches = []
    for measurement in measurements:
        match = re.fullmatch(r'\s*(' + _SCALAR + r')\s*(GB|GiB)?\s*', measurement)
        if not match:
            raise ValueError('unsupported full-file size cell')
        unit = match[2] or (units[0] if len(units) == 1 else None)
        if unit is None or (units and unit not in units):
            raise ValueError('size unit is not bound to its source cell')
        matches.append((match[1], unit))
    if (str(point['size_raw']), point['size_unit']) not in matches:
        raise ValueError('size value or unit differs from its source column')


def _number(raw):
    value = str(raw).strip().rstrip('%').strip()
    if not re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)', value):
        raise ValueError('not a scalar number')
    return float(value)


def _evidence(lines, point, prefix):
    line = point['size_line' if prefix == 'size' else 'line']
    quote = point['size_quote' if prefix == 'size' else 'quote']
    raw = str(point['size_raw' if prefix == 'size' else 'raw'])
    if isinstance(line, bool) or not isinstance(line, int) or line < 1 or line > len(lines) or lines[line - 1] != quote:
        raise ValueError('source quote or line changed')
    # Numeric boundaries prevent a reviewed 8.0 silently matching 88.0.
    if not re.search(r'(?<![+\-−–\d.])' + re.escape(raw) + r'(?![\d.])', quote):
        raise ValueError('published value absent from cited line')
    value = point['size_value' if prefix == 'size' else 'score']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('nonfinite value')
    if _number(raw) != value:
        raise ValueError('numeric value differs from source text')


def group_benchmarks(sources):
    """Index exact task variants and units, retaining every independent result."""
    groups = {}
    for si, source in enumerate(sources):
        for mi, metric in enumerate(source['metrics']):
            key = (metric['name'], metric['unit'])
            group = groups.setdefault(key, dict(name=key[0], unit=key[1], evaluations=[]))
            group['evaluations'].append(dict(source=si, metric=mi))
    for group in groups.values():
        points = [point for e in group['evaluations'] for point in sources[e['source']]['metrics'][e['metric']]['points']]
        group['comparison'] = classify_group(points, metric='bench:' + group['name'])
    return list(groups.values())


def size_basis_text(sources):
    """Say what the size axis is. Resident (in-memory) size is plotted only where a card publishes it; none does for these results."""
    kinds = sorted({x['size_basis'] for x in sources})
    resident = any(x.get('size_basis_kind') == 'resident' for x in sources)
    head = 'Size axis: ' + ('; '.join(kinds) if kinds else 'file size as published') + '.'
    ngram = ' This size includes an n-gram (Engram) table that is not loaded into GPU memory as weights.' if any(x.get('includes_ngram') for x in sources) else ''
    note = '' if resident else ' It is not resident (in-memory) size: these cards do not publish one for these results.'
    return html.escape(head + ngram + note)


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
                    _bound_evidence(lines, source, metric, point, meta['sha'])
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
                sb = _SOURCE_BINDINGS.get(repo, {}).get('size_basis')
                sources.append(dict(repo=repo, label=source['label'], context=source['context'], notes=source.get('notes', []),
                                    size_basis=sb[0] if sb else 'file size as published', size_basis_kind='disk', includes_ngram=bool(sb and sb[1]),
                                    metrics=metrics, revision=meta['sha'], card_sha256=source['card_sha256']))
        except (KeyError, ValueError, OSError, TypeError) as exc:
            withheld.append(dict(repo=repo, reason=str(exc)))
    return dict(model=cfg.get('name', cfg.get('model', 'Model')), sources=sources, groups=group_benchmarks(sources), withheld=withheld)


def static_chart(payload):
    if not payload['sources']:
        return '<p>No reviewed benchmark scores with verified file sizes are currently available.</p>'
    metric = payload['sources'][0]['metrics'][0]
    points = [dict(p, _source=e['source']) for e in payload['groups'][0]['evaluations']
              for p in payload['sources'][e['source']]['metrics'][e['metric']]['points']]
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
    colors = ('var(--accent)', '#a36bcc', '#bd772b', '#4379bd', '#b65c7d', '#58893b')
    for i, p in enumerate(points):
        xx, yy = x(p['size']), y(p['score'])
        shape = '<rect x="-5" y="-5" width="10" height="10" transform="rotate(45)" class="bm-dot"/>' if p['reference'] else '<circle r="6" class="bm-dot"/>'
        shape = shape.replace('class="bm-dot"', f'class="bm-dot" style="fill:{colors[p["_source"] % len(colors)]}"')
        svg.append(f'<svg x="{xx}%" y="{yy}" width="1" height="1" overflow="visible">{shape}<title>{E(payload["sources"][p["_source"]]["label"])} · {E(p["variant"])}: {E(str(p["raw"]))}</title></svg>')
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
    context = E('Evaluation source: ' + payload['sources'][0]['label']) if payload['sources'] else ''
    notes = ''.join('<p><strong>' + E(s['label']) + '</strong></p><p>' + E(s['context']) + '</p><ul>' + ''.join('<li>' + E(n) + '</li>' for n in s['notes']) + '</ul>' for s in payload['sources'])
    comparison = payload['groups'][0]['comparison'] if payload['sources'] else None
    classification = E(comparison['label']) if comparison else ''
    reason = E(comparison['explanation']) if comparison else ''
    withheld = f'<p class="dim">{len(payload["withheld"])} source table(s) awaiting review after a source change.</p>' if payload['withheld'] else ''
    return '''<section id="benchmarks" class="bm-pilot" aria-labelledby="bm-title">
<h2 id="bm-title">Benchmark score vs file size</h2>
<p id="bm-classification" class="dim">''' + classification + '''</p>
<p id="bm-size-basis" class="dim">''' + size_basis_text(payload['sources']) + '''</p>

<div class="bm-controls" id="bm-controls" hidden>
<label for="bm-metric">Benchmark<select id="bm-metric"></select></label>
<label id="bm-source-choice" for="bm-source" hidden>Source<select id="bm-source"></select></label>
<label class="bm-check"><input id="bm-zero" type="checkbox"> Start score axis at zero</label>
<label for="bm-limit">File-size limit (GB)<input id="bm-limit" type="number" min="0" step="any" placeholder="Any size"></label>
</div><p id="bm-context" class="dim" hidden>''' + context + '''</p><p id="bm-status" class="dim" role="status"></p>
<p id="bm-axis" class="dim" hidden></p><div id="bm-plot" class="bm-plot">''' + static_chart(payload) + '''</div>
<div id="bm-sources" class="bm-sources" hidden aria-label="Evaluation sources"></div>
<p class="bm-legend" id="bm-legend">● Quantized build &nbsp; ◆ Reference · Higher score ↑</p>
<div id="bm-selection" hidden><label for="bm-pick">Inspect result<select id="bm-pick"></select></label><div id="bm-detail" class="bm-detail" aria-live="polite"></div></div>
<details id="bm-notes"><summary>Evaluation &amp; size notes</summary><p id="bm-reason">''' + reason + '''</p><p>File size is not total VRAM usage. Scores retain their published scale; no cross-source ranking is inferred. Source links may open a mutable model card; captured card hashes are recorded in the data.</p><div id="bm-source-notes">''' + notes + '''</div></details>''' + withheld + '''
<details open id="bm-evidence"><summary>Source-linked benchmark values</summary><div class="tablebox"><table class="disagree"><thead><tr><th>Evaluation</th><th>Benchmark</th><th>Build</th><th>File GB</th><th>Published score</th><th>Sources</th></tr></thead><tbody>''' + table + '''</tbody></table></div></details>
<script id="bm-data" type="application/json">''' + data + '''</script></section>'''
