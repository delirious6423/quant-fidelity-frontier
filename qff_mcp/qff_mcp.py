#!/usr/bin/env python3
"""QFF MCP server: read-only tools over the Quant Fidelity Frontier data.

Every value it returns is a publisher's own claim, copied verbatim from a model card and pinned to the card line it
came from (`src`). Nothing here was re-measured. Direct numerical rankings require verified compatible protocols,
independently of quantization or measurement publisher identity.

Run:
  python3 qff_mcp.py                       # stdio (Claude Desktop / Claude Code / any MCP client)
  python3 qff_mcp.py --http 8000           # streamable HTTP at /mcp (for hosting, e.g. a Hugging Face Space)

Data source (QFF_DATA): a folder holding models.json + data.json files (the assembled site, default ../_site), or the
public site URL (e.g. https://delirious6423.github.io/quant-fidelity-frontier/). The cache refreshes every five minutes.
"""
import json, math, os, re, sys, urllib.request
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

try:
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError as _ToolError
    _SDK_V2 = True
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError as _ToolError
    _SDK_V2 = False
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("QFF_DATA") or os.path.join(HERE, "..", "_site")
# The Space package copies comparison.py alongside this server; a local checkout loads it from web/.
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "web"))
try:
    from comparison import classify_pair, classify_group, summarize_comparisons
except ImportError:
    classify_pair = classify_group = summarize_comparisons = None

CAVEAT = ("Published claims, not QFF measurements. Direct numerical comparison requires verified compatible protocols. "
          "A lower KLD across different base models does not rank model quality or quantization fidelity.")
BENCHMARK_CAVEAT = ("Publisher-reported task scores, not QFF measurements. Matching benchmark names and score units "
                   "do not verify matching evaluation settings. Independent reports are retained without ranking, "
                   "averaging or converting score units. File size is not total VRAM usage.")
_cache = {}
_cache_lock = threading.RLock()
_loaded_at = None
REFRESH_SECONDS = 300
MAX_MANIFEST_BYTES = 1_000_000
MAX_MODEL_BYTES = 32_000_000
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)


class QFFError(ValueError, _ToolError):
    """An expected, safe-to-display failure on both supported SDK generations."""


def _finite_filter(name, value, nonnegative=False):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or (nonnegative and value < 0)):
        raise QFFError(f'{name} must be a finite ' + ('nonnegative ' if nonnegative else '') + 'number')


def _http_security(port):
    """Allow this listener and its explicitly configured public origin only."""
    hosts = [host + suffix for host in ('127.0.0.1', 'localhost', '[::1]')
             for suffix in ('', f':{port}')]
    origins = ['http://' + host for host in hosts]
    public = os.environ.get('QFF_PUBLIC_URL', '')
    space_host = os.environ.get('SPACE_HOST', '')
    urls = ([public] if public else []) + (['https://' + space_host] if space_host else [])
    for url in urls:
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment
                or any(c in url for c in ('*', '%', '\\')) or any(c.isspace() for c in url)):
            raise QFFError('QFF_PUBLIC_URL must be an exact HTTPS origin; SPACE_HOST must be an exact hostname')
        # Accessing .port validates its syntax and range before accepting it.
        parsed.port
        hosts.append(parsed.netloc)
        origins.append('https://' + parsed.netloc)
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=hosts, allowed_origins=origins)


def _safe_relative(rel):
    if not isinstance(rel, str) or not rel or '\\' in rel or unquote(rel) != rel:
        raise QFFError('Data path must be a plain relative JSON path')
    parsed = urlsplit(rel)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or rel.startswith('/'):
        raise QFFError('Data path must remain within the configured data source')
    if any(part in ('', '.', '..') for part in rel.split('/')) or PurePosixPath(rel).suffix != '.json':
        raise QFFError('Data path must be a relative JSON file without traversal')
    return rel


def _remote_in_root(url):
    base, target = urlsplit(SRC), urlsplit(url)
    root = base.path.rstrip('/') + '/'
    return (target.scheme, target.netloc) == (base.scheme, base.netloc) and target.path.startswith(root) and not target.query and not target.fragment


class _SourceRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _remote_in_root(newurl):
            raise QFFError('Data redirect left the configured source')
        # Also reject encoded traversal and path segments even on the same host.
        relative = urlsplit(newurl).path[len(urlsplit(SRC).path.rstrip('/') + '/'):]
        _safe_relative(relative)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

def _read(rel):
    rel = _safe_relative(rel)
    limit = MAX_MANIFEST_BYTES if rel == 'models.json' else MAX_MODEL_BYTES
    if re.match(r"https?://", SRC):
        url = SRC.rstrip("/") + "/" + rel
        opener = urllib.request.build_opener(_SourceRedirects())
        with opener.open(urllib.request.Request(url, headers={"User-Agent": "qff-mcp/1.0"}), timeout=30) as source:
            raw = source.read(limit + 1)
    else:
        root = Path(SRC).resolve()
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise QFFError('Data path resolved outside the configured source')
        with path.open('rb') as source:
            raw = source.read(limit + 1)
    if len(raw) > limit:
        raise QFFError(f'Data JSON exceeds the {limit}-byte limit')
    return json.loads(raw.decode('utf-8'))

def _manifest():
    global _loaded_at
    with _cache_lock:
        if "m" not in _cache or _loaded_at is None or time.monotonic() - _loaded_at >= REFRESH_SECONDS:
            manifest = _read("models.json")
            if not isinstance(manifest, dict) or not isinstance(manifest.get('models'), list):
                raise QFFError('Invalid QFF model manifest')
            seen = set()
            for model in manifest['models']:
                if not isinstance(model, dict) or not all(isinstance(model.get(k), str) and model[k] for k in ('model', 'slug', 'data')):
                    raise QFFError('Invalid QFF model entry')
                _safe_relative(model['data'])
                if model['data'] == 'models.json' or model['slug'] in seen:
                    raise QFFError('Duplicate or invalid QFF model path')
                seen.add(model['slug'])
            # A refresh failure never replaces the last valid snapshot or marks
            # it fresh. The request raises and the next call retries.
            _cache.clear()
            _cache['m'] = manifest
            _loaded_at = time.monotonic()
        return _cache["m"]

def _model(model):
    """Resolve a model by id, slug or name (case-insensitive, partial ok when unique)."""
    with _cache_lock:
        ms = _manifest()["models"]; q = (model or "").strip().lower()
        hit = [m for m in ms if q in (m["model"].lower(), m["slug"].lower(), m["model"].split("/")[-1].lower())]
        if not hit: hit = [m for m in ms if q and q in m["model"].lower()]
        if len(hit) != 1:
            raise QFFError(f"model {model!r} matches {len(hit)} models; use one of: " + ", ".join(m["model"] for m in ms))
        m = hit[0]
        if m["data"] not in _cache:
            data = _read(m["data"])
            if not isinstance(data, dict) or data.get('model') != m['model'] or not all(isinstance(data.get(k), list) for k in ('sets', 'rows', 'repos')):
                raise QFFError('Model data does not match its manifest identity or schema')
            _cache[m["data"]] = data
        return m, _cache[m["data"]]

def _metrics(ms, keys=None):
    return {k: v.get("raw") for k, v in ms.items() if v.get("v") is not None and (not keys or k in keys)}

def _unavailable_comparison():
    return {"classification": "different_or_unverified_protocol",
            "label": "Different or Unverified Protocol — Not Directly Comparable",
            "explanation": "The comparison engine is unavailable; compatibility cannot be verified.",
            "evidence": [], "verified_matching_conditions": [], "differences": [],
            "unknowns": ["comparison_engine"], "ranking_permitted": False}

def _comparison(points, metric=None):
    return classify_group(points, metric=metric) if classify_group else _unavailable_comparison()

def _set_comparison(S, metric):
    # Recompute from evidence, never trust a cached/legacy comparable boolean.
    return _comparison([p for p in S["points"] if p.get("m", {}).get(metric, {}).get("v") is not None], metric)

def _set_summary(S):
    comparisons = {k: _set_comparison(S, k) for k in S.get("metrics", [])}
    comparison = summarize_comparisons(comparisons) if summarize_comparisons else _unavailable_comparison()
    return {"id": S["id"], "title": S["title"],
            "comparable": bool(comparisons) and all(c["ranking_permitted"] for c in comparisons.values()),
            "comparison": comparison, "comparisons": comparisons,
            "ranking_permitted": comparison["ranking_permitted"], "reference": S.get("reference"),
            "data": S.get("data"), "x": (S.get("x") or {}).get("label"),
            "metrics": S.get("metrics"), "quants": len(S["points"])}

def _rows_comparisons(rows):
    metrics = sorted({key for row in rows for key, claim in row.get("metrics", {}).items()
                      if claim.get("v") is not None})
    results = {metric: _comparison([row for row in rows if row.get("metrics", {}).get(metric, {}).get("v") is not None], metric)
               for metric in metrics}
    summary = summarize_comparisons(results) if summarize_comparisons else _unavailable_comparison()
    return {"comparison": summary, "comparisons": results}

def _winner_comparison(entries, requested_metric):
    resolved_metrics = sorted({entry["metric"] for entry in entries})
    if len(resolved_metrics) <= 1:
        # A prefix can resolve to a named series even when every winner uses
        # that same series. Classify its actual overlay, never the prefix.
        metric = resolved_metrics[0] if resolved_metrics else requested_metric
        return _comparison([entry["best"] for entry in entries], metric)
    evidence = {}
    for entry in entries:
        for source in entry["comparison"].get("evidence", []):
            evidence[json.dumps(source, sort_keys=True)] = source
    return {"classification": "different_or_unverified_protocol",
            "label": "Different or Unverified Protocol — Not Directly Comparable",
            "explanation": "The requested metric prefix resolves to different metric series. Equivalence across those series has not been established; winners remain comparable only within their own verified sets.",
            "evidence": list(evidence.values()), "verified_matching_conditions": [], "differences": [],
            "unknowns": [{"condition": "metric_selection", "reason": "Different resolved metric series require an explicit equivalence assessment", "metrics": resolved_metrics}],
            "ranking_permitted": False, "metric": requested_metric, "resolved_metrics": resolved_metrics}

def _provenance(p):
    return {k: p.get(k) for k in ("measurement_id", "measurement_owner", "quantization_owner", "protocol", "claim_provenance")}

def _row(r):
    return {**_provenance(r), "repo": r["repo"], "variant": r["variant"], "kind": r["kind"], "measured_by": r.get("measured_by"), "fmt": r.get("fmt"),
            "size_gb": r.get("size_gb"), "size_raw": r.get("size_raw"), "size_basis": r.get("basis"), "bpw": r.get("bpw"),
            "claims": _metrics(r.get("metrics", {})), "src": r.get("src"),
            **({"copied_from": r["copied_from"]} if r.get("copied_from") else {}),
            **({"measured": False} if r.get("kind") == "quoted" else {})}

def _point(p, k=None):
    return {**_provenance(p), "variant": p["variant"], "repo": p["repo"], "fmt": p.get("fmt"), "x": p.get("x"), "size_gb": p.get("size_gb"),
            "bpw": p.get("bpw"), "claims": _metrics(p.get("m", {})), "src": p.get("src"),
            **({"reference_build": True} if p.get("is_ref") else {}), **({"repo_not_named_on_card": True} if p.get("inferred") else {})}

def _metric_for(S, metric):
    ks = S.get("metrics") or []
    if metric and metric in ks: return metric
    if metric: raise QFFError(f"set {S['id']!r} has metrics {ks}, not {metric!r}")
    return ks[0] if ks else None

def _loss(point, metric):
    value = point["m"][metric]["v"]
    protocol = point.get("protocol") or {}
    overlay = (protocol.get("metrics") or {}).get(metric) or {}
    definition = overlay.get("metric_definition", protocol.get("metric_definition", {}))
    if isinstance(definition, list):
        definition = definition[0] if definition else {}
    definition = definition.get("value", {}) if isinstance(definition, dict) else {}
    direction = definition.get("direction") if isinstance(definition, dict) else None
    if isinstance(direction, dict):
        direction = direction.get("optimization")
    if direction in ("higher", "maximize", "higher_is_better"):
        return -value
    if direction in ("lower", "minimize", "lower_is_better"):
        return value
    raise QFFError(f"unsupported metric direction {direction!r}; direct ranking is unavailable")

def _frontier(P, k):
    """Pareto set: no other point is no larger AND better (lower) on k. Reference builds are never ranked."""
    P = [p for p in P if p.get("x") is not None and k in p.get("m", {}) and p["m"][k].get("v") is not None and not p.get("is_ref")]
    out = []
    for p in P:
        dom = next((o for o in P if o is not p and ((o["x"] <= p["x"] and _loss(o, k) < _loss(p, k)) or (o["x"] < p["x"] and _loss(o, k) <= _loss(p, k)))), None)
        out.append((p, dom))
    return out


def _page(items, limit, offset):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise QFFError('limit must be an integer between 1 and 200')
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise QFFError('offset must be a nonnegative integer')
    selected = items[offset:offset + limit]
    return selected, dict(matches=len(items), limit=limit, offset=offset,
                          next_offset=offset + limit if offset + limit < len(items) else None)


def _benchmark_catalog(model=''):
    """Read the reviewed model payloads, not legacy bench:* metric guesses.

    Keep a single cache generation throughout a call. Source and point indices
    match the site's shared chart IDs; they identify this captured snapshot.
    """
    with _cache_lock:
        models = [_model(model)[0]] if model else _manifest()['models']
        groups, withheld = {}, []
        for entry in models:
            m, data = _model(entry['slug'])
            benchmarks = data.get('benchmarks')
            if benchmarks is None:
                continue  # Older payloads remain usable for the seven original tools.
            if not isinstance(benchmarks, dict) or not isinstance(benchmarks.get('sources'), list):
                raise QFFError('Invalid reviewed benchmark payload')
            withheld.extend(dict(source, model=m['model']) for source in benchmarks.get('withheld', []))
            for si, source in enumerate(benchmarks['sources']):
                for mi, metric in enumerate(source['metrics']):
                    name, unit = metric['name'], metric['unit']
                    if not all(isinstance(value, str) and value for value in (name, unit)):
                        raise QFFError('Benchmark names and units must be explicit')
                    group = groups.setdefault((name, unit), dict(name=name, unit=unit, points=[]))
                    for pi, point in enumerate(metric['points']):
                        if not all(isinstance(point.get(k), (int, float)) and not isinstance(point[k], bool)
                                   and math.isfinite(point[k]) for k in ('size', 'score')) or point['size'] <= 0:
                            raise QFFError('Benchmark scores and positive file sizes must be finite numbers')
                        group['points'].append(dict(point, id=f'{m["slug"]}:{si}:{mi}:{pi}',
                            model=m['model'], slug=m['slug'], benchmark=name, unit=unit, size_gb=point['size'],
                            source_id=f'{m["slug"]}:{si}', source_repo=source['repo'],
                            source_label=source['label'], context=source['context'], notes=source.get('notes', []),
                            card_sha256=source['card_sha256'], revision=source['revision'],
                            captured_at=data.get('captured_at')))
        return list(groups.values()), withheld


server = _Server("qff", instructions=(
    "QFF (Quant Fidelity Frontier): what Hugging Face quant publishers claim about their quants (KLD, perplexity, "
    "top-1 agreement, benchmarks, sizes), each claim pinned to its model-card line. Claims, not measurements. "
    "Start with list_models, then search_quants or list_sets for fidelity. Use list_benchmarks and search_benchmarks "
    "for reviewed task scores across models, keeping exact benchmark variants and units separate. "
    "Benchmark results retain every independent source report and do not establish a ranking. "
    "Use compare_measurements to check evidence-based "
    "compatibility across publishers or sets. Only rank when ranking_permitted is true; frontier and best_under "
    "recompute compatibility. Missing methodology is unverified. Cite the src link of every number you report."))

@server.tool(annotations=READ_ONLY)
def list_models() -> dict:
    """List the base models QFF covers, with capture date and counts."""
    m = _manifest()
    return {"about": m.get("about"), "models": m["models"], "comparison_policy": "evidence_verified_protocol", "caveat": CAVEAT}


@server.tool(annotations=READ_ONLY)
def list_benchmarks(query: str = '', model: str = '', min_models: int = 1,
                    limit: int = 50, offset: int = 0) -> dict:
    """Discover reviewed benchmark names, exact score units, covered models, source counts and point counts.
    query matches all words in the benchmark name; model optionally limits one model; min_models=2 finds
    shared benchmarks. Each name/unit combination stays separate. Pagination uses limit (1–200) and offset.
    Matching names do not establish comparable evaluation protocols or authorize ranking."""
    if isinstance(min_models, bool) or not isinstance(min_models, int) or min_models < 1:
        raise QFFError('min_models must be a positive integer')
    groups, withheld = _benchmark_catalog(model)
    words, rows = query.casefold().split(), []
    for group in groups:
        points = group['points']
        models = sorted({p['model'] for p in points})
        if len(models) < min_models or not all(word in group['name'].casefold() for word in words):
            continue
        rows.append(dict(name=group['name'], unit=group['unit'], models=models, model_count=len(models),
                         source_count=len({p['source_id'] for p in points}), point_count=len(points),
                         reference_count=sum(bool(p.get('reference')) for p in points)))
    rows.sort(key=lambda row: (row['name'].casefold(), row['unit']))
    selected, page = _page(rows, limit, offset)
    return dict(benchmarks=selected, **page, withheld_sources=withheld,
                ranking_permitted=False, caveat=BENCHMARK_CAVEAT)


@server.tool(annotations=READ_ONLY)
def search_benchmarks(benchmark: str, unit: str = '', model: str = '', query: str = '', publisher: str = '',
                      max_size_gb: float | None = None, min_score: float | None = None,
                      include_reference: bool = True, limit: int = 50, offset: int = 0) -> dict:
    """Read all independent reports for one EXACT benchmark name from list_benchmarks, across models by default.
    unit is required if the name has multiple score scales; no score conversion, averaging or ranking is performed.
    model optionally selects one model. query matches all words across model/build/source text. publisher is the
    reporting card's owner (source_repo), not the quantized build's owner. max_size_gb limits full file GB;
    min_score uses the selected published unit. Includes references by default. limit is 1–200; use next_offset
    to paginate. Each point preserves raw score/size, both source links, captured card hash, revision and context."""
    _finite_filter('max_size_gb', max_size_gb, nonnegative=True)
    _finite_filter('min_score', min_score)
    groups, withheld = _benchmark_catalog(model)
    matching = [g for g in groups if g['name'] == benchmark]
    if not matching:
        raise QFFError(f'No reviewed benchmark {benchmark!r}; use an exact name from list_benchmarks')
    units = sorted({g['unit'] for g in matching})
    if not unit and len(units) != 1:
        raise QFFError(f'Benchmark {benchmark!r} has multiple units {units}; select unit explicitly')
    selected_unit = unit or units[0]
    if selected_unit not in units:
        raise QFFError(f'Benchmark {benchmark!r} has units {units}, not {selected_unit!r}')
    words, rows = query.casefold().split(), []
    for group in matching:
        if group['unit'] != selected_unit:
            continue
        for point in group['points']:
            hay = ' '.join(str(point.get(k) or '') for k in ('model', 'variant', 'repo', 'source_repo', 'source_label')).casefold()
            if not all(word in hay for word in words): continue
            if publisher and point['source_repo'].split('/')[0].casefold() != publisher.casefold(): continue
            if not include_reference and point.get('reference'): continue
            if max_size_gb is not None and point['size_gb'] > max_size_gb: continue
            if min_score is not None and point['score'] < min_score: continue
            rows.append(point)
    selected, page = _page(rows, limit, offset)
    return dict(benchmark=benchmark, unit=selected_unit, points=selected, **page,
                model_count=len({p['model'] for p in rows}), source_count=len({p['source_id'] for p in rows}),
                order='captured model/source/row order; not a ranking', withheld_sources=withheld,
                ranking_permitted=False, caveat=BENCHMARK_CAVEAT)


@server.tool(annotations=READ_ONLY)
def search_quants(model: str, query: str = "", fmt: str = "", max_size_gb: float | None = None,
                  has_metric: str = "", publisher: str = "", limit: int = 25) -> dict:
    """Search every claim row for a model. query matches repo or variant text (all words must match);
    fmt filters by format (GGUF, EXL3, NVFP4, MLX, AWQ, GPTQ, FP8 ...); has_metric keeps rows with that claim
    (mean_kld, ppl, top1_pct, kld_99, or a bench: prefix); publisher filters by repo owner.
    Rows copied from another card are marked copied_from."""
    _finite_filter('max_size_gb', max_size_gb, nonnegative=True)
    m, d = _model(model); words = query.lower().split()
    rows = []
    for r in d["rows"]:
        hay = f"{r['repo']} {r['variant']} {r.get('fmt') or ''}".lower()
        if words and not all(w in hay for w in words): continue
        if fmt and (r.get("fmt") or "").lower() != fmt.lower(): continue
        if publisher and r["repo"].split("/")[0].lower() != publisher.lower(): continue
        if max_size_gb is not None and (r.get("size_gb") is None or r["size_gb"] > max_size_gb): continue
        ks = [k for k, v in r.get("metrics", {}).items() if v.get("v") is not None]
        if has_metric and not any(k == has_metric or k.startswith(has_metric) for k in ks): continue
        rows.append(r)
    selected = rows[:max(1, min(limit, 200))]
    return {"model": m["model"], "matches": len(rows), "rows": [_row(r) for r in selected],
            **_rows_comparisons(selected), "comparison_scope": "returned rows", "caveat": CAVEAT}

@server.tool(annotations=READ_ONLY)
def get_repo(model: str, repo: str) -> dict:
    """Everything QFF holds for one repo: format, method, methodology as stated, flags, card pin, and every claim row
    it publishes (own files, its comparison builds, and other publishers' files it measured)."""
    m, d = _model(model)
    info = next((x for x in d["repos"] if x["repo"].lower() == repo.lower()), None)
    if not info: raise QFFError(f"{repo!r} is not in QFF for {m['model']}")
    rows = [r for r in d["rows"] if r["repo"].lower() == repo.lower() or (r.get("measured_by") or "").lower() == repo.lower()]
    keep = {k: info.get(k) for k in ("repo", "fmt", "runtime", "method", "methodology", "flags", "sha", "sha256", "status", "copied_from") if info.get(k) is not None}
    return {"model": m["model"], "repo": keep, "rows": [_row(r) for r in rows], **_rows_comparisons(rows), "caveat": CAVEAT}

@server.tool(annotations=READ_ONLY)
def list_sets(model: str, comparable_only: bool = True) -> dict:
    """Measurement sets with evidence-based classification and per-metric ranking eligibility.
    Different publishers can be comparable; publisher identity alone never establishes compatibility.
    comparable_only includes sets with at least one verified comparable metric; inspect comparisons for that metric."""
    m, d = _model(model)
    out = [_set_summary(S) for S in d["sets"]]
    if comparable_only:
        out = [S for S in out if any(c["ranking_permitted"] for c in S["comparisons"].values())]
    out.sort(key=lambda s: (s["id"] != "all", s["title"].lower()))
    return {"model": m["model"], "sets": out, "caveat": CAVEAT}

@server.tool(annotations=READ_ONLY)
def frontier(model: str, set_id: str, metric: str = "", max_x: float | None = None) -> dict:
    """Efficiency frontier inside one comparable set: quants that no other quant in the set beats on both size and
    the metric (top-1 higher is better; KLD/PPL lower is better). Each other quant lists the quant that beats it. max_x caps the set's x axis
    (GB or bits per weight, see the set's x label)."""
    _finite_filter('max_x', max_x, nonnegative=True)
    m, d = _model(model)
    S = next((s for s in d["sets"] if s["id"] == set_id), None)
    if not S: raise QFFError(f"no set {set_id!r}; call list_sets")
    k = _metric_for(S, metric)
    comparison = _set_comparison(S, k)
    if not comparison["ranking_permitted"]:
        raise QFFError(f"set {set_id!r} is not comparable for {k!r}: " + json.dumps(comparison, ensure_ascii=False))
    P = [p for p in S["points"] if max_x is None or (p.get("x") is not None and p["x"] <= max_x)]
    F = _frontier(P, k)
    eff = sorted([p for p, dom in F if dom is None], key=lambda p: p["x"])
    beaten = [{**_point(p), "beaten_by": dom["variant"]} for p, dom in F if dom is not None]
    refs = [_point(p) for p in P if p.get("is_ref")]
    return {"model": m["model"], "set": {"id": S["id"], "title": S["title"], "reference": S.get("reference"), "data": S.get("data"),
            "x": (S.get("x") or {}).get("label")}, "metric": k, "efficient": [_point(p) for p in eff], "beaten": beaten,
            "reference_builds": refs, "comparison": comparison, "ranking_permitted": True, "caveat": CAVEAT}

@server.tool(annotations=READ_ONLY)
def best_under(model: str, max_size_gb: float, metric: str = "kld") -> dict:
    """For a size budget, the best quant under it in EACH comparable set that reports `metric` (higher top-1, lower KLD/PPL) and
    measures size in GB. Results from different sets are listed side by side and must not be ranked against each other."""
    _finite_filter('max_size_gb', max_size_gb, nonnegative=True)
    m, d = _model(model); out = []; excluded = []
    for S in d["sets"]:
        if (S.get("x") or {}).get("unit") != "GB": continue
        ks = S.get("metrics") or []
        k = metric if metric in ks else next((x for x in ks if x.startswith(metric)), None)
        if not k: continue
        comparison = _set_comparison(S, k)
        if not comparison["ranking_permitted"]:
            excluded.append({"set": S["id"], "metric": k, "comparison": comparison})
            continue
        P = [p for p in S["points"] if not p.get("is_ref") and p.get("x") is not None and p["x"] <= max_size_gb and p.get("m", {}).get(k, {}).get("v") is not None]
        if not P: continue
        b = min(P, key=lambda p: (_loss(p, k), p["x"]))
        out.append({"set": S["id"], "set_title": S["title"], "metric": k, "best": _point(b), "candidates": len(P), "comparison": comparison, "ranking_permitted": True})
    out.sort(key=lambda x: -x["candidates"])
    return {"model": m["model"], "max_size_gb": max_size_gb, "per_set": out, "excluded_sets": excluded,
            "comparison": _winner_comparison(out, metric),
            "caveat": CAVEAT + " Each entry is the best within its own set only."}

@server.tool(annotations=READ_ONLY)
def compare_measurements(model: str, left_id: str, right_id: str, metric: str = "kld", other_model: str = "") -> dict:
    """Classify two measurement IDs, including different quant publishers, evaluators, sets or base models.
    Find measurement_id with search_quants. Returns protocol evidence, verified matches, differences, unknowns,
    and whether direct numerical ranking is permitted. Copied claims retain their original provenance."""
    left_model, left_data = _model(model)
    right_model, right_data = _model(other_model or model)
    def find(data, identifier):
        hit = [r for r in data["rows"] if r.get("measurement_id") == identifier]
        if not hit:
            hit = [p for S in data["sets"] for p in S["points"] if p.get("measurement_id") == identifier]
        if not hit:
            raise QFFError(f"unknown measurement_id {identifier!r}; call search_quants")
        # The same claim may appear in more than one chart; its measurement ID remains stable.
        first = hit[0]
        if any(p.get("protocol") != first.get("protocol") for p in hit[1:]):
            raise QFFError(f"measurement_id {identifier!r} has contradictory protocol records")
        return first
    left, right = find(left_data, left_id), find(right_data, right_id)
    # Canonical chart keys may differ from the exact source-column key. A matching
    # chart point is the authoritative mapping; never infer a benchmark family.
    for data, identifier, measurement in ((left_data, left_id, left), (right_data, right_id, right)):
        has_metric = any(values.get(metric, {}).get("v") is not None
                         for values in (measurement.get("m", {}), measurement.get("metrics", {})))
        if not has_metric:
            has_metric = any(p.get("measurement_id") == identifier and p.get("m", {}).get(metric, {}).get("v") is not None
                             for S in data["sets"] for p in S["points"])
        if not has_metric:
            raise QFFError(f"measurement {identifier!r} has no published {metric!r} claim; use its exact metric key")
    result = classify_pair(left, right, metric=metric) if classify_pair else _unavailable_comparison()
    return {"left_model": left_model["model"], "right_model": right_model["model"], "metric": metric,
            "left": _row(left) if "metrics" in left else _point(left),
            "right": _row(right) if "metrics" in right else _point(right),
            **result, "caveat": CAVEAT}

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1]) if len(sys.argv) > sys.argv.index("--http") + 1 else 8000
        security = _http_security(port)
        if _SDK_V2:
            server.run("streamable-http", host="0.0.0.0", port=port,
                       transport_security=security, json_response=True)
        else:
            server.settings.transport_security = security
            server.settings.json_response = True
            server.settings.host, server.settings.port = "0.0.0.0", port
            server.run("streamable-http")
    else:
        server.run("stdio")
