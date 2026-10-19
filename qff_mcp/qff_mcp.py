#!/usr/bin/env python3
"""QFF MCP server: read-only tools over the Quant Fidelity Frontier data.

Every value it returns is a publisher's own claim, copied verbatim from a model card and pinned to the card line it
came from (`src`). Nothing here was re-measured. Direct numerical rankings require verified compatible protocols,
independently of quantization or measurement publisher identity.

Run:
  python3 qff_mcp.py                       # stdio (Claude Desktop / Claude Code / any MCP client)
  python3 qff_mcp.py --http 8000           # streamable HTTP at /mcp (for hosting, e.g. a Hugging Face Space)

Data source (QFF_DATA): a folder holding models.json + data.json files (the assembled site, default ../_site), or the
public site URL (e.g. https://delirious6423.github.io/quant-fidelity-frontier/). Data is loaded once and cached.
"""
import json, os, re, sys, urllib.request

try:                                                   # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                                    # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("QFF_DATA") or os.path.join(HERE, "..", "_site")
# The Space package copies comparison.py alongside this server; local checkout loads it from the repo root.
sys.path.insert(0, os.path.dirname(HERE))
try:
    from comparison import classify_pair, classify_group, summarize_comparisons
except ImportError:
    classify_pair = classify_group = summarize_comparisons = None

CAVEAT = ("Published claims, not QFF measurements. Direct numerical comparison requires verified compatible protocols. "
          "A lower KLD across different base models does not rank model quality or quantization fidelity.")
_cache = {}

def _read(rel):
    if re.match(r"https?://", SRC):
        url = SRC.rstrip("/") + "/" + rel
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "qff-mcp/1.0"}), timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    with open(os.path.join(SRC, rel), encoding="utf-8") as source:
        return json.load(source)

def _manifest():
    if "m" not in _cache: _cache["m"] = _read("models.json")
    return _cache["m"]

def _model(model):
    """Resolve a model by id, slug or name (case-insensitive, partial ok when unique)."""
    ms = _manifest()["models"]; q = (model or "").strip().lower()
    hit = [m for m in ms if q in (m["model"].lower(), m["slug"].lower(), m["model"].split("/")[-1].lower())]
    if not hit: hit = [m for m in ms if q and q in m["model"].lower()]
    if len(hit) != 1:
        raise ValueError(f"model {model!r} matches {len(hit)} models; use one of: " + ", ".join(m["model"] for m in ms))
    m = hit[0]
    if m["data"] not in _cache: _cache[m["data"]] = _read(m["data"])
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
    if metric: raise ValueError(f"set {S['id']!r} has metrics {ks}, not {metric!r}")
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
    raise ValueError(f"unsupported metric direction {direction!r}; direct ranking is unavailable")

def _frontier(P, k):
    """Pareto set: no other point is no larger AND better (lower) on k. Reference builds are never ranked."""
    P = [p for p in P if p.get("x") is not None and k in p.get("m", {}) and p["m"][k].get("v") is not None and not p.get("is_ref")]
    out = []
    for p in P:
        dom = next((o for o in P if o is not p and ((o["x"] <= p["x"] and _loss(o, k) < _loss(p, k)) or (o["x"] < p["x"] and _loss(o, k) <= _loss(p, k)))), None)
        out.append((p, dom))
    return out

server = _Server("qff", instructions=(
    "QFF (Quant Fidelity Frontier): what Hugging Face quant publishers claim about their quants (KLD, perplexity, "
    "top-1 agreement, benchmarks, sizes), each claim pinned to its model-card line. Claims, not measurements. "
    "Start with list_models, then search_quants or list_sets. Use compare_measurements to check evidence-based "
    "compatibility across publishers or sets. Only rank when ranking_permitted is true; frontier and best_under "
    "recompute compatibility. Missing methodology is unverified. Cite the src link of every number you report."))

@server.tool()
def list_models() -> dict:
    """List the base models QFF covers, with capture date and counts."""
    m = _manifest()
    return {"about": m.get("about"), "models": m["models"], "comparison_policy": "evidence_verified_protocol", "caveat": CAVEAT}

@server.tool()
def search_quants(model: str, query: str = "", fmt: str = "", max_size_gb: float | None = None,
                  has_metric: str = "", publisher: str = "", limit: int = 25) -> dict:
    """Search every claim row for a model. query matches repo or variant text (all words must match);
    fmt filters by format (GGUF, EXL3, NVFP4, MLX, AWQ, GPTQ, FP8 ...); has_metric keeps rows with that claim
    (mean_kld, ppl, top1_pct, kld_99, or a bench: prefix); publisher filters by repo owner.
    Rows copied from another card are marked copied_from."""
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

@server.tool()
def get_repo(model: str, repo: str) -> dict:
    """Everything QFF holds for one repo: format, method, methodology as stated, flags, card pin, and every claim row
    it publishes (own files, its comparison builds, and other publishers' files it measured)."""
    m, d = _model(model)
    info = next((x for x in d["repos"] if x["repo"].lower() == repo.lower()), None)
    if not info: raise ValueError(f"{repo!r} is not in QFF for {m['model']}")
    rows = [r for r in d["rows"] if r["repo"].lower() == repo.lower() or (r.get("measured_by") or "").lower() == repo.lower()]
    keep = {k: info.get(k) for k in ("repo", "fmt", "runtime", "method", "methodology", "flags", "sha", "sha256", "status", "copied_from") if info.get(k) is not None}
    return {"model": m["model"], "repo": keep, "rows": [_row(r) for r in rows], **_rows_comparisons(rows), "caveat": CAVEAT}

@server.tool()
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

@server.tool()
def frontier(model: str, set_id: str, metric: str = "", max_x: float | None = None) -> dict:
    """Efficiency frontier inside one comparable set: quants that no other quant in the set beats on both size and
    the metric (top-1 higher is better; KLD/PPL lower is better). Each other quant lists the quant that beats it. max_x caps the set's x axis
    (GB or bits per weight, see the set's x label)."""
    m, d = _model(model)
    S = next((s for s in d["sets"] if s["id"] == set_id), None)
    if not S: raise ValueError(f"no set {set_id!r}; call list_sets")
    k = _metric_for(S, metric)
    comparison = _set_comparison(S, k)
    if not comparison["ranking_permitted"]:
        raise ValueError(f"set {set_id!r} is not comparable for {k!r}: " + json.dumps(comparison, ensure_ascii=False))
    P = [p for p in S["points"] if max_x is None or (p.get("x") is not None and p["x"] <= max_x)]
    F = _frontier(P, k)
    eff = sorted([p for p, dom in F if dom is None], key=lambda p: p["x"])
    beaten = [{**_point(p), "beaten_by": dom["variant"]} for p, dom in F if dom is not None]
    refs = [_point(p) for p in P if p.get("is_ref")]
    return {"model": m["model"], "set": {"id": S["id"], "title": S["title"], "reference": S.get("reference"), "data": S.get("data"),
            "x": (S.get("x") or {}).get("label")}, "metric": k, "efficient": [_point(p) for p in eff], "beaten": beaten,
            "reference_builds": refs, "comparison": comparison, "ranking_permitted": True, "caveat": CAVEAT}

@server.tool()
def best_under(model: str, max_size_gb: float, metric: str = "kld") -> dict:
    """For a size budget, the best quant under it in EACH comparable set that reports `metric` (higher top-1, lower KLD/PPL) and
    measures size in GB. Results from different sets are listed side by side and must not be ranked against each other."""
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

@server.tool()
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
            raise ValueError(f"unknown measurement_id {identifier!r}; call search_quants")
        # The same claim may appear in more than one chart; its measurement ID remains stable.
        first = hit[0]
        if any(p.get("protocol") != first.get("protocol") for p in hit[1:]):
            raise ValueError(f"measurement_id {identifier!r} has contradictory protocol records")
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
            raise ValueError(f"measurement {identifier!r} has no published {metric!r} claim; use its exact metric key")
    result = classify_pair(left, right, metric=metric) if classify_pair else _unavailable_comparison()
    return {"left_model": left_model["model"], "right_model": right_model["model"], "metric": metric,
            "left": _row(left) if "metrics" in left else _point(left),
            "right": _row(right) if "metrics" in right else _point(right),
            **result, "caveat": CAVEAT}

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1]) if len(sys.argv) > sys.argv.index("--http") + 1 else 8000
        try: server.run("streamable-http", host="0.0.0.0", port=port)
        except TypeError:
            server.settings.host, server.settings.port = "0.0.0.0", port; server.run("streamable-http")
    else:
        server.run("stdio")
