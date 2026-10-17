#!/usr/bin/env python3
"""QFF MCP server: read-only tools over the Quant Fidelity Frontier data.

Every value it returns is a publisher's own claim, copied verbatim from a model card and pinned to the card line it
came from (`src`). Nothing here was re-measured. Numbers from different measurement sets are not comparable: a set is
one measurer using one reference, dataset and method.

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
CAVEAT = "Publisher claims, not measurements. Compare numbers only within one measurement set."
_cache = {}

def _read(rel):
    if re.match(r"https?://", SRC):
        url = SRC.rstrip("/") + "/" + rel
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "qff-mcp/1.0"}), timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    return json.load(open(os.path.join(SRC, rel), encoding="utf-8"))

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

def _row(r):
    return {"repo": r["repo"], "variant": r["variant"], "kind": r["kind"], "measured_by": r.get("measured_by"), "fmt": r.get("fmt"),
            "size_gb": r.get("size_gb"), "size_raw": r.get("size_raw"), "size_basis": r.get("basis"), "bpw": r.get("bpw"),
            "claims": _metrics(r.get("metrics", {})), "src": r.get("src"),
            **({"copied_from": r["copied_from"]} if r.get("copied_from") else {}),
            **({"measured": False} if r.get("kind") == "quoted" else {})}

def _point(p, k=None):
    return {"variant": p["variant"], "repo": p["repo"], "fmt": p.get("fmt"), "x": p.get("x"), "size_gb": p.get("size_gb"),
            "bpw": p.get("bpw"), "claims": _metrics(p.get("m", {})), "src": p.get("src"),
            **({"reference_build": True} if p.get("is_ref") else {}), **({"repo_not_named_on_card": True} if p.get("inferred") else {})}

def _metric_for(S, metric):
    ks = S.get("metrics") or []
    if metric and metric in ks: return metric
    if metric: raise ValueError(f"set {S['id']!r} has metrics {ks}, not {metric!r}")
    return ks[0] if ks else None

def _loss(point, metric):
    value = point["m"][metric]["v"]
    return -value if metric == "top1" or metric.startswith("top1:") else value

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
    "Start with list_models, then search_quants or list_sets. Only compare numbers inside one measurement set "
    "(frontier, best_in_set); cite the src link of every number you report."))

@server.tool()
def list_models() -> dict:
    """List the base models QFF covers, with capture date and counts."""
    m = _manifest()
    return {"about": m.get("about"), "models": m["models"], "caveat": CAVEAT}

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
        rows.append(_row(r))
    return {"model": m["model"], "matches": len(rows), "rows": rows[:max(1, min(limit, 200))], "caveat": CAVEAT}

@server.tool()
def get_repo(model: str, repo: str) -> dict:
    """Everything QFF holds for one repo: format, method, methodology as stated, flags, card pin, and every claim row
    it publishes (own files, its comparison builds, and other publishers' files it measured)."""
    m, d = _model(model)
    info = next((x for x in d["repos"] if x["repo"].lower() == repo.lower()), None)
    if not info: raise ValueError(f"{repo!r} is not in QFF for {m['model']}")
    rows = [_row(r) for r in d["rows"] if r["repo"].lower() == repo.lower() or (r.get("measured_by") or "").lower() == repo.lower()]
    keep = {k: info.get(k) for k in ("repo", "fmt", "runtime", "method", "methodology", "flags", "sha", "sha256", "status", "copied_from") if info.get(k) is not None}
    return {"model": m["model"], "repo": keep, "rows": rows, "caveat": CAVEAT}

@server.tool()
def list_sets(model: str, comparable_only: bool = True) -> dict:
    """Measurement sets for a model. A comparable set is one measurer with one reference, dataset and method, so its
    points can be ranked against each other. The 'all' overview mixes methods and is not comparable."""
    m, d = _model(model)
    out = [{"id": S["id"], "title": S["title"], "comparable": bool(S.get("comparable")), "reference": S.get("reference"),
            "data": S.get("data"), "x": (S.get("x") or {}).get("label"), "metrics": S.get("metrics"), "quants": len(S["points"])}
           for S in d["sets"] if S.get("comparable") or not comparable_only]
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
    if not S.get("comparable"): raise ValueError(f"set {set_id!r} is not comparable (mixed methods); pick a comparable set")
    k = _metric_for(S, metric)
    P = [p for p in S["points"] if max_x is None or (p.get("x") is not None and p["x"] <= max_x)]
    F = _frontier(P, k)
    eff = sorted([p for p, dom in F if dom is None], key=lambda p: p["x"])
    beaten = [{**_point(p), "beaten_by": dom["variant"]} for p, dom in F if dom is not None]
    refs = [_point(p) for p in P if p.get("is_ref")]
    return {"model": m["model"], "set": {"id": S["id"], "title": S["title"], "reference": S.get("reference"), "data": S.get("data"),
            "x": (S.get("x") or {}).get("label")}, "metric": k, "efficient": [_point(p) for p in eff], "beaten": beaten,
            "reference_builds": refs, "caveat": CAVEAT}

@server.tool()
def best_under(model: str, max_size_gb: float, metric: str = "kld") -> dict:
    """For a size budget, the best quant under it in EACH comparable set that reports `metric` (higher top-1, lower KLD/PPL) and
    measures size in GB. Results from different sets are listed side by side and must not be ranked against each other."""
    m, d = _model(model); out = []
    for S in d["sets"]:
        if not S.get("comparable") or (S.get("x") or {}).get("unit") != "GB": continue
        ks = S.get("metrics") or []
        k = metric if metric in ks else next((x for x in ks if x.startswith(metric)), None)
        if not k: continue
        P = [p for p in S["points"] if not p.get("is_ref") and p.get("x") is not None and p["x"] <= max_size_gb and p.get("m", {}).get(k, {}).get("v") is not None]
        if not P: continue
        b = min(P, key=lambda p: (_loss(p, k), p["x"]))
        out.append({"set": S["id"], "set_title": S["title"], "metric": k, "best": _point(b), "candidates": len(P)})
    out.sort(key=lambda x: -x["candidates"])
    return {"model": m["model"], "max_size_gb": max_size_gb, "per_set": out,
            "caveat": CAVEAT + " Each entry is the best within its own set only."}

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(sys.argv[sys.argv.index("--http") + 1]) if len(sys.argv) > sys.argv.index("--http") + 1 else 8000
        try: server.run("streamable-http", host="0.0.0.0", port=port)
        except TypeError:
            server.settings.host, server.settings.port = "0.0.0.0", port; server.run("streamable-http")
    else:
        server.run("stdio")
