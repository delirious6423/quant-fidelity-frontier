#!/usr/bin/env python3
"""Build the comparison site from verified/part*.py + snapshots.
Outputs: site/data.json, site/index.html (full doc for GitHub Pages), site/artifact.html (body-only for claude.ai)."""
import json, glob, os, re, html, math, datetime, collections

CODE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]
import qcl_fmt as FMT
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")          # data root: one folder per model (models/<slug>/); code stays here
IDX = json.load(open(f"{ROOT}/snapshots/index.json"))
SIZES = json.load(open(f"{ROOT}/snapshots/hf_sizes_summary.json"))
CFG = json.load(open(f"{ROOT}/model.json")) if os.path.exists(f"{ROOT}/model.json") else {}
MODEL = CFG.get("model", "Qwen/Qwen3.8-Flash-Next"); MNAME = CFG.get("name", MODEL.split("/")[-1]); SLUG = CFG.get("slug", "")
SIZE_NOTE = CFG.get("size_note", 'This model carries a 51B-parameter n-gram table that some files keep on disk. A size marked "GPU weights" or "resident" leaves it out. GiB figures are converted to GB.')
COVER_NOTE = CFG.get("coverage_note", "claims made only off-site (other than Unsloth's docs page) or in images other than turboderp's")
PIN_NOTE = CFG.get("pin_note", "")
SITE_URL = "https://delirious6423.github.io/quant-fidelity-frontier/" + (SLUG + "/" if SLUG else "")
REPO_URL = "https://github.com/delirious6423/quant-fidelity-frontier"
_SITE = json.load(open(os.path.join(CODE, "site.json"))) if os.path.exists(os.path.join(CODE, "site.json")) else {}
SUPPORT_HTML = (f' · <a href="{html.escape(_SITE["support_url"])}" rel="noopener" data-umami-event="support">{html.escape(_SITE.get("support_label") or "Support QFF")}</a>'
                f' (<a href="{REPO_URL}/blob/main/docs/SPONSORS.md">sponsor policy</a>)') if _SITE.get("support_url") else ""
_AN = _SITE.get("analytics") or {}
ANALYTICS = (f'<script defer src="{html.escape(_AN.get("script") or "https://cloud.umami.is/script.js")}" data-website-id="{html.escape(_AN["website_id"])}" data-domains="delirious6423.github.io"></script>\n'
             if _AN.get("website_id") else "")   # cookie-free; data-domains keeps local test runs out of the stats
GIB = 1.073741824
E = html.escape

def load():
    R, seen = [], {}
    for p in sorted(glob.glob(f"{ROOT}/verified/part*.py")):
        ns = {}; exec(open(p).read(), ns)
        for r in ns["R"]:
            r["_part"] = os.path.basename(p)
            R.append(r)
    reviewed = {}
    for r in R:
        if r.get("auto"): continue
        if r["repo"] in reviewed: raise SystemExit(f"duplicate reviewed entry for {r['repo']}: {reviewed[r['repo']]} and {r['_part']}")
        reviewed[r["repo"]] = r["_part"]
    return [r for r in R if not (r.get("auto") and r["repo"] in reviewed)]

MINUS = re.compile(r"[\u2212\u2013\u2012\u2014\uFE63\uFF0D](?=\s*\.?\d)")
def num(s):
    if s is None: return None
    if isinstance(s, (int, float)): return float(s)
    s = MINUS.sub("-", str(s))
    m = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", str(s).replace(",", ""))
    return float(m.group()) if m else None

def err(s):
    if not isinstance(s, str): return None
    s = MINUS.sub("-", s)
    m = re.search(r"(?:±|\+/-)\s*(\d+(?:\.\d+)?|\.\d+)", s)
    return float(m.group(1)) if m else None

def to_gb(size, unit):
    if size is None: return None
    if unit == "bytes": g = size / 1e9; return round(g, 2) if g >= 0.1 else float(f"{g:.2g}")
    f = {"GiB": GIB, "MiB": GIB / 1024, "MB": 1e-3, "TiB": GIB * 1024, "TB": 1e3, "KiB": GIB / 1024 ** 2}.get(unit)
    if f: g = size * f; return round(g, 2) if g >= 0.1 else float(f"{g:.2g}")
    return float(size)

BASIS_LABEL = {"text_weights": "text weights (excl. vision, MTP)", "disk": "download", "hf_listing_total": "download (HF listing)", "vram": "GPU weights",
               "resident": "resident", "ram_in_use": "RAM in use", "trunk_only": "trunk only", "hf_branch_total": "download (HF branch listing)", None: "unknown"}

REF_OVERRIDES = json.load(open(f"{ROOT}/audit/ref_overrides.json")) if os.path.exists(f"{ROOT}/audit/ref_overrides.json") else {}
def ref_class(meth, repo=None):
    """Reference class for badges. Conservative: 'bf16' only when the stated reference IS the BF16/full-precision model."""
    if repo in REF_OVERRIDES: return REF_OVERRIDES[repo]["class"]
    if not meth: return "none"
    r = (meth.get("reference") or "").lower().strip()
    if r == "" or r.startswith("not stated"): return "unstated"
    if "same source" in r or "unspecified" in r or "not named" in r: return "unclear"
    if re.match(r"^(none|no )", r) or re.search(r"\bno (\w+ )?(bf16|fp8|reference|base)|\bnot (\w+ ){0,2}bf16|without (a )?bf16", r): return "other"
    head = re.split(r"[(;,]", r)[0]
    if re.search(r"q8_?0", head) or re.search(r"q8_?0\)? is the reference", r): return "q8"
    # the stated reference names the full-precision model itself (anywhere in the string), not a component of it
    BF = r"bf16|bfloat16|fp32|full[- ]precision|unquanti[sz]ed|original (model|weights)|qwen/qwen3\.8-flash-next(?![-\w])"
    comp = r"(n-?gram|table|embedding|ple|lm[_ ]head|kv)"
    if re.search(BF, r) and not re.search(rf"(bf16|bfloat16)\s+{comp}\b|\b{comp}\s+(in\s+)?(bf16|bfloat16)", r) and not re.search(r"\b(glm|other model)\b", r):
        return "bf16"
    return "other"

def canon(m):
    """Map published metric dict to canonical numeric fields (value + original string)."""
    out = {}
    for k, v in (m or {}).items():
        key = k
        if k == "mean_kl_mnats":
            out["mean_kld"] = {"v": num(v) / 1000.0, "raw": f"{v} mnats", "unit": "nats (converted from millinats)"}
            continue
        frac = re.match(r"^\s*(\d+)\s*(?:of|/)\s*(\d+)", str(v)) if key.startswith("top1") else None
        if frac and int(frac.group(2)) > 0:
            out[key] = {"v": round(100 * int(frac.group(1)) / int(frac.group(2)), 3), "err": None, "raw": v, "unit": "% (from count)"}
            continue
        val = num(v)
        if val is not None and re.match(r"^(top1|same_top1)", key) and "%" not in str(v) and 0 < val <= 1:
            out[key] = {"v": round(val * 100, 4), "err": None, "raw": v, "unit": "% (from fraction)"}; continue
        out[key] = {"v": val, "err": err(v), "raw": v}
    return out

def hf_line(repo, sha, L, src=None):
    if src == "unsloth_docs": return "https://unsloth.ai/docs/models/qwen3.8-next"
    return f"https://huggingface.co/{repo}/blob/{sha}/README.md" + (f"#L{L}" if L else "")

COPIES = json.load(open(f"{ROOT}/audit/copies.json")) if os.path.exists(f"{ROOT}/audit/copies.json") else {}
if os.path.exists(f"{ROOT}/audit/copy_overrides.json"): COPIES.update(json.load(open(f"{ROOT}/audit/copy_overrides.json")))   # reviewed by hand
def copied_from(repo):
    c = COPIES.get(repo)
    return c["source"] if c and (c.get("manual") or (c["share"] >= 0.6 and c["claim_lines"] >= 4)) else None

OTHER_MODEL = []
_card_cache = {}
def CARD_TEXT(repo):
    if repo not in _card_cache:
        m = IDX["repos"].get(repo); _card_cache[repo] = open(f"{ROOT}/{m['file']}", encoding="utf-8", errors="replace").read() if m else ""
    return _card_cache[repo]

def build_rows(R):
    rows, repos = [], []
    for r in R:
        meta = IDX["repos"][r["repo"]]; sha = meta["sha"]
        meth = r.get("methodology")
        rc = ref_class(meth, r["repo"])
        if rc == "none" and any((v.get("m") or {}) for v in r.get("variants", [])): rc = "unstated"
        cf = copied_from(r["repo"])
        if cf and COPIES[r["repo"]].get("manual"): pass   # flags written by hand
        elif cf and not any("copied" in (f or "").lower() for f in (r.get("flags") or [])):
            r["flags"] = list(r.get("flags") or []) + [f"Card's numbers match {cf}'s card ({COPIES[r['repo']]['shared_lines']} of {COPIES[r['repo']]['claim_lines']} claim lines identical, detected automatically); treated as copied, not as an independent measurement"]
        repos.append({k: r.get(k) for k in ["repo", "fmt", "runtime", "weights_modified", "method", "methodology", "flags"]}
                     | {"sha": sha, "lastModified": meta["lastModified"], "ref_class": rc,
                        "n_variants": len(r.get("variants", [])), "copied_from": cf, "auto": bool(r.get("auto"))})
        def add(e, kind, measured_by, of):
            eflags = [str(f) for f in (e.get("flags") or [])]
            if any(re.search(r"not this model", f, re.I) for f in eflags) or re.search(r"\bGLM\b", str(e.get("name") or e.get("variant") or "")):
                OTHER_MODEL.append((r["repo"], e.get("name") or e.get("variant"))); return      # measured on a different model: not a row
            vname = str(e.get("name") or e.get("variant") or "")
            quoted = kind in ("cross", "ladder") and (e.get("measured") is False or any(re.search(r"quoted|not measured|reported by|publisher-reported|as published by", f, re.I) for f in eflags)
                                                     or bool(re.search(r"\bquoted\b|-reported\b", vname, re.I)))
            if quoted: kind = "quoted"
            if e.get("size") is not None and not e.get("unit"):
                raise SystemExit(f"size without unit: {r['repo']} {kind} {e.get('name') or e.get('variant')} {e.get('size')} (L{e.get('L')})")
            m = canon(e.get("m"))
            rows.append({
                "repo": of or r["repo"], "measured_by": measured_by, "kind": kind,
                "variant": e.get("name") or e.get("variant"), "fmt": r.get("fmt") if kind != "cross" else ((lambda f: None if f == "other" else f)(FMT.fmt_of(f"{e.get('of') or ''} {e.get('name') or e.get('variant') or ''}"))),
                "size_gb": to_gb(e.get("size"), e.get("unit")), "size_raw": f"{e.get('size')} {e.get('unit') or ''}".strip() if e.get("size") is not None else None,
                "basis": e.get("basis"), "bpw": e.get("bpw"),
                "metrics": m, "ref_class": rc if kind != "cross" else rc,
                "weights_modified": bool(r.get("weights_modified")) if kind == "own" else None,
                "src": hf_line(r["repo"], sha, e.get("L"), e.get("src")), "L": e.get("L"), "sha": sha, "copied_from": cf, "flags": eflags,
                "measured": e.get("measured"),
                "src_label": ("unsloth.ai docs" if e.get("src") == "unsloth_docs" else f"L{e.get('L')} @ {sha[:7]}") if e.get("L") else f"@ {sha[:7]}",
            })
        for v in r.get("variants", []): add(v, "own", r["repo"], None)
        for v in r.get("ladder", []) or []:
            add(v, "ladder", r["repo"], None)
        for v in r.get("cross", []) or []: add(v, "cross", r["repo"], v.get("of"))
    # attribution check: a target repo the card never names is an inference (e.g. from UD- file names)
    for row in rows:
        if row["kind"] in ("cross", "quoted") and re.fullmatch(r"[\w.\-]+/[\w.\-]+", row["repo"]):
            txt = re.sub(r"[^a-z0-9/]+", "", CARD_TEXT(row["measured_by"]).lower())
            row["inferred"] = re.sub(r"[^a-z0-9/]+", "", owner(row["repo"]).lower()) not in txt
    modmap = {x["repo"]: bool(x.get("weights_modified")) for x in repos}
    for r in rows:
        if r["kind"] in ("cross", "quoted") and r["repo"] in modmap: r["weights_modified"] = modmap[r["repo"]]
    own_vals = collections.defaultdict(dict)
    for r in rows:
        if r["kind"] == "own":
            q = qid(r["repo"], r["variant"])
            for k, m in r["metrics"].items(): own_vals[q][k] = m["raw"]
    for r in rows:
        if r["kind"] != "cross" or r.get("copied_from"): continue
        if owner(r["measured_by"]) == owner(r["repo"]): continue     # a publisher's own card reprinting its own result is not a rival restatement
        q = qid(r["repo"], r["variant"]); ov = own_vals.get(q)
        if not ov: continue
        def fam(k):
            for c, rx in CANON:
                if rx.match(k): return c
            return k
        ovf = {fam(k): v for k, v in ov.items()}
        shared = [k for k in r["metrics"] if fam(k) in ovf]
        if shared and all(same_value(r["metrics"][k]["raw"], ovf[fam(k)]) for k in shared):
            r["copied_from"] = r["repo"]; r["restated"] = True
    created = {rid: m.get("createdAt") or "" for rid, m in IDX["repos"].items()}
    owns = [r for r in rows if r["kind"] == "own" and not r.get("copied_from") and len([k for k in r["metrics"] if r["metrics"][k]["v"] is not None]) >= 2]
    for r in owns:
        for o in owns:
            if o is r or owner(o["repo"]) == owner(r["repo"]) or created.get(o["repo"], "") >= created.get(r["repo"], ""): continue
            ks = [k for k in r["metrics"] if k in o["metrics"]]
            if len(ks) >= 2 and len(ks) == len(r["metrics"]) and all(same_value(r["metrics"][k]["raw"], o["metrics"][k]["raw"]) for k in ks):
                r["copied_from"] = o["repo"]; r["restated"] = True; break
    return rows, repos

def same_value(a, b):
    """Equal to the precision of the less precise published string (catches rounded copies)."""
    def p(v):
        mm = re.search(r"\d[\d,]*(?:\.\d+)?|\.\d+", str(v))
        if not mm: return None, None
        s = mm.group().replace(",", "")
        d = len(s.split(".")[1]) if "." in s else 0
        return float(s), d
    (x, dx), (y, dy) = p(a), p(b)
    if x is None or y is None: return False
    d = min(dx, dy); return abs(x - y) <= 0.5 * 10 ** (-d) + 1e-12

def norm_raw(v):
    mm = re.search(r"\d[\d,]*(?:\.\d+)?|\.\d+", str(v))
    return f"{float(mm.group().replace(',', '')):.6g}" if mm else str(v)

# ---------- rendering helpers ----------
def fmtv(x, nd=4):
    if x is None: return ""
    if abs(x) >= 100: return f"{x:.1f}"
    if abs(x) >= 10: return f"{x:.2f}"
    if abs(x) < 0.1 and x: return f"{x:.2g}"
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") if nd else str(x)

def owner(repo): return repo.split("/")[0] if "/" in repo else repo
def short(repo):
    if "/" not in repo: return ""
    n = repo.split("/")[1]; base = MNAME
    return re.sub(re.escape(base) + r"[-_]?", "", n, flags=re.I) or n

REF_TXT = {"bf16": "vs BF16", "q8": "vs a Q8_0", "unclear": "reference unclear", "other": "other or no reference", "unstated": "reference not stated", "none": "no quality claim"}

def kld_chart(rows):
    """Static SVG: own-claim mean KLD vs size, log y. Filled = BF16 reference; hollow = other/unstated."""
    pts = [r for r in rows if r["kind"] in ("own", "ladder") and r["size_gb"] and "mean_kld" in r["metrics"] and r["metrics"]["mean_kld"]["v"]]
    if not pts: return "", ""
    W, H, ml, mr, mt, mb = 760, 420, 58, 16, 18, 46
    xs = [p["size_gb"] for p in pts]; ys = [p["metrics"]["mean_kld"]["v"] for p in pts]
    x0, x1 = 20, math.ceil(max(xs) / 20) * 20
    y0, y1 = 10 ** math.floor(math.log10(min(ys))), 10 ** math.ceil(math.log10(max(ys)))
    X = lambda v: ml + (v - x0) / (x1 - x0) * (W - ml - mr)
    Y = lambda v: mt + (math.log10(y1) - math.log10(v)) / (math.log10(y1) - math.log10(y0)) * (H - mt - mb)
    pubs = sorted({owner(p["measured_by"]) for p in pts})
    pal = [f"var(--s{i % 8})" for i in range(len(pubs))]
    col = dict(zip(pubs, pal))
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="kldt kldd" class="chart"><title id="kldt">Mean KLD versus size, as claimed by each publisher</title>'
         f'<desc id="kldd">Scatter of {len(pts)} self-reported mean KLD figures against size in GB. Log scale. Methods differ between publishers.</desc>']
    for gx in range(x0, x1 + 1, 20):
        g.append(f'<line x1="{X(gx):.1f}" x2="{X(gx):.1f}" y1="{mt}" y2="{H-mb}" class="grid"/><text x="{X(gx):.1f}" y="{H-mb+16}" class="tick" text-anchor="middle">{gx}</text>')
    d = y0
    while d <= y1 * 1.0001:
        for mlt in (1, 2, 5):
            v = d * mlt
            if y0 <= v <= y1:
                g.append(f'<line x1="{ml}" x2="{W-mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid{"" if mlt == 1 else " minor"}"/>')
                if mlt in (1, 2, 5): g.append(f'<text x="{ml-6}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:g}</text>')
        d *= 10
    g.append(f'<text x="{(ml+W-mr)/2}" y="{H-8}" class="axis" text-anchor="middle">Size, GB (as published; GiB converted)</text>')
    g.append(f'<text transform="translate(14 {(mt+H-mb)/2}) rotate(-90)" class="axis" text-anchor="middle">Mean KLD (nats, log scale)</text>')
    for p in sorted(pts, key=lambda p: (owner(p["measured_by"]), p["size_gb"])):
        c = col[owner(p["measured_by"])]; filled = p["ref_class"] == "bf16"
        t = f'{p["measured_by"]} · {p["variant"]} · {p["size_raw"]} ({BASIS_LABEL.get(p["basis"], p["basis"])}) · KLD {p["metrics"]["mean_kld"]["raw"]} · {REF_TXT[p["ref_class"]]}'
        shape = (f'<circle cx="{X(p["size_gb"]):.1f}" cy="{Y(p["metrics"]["mean_kld"]["v"]):.1f}" r="4.6" '
                 f'style="fill:{c if filled else "var(--bg)"};stroke:{c}" stroke-width="1.8"><title>{E(t)}</title></circle>')
        g.append(shape)
    g.append("</svg>")
    legend = "".join(f'<li><span class="sw" style="background:{col[p]}"></span>{E(p)}</li>' for p in pubs)
    return "\n".join(g), legend


def qid(repo, variant):
    return (repo.split("/")[0] if "/" in repo else repo).lower() + "|" + re.sub(r"[^a-z0-9.]", "", str(variant).lower().replace("(this repo)", "").replace("-m64", "").replace("ad-4.27bpw q4_k_m", "ad-4.27bpw-q4_k_m"))

def build_sets(rows, repos):
    global RAW
    ns = {}; exec(open(f"{ROOT}/sets.py").read(), ns)
    fmt_of = {x["repo"]: x["fmt"] for x in repos}
    out = []
    def pt(r, mmap, xfield):
        m = {}
        for canon_k, src_k in mmap.items():
            mv = r["metrics"].get(src_k)
            if mv and mv.get("v") is not None: m[canon_k] = {"v": mv["v"], "raw": str(mv["raw"])}
        x = r.get(xfield)
        return {"qid": qid(r["repo"], r["variant"]), "repo": r["repo"], "pub": owner(r["repo"]), "variant": str(r["variant"]), "inferred": bool(r.get("inferred")),
                "fmt": fmt_of.get(r["repo"]) or "", "kind": r["kind"], "x": x, "size_gb": r.get("size_gb"), "size_raw": r.get("size_raw"),
                "bpw": r.get("bpw"), "basis": BASIS_LABEL.get(r.get("basis"), r.get("basis")), "m": m,
                "src": r["src"], "src_label": r["src_label"], "measured_by": r["measured_by"]}
    for st in ns["SETS"]:
        pts, seen = [], set()
        for r in rows:
            if r["measured_by"] in st["by"] and r["kind"] in st["kinds"] and not r.get("copied_from"):
                p = pt(r, st["metrics"], st["x"]["field"])
                if p["qid"] in seen or p["x"] is None or not p["m"]: continue
                seen.add(p["qid"]); pts.append(p)
        bp = st.get("baseline_point")
        if bp:
            src_repo = st["by"][0]; R0 = next(x for x in RAW if x["repo"] == src_repo)
            base = dict(R0.get("baseline") or {}); base.update({k: v for k, v in (R0.get("baseline2") or {}).items()})
            m = {}
            for ck, sk in st["metrics"].items():
                if sk in base and num(base[sk]) is not None: m[ck] = {"v": num(base[sk]), "raw": str(base[sk])}
            gb = bp.get("size_gb") or (SIZES.get(bp["repo"], {}).get("total_GB"))
            if bp.get("size_L"):   # a typed baseline size must be on its cited card line
                ln = CARD_TEXT(src_repo).split("\n")[bp["size_L"] - 1]
                if not re.search(rf"\b{re.escape(fmtv(gb, 1) if gb % 1 else str(int(gb)))}\b", ln): raise SystemExit(f"baseline size {gb} not on {src_repo} L{bp['size_L']}: {ln}")
            sha = IDX["repos"][src_repo]["sha"]
            pts.append({"qid": qid(bp["repo"], bp["variant"]), "repo": bp["repo"], "pub": owner(bp["repo"]), "variant": bp["variant"], "fmt": "",
                        "kind": "baseline", "x": gb, "size_raw": bp.get("size_raw") or (f"{gb} GB (HF listing)" if gb else None), "bpw": None, "basis": "download",
                        "m": m, "src": hf_line(src_repo, sha, base.get("L")), "src_label": f"L{base.get('L')} @ {sha[:7]}", "measured_by": src_repo})
        avail = [k for k in st["metrics"] if sum(1 for p in pts if k in p["m"]) >= 2]
        out.append({k: st[k] for k in ("id", "title", "reference", "data", "x", "note")} |
                   {"refs": {k: list(v) for k, v in st["refs"].items()}, "metrics": avail, "points": pts, "comparable": True})
    curated_by = {b for st in ns["SETS"] for b in st["by"]}
    for a in sorted(auto_sets(rows, repos, curated_by), key=lambda x: -len(x["points"])):
        aq = {p["qid"] for p in a["points"]}; aown = a["title"].split(" · ")[0]
        avals = [pv["raw"] for p in a["points"] for pv in p["m"].values()]
        def repeats(o):
            ovals = [pv["raw"] for p in o["points"] for pv in p["m"].values()]
            return sum(1 for x in avals if any(same_value(x, y) and len(re.sub(r"[^0-9]", "", str(x))) >= 3 for y in ovals))
        dup = any(o["title"].split(" · ")[0] == aown and (len(aq & {p["qid"] for p in o["points"]}) >= 0.7 * len(aq) or repeats(o) >= max(2, 0.6 * len(avals))) for o in out)
        if not dup: out.append(a)
    # all self-reported claims (not comparable)
    allp = []
    for r in rows:
        if r["size_gb"] is None or r.get("copied_from"): continue
        if r["kind"] == "quoted": continue
        p = pt(r, {c: k for c in ("kld", "top1") for k in [fam_key(r, c)] if k}, "size_gb")
        if not p["m"]: continue
        p["qid"] = p["qid"] + "@" + owner(r["measured_by"]).lower()
        allp.append(p)
    out.append({"id": "all", "title": "Every claim together (not comparable)", "reference": "Mixed", "data": "Mixed",
                "x": {"field": "size_gb", "label": "Size, GB as published (download, GPU or resident; see each point)", "unit": "GB"},
                "note": "Every KLD and top-1 figure with a size, from every card. Methods, references and size definitions differ, so read this only for orientation.",
                "refs": {}, "metrics": ["kld", "top1"], "points": allp, "comparable": False})
    return out, ns["METRICS"]


def link_sets(sets, min_anchors=3):
    """Test equating between comparable sets: for each (source, base) pair sharing >= min_anchors quants on a metric,
    fit log(y_base) = c + s*log(y_src) by OLS (top-1 uses disagreement 100-top1), report leave-one-out error, and
    map the source's other quants onto the base scale with a band of +/- the worst LOO error."""
    import math
    def tr(k, v): return math.log(100 - v) if k.startswith("top1") else math.log(v)
    def inv(k, z): return 100 - math.exp(z) if k.startswith("top1") else math.exp(z)
    def ols(X, Y):
        n = len(X); mx = sum(X) / n; my = sum(Y) / n; sxx = sum((x - mx) ** 2 for x in X)
        if sxx == 0: return None
        k = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sxx; return my - k * mx, k
    comp = [x for x in sets if x["comparable"]]
    for B in comp:
        B["linked"] = []
        for A in comp:
            if A is B: continue
            for k in set(A["metrics"]) & set(B["metrics"]):
                if k not in ("kld", "top1"): continue
                a = {p["qid"]: p for p in A["points"] if k in p["m"] and (k == "top1" and p["m"][k]["v"] < 100 or k != "top1" and p["m"][k]["v"] > 0)}
                b = {p["qid"]: p for p in B["points"] if k in p["m"] and (k == "top1" and p["m"][k]["v"] < 100 or k != "top1" and p["m"][k]["v"] > 0)}
                if A["title"].split(" · ")[0] == B["title"].split(" · ")[0]: continue      # same publisher: not independent
                anc = sorted(set(a) & set(b))
                ident = sum(1 for q in anc if same_value(a[q]["m"][k]["raw"], b[q]["m"][k]["raw"]))
                anc = [q for q in anc if not same_value(a[q]["m"][k]["raw"], b[q]["m"][k]["raw"])]
                if ident >= len(anc): continue                                             # mostly copied values: not a measurement link
                if len(anc) < min_anchors: continue
                X = [tr(k, a[q]["m"][k]["v"]) for q in anc]; Y = [tr(k, b[q]["m"][k]["v"]) for q in anc]
                f = ols(X, Y)
                if not f: continue
                res = []
                for i in range(len(anc)):
                    fi = ols(X[:i] + X[i+1:], Y[:i] + Y[i+1:])
                    if not fi: res = None; break
                    res.append((fi[0] + fi[1] * X[i]) - Y[i])
                if not res: continue
                worst = max(abs(r) for r in res); rms = (sum(r * r for r in res) / len(res)) ** 0.5
                if math.exp(worst) - 1 > 0.25: continue                                    # validation failed: worst LOO error above 25%
                xf = B["x"]["field"]; pts = []
                for q, p in a.items():
                    if q in b: continue
                    xv = p.get(xf) if xf != "size_gb" else p.get("size_gb")
                    if xv is None: continue
                    z = f[0] + f[1] * tr(k, p["m"][k]["v"])
                    est, lo, hi = inv(k, z), inv(k, z - worst), inv(k, z + worst)
                    pts.append(p | {"x": xv, "est": {"k": k, "v": est, "lo": min(lo, hi), "hi": max(lo, hi), "src_raw": p["m"][k]["raw"]}})
                B["linked"].append({"from": A["id"], "from_title": A["title"], "metric": k, "anchors": [b[q]["variant"] for q in anc],
                                    "fit": {"c": round(f[0], 4), "s": round(f[1], 4)}, "loo_worst_pct": round((math.exp(worst) - 1) * 100, 1),
                                    "loo_rms_pct": round((math.exp(rms) - 1) * 100, 1), "points": pts})
    return sets


CANON = [("kld", re.compile(r"^(mean_kld|kld|kl|kld:english|mean_kl|kl_mean|kld_mean|approx_kld|kld:held.?out|kld:prose)$")),
         ("top1", re.compile(r"^(top1_pct|top1|top1:english|top1_pct_vs_bf16|top1_agree_pct|top1:held.?out|same_top1)$")),
         ("ppl", re.compile(r"^(ppl|ppl:english|ppl:wikitext)$")),
         ("ppl_delta", re.compile(r"^(ppl_delta_pct)$"))]

def fam_key(r, fam):
    """First metric key on a row that belongs to a canonical family (kld, top1, ppl, ppl_delta)."""
    rx = dict(CANON)[fam]
    for k in r["metrics"]:
        if rx.match(k) and r["metrics"][k].get("v") is not None: return k
    return None

def auto_sets(rows, repos, curated_by):
    """One set per measurer (or per owner+identical stated method across sibling repos) with >=2 measured quants."""
    meth = {x["repo"]: x.get("methodology") or {} for x in repos}
    groups = collections.defaultdict(list)
    for r in rows:
        if r["measured_by"] in curated_by or r.get("copied_from"): continue
        if r["kind"] not in ("own", "ladder", "cross"): continue     # quoted rows (kind "quoted") never enter a set
        if r["kind"] == "cross" and r.get("inferred"): continue      # target file not named on the card
        mm = meth.get(r["measured_by"]) or {}
        key = (owner(r["measured_by"]), str(mm.get("reference")), str(mm.get("dataset")))
        groups[key].append(r)
    out = []
    for (own, ref, data), rs in groups.items():
        if ref in ("None", "not stated") and data in ("None", "not stated") and len({r["measured_by"] for r in rs}) > 1:
            # without a stated method we cannot assume sibling repos used the same one: split them
            parts = collections.defaultdict(list)
            for r in rs: parts[(own, r["measured_by"])].append(r)
            subgroups = list(parts.values())
        else:
            subgroups = [rs]
        for g in subgroups:
            mmap = {}
            for canon, rx in CANON:
                for r in g:
                    for k in r["metrics"]:
                        if rx.match(k) and canon not in mmap: mmap[canon] = k
            pts, seen = [], set()
            for r in g:
                m = {}
                for canon, rx in CANON:
                    for k, mv in r["metrics"].items():
                        if rx.match(k) and mv.get("v") is not None and canon not in m: m[canon] = {"v": mv["v"], "raw": str(mv["raw"])}
                xfield = "size_gb"
                if r.get("size_gb") is None: continue
                q = qid(r["repo"], r["variant"])
                if not m or q in seen: continue
                seen.add(q)
                pts.append({"qid": q, "repo": r["repo"], "pub": owner(r["repo"]), "variant": str(r["variant"]), "fmt": "", "kind": r["kind"], "inferred": bool(r.get("inferred")),
                            "x": r["size_gb"], "size_gb": r["size_gb"], "size_raw": r.get("size_raw"), "bpw": r.get("bpw"),
                            "basis": BASIS_LABEL.get(r.get("basis"), r.get("basis")), "m": m, "src": r["src"], "src_label": r["src_label"], "measured_by": r["measured_by"]})
            bc = collections.Counter(p["basis"] for p in pts)
            ownb = collections.Counter(p["basis"] for p in pts if p["kind"] == "own")
            if bc:
                main_basis = (ownb or bc).most_common(1)[0][0]                # the title repo's own basis first
                pts = [p for p in pts if p["basis"] == main_basis]          # one size basis per set
            avail = [c for c, _ in CANON if sum(1 for p in pts if c in p["m"]) >= 2]
            if len(pts) < 2 or not avail: continue
            bys = sorted({p["measured_by"] for p in pts})
            mm = meth.get(bys[0]) or {}
            bases = sorted({p["basis"] for p in pts if p["basis"]})
            out.append({"id": "a-" + re.sub(r"[^a-z0-9]+", "-", (own + "-" + short(bys[0])).lower()).strip("-")[:48], "title": f"{own} · {short(bys[0]) if len(bys) == 1 else 'model cards'}",
                        "reference": str(mm.get("reference") or "Not stated"), "data": str(mm.get("dataset") or "Not stated"),
                        "x": {"field": "size_gb", "label": "Size, GB (" + ", ".join(bases) + ")", "unit": "GB"},
                        "note": "Set assembled automatically from one publisher's card" + ("s" if len(bys) > 1 else "") + ". It includes other publishers' files only where this card's author ran them in the same test; figures the card quotes from elsewhere are left out. Check the source lines.",
                        "refs": {}, "metrics": avail, "points": pts, "comparable": True, "auto": True})
    return out

# ---------- page ----------
def _others():
    """Sibling model pages: every other models/<slug>/model.json, in watch-list order. A hand-set entry in this model's
    "others" only adds its claude.ai preview link ("artifact"); nothing needs editing when a model is added."""
    arts = {o.get("name"): o.get("artifact") for o in CFG.get("others", [])}
    order, status = [], {}
    wl = os.path.join(CODE, "watchlist.json")
    if os.path.exists(wl):
        for m in json.load(open(wl)).get("models", []): order.append(m["slug"]); status[m["slug"]] = m.get("status")
    out = []
    for mj in glob.glob(os.path.join(CODE, "models", "*", "model.json")):
        c = json.load(open(mj)); slug = c.get("slug")
        if not slug or slug == SLUG or status.get(slug, "live") != "live": continue   # link only models that are on the site
        nm = c.get("name") or c.get("model", "").split("/")[-1]
        out.append({"name": nm, "href": f"../{slug}/", "artifact": arts.get(nm), "_k": order.index(slug) if slug in order else 999})
    return [{k: v for k, v in o.items() if k != "_k"} for o in sorted(out, key=lambda o: (o["_k"], o["name"]))]
OTHERS = _others()
def render(rows, repos, stats, verify_counts, highlights, exdata=''):
    OTHER_MODELS = "".join(f' · <a href="{E(o["href"])}">{E(o["name"])}</a>' for o in OTHERS)
    captured = IDX["captured_at"][:10]
    own = [r for r in rows if r["kind"] == "own"]
    def mcell(r, k, pct=False):
        m = r["metrics"].get(k) if k else None
        if not m: return '<td class="n dim">—</td>'
        note = "" if k in ("mean_kld", "top1_pct", "ppl") else f'<small class="dim"> {E(k)}</small>'
        return f'<td class="n" data-v="{m["v"] if m["v"] is not None else ""}" title="{E(str(m["raw"]))} (published as {E(k)}{", " + E(m["unit"]) if m.get("unit") else ""})">{E(str(m["raw"]))}{note}</td>'
    MAIN = {"mean_kld", "top1_pct", "ppl", "ppl_delta_pct", "ppl_ratio"}
    def bench(r):
        shown_k = {fam_key(r, "kld"), fam_key(r, "top1"), fam_key(r, "ppl")}
        items = [(k.replace("bench:", ""), v["raw"]) for k, v in r["metrics"].items() if k not in MAIN and k not in shown_k]
        return '<td class="bench">' + ("; ".join(f"{E(a)} {E(str(b))}" for a, b in items) or '<span class="dim">—</span>') + "</td>"
    tr = []
    for r in sorted(rows, key=lambda r: (r["kind"] != "own", owner(r["repo"]).lower(), r["repo"], -(r["size_gb"] or 0))):
        repo_meta = next((x for x in repos if x["repo"] == r["measured_by"]), {})
        fmt = r["fmt"] or next((x["fmt"] for x in repos if x["repo"] == r["repo"]), "") or ""
        who = (f'<span class="chip x">measured by {E(owner(r["measured_by"]))}</span>' if r["kind"] == "cross" else
               f'<span class="chip q" title="The card quotes this figure from elsewhere; {E(owner(r["measured_by"]))} did not measure it">quoted by {E(owner(r["measured_by"]))}, not measured</span>' if r["kind"] == "quoted" else "")
        if r.get("inferred"): who += '<span class="chip q" title="The card names the file but not the repository; attributed from the file name">repo not named on card</span>'
        if r.get("flags"): who += "".join(f'<small class="eflag">{E(f)}</small>' for f in r["flags"] if not re.search(r"^quoted, not measured$", f))
        lad = '<span class="chip l">publisher comparison build</span>' if r["kind"] == "ladder" else ""
        mod = '<span class="chip w">modified weights</span>' if r.get("weights_modified") else ""
        has_q = any(k for k in r["metrics"])
        pubcell = (f'<td class="pub"><a href="https://huggingface.co/{E(r["repo"])}">{E(owner(r["repo"]))}<span class="slash">/</span>{E(short(r["repo"]))}</a> {who}{lad}{mod}</td>' if "/" in r["repo"] else f'<td class="pub"><span class="unl">{E(r["repo"])}</span> {who}{lad}{mod}</td>')
        tr.append(
            f'<tr data-kind="{r["kind"]}" data-fmt="{E(fmt)}" data-pub="{E(owner(r["repo"]))}" data-ref="{r["ref_class"]}" data-q="{1 if has_q else 0}" data-mod="{1 if r.get("weights_modified") else 0}">'
                + pubcell +
            f'<td class="var">{E(str(r["variant"]))}</td><td>{E(fmt)}</td>'
            f'<td class="n" data-v="{r["size_gb"] or ""}">{fmtv(r["size_gb"],1) if r["size_gb"] else "—"}</td>'
            f'<td class="basis">{E(BASIS_LABEL.get(r["basis"], str(r["basis"])) if r["size_gb"] else "")}</td>'
            f'{mcell(r, fam_key(r, "kld"))}{mcell(r, fam_key(r, "top1"))}{mcell(r, fam_key(r, "ppl"))}'
            f'<td class="n">{E(str((r["metrics"].get("ppl_delta_pct") or r["metrics"].get("ppl_ratio") or {}).get("raw","—")))}</td>'
            f'{bench(r)}<td><span class="ref r-{r["ref_class"]}">{E(REF_TXT[r["ref_class"]])}</span></td>'
            f'<td class="src"><a href="{E(r["src"])}">{E(r["src_label"])}</a></td></tr>')
    chart, legend = kld_chart(rows)
    fmts = sorted({x["fmt"] for x in repos if x["fmt"]})
    cards = []
    noclaim = [x for x in repos if x.get("auto") or not any(r["metrics"] for r in rows if r["measured_by"] == x["repo"])]
    noclaim_ids = {x["repo"] for x in noclaim}
    nc_rows = "".join(f'<tr><td><a href="https://huggingface.co/{E(x["repo"])}">{E(x["repo"])}</a></td><td>{"not yet reviewed" if x.get("auto") else "read in full: no quality claims"}</td><td>{E(x["fmt"] or "")}</td>'
                      f'<td>{"modified weights" if x.get("weights_modified") else ""}</td><td>{E((x.get("flags") or [""])[0])}</td>'
                      f'<td class="src"><a href="https://huggingface.co/{E(x["repo"])}/blob/{x["sha"]}/README.md">{x["sha"][:7]}</a></td></tr>'
                      for x in sorted(noclaim, key=lambda x: x["repo"].lower()))
    for x in sorted(repos, key=lambda x: x["repo"].lower()):
        if x["repo"] in noclaim_ids: continue
        mth = x.get("methodology") or {}
        md = "".join(f"<dt>{E(a)}</dt><dd>{E(str(mth.get(b)))}</dd>" for a, b in
                     [("Reference", "reference"), ("Eval data", "dataset"), ("Context", "ctx"), ("Tool", "tool"), ("Unit", "unit_note")] if mth.get(b))
        fl = "".join(f"<li>{E(f)}</li>" for f in (x.get("flags") or []))
        cards.append(f'<article class="card" data-f="{1 if x.get("flags") else 0}" id="{E(x["repo"].replace("/","--"))}"><header><h3><a href="https://huggingface.co/{E(x["repo"])}">{E(x["repo"])}</a></h3>'
                     f'<p class="meta"><span class="chip">{E(x["fmt"] or "")}</span> <span class="chip">{E(x["runtime"] or "")}</span>'
                     f'{" <span class=\"chip w\">modified weights</span>" if x.get("weights_modified") else ""} '
                     f'<span class="ref r-{x["ref_class"]}">{E(REF_TXT[x["ref_class"]])}</span></p></header>'
                     f'<p class="method">{E(x.get("method") or "")}</p>'
                     + (f'<dl>{md}</dl>' if md else '<p class="dim">No evaluation method stated.</p>')
                     + (f'<ul class="flags">{fl}</ul>' if fl else "")
                     + f'<p class="snap">Card snapshot <a href="https://huggingface.co/{E(x["repo"])}/blob/{x["sha"]}/README.md"><code>{x["sha"][:12]}</code></a> · updated {x["lastModified"][:10]}</p></article>')
    hl = "".join(highlights)
    s = stats
    body = f"""
<header class="top"><div class="wrap">
  <p class="eyebrow"><a class="home" href="../" title="All models">QFF</a> · {E(MODEL)}{OTHER_MODELS}</p>
  <h1>Every Hugging Face quant of {E(MNAME)}, compared by what its publisher claims</h1>
  <p class="lede">All {s["repos"]} repositories Hugging Face tags as quantizations of this model on {captured}. {s["with_quality"]} publish quality numbers about their own files; {verify_counts["values"]} published values are copied from the card text at a pinned commit and checked against it mechanically. Nothing here was re-measured by us.</p>
  <dl class="stats">
    <div><dt>Repos</dt><dd>{s["repos"]}</dd></div>
    <div><dt>Publish a quality number</dt><dd>{s["with_quality"]}</dd></div>
    <div><dt>Of those, against BF16</dt><dd>{s["bf16_ref"]}</dd></div>
    <div><dt>Read, no quality number</dt><dd>{s["no_quality"]}</dd></div>
    <div><dt>Not yet reviewed</dt><dd>{s["unreviewed"]}</dd></div>
    <div><dt>Measurements of rivals' files</dt><dd>{s["cross"]}</dd></div>
    <div><dt>Cards captured</dt><dd>{captured}</dd></div>
  </dl>
</div></header>
<nav class="toc" aria-label="Sections"><div class="wrap"><a href="#read-first">Read first</a><a href="#chart">Compare</a><a href="#head-to-head">Head-to-head</a><a href="#ledger">Ledger</a><a href="#publishers">Publisher notes</a><a href="#no-claims">No claims</a><a href="#method">Method</a><a class="ext" href="{REPO_URL}" rel="noopener" data-umami-event="github">GitHub ↗</a></div></nav>
<main class="wrap">
<section id="read-first" class="rules">
  <h2>Read this first</h2>
  <ol>
    <li><strong>These are claims, not our measurements.</strong> Each publisher ran its own test. A lower KLD on one card is not evidence of a better file than a higher KLD on another.</li>
    <li><strong>Methods rarely match.</strong> Publishers use different reference models (BF16, their own Q8_0, someone else's Q8_0), different text, context lengths and tools. Each row shows its reference; the cards below give the rest.</li>
    <li><strong>Sizes mean different things.</strong> {E(SIZE_NOTE)}</li>
  </ol>
</section>
<section id="chart"><h2>Compare quants inside one measurement</h2>
<p class="note">Every point in a set was measured the same way by the same party, so points in one set can be compared with each other; points from different sets cannot. Your ticks carry across sets, so you can follow a quant through every set that measured it. Quants that publish no comparable numbers are listed under <a href="#no-claims">Repos without extracted claims</a>.</p>
<div class="explorer" id="explorer">
  <div class="ex-top">
    <div class="ex-field"><label for="exset">Measurement set</label><select id="exset"></select></div>
    <div class="ex-field ex-find"><label for="exfind">Find a quant in any set</label><input id="exfind" type="search" placeholder="e.g. UD-Q4_K_XL, nvfp4, bartowski   ( / )" autocomplete="off" aria-controls="exfindres"><div id="exfindres" class="ex-findres" hidden aria-live="polite"></div></div>
  </div>
  <div class="ex-bar">
    <div class="ex-metrics" role="radiogroup" aria-label="Metric" id="exmetric"></div>
    <div class="ex-toggles"><label class="cb" for="exlabels"><input id="exlabels" type="checkbox" checked> Labels</label><label class="cb" for="exfront"><input id="exfront" type="checkbox"> Efficient frontier</label><span class="seg" role="group" aria-label="Y axis scale"><button type="button" id="exlog" aria-pressed="true">Log</button><button type="button" id="exlin" aria-pressed="false">Linear</button></span><button type="button" id="exlink" class="tiny">Copy link to view</button></div>
  </div>
  <div class="ex-range" id="exrange" role="group" aria-label="Size and quality limits">
    <label class="rg" for="exxr"><span>Max size</span><input id="exxr" type="range" min="0" max="1000" step="1" value="1000"><output id="exxo" for="exxr">any</output></label>
    <label class="rg" for="exyr"><span id="exylab">Max metric</span><input id="exyr" type="range" min="0" max="1000" step="1" value="1000"><output id="exyo" for="exyr">any</output></label>
    <span class="presets" id="expresets" aria-label="Size budget presets"></span>
    <label class="rg bs" for="exbasis" id="exbasiswrap" hidden><span>Size type</span><select id="exbasis"></select></label>
    <label class="cb" for="exonly"><input id="exonly" type="checkbox"> Only efficient quants</label>
    <button type="button" id="exrclear" class="tiny" hidden>Clear limits</button>
  </div>
  <p class="ex-meta" id="exmeta"></p>
  <div class="ex-body">
    <div class="ex-chart"><div id="exsvg" class="chartbox"><noscript>The interactive chart needs JavaScript. All numbers are in the ledger below.</noscript></div><div id="extip" class="ex-tip" role="tooltip" hidden></div><p class="ex-chartnote dim" id="exchartnote"></p><div class="legend" id="exlegend" aria-label="Show or hide a repository"></div></div>
    <aside class="ex-picker" aria-label="Choose quants">
      <div class="ex-pick-head"><b>Quants</b><span id="excount" class="count" aria-live="polite"></span><button type="button" id="exexpand" class="tiny" aria-pressed="false">Expand list</button></div>
      <input id="exq" type="search" placeholder="Filter by name, publisher, format…" aria-label="Filter quants in this set">
      <div id="exfmts" class="ex-fmts" aria-label="Formats"></div>
      <div class="ex-pick-actions"><button type="button" id="exall">Tick matches</button><button type="button" id="exnone">Untick matches</button><button type="button" id="exreset" hidden>Reset</button></div>
      <div id="expick" class="ex-picklist"></div>
    </aside>
  </div>
  <div class="ex-tablehead"><span id="extcap" class="dim"></span><button type="button" id="excopy" class="tiny">Copy as CSV</button></div>
  <div class="tablebox ex-tablebox"><table class="disagree" id="extable"></table></div>
</div>
<script type="application/json" id="exdata">__EXDATA__</script>
</section>
<section id="head-to-head"><h2>When one publisher measured several</h2>
<p class="note">These are the only tables where different publishers' files went through the same test. They are still one party's measurement, often of competitors.</p>
{hl}
</section>

<section id="ledger"><h2>The ledger</h2>
<div class="filters" role="group" aria-label="Filter the ledger">
  <label for="q">Search</label><input id="q" type="search" placeholder="publisher, quant type, metric…">
  <label for="ffmt">Format</label><select id="ffmt"><option value="">All</option>{"".join(f'<option>{E(f)}</option>' for f in fmts)}</select>
  <label for="fref">Reference</label><select id="fref"><option value="">Any</option><option value="bf16">vs BF16</option><option value="q8">vs a Q8_0</option><option value="unclear">unclear</option><option value="other">other or no reference</option><option value="unstated">not stated</option><option value="none">no quality claim</option></select>
  <label class="cb" for="fq"><input id="fq" type="checkbox"> Only rows with a quality number</label>
  <label class="cb" for="fx"><input id="fx" type="checkbox" checked> Include rival and quoted rows</label>
  <label class="cb" for="fm"><input id="fm" type="checkbox" checked> Include modified weights</label>
  <button type="button" id="freset" class="tiny">Reset</button><span id="count" class="count" aria-live="polite"></span>
</div>
<div class="tablebox"><table id="t">
<thead><tr><th data-sort="s">Repository</th><th data-sort="s">Variant</th><th data-sort="s">Format</th><th class="n" data-sort="n">Size GB</th><th>Size is</th><th class="n" data-sort="n">Mean KLD</th><th class="n" data-sort="n">Top-1</th><th class="n" data-sort="n">PPL</th><th class="n" title="Change in perplexity against the baseline the row's card states (BF16, a sibling build, or a previous build); check the source line">PPL change<br><small>vs the card's baseline</small></th><th>Other metrics (as published)</th><th>Reference</th><th>Source line</th></tr></thead>
<tbody>{"".join(tr)}</tbody></table></div>
</section>
<section id="publishers"><h2>Publisher notes</h2>
<p class="note">Method, evaluation setup and problems we found, per repository. Line numbers refer to the card snapshot linked at the bottom of each note.</p>
<div class="filters"><label for="cq">Search</label><input id="cq" type="search" placeholder="publisher, format, flag text…"><label class="cb" for="cflag"><input id="cflag" type="checkbox"> Only notes with flags</label><span id="ccount" class="count" aria-live="polite"></span></div>
<div class="cards" id="cards">{"".join(cards)}</div>
</section>
<section id="no-claims"><h2>Repos without extracted claims ({len(noclaim)})</h2>
<p class="note">Two different situations, marked in the Status column. <b>Read in full: no quality claims</b> means a reviewer read the whole card and found no quality number about the quant itself. <b>Not yet reviewed</b> means the card has not been read in full yet and may contain claims; it is listed so the catalogue is complete, not as a finding.</p>
<div class="filters"><label for="ncq">Search</label><input id="ncq" type="search" placeholder="publisher or format…"><label for="ncs">Status</label><select id="ncs"><option value="">Any</option><option>not yet reviewed</option><option>read in full: no quality claims</option></select><span id="nccount" class="count" aria-live="polite"></span></div>
<div class="tablebox"><table class="disagree" id="nctable"><thead><tr><th>Repository</th><th>Status</th><th>Format</th><th>Weights</th><th>Note</th><th>Card</th></tr></thead><tbody>{nc_rows}</tbody></table></div>
</section>
<section id="method" class="method-sec"><h2>How this was built</h2>
<p>We listed every repository Hugging Face tags as a quantization of <a href="https://huggingface.co/{E(MODEL)}">{E(MODEL)}</a> ({s["repos"]} on {captured}) and saved each raw <code>README.md</code> at its pinned commit into a local ledger, so nothing is re-fetched. Cards that reproduce another publisher's numbers are detected by comparing claim lines across all cards and are not counted as independent measurements. Results published only as chart images were transcribed from their labels and checked against the marker positions in the image. Numbers were extracted with the line they came from, then a script confirmed that {verify_counts["values"]} published values appear verbatim on their cited lines ({verify_counts["fails"]} failures). For every card that carried measured claims at the first pass (196 of {s["repos"]}), a second reader re-extracted the numbers without seeing our data, and disagreements were resolved against the source text; the remaining cards were each read in full by one reader. Rows a card quotes from elsewhere are marked "quoted" and are not counted as measurements; rows measured on a different model are left out ({s["other_model"]}). File sizes without a card figure come from Hugging Face's file listing at the same commit.</p>
{('<p>' + E(PIN_NOTE) + '</p>') if PIN_NOTE else ''}
<p>Not covered: {E(COVER_NOTE)}, and repos created or changed after {captured}. Found an error or a new quant? <a href="{REPO_URL}">Open an issue or pull request</a>; each claim lives in a plain data file with its source line.</p>
</section>
</main>
<footer class="foot"><div class="wrap"><p>QFF (Quant Fidelity Frontier): what publishers claim, pinned to the line they said it · data captured {captured} · claims belong to their publishers · <a href="{REPO_URL}">source and data</a>{SUPPORT_HTML}</p></div></footer>
"""
    return body.replace("__EXDATA__", exdata)

CSS = open(os.path.join(CODE, "web", "site_style.css")).read() if os.path.exists(os.path.join(CODE, "web", "site_style.css")) else ""
JS = "\n".join(open(os.path.join(CODE, "web", f)).read() for f in ("site_script.js", "site_explorer.js"))

def main():
    global RAW
    import subprocess
    DEV = os.environ.get("QCL_DEV") == "1"     # dev preview: gates skipped, output goes to site_dev/ only (never site/)
    OUT = f"{ROOT}/site_dev" if DEV else f"{ROOT}/site"
    GATE_OUT = {}
    if not DEV:
        import gatecache
        GATE_OUT = gatecache.run_all()          # parallel; reused only when every input is byte-identical
        for gate, (rc, out, hit) in GATE_OUT.items():
            if rc != 0: raise SystemExit(f"REFUSING TO BUILD: {gate} failed (exit {rc})\n" + out[-3000:])
    R = load(); RAW = R; rows, repos = build_rows(R)
    # sign agreement: a published value written with a leading minus must parse negative, and vice versa
    sign_bad = []
    for r in rows:
        for k, m in r["metrics"].items():
            raw = re.sub(r"\*\*|`", "", str(m["raw"])).strip()
            if m.get("v") is None or not raw: continue
            neg = bool(re.match(r"^[-\u2212\u2013]\s*\.?\d", raw))
            if neg != (m["v"] < 0): sign_bad.append((r["repo"], r["variant"], k, raw, m["v"]))
    if sign_bad: raise SystemExit("sign mismatch between raw and parsed values:\n" + "\n".join(map(str, sign_bad)))
    bad = []
    for r in rows:
        for k, m in r["metrics"].items():
            v = m.get("v")
            if v is None: continue
            if (k in ("mean_kld",) or k.startswith("kld") or k.startswith("kl:")) and "99" not in k and not re.search(r"delta|change|diff|gap|ratio|max|p99|p999", k) and not (0 <= v < 5): bad.append((r["repo"], r["variant"], k, m["raw"], v))
            if k.startswith("top1") and not re.search(r"delta|change|diff|gap", k) and not (0 <= v <= 100): bad.append((r["repo"], r["variant"], k, m["raw"], v))
        small_ok = bool(re.search(r"control vector|lora|adapter|mtp|draft|sidecar|n-?gram|ple table|direction", (str(r.get("fmt") or "") + " " + str(r["variant"])), re.I))
        if r["size_gb"] is not None and not ((0 < r["size_gb"] if small_ok else 1 <= r["size_gb"]) and r["size_gb"] <= 400): bad.append((r["repo"], r["variant"], "size_gb", r["size_raw"], r["size_gb"]))
    if bad:
        raise SystemExit("implausible parsed values:\n" + "\n".join(map(str, bad)))
    import subprocess
    vout = GATE_OUT["verify.py"][1] if "verify.py" in GATE_OUT else subprocess.run(["python3", f"{CODE}/gates/verify.py"], capture_output=True, text=True).stdout
    vc = {"values": int(re.search(r"verified verbatim: (\d+)", vout).group(1)), "fails": int(re.search(r"failures: (\d+)", vout).group(1))}
    if not DEV: os.makedirs(f"{ROOT}/audit", exist_ok=True); json.dump(vc, open(f"{ROOT}/audit/verify_counts.json", "w"))
    hl = []
    if os.path.exists(f"{ROOT}/highlights.py"):
        ns = {"rows": rows, "repos": repos, "E": E, "fmtv": fmtv, "BASIS_LABEL": BASIS_LABEL}; exec(open(f"{ROOT}/highlights.py").read(), ns); hl = ns["render_highlights"]()
    has_q = {x["repo"] for x in repos if any(r["metrics"] for r in rows if r["measured_by"] == x["repo"] and r["kind"] == "own" and not r.get("copied_from"))}
    stats = {"repos": len(repos), "variants": sum(1 for r in rows if r["kind"] == "own"),
             "values": sum(len(r["metrics"]) + (1 if r["size_gb"] else 0) for r in rows),
             "with_quality": len(has_q),
             "bf16_ref": sum(1 for x in repos if x["repo"] in has_q and x["ref_class"] == "bf16"),
             "unreviewed": sum(1 for x in repos if x.get("auto")),
             "no_quality": len(repos) - len(has_q) - sum(1 for x in repos if x.get("auto")),
             "cross": sum(1 for r in rows if r["kind"] == "cross" and not r.get("copied_from")),
             "quoted": sum(1 for r in rows if r["kind"] == "quoted"), "other_model": len(OTHER_MODEL)}
    sets, metrics = build_sets(rows, repos)
    # linked estimates disabled pending provenance/transfer validation (external review F06/F07)
    for s_ in sets: s_['linked'] = []
    # unique set ids (explorer state, deep links) and a format on every point (picker filters)
    fmt_of = {x["repo"]: x["fmt"] for x in repos}; seen_ids = collections.Counter()
    for s_ in sets:
        seen_ids[s_["id"]] += 1
        if seen_ids[s_["id"]] > 1: s_["id"] = f'{s_["id"]}-{seen_ids[s_["id"]]}'
        for p_ in s_["points"]:
            if not p_.get("fmt"): p_["fmt"] = fmt_of.get(p_["repo"]) or ""
            # the yardstick of a measurement is not a candidate for "efficient"
            p_["is_ref"] = p_["kind"] == "baseline" or bool(re.search(r"\breference\b|\bbaseline\b|\bteacher\b|unquanti[sz]ed|^\s*(bf16|fp16|fp32|base)\s*(\(.*\))?\s*$", p_["variant"], re.I))
    assert len({s_["id"] for s_ in sets}) == len(sets), "duplicate set ids"
    exdata = json.dumps({'sets': sets, 'metrics': metrics}, separators=(',', ':')).replace('</', '<\\/')
    body = render(rows, repos, stats, vc, hl, exdata)
    os.makedirs(f"{OUT}", exist_ok=True)
    json.dump({"model": MODEL, "captured_at": IDX["captured_at"], "repos": repos, "rows": rows, "sets": sets},
              open(f"{OUT}/data.json", "w"), separators=(",", ":"))
    title = f"{MNAME} QFF"
    desc = (f"Every Hugging Face quant of {MNAME} (GGUF, NVFP4, FP8, MXFP4, AWQ, AutoRound, EXL3, MLX) compared by each publisher's own "
            f"KLD, top-1, perplexity and benchmark claims, with methodology, red flags and source lines. {stats['repos']} repos.")
    head_common = f'<title>{title}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
    fonts = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">'
    abody = body.replace('<a class="home" href="../" title="All models">QFF</a>', '<abbr title="Quant Fidelity Frontier">QFF</abbr>')
    for o in OTHERS:                                   # inside claude.ai the sibling site is another artifact, not a relative path
        if o.get("artifact"): abody = abody.replace(f'<a href="{E(o["href"])}">{E(o["name"])}</a>', f'<a href="{E(o["artifact"])}" target="_blank" rel="noopener">{E(o["name"])}</a>')
    open(f"{OUT}/artifact.html", "w").write(head_common + fonts + f"<style>{CSS}</style>\n" + abody + f"<script>{JS}</script>")
    ld = {"@context": "https://schema.org", "@type": "Dataset", "name": f"QFF (Quant Fidelity Frontier): {MNAME} quantizations",
          "description": desc, "url": SITE_URL, "license": "https://creativecommons.org/licenses/by/4.0/",
          "creator": {"@type": "Person", "name": "delirious6423", "url": "https://github.com/delirious6423"},
          "dateModified": IDX["captured_at"][:10], "isBasedOn": f"https://huggingface.co/{MODEL}",
          "keywords": [MNAME, "quantization", "GGUF", "KLD", "NVFP4", "llama.cpp", "unsloth", "bartowski", "Hugging Face"],
          "distribution": [{"@type": "DataDownload", "encodingFormat": "application/json", "contentUrl": SITE_URL + "data.json"}]}
    full = (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'<title>{MNAME} quant comparison · QFF</title>\n<meta name="description" content="{E(desc)}">\n'
            f'<link rel="canonical" href="{SITE_URL}">\n<meta property="og:title" content="{MNAME} quant comparison">\n<meta property="og:description" content="{E(desc)}">\n'
            f'<meta property="og:type" content="website"><meta property="og:url" content="{SITE_URL}">\n'
            f'<script type="application/ld+json">{json.dumps(ld)}</script>\n{ANALYTICS}{fonts}\n<style>:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}body{{margin:0;font-size:14px}}{CSS}</style></head>\n<body>{body}<script>{JS}</script></body></html>')
    open(f"{OUT}/index.html", "w").write(full)
    open(f"{OUT}/sitemap.xml", "w").write(f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{SITE_URL}</loc><lastmod>{IDX["captured_at"][:10]}</lastmod></url></urlset>\n')
    open(f"{OUT}/robots.txt", "w").write(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}sitemap.xml\n")
    print(json.dumps(stats), "| rows", len(rows))

if __name__ == "__main__":
    main()
