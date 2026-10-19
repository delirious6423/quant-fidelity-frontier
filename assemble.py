#!/usr/bin/env python3
"""Assemble the static GitHub Pages site from every model's build output.

  python3 assemble.py [OUT_DIR]        # default _site/

Output is plain HTML + CSS + JS + JSON (data is also inlined in each index.html), so GitHub Pages only serves files:
no server code runs there. Python runs only at build time (locally or in GitHub Actions)."""
import glob, json, os, shutil, sys
CODE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CODE, "_site")
SKIP = {"artifact.html"}                       # claude.ai preview only, not part of the public site
def copy(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f in SKIP: continue
        p = os.path.join(src, f)
        if os.path.isfile(p): shutil.copy2(p, os.path.join(dst, f))
shutil.rmtree(OUT, ignore_errors=True)
WL = os.path.join(CODE, "watchlist.json")
STATUS = {m["slug"]: m.get("status") for m in json.load(open(WL))["models"]} if os.path.exists(WL) else {}
ORDER = list(STATUS)
for mj in sorted(glob.glob(os.path.join(CODE, "models", "*", "model.json")), key=lambda p: (ORDER.index(os.path.basename(os.path.dirname(p))) if os.path.basename(os.path.dirname(p)) in ORDER else 999, p)):
    slug = json.load(open(mj)).get("slug"); site = os.path.join(os.path.dirname(mj), "site")
    if STATUS.get(slug, "live") != "live": continue          # a model mid-ingest stays off the public site until it goes live
    if slug and os.path.isdir(site): copy(site, os.path.join(OUT, slug))
import re                                      # consolidate model sitemaps; robots rules require the hostname root
locs = []
for sm in sorted(glob.glob(os.path.join(OUT, "**", "sitemap.xml"), recursive=True)):
    locs += re.findall(r"<url>.*?</url>", open(sm).read())
    if os.path.dirname(sm) != OUT: os.remove(sm)
for rb in glob.glob(os.path.join(OUT, "*", "robots.txt")): os.remove(rb)
BASE = "https://delirious6423.github.io/quant-fidelity-frontier/"
open(os.path.join(OUT, "robots.txt"), "w").write(f"User-agent: *\nAllow: /\nSitemap: {BASE}sitemap.xml\n")
locs.insert(0, f"<url><loc>{BASE}</loc></url>")
open(os.path.join(OUT, "sitemap.xml"), "w").write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + "".join(locs) + "</urlset>")
man = []                                       # machine-readable index for agents and the MCP server
_wl = os.path.join(CODE, "watchlist.json")        # when the hourly scan last checked the Hub for each model
_checked = {m["slug"]: m.get("last_scan_at") for m in (json.load(open(_wl))["models"] if os.path.exists(_wl) else [])}
for dj in sorted(glob.glob(os.path.join(OUT, "**", "data.json"), recursive=True), key=lambda p: (ORDER.index(os.path.basename(os.path.dirname(p))) if os.path.basename(os.path.dirname(p)) in ORDER else 999, p)):
    d = json.load(open(dj)); rel = os.path.relpath(os.path.dirname(dj), OUT)
    man.append({"model": d["model"], "slug": rel, "data": rel + "/data.json",
                "captured_at": d.get("captured_at"), "checked_at": _checked.get(rel), "repos": len(d.get("repos", [])), "rows": len(d.get("rows", [])), "sets": len(d.get("sets", []))})
json.dump({"name": "QFF (Quant Fidelity Frontier)", "about": "Publisher claims about Hugging Face quants, each pinned to its source line. Claims, not measurements.", "models": man},
          open(os.path.join(OUT, "models.json"), "w"), indent=1)
import html as H                              # landing page: one card per model, built from the manifest
_S = json.load(open(os.path.join(CODE, "site.json"))) if os.path.exists(os.path.join(CODE, "site.json")) else {}
_A = _S.get("analytics") or {}
AN = (f'<script defer src="{H.escape(_A.get("script") or "https://cloud.umami.is/script.js")}" data-website-id="{H.escape(_A["website_id"])}" data-domains="delirious6423.github.io"></script>' if _A.get("website_id") else "")
SUP = (f' · <a href="{H.escape(_S["support_url"])}" rel="noopener" data-umami-event="support">{H.escape(_S.get("support_label") or "Support QFF")}</a>'
       f' (<a href="https://github.com/delirious6423/quant-fidelity-frontier/blob/main/docs/SPONSORS.md">sponsor policy</a>)') if _S.get("support_url") else ""
from homepage import build_homepage
from benchmark_view import build_benchmark_view
# Its chart catalog is not a model; generate it after the model manifest above.
build_benchmark_view(OUT, man, BASE, AN, SUP)
build_homepage(OUT, man, BASE, AN, SUP)
from discovery import build_discovery
build_discovery(OUT, BASE, man, AN)
open(os.path.join(OUT, ".nojekyll"), "w").close()   # serve files as-is; no Jekyll processing
bad = [p for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p) and not p.endswith((".html", ".json", ".txt", ".xml", ".css", ".js", ".svg", ".png", ".ico", ".nojekyll"))]
assert not bad, f"non-static files in site: {bad[:5]}"
big = [(p, os.path.getsize(p)) for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p) and os.path.getsize(p) > 50e6]
assert not big, f"files over 50 MB (GitHub warns at 50, rejects at 100): {big}"
tot = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p))
assert tot < 1_000_000_000, "published site exceeds the GitHub Pages 1 GB limit"
print(json.dumps({"out": OUT, "files": sum(1 for p in glob.glob(os.path.join(OUT, '**', '*'), recursive=True) if os.path.isfile(p)), "MB": round(tot / 1e6, 1)}))
