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
import re                                      # one sitemap and one robots.txt, at the site root (crawlers ignore nested ones)
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
for dj in sorted(glob.glob(os.path.join(OUT, "**", "data.json"), recursive=True), key=lambda p: (ORDER.index(os.path.basename(os.path.dirname(p))) if os.path.basename(os.path.dirname(p)) in ORDER else 999, p)):
    d = json.load(open(dj)); rel = os.path.relpath(os.path.dirname(dj), OUT)
    man.append({"model": d["model"], "slug": rel, "data": rel + "/data.json",
                "captured_at": d.get("captured_at"), "repos": len(d.get("repos", [])), "rows": len(d.get("rows", [])), "sets": len(d.get("sets", []))})
json.dump({"name": "QFF (Quant Fidelity Frontier)", "about": "Publisher claims about Hugging Face quants, each pinned to its source line. Claims, not measurements.", "models": man},
          open(os.path.join(OUT, "models.json"), "w"), indent=1)
import html as H                              # landing page: one card per model, built from the manifest
_S = json.load(open(os.path.join(CODE, "site.json"))) if os.path.exists(os.path.join(CODE, "site.json")) else {}
SUP = (f' · <a href="{H.escape(_S["support_url"])}" rel="noopener">{H.escape(_S.get("support_label") or "Support QFF")}</a>'
       f' (<a href="https://github.com/delirious6423/quant-fidelity-frontier/blob/main/docs/SPONSORS.md">sponsor policy</a>)') if _S.get("support_url") else ""
cards = "".join(f'<a class="card" href="{H.escape(m["slug"])}/"><h2>{H.escape(m["model"].split("/")[-1])}</h2><p class="org">{H.escape(m["model"])}</p>'
                f'<p>{m["repos"]:,} quant repos · {m["rows"]:,} claim rows · {m["sets"]} measurement sets</p><p class="dim">Captured {H.escape((m["captured_at"] or "")[:10])}</p></a>' for m in man)
open(os.path.join(OUT, "index.html"), "w").write(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QFF · Quant Fidelity Frontier</title><meta name="description" content="Every Hugging Face quant of a model, compared by what its publisher claims, each claim pinned to its model-card line.">
<link rel="canonical" href="{BASE}"><style>
:root{{--bg:#f7f7f4;--fg:#1d2124;--dim:#5d666d;--card:#fff;--line:#d9dcd6;--acc:#0e7c86}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111416;--fg:#e8eaeb;--dim:#9aa3a9;--card:#181c1f;--line:#2b3135;--acc:#5cc3cc;color-scheme:dark}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:880px;margin:0 auto;padding:48px max(16px,env(safe-area-inset-left)) 64px max(16px,env(safe-area-inset-right))}}
h1{{font-size:clamp(28px,5vw,40px);line-height:1.15;margin:0 0 8px;text-wrap:balance}} .lede{{color:var(--dim);max-width:62ch;margin:0 0 32px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}}
.card{{display:block;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;color:inherit;text-decoration:none}}
.card:hover,.card:focus-visible{{border-color:var(--acc)}} .card h2{{margin:0 0 2px;font-size:20px;color:var(--acc)}} .card p{{margin:4px 0}}
.org,.dim{{color:var(--dim);font-size:14px}} footer{{margin-top:40px;color:var(--dim);font-size:14px}} footer a{{color:var(--acc)}} .top{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}} .gh{{color:var(--fg);border:1px solid var(--line);border-radius:999px;padding:2px 12px;text-decoration:none}}
</style></head><body><main><p class="dim top">QFF · Quant Fidelity Frontier <a class="gh" href="https://github.com/delirious6423/quant-fidelity-frontier">GitHub ↗</a></p>
<h1>Every Hugging Face quant, compared by what its publisher claims</h1>
<p class="lede">Each number is copied from the publisher's own model card and links to the line it came from. Nothing is re-measured; numbers are comparable only within one measurement set.</p>
<div class="grid">{cards}</div>
<footer><a href="models.json">models.json</a> (machine-readable index) · <a href="https://github.com/delirious6423/quant-fidelity-frontier">source and data</a>{SUP}</footer></main></body></html>""")
open(os.path.join(OUT, ".nojekyll"), "w").close()   # serve files as-is; no Jekyll processing
bad = [p for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p) and not p.endswith((".html", ".json", ".txt", ".xml", ".css", ".js", ".svg", ".png", ".ico", ".nojekyll"))]
assert not bad, f"non-static files in site: {bad[:5]}"
big = [(p, os.path.getsize(p)) for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p) and os.path.getsize(p) > 50e6]
assert not big, f"files over 50 MB (GitHub warns at 50, rejects at 100): {big}"
tot = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p))
print(json.dumps({"out": OUT, "files": sum(1 for p in glob.glob(os.path.join(OUT, '**', '*'), recursive=True) if os.path.isfile(p)), "MB": round(tot / 1e6, 1)}))
