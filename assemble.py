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
copy(os.path.join(CODE, "site"), OUT)          # the root model (model.json at repo root)
for mj in sorted(glob.glob(os.path.join(CODE, "models", "*", "model.json"))):
    slug = json.load(open(mj)).get("slug"); site = os.path.join(os.path.dirname(mj), "site")
    if slug and os.path.isdir(site): copy(site, os.path.join(OUT, slug))
import re                                      # one sitemap and one robots.txt, at the site root (crawlers ignore nested ones)
locs = []
for sm in sorted(glob.glob(os.path.join(OUT, "**", "sitemap.xml"), recursive=True)):
    locs += re.findall(r"<url>.*?</url>", open(sm).read())
    if os.path.dirname(sm) != OUT: os.remove(sm)
for rb in glob.glob(os.path.join(OUT, "*", "robots.txt")): os.remove(rb)
open(os.path.join(OUT, "sitemap.xml"), "w").write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + "".join(locs) + "</urlset>")
man = []                                       # machine-readable index for agents and the MCP server
for dj in sorted(glob.glob(os.path.join(OUT, "**", "data.json"), recursive=True)):
    d = json.load(open(dj)); rel = os.path.relpath(os.path.dirname(dj), OUT)
    man.append({"model": d["model"], "slug": "" if rel == "." else rel, "data": ("" if rel == "." else rel + "/") + "data.json",
                "captured_at": d.get("captured_at"), "repos": len(d.get("repos", [])), "rows": len(d.get("rows", [])), "sets": len(d.get("sets", []))})
json.dump({"name": "QFF (Quant Fidelity Frontier)", "about": "Publisher claims about Hugging Face quants, each pinned to its source line. Claims, not measurements.", "models": man},
          open(os.path.join(OUT, "models.json"), "w"), indent=1)
open(os.path.join(OUT, ".nojekyll"), "w").close()   # serve files as-is; no Jekyll processing
bad = [p for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p) and not p.endswith((".html", ".json", ".txt", ".xml", ".css", ".js", ".svg", ".png", ".ico", ".nojekyll"))]
assert not bad, f"non-static files in site: {bad[:5]}"
big = [(p, os.path.getsize(p)) for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p) and os.path.getsize(p) > 50e6]
assert not big, f"files over 50 MB (GitHub warns at 50, rejects at 100): {big}"
tot = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p))
print(json.dumps({"out": OUT, "files": sum(1 for p in glob.glob(os.path.join(OUT, '**', '*'), recursive=True) if os.path.isfile(p)), "MB": round(tot / 1e6, 1)}))
