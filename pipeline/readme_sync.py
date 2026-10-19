#!/usr/bin/env python3
"""Keep README.md's facts generated, not typed.

  python3 pipeline/readme_sync.py            # rewrite the model table from each live model's built site data
  python3 pipeline/readme_sync.py --check    # exit 1 if the table is stale, a live model is missing, or a link is dead
  python3 pipeline/readme_sync.py --check --structure   # CI: same, but counts may lag (scans commit data before QA publishes)

The table between <!-- qff:models:start --> and <!-- qff:models:end --> lists every live model (pipeline/qcl_live.py)
with its repo and claimed-value counts, computed exactly as the site computes them from models/<slug>/site/data.json
(repos = repositories listed; values = each row's metrics plus its size). Run it after build.py; the QA worker runs it
before every publish commit, so README and site change together. --check also verifies that every relative link and
every repo path in backticks points at a file that exists, so the prose cannot point at something that was renamed."""
import json, os, re, subprocess, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(CODE, "README.md")
BASE = "https://delirious6423.github.io/quant-fidelity-frontier/"
START, END = "<!-- qff:models:start -->", "<!-- qff:models:end -->"

def live():   # in watch-list order, the order of the site's landing page
    ds = [l for l in subprocess.run([sys.executable, f"{CODE}/pipeline/qcl_live.py"], capture_output=True, text=True, check=True).stdout.split() if l]
    order = [m["slug"] for m in json.load(open(os.path.join(CODE, "watchlist.json")))["models"]]
    return sorted(ds, key=lambda d: (order.index(os.path.basename(d)) if os.path.basename(d) in order else 999, d))

def table():
    rows, missing = [], []
    for d in live():
        slug = os.path.basename(d); dj = os.path.join(CODE, d, "site", "data.json")
        if not os.path.exists(dj): missing.append(slug); continue
        x = json.load(open(dj))
        vals = sum(len(r.get("metrics") or {}) + (1 if r.get("size_gb") else 0) for r in x.get("rows", []))
        rows.append((x["model"], slug, len(x.get("repos", [])), vals))
    body = "| Model | Repos | Claimed values |\n|---|---|---|\n" + "".join(f"| [{m}]({BASE}{s}/) | {n:,} | {v:,} |\n" for m, s, n, v in rows)
    return body, missing, [s for _, s, _, _ in rows]

def dead_links(text, base=CODE):   # relative to the file's own folder; URL paths and the built _site are not repo files
    bad = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"[a-z]+:", target) or target.startswith(("#", "/")): continue
        if not os.path.exists(os.path.join(base, target.split("#")[0])): bad.append(target)
    for code in re.findall(r"`([^`\s]+)`", text):   # repo paths quoted in prose, e.g. `qff_mcp/README.md`, `models/*/cache/cards/`
        if "/" in code and not code.startswith(("http", "-", "/")) and "_site" not in code and re.fullmatch(r"[\w.*/-]+", code):
            import glob
            if not (glob.glob(os.path.join(base, code.rstrip("/"))) or glob.glob(os.path.join(CODE, code.rstrip("/")))): bad.append(code)
    return bad

text = open(README).read()
if START not in text or END not in text: print(f"README.md has no {START} ... {END} block"); sys.exit(1)
body, missing, slugs = table()
new = text[:text.index(START) + len(START)] + "\n" + body + text[text.index(END):]
if "--check" in sys.argv:
    problems = []
    if missing: problems.append(f"no built site for live model(s) {missing}; run build.py first")
    listed = re.findall(re.escape(BASE) + r"([\w.-]+)/\)", text[text.index(START):text.index(END)])
    if sorted(listed) != sorted(slugs) and not missing: problems.append(f"table lists {sorted(listed)}, live models are {sorted(slugs)}")
    if new != text and not missing and "--structure" not in sys.argv: problems.append("model counts are stale; run pipeline/readme_sync.py")
    problems += [f"dead link or path: {b}" for b in dead_links(text)]
    mr = os.path.join(CODE, "qff_mcp", "README.md")   # the MCP README's tool table must match the server's public tools
    if os.path.exists(mr):
        doc = set(re.findall(r"^\| `(\w+)", open(mr).read(), re.M))
        code = set(re.findall(r"^def ([a-z]\w*)\(", open(os.path.join(CODE, "qff_mcp", "qff_mcp.py")).read(), re.M)) - {"main"}
        if doc != code: problems.append(f"qff_mcp/README.md tools {sorted(doc)} differ from qff_mcp.py tools {sorted(code)}")
        problems += [f"dead link or path in qff_mcp/README.md: {b}" for b in dead_links(open(mr).read(), os.path.dirname(mr))]
    print("\n".join(problems) or "README ok"); sys.exit(1 if problems else 0)
if missing: print(f"skipped: no built site for {missing}"); sys.exit(1)
if new != text: open(README, "w").write(new)
print(json.dumps({"models": slugs, "changed": new != text}))
