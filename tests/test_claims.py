#!/usr/bin/env python3
"""Regression checks for the cold external review (audit/fable_review.md, F1-F24) on the BUILT site.
Usage: python3 test_claims.py [site_dir]   (default: site/). Exit 1 on any failure."""
import json, os, re, sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")
if len(sys.argv) >= 2: SITE = os.path.abspath(sys.argv[1])
else:   # the reviewed Flash-Next data, re-pinned against hourly card drift, built into a throwaway copy (tests/frozen.py)
    import subprocess; sys.path.insert(0, os.path.join(CODE, "tests")); import frozen
    _copy = frozen.frozen_copy()
    _b = subprocess.run([sys.executable, os.path.join(CODE, "web", "build.py")], env={**os.environ, "QCL_ROOT": str(_copy)}, capture_output=True, text=True)
    if _b.returncode: sys.exit("frozen build failed: " + (_b.stdout + _b.stderr)[-800:])
    SITE = os.path.join(_copy, "site")
d = json.load(open(f"{SITE}/data.json")); html = open(f"{SITE}/index.html").read()
rows, repos, sets = d["rows"], d["repos"], {s["id"]: s for s in d["sets"]}
rep = {x["repo"]: x for x in repos}
res = []
def check(cond, name, detail=""):
    res.append((bool(cond), name, detail))

# F1 quoted rows are never presented as measurements
bad = [(r["measured_by"], r["variant"]) for r in rows if r["kind"] == "cross" and any(re.search(r"quoted|not measured|reported by|publisher-reported", f, re.I) for f in r.get("flags") or [])]
check(not bad, "F1 no flagged-quoted row is kind=cross", bad[:3])
bad = [(r["measured_by"], r["variant"]) for r in rows if r["kind"] == "cross" and re.search(r"\bquoted\b|-reported\b", str(r["variant"]), re.I)]
check(not bad, "F1 no row named 'quoted'/'-reported' is kind=cross", bad[:3])
check(any(r["kind"] == "quoted" for r in rows) and "quoted by" in html, "F1 quoted rows exist and are labelled 'quoted by'")
check("an older build" not in html and "quotes 117.13 GiB" not in html, "F1 AesSedai/agentionai pseudo-disagreement removed from head-to-head")
for s in sets.values():
    q = [p for p in s["points"] if p["kind"] == "quoted"]
    check(not q, f"F1 set {s['id']} has no quoted points", [p["variant"] for p in q][:3])

# F2 units
check(all(r["size_raw"] is None or re.search(r"[A-Za-z]", r["size_raw"]) for r in rows), "F2 every size carries its unit", [r["size_raw"] for r in rows if r["size_raw"] and not re.search(r"[A-Za-z]", r["size_raw"])][:3])
jf = [p for p in sets["jifarthing"]["points"] if "IQ4_KT_v2" in p["variant"]]
check(jf and all(abs(p["x"] - 80.72 * 1.073741824) < 0.02 for p in jf), "F2 ji-farthing IQ4_KT_v2 plotted as 86.67 GB (80.72 GiB)", [p["x"] for p in jf])

# F3 reference classes
must_bf16 = ["scorpoon/Qwen3.8-Flash-Next-exl3-SC", "pipenetwork/Qwen3.8-Flash-Next-MLX-6bit", "perkel/Qwen3.8-Flash-Next-MC", "ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4",
             "RedHatAI/Qwen3.8-Flash-Next-NVFP4", "amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8", "IHaveNoClueAndIMustPost/Qwen3.8-Flash-Next-Simple-Q3_K"]
check(all(rep[r]["ref_class"] == "bf16" for r in must_bf16 if r in rep), "F3 stated-BF16 references classed bf16", [(r, rep[r]["ref_class"]) for r in must_bf16 if r in rep and rep[r]["ref_class"] != "bf16"])
must_not = {"Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata": "q8", "aniolekx/Qwen3.8-Flash-Next-E9-MLX-8bit": "other", "Lottolabs/Qwen3.8-Flash-Next-Coder-P150": "other",
            "Delgon/Qwen3.8-Flash-Next-PLE-GGUF": "other", "codelight-ai/Qwen3.8-Flash-Next-NVFP4-PLE-INT3": "other"}
check(all(rep[r]["ref_class"] == c for r, c in must_not.items() if r in rep), "F3 negated / component / Q8_0 references not classed bf16", [(r, rep[r]["ref_class"]) for r, c in must_not.items() if r in rep and rep[r]["ref_class"] != c])

# F4 method text
check("Estimated points in the explorer" not in html, "F4 no claim that estimates are shown")
check("re-extracted every card" not in html, "F4 no claim that every card was blind re-extracted")

# F5 / F22 strata keeps the mix row; no auto set reprints a curated table
st = sets["strata"]
check(any("mix" in p["variant"].lower() for p in st["points"]), "F5 strata set includes the NVFP4 + Q8_0 mix row")
check(not [i for i in sets if i.startswith("a-maximilian228")], "F5 no duplicate Maximilian228 auto sets", [i for i in sets if i.startswith("a-maximilian228")])

# F6 every auto set contains its own publisher's quant
bad = [s["id"] for s in sets.values() if s.get("auto") and not any(p["kind"] == "own" for p in s["points"])]
check(not bad, "F6 every auto set contains the title repo's own quant", bad)

# F7 auto sets include rival files the measurer ran in the same test
b = sets.get("a-beamster-sushi-2bpw")
check(b and any(p["kind"] == "cross" for p in b["points"]), "F7 beamster set includes the rival packs beamster scored")

# F8 other-model rows are not rows
check(not [r for r in rows if re.search(r"\bGLM\b", str(r["variant"]))], "F8 no GLM 5.1 rows in the ledger")

# F9 showkieo is a copy, with no set of its own
check(not [i for i in sets if "showkieo" in i], "F9 showkieo has no comparable set")
check(rep["showkieo/Qwen3.8-Flash-Next-Sushi-3bpw"]["copied_from"] == "beamster/Qwen3.8-Flash-Next-Sushi-3bpw", "F9 showkieo marked as copy of beamster")
check(not any("disagree" in f for x in ("beamster/Qwen3.8-Flash-Next-Sushi-3bpw", "showkieo/Qwen3.8-Flash-Next-Sushi-3bpw") for f in (rep[x]["flags"] or [])), "F9 beamster/showkieo not described as disagreeing cards")

# F10 attributions the card does not make are marked
td = [r for r in rows if r["measured_by"] == "turboderp/Qwen3.8-Flash-Next-exl3" and r["repo"] == "unsloth/Qwen3.8-Flash-Next-GGUF"]
check(td and all(r.get("inferred") for r in td), "F10 turboderp's UD rows marked 'repo not named on card'")
check("repo not named on card" in html, "F10 inference is visible in the ledger")

# F12 copy source = largest overlap
cp = json.load(open(f"{ROOT}/audit/copies.json"))
k = [v for kk, v in cp.items() if kk.startswith("Kashapps/")]
check(k and "Uncensored-ROCmFP4-FAST" in k[0]["source"], "F12 Kashapps copy source is kingjones777 Uncensored-FAST", k[0]["source"] if k else None)

# F13 references are flagged so the explorer never ranks them
check(any(p.get("is_ref") for p in st["points"] if "reference" in p["variant"].lower()), "F13 Q8_0 (reference) flagged is_ref")
check(all(p.get("is_ref") for p in sets["nvidia"]["points"] if p["kind"] == "baseline"), "F13 nvidia FP8 baseline flagged is_ref")

# F14 / F19 no speculation about a publisher's method
check("suggests the reference is not BF16" not in html, "F14 unsloth note does not speculate")
check("ctx 512 (current build" not in html, "F19 head-to-head does not state an AesSedai method the card lacks")

# F15 stat counts only measured rival rows
n = sum(1 for r in rows if r["kind"] == "cross" and not r.get("copied_from"))
check(re.search(rf"Measurements of rivals' files</dt><dd>{n}</dd>", html), f"F15 header stat equals measured, uncopied rival rows ({n})")

# F16 no raw basis keys rendered
check(not re.search(r">\s*(hf_branch_total|hf_listing_total|text_weights|ram_in_use|trunk_only)\s*<", html), "F16 no raw basis keys in the page")

# F17 PPL change header says baseline varies
check("vs the card's baseline" in html, "F17 PPL change column names its baseline")

# F18 one file, one basis
dr = {(r["variant"], r["size_raw"]): r["basis"] for r in rows if r["measured_by"].startswith("TheDrainFlorist/") and r["size_raw"]}
by = {}
for (v, sz), bs in dr.items(): by.setdefault((v, sz), set()).add(bs)
check(len({r["basis"] for r in rows if r["measured_by"].startswith("TheDrainFlorist/") and r["size_raw"]}) == 1, "F18 TheDrainFlorist shared table uses one size basis")

# F21 kl / mean_kl keys reach the main KLD column (sortable)
y = [r for r in rows if r["measured_by"].startswith("yamz-labs/") and "kl" in r["metrics"]]
check(not y or 'published as kl' in html, "F21 'kl' metrics rendered in the main KLD column")

# F24 ISTA baseline size cited
check("354 GB (card L123)" in html or any(p.get("size_raw") == "354 GB (card L123)" for p in sets["ista"]["points"]), "F24 ISTA baseline size cites its card line")

for ok, n, det in res: print(("[PASS] " if ok else "[FAIL] ") + n + ("" if ok or not det else f" — {det}"))
sys.exit(0 if all(ok for ok, _, _ in res) else 1)
