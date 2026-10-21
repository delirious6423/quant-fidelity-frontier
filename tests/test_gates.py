#!/usr/bin/env python3
"""Acceptance tests for the publication gates (from the external review's counterfactuals).
Each test mutates a throwaway copy of the project and asserts the gates react. Exit 1 if any test fails."""
import os, shutil, subprocess, tempfile, json, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
_BASE = shutil.ignore_patterns("dist", "site", "claims", "data", "__pycache__", ".git", "_site", "work", "maps")
def IGNORE(d, names):   # copy only the Flash-Next data root (the tests mutate it); skip every other model
    skip = set(_BASE(d, names))
    if os.path.basename(d) == "models": skip |= {n for n in names if n != "qwen3.8-flash-next"}
    return skip
FN = "models/qwen3.8-flash-next"   # data root the gates default to
results = []

def fresh():
    d = tempfile.mkdtemp(prefix="qcl-test-")
    shutil.copytree(ROOT, f"{d}/p", ignore=IGNORE)
    return f"{d}/p"

def run(p, cmd):
    r = subprocess.run(cmd, cwd=p, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

def edit(path, old, new):
    s = open(path).read(); assert old in s, f"mutation anchor not found in {path}: {old[:60]}"
    open(path, "w").write(s.replace(old, new, 1))

def test(name, fn):
    try: fn(); results.append((name, "PASS", ""))
    except AssertionError as e: results.append((name, "FAIL", str(e)))

def t_absent_value_blocks_build():
    p = fresh()
    edit(f"{p}/{FN}/verified/part1.py", 'mean_kld="0.020218 ± 0.000145"', 'mean_kld="0.1234567 ± 0.000145"')
    rc, out = run(p, "python3 gates/verify.py"); assert rc != 0, "verify.py passed a value absent from its source line"
    rc, out = run(p, "python3 web/build.py"); assert rc != 0 and "REFUSING TO BUILD" in out, "build.py wrote a site despite a failing verifier"
    assert not os.path.exists(f"{p}/{FN}/site/index.html"), "site/index.html was written"
    rc, out = run(p, "python3 docs/history/legacy-tools/make_pack.py"); assert rc != 0 and "REFUSING TO PACK" in out, "make_pack.py produced a pack despite a failing gate"

def t_swapped_columns_caught():
    p = fresh()
    edit(f"{p}/{FN}/verified/part1.py", 'ppl="4.596526 ± 0.027937", ppl_delta_pct="+0.1270%", mean_kld="0.020218 ± 0.000145"',
         'ppl="0.020218 ± 0.000145", ppl_delta_pct="+0.1270%", mean_kld="4.596526 ± 0.027937"')
    rc, out = run(p, "python3 gates/verify_columns.py"); assert rc != 0 and "MISMATCH" in out, "KLD/PPL swap from the same row was not caught"

def t_image_baseline_checked():
    p = fresh()
    edit(f"{p}/{FN}/verified/part3_images.py", "baseline={'ppl': '1.38474'", "baseline={'ppl': '999.999'")
    rc, out = run(p, "python3 gates/verify_images.py"); assert rc != 0, "changing the image-sourced BF16 baseline still passed"

def t_unicode_minus_sign():
    p = fresh()
    rc, out = run(p, "python3 web/build.py"); assert rc == 0, "clean build failed: " + out[-500:]
    d = json.load(open(f"{p}/{FN}/site/data.json"))
    bad = [(r["repo"], k, m["raw"], m["v"]) for r in d["rows"] for k, m in r["metrics"].items()
           if m.get("v") is not None and re.match(r"^\s*(\*\*)?[−\-]\s*\.?\d", str(m["raw"])) and m["v"] >= 0]
    assert not bad, f"negative published values parsed as non-negative: {bad[:3]}"

def t_no_invisible_metrics():
    p = ROOT
    d = json.load(open(f"{p}/{FN}/site/data.json")); html = open(f"{p}/{FN}/site/index.html").read()
    missing = []
    for r in d["rows"]:
        for k, m in r["metrics"].items():
            if str(m["raw"]) and str(m["raw"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") not in html:
                missing.append((r["repo"], k, m["raw"]))
    assert not missing, f"{len(missing)} metric values not rendered, e.g. {missing[:3]}"

def t_estimates_disabled():
    d = json.load(open(f"{ROOT}/{FN}/site/data.json"))
    assert all(not s.get("linked") for s in d["sets"]), "linked estimates are present in the published data"

def t_auto_sets_single_basis_no_cross():
    d = json.load(open(f"{ROOT}/{FN}/site/data.json"))
    for s in d["sets"]:
        if not s.get("auto"): continue
        assert len({p["basis"] for p in s["points"]}) <= 1, f"auto set {s['id']} mixes size bases"
        # F7: build.py admits cross rows the card's author ran (kind "cross") but never quoted ones (kind "quoted",
        # i.e. entry measured=False) or restated/copied rows
        bad = [p.get("variant") for p in s["points"] if p["kind"] == "quoted" or p.get("measured") is False or p.get("copied_from")]
        assert not bad, f"auto set {s['id']} contains quoted/restated cross rows: {bad[:3]}"

def t_unique_explorer_ids():
    js = os.path.join(ROOT, "web", "site_explorer.js")
    assert "'pk-' + S.id + '-' + S.points.indexOf(p)" in open(js).read(), "checkbox ids are not unique per set/point"

# ---- gate-gap mutations from the external review (audit/fable_review.md F11 + gate-gap table) ----
def caught(out, marker, repo):
    return any(marker in l and repo in l for l in out.split("\n"))

def t_pivot_column_swap():
    # perkel MCTiny <-> MCSmall held-out KL / top-1, same row L66, variants are columns
    p = fresh(); f = f"{p}/{FN}/verified/part4_0.py"
    edit(f, '"kl:held-out":"0.1454","top1:held-out":"86.7%"', '"kl:held-out":"@T1","top1:held-out":"@T2"')
    edit(f, '"kl:held-out":"0.0613","top1:held-out":"91.7%"', '"kl:held-out":"0.1454","top1:held-out":"86.7%"')
    edit(f, '"kl:held-out":"@T1","top1:held-out":"@T2"', '"kl:held-out":"0.0613","top1:held-out":"91.7%"')
    rc, out = run(p, "python3 gates/verify_columns.py")
    assert rc != 0 and caught(out, "ROWLABEL", "perkel/"), "pivoted-table column swap (perkel Tiny/Small) was not caught"

def t_sign_flip():
    p = fresh()
    edit(f"{p}/{FN}/verified/part4_5.py", '"ppl_delta_pct":"−0.16 %"', '"ppl_delta_pct":"+0.16 %"')
    rc, out = run(p, "python3 gates/verify.py")
    assert rc != 0 and caught(out, "SIGN", "PaoAI/"), "sign flip (PaoAI -0.16 % -> +0.16 %) was not caught"

def t_unit_flip():
    p = fresh()
    edit(f"{p}/{FN}/verified/part1.py", 'dict(name="Q5_K_M", size=153.07, unit="GiB"', 'dict(name="Q5_K_M", size=153.07, unit="GB"')
    rc, out = run(p, "python3 gates/verify.py")
    assert rc != 0 and caught(out, "UNIT", "AesSedai/"), "GiB -> GB unit flip (AesSedai Q5_K_M) was not caught"

def t_variant_name_swap():
    p = fresh(); f = f"{p}/{FN}/verified/part1.py"
    edit(f, 'dict(name="Q5_K_M", size=153.07', 'dict(name="@SWAP", size=153.07')
    edit(f, 'dict(name="Q4_K_M", size=131.98', 'dict(name="Q5_K_M", size=131.98')
    edit(f, 'dict(name="@SWAP", size=153.07', 'dict(name="Q4_K_M", size=153.07')
    rc, out = run(p, "python3 gates/verify_columns.py")
    assert rc != 0 and caught(out, "ROWLABEL", "AesSedai/"), "variant-name swap keeping L (AesSedai Q5_K_M/Q4_K_M) was not caught"

def t_quoted_without_measured():
    # mensaprodigy REAP-384-mlx-mtp: card L127 says the figures are 'reported by Sh0wie'
    p = fresh()
    edit(f"{p}/{FN}/verified/part4_2.py", 'variant="Full Flash-Next Q4 (512 experts)", measured=False,', 'variant="Full Flash-Next Q4 (512 experts)",')
    rc, out = run(p, "python3 gates/verify.py")
    assert rc != 0 and caught(out, "QUOTED", "mensaprodigy/"), "cross entry in quoted context without measured= was not caught"

def t_size_without_unit():
    p = fresh()
    edit(f"{p}/{FN}/verified/part1.py", 'dict(name="Q5_K_M", size=153.07, unit="GiB", ', 'dict(name="Q5_K_M", size=153.07, ')
    rc, out = run(p, "python3 gates/verify.py")
    assert rc != 0 and caught(out, "no unit", "AesSedai/"), "size without unit (AesSedai Q5_K_M) was not caught"

def t_gate_cache_never_masks():
    p = fresh()
    rc, out = run(p, "python3 web/build.py"); assert rc == 0, "clean build failed: " + out[-300:]          # warms the cache
    import json as _j
    idx = _j.load(open(f"{p}/{FN}/snapshots/index.json"))["repos"]; f = f"{p}/{FN}/" + idx["AesSedai/Qwen3.8-Flash-Next-GGUF"]["file"]
    s = open(f).read(); s2 = re.sub(r"(\d)\.(\d)", r"\1.9\2", s, count=40); assert s2 != s; open(f, "w").write(s2)   # edit card bytes only
    rc, out = run(p, "python3 web/build.py"); assert rc != 0 and "REFUSING TO BUILD" in out, "a warm gate cache let an edited card through"

for name, fn in [("gate cache never masks an edited card", t_gate_cache_never_masks), ("absent value blocks build and pack", t_absent_value_blocks_build), ("swapped KLD/PPL columns caught", t_swapped_columns_caught),
                 ("image baseline checked", t_image_baseline_checked), ("unicode minus keeps sign", t_unicode_minus_sign),
                 ("no invisible metrics in ledger", t_no_invisible_metrics), ("linked estimates disabled", t_estimates_disabled),
                 ("auto sets: one basis, no quoted cross rows", t_auto_sets_single_basis_no_cross), ("unique explorer ids", t_unique_explorer_ids),
                 ("pivoted-table column swap caught (perkel Tiny/Small)", t_pivot_column_swap), ("sign flip caught (PaoAI)", t_sign_flip),
                 ("GiB->GB unit flip caught (AesSedai)", t_unit_flip), ("variant-name swap keeping L caught (AesSedai)", t_variant_name_swap),
                 ("quoted-context entry without measured= caught (mensaprodigy)", t_quoted_without_measured),
                 ("size without unit caught (AesSedai)", t_size_without_unit)]:
    test(name, fn)
for n, st, msg in results: print(f"[{st}] {n}" + (f" — {msg}" if msg else ""))
sys.exit(0 if all(st == "PASS" for _, st, _ in results) else 1)
