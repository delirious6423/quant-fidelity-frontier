#!/usr/bin/env python3
"""Gate 1b: compare the blind auditor's extraction (audit/blind.json) with ours (verified/part*.py).
Compares sets of numeric tokens per repo. Reports numbers the auditor found that we lack (possible misses)
and numbers we have that the auditor lacks (possible mis-extractions)."""
import json, glob, re, os, collections
ROOT = os.environ.get("QCL_ROOT") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models", "qwen3.8-flash-next")

def toks(s):
    out = set()
    for t in re.findall(r"\d[\d,]*(?:\.\d+)?|\.\d+", str(s)):
        t = t.replace(",", "")
        try: out.add(round(float(t), 6))
        except ValueError: pass
    return out

ours = collections.defaultdict(set)
for p in sorted(glob.glob(f"{ROOT}/verified/part*.py")):
    ns = {}; exec(open(p).read(), ns)
    for r in ns["R"]:
        def eat(e, key):
            for v in (e.get("m") or {}).values(): ours[key] |= toks(v)
            for k in ("size", "bpw", "mem"):
                if e.get(k) is not None: ours[key] |= toks(e[k])
        for v in r.get("variants", []): eat(v, r["repo"])
        for v in r.get("ladder", []) or []: eat(v, r["repo"])
        for v in r.get("cross", []) or []: eat(v, r["repo"])
        for bk in ("baseline", "baseline2"):
            b = r.get(bk)
            if isinstance(b, dict):
                for k, v in b.items():
                    if k not in ("L", "L_extra", "src"): ours[r["repo"]] |= toks(v)

import sys
files = sys.argv[1:] or [f"{ROOT}/audit/blind.json"]
blind = {}
for fpath in files: blind.update(json.load(open(fpath)))
ours = {k: v for k, v in ours.items() if k in blind}
theirs = {r: set().union(*[toks(e["value"]) for e in es]) if es else set() for r, es in blind.items()}
blind_lines = {r: {round(float(t),6): e for e in es for t in [x.replace(",","") for x in re.findall(r"\d[\d,]*(?:\.\d+)?|\.\d+", str(e["value"]))]} for r, es in blind.items()}

report, tot_a, tot_b, tot_both = [], 0, 0, 0
for repo in sorted(set(ours) | set(theirs)):
    a, b = ours.get(repo, set()), theirs.get(repo, set())
    tot_a += len(a); tot_b += len(b); tot_both += len(a & b)
    only_blind = sorted(b - a); only_ours = sorted(a - b)
    if only_blind or only_ours:
        report.append(f"## {repo}\n  agreed: {len(a & b)}\n  auditor only: " +
                      ", ".join(f"{x:g}[L{blind_lines[repo][x]['line']} {blind_lines[repo][x]['metric'][:30]}]" for x in only_blind[:40]) +
                      f"\n  ours only: {', '.join(f'{x:g}' for x in only_ours[:40])}")
out_name = "blind_diff.md" if len(files) == 1 and files[0].endswith("blind.json") else "blind_diff_new.md"
open(f"{ROOT}/audit/{out_name}", "w").write("\n".join(report))
print(f"ours {tot_a} values, auditor {tot_b}, agreed {tot_both}; repos with differences: {len(report)}")
