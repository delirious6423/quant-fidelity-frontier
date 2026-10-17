#!/usr/bin/env python3
"""claims/*.json (raw extraction, untouched) -> data/claims_normalized.json.
Every metric becomes {key, value, err, raw, unit}; `raw` is the exact published string for verbatim checks."""
import json, glob, os, re

ROOT = os.environ.get("QCL_ROOT") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qwen3.8-flash-next")
NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

def parse(x):
    """Return (value, err, unit) from a number or a published string."""
    if isinstance(x, bool) or x is None: return None
    if isinstance(x, (int, float)): return (float(x), None, None)
    if isinstance(x, str):
        s = x.strip()
        m = re.fullmatch(rf"({NUM})\s*(%?)\s*(?:±|\+/-|\+-)\s*({NUM})\s*(%?)", s)
        if m: return (float(m.group(1)), float(m.group(3)), "%" if (m.group(2) or m.group(4)) else None)
        m = re.fullmatch(rf"({NUM})\s*(%?)", s)
        if m: return (float(m.group(1)), None, "%" if m.group(2) else None)
    return None

def metrics_of(m, prefix=""):
    out, notes = [], []
    for k, v in (m or {}).items():
        if k == "bench":
            for bk, bv in (v or {}).items():
                p = parse(bv)
                if p: out.append({"key": f"{prefix}bench:{bk}", "value": p[0], "err": p[1], "unit": p[2], "raw": bv})
                elif bv is not None: notes.append(f"{prefix}bench:{bk}={bv}")
            continue
        if isinstance(v, dict):                       # e.g. ji-farthing "code" sub-eval
            o, n = metrics_of(v, prefix=f"{prefix}{k}:"); out += o; notes += n; continue
        p = parse(v)
        if p: out.append({"key": prefix + k, "value": p[0], "err": p[1], "unit": p[2], "raw": v})
        elif v is not None: notes.append(f"{prefix}{k}: {v}")
    return out, notes

repos = []
for f in sorted(glob.glob(os.path.join(ROOT, "claims", "*.json"))):
    if os.path.basename(f).startswith("_"): continue
    d = json.load(open(f))
    vs = []
    for v in d.get("variants") or []:
        ms, notes = metrics_of(v.get("metrics"))
        vs.append({**{k: v.get(k) for k in ["name", "size_gb", "size_unit", "bpw", "qualitative"]},
                   "metrics": ms, "metric_notes": notes})
    repos.append({**{k: d.get(k) for k in ["repo", "url", "format", "runtime", "weights_modified", "method",
                  "imatrix_or_calib_data", "methodology", "headline_claims", "red_flags", "evidence_urls"]},
                  "baseline_numbers": d.get("baseline_numbers"), "variants": vs})

base = json.load(open(os.path.join(ROOT, "claims", "_base_model.json")))
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
json.dump({"model": "Qwen/Qwen3.8-Flash-Next", "base": base, "repos": repos},
          open(os.path.join(ROOT, "data", "claims_normalized.json"), "w"), indent=1)
n = sum(len(v["metrics"]) for r in repos for v in r["variants"])
print(f"{len(repos)} repos, {sum(len(r['variants']) for r in repos)} variants, {n} metric values")
