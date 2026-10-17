#!/usr/bin/env python3
"""Gate 2: automated consistency checks on claims/*.json. Exit 1 on any ERROR."""
import json, sys, glob, os

ROOT = os.environ.get("QCL_ROOT") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "qwen3.8-flash-next")
FORMATS = {"GGUF","NVFP4","FP8","MXFP4","AWQ","GPTQ","AutoRound","EXL3","MLX","other"}
LOWER_BETTER = ["mean_kld","kld_p99","kld_p99_9","ppl","delta_p_rms"]
errors, warns = [], []

def E(repo, msg): errors.append(f"ERROR {repo}: {msg}")
def W(repo, msg): warns.append(f"WARN  {repo}: {msg}")

files = sorted(f for f in glob.glob(os.path.join(ROOT,"claims","*.json")) if not os.path.basename(f).startswith("_"))
for f in files:
    try:
        d = json.load(open(f))
    except Exception as ex:
        E(f, f"invalid JSON: {ex}"); continue
    r = d.get("repo", f)
    for k in ["repo","url","format","variants","evidence_urls","methodology","weights_modified"]:
        if k not in d: E(r, f"missing field {k}")
    if d.get("format") not in FORMATS: W(r, f"format '{d.get('format')}' not in vocabulary")
    if d.get("url","").rstrip("/") != f"https://huggingface.co/{r}": W(r, "url does not match repo id")
    if not d.get("evidence_urls"): E(r, "no evidence_urls")
    vs = d.get("variants") or []
    has_metric = False
    for v in vs:
        m = v.get("metrics") or {}
        nums = {k: x for k, x in m.items() if k != "bench" and x is not None}
        if nums or m.get("bench"): has_metric = True
        if v.get("size_gb") is not None and v.get("size_unit") in (None, "not stated"):
            W(r, f"{v.get('name')}: size without unit")
        for k, x in nums.items():
            if not isinstance(x, (int, float)): E(r, f"{v.get('name')}.{k} not numeric: {x!r}")
        t = m.get("top1_agree_pct")
        if isinstance(t,(int,float)) and not (0 <= t <= 100): E(r, f"{v.get('name')} top1 out of range {t}")
    meth = d.get("methodology") or {}
    if has_metric and all(meth.get(k) in (None,"not stated") for k in ["reference","eval_dataset","tool"]):
        W(r, "metrics present but no methodology stated (shown as 'unverifiable method')")
    # monotonic: bigger size should not be worse on lower-better metrics (within this publisher)
    sized = [v for v in vs if isinstance(v.get("size_gb"),(int,float))]
    sized.sort(key=lambda v: v["size_gb"])
    for key in LOWER_BETTER:
        seq = [(v["name"], v["size_gb"], (v.get("metrics") or {}).get(key)) for v in sized]
        seq = [s for s in seq if isinstance(s[2],(int,float))]
        for a, b in zip(seq, seq[1:]):
            if b[2] > a[2] * 1.05:
                W(r, f"non-monotonic {key}: {a[0]} ({a[1]}) {a[2]} -> {b[0]} ({b[1]}) {b[2]}")

print(f"{len(files)} repo files checked")
for w in warns: print(w)
for e in errors: print(e)
print(f"{len(errors)} errors, {len(warns)} warnings")
sys.exit(1 if errors else 0)
