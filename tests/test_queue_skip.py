#!/usr/bin/env python3
"""Cards that cannot state a quality claim are filed by code, not read by two mapping workers (pipeline/qcl_map.skip_reason,
used by pipeline/qcl_queue.py): no numeric candidates, or no quality-metric term anywhere in the card text.

Three checks: the predicate itself; the queue builder on a synthetic model root (skipped cards get identical "no claims"
maps for runs A and B and never enter a queue); and recall on the committed data: every run-A map that files a quality
value comes from a card the vocabulary matches, so the skip would not have hidden one committed claim."""
import glob, hashlib, json, os, re, subprocess, sys, tempfile
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE, "pipeline"))
import qcl_map as Q

# 1. the predicate
c = {"c1": {"L": 3, "raw": "4.2", "header": "Size", "label": "Q4_K_M"}}
assert "no numbers" in Q.skip_reason({}, "Perplexity 5.1 on wikitext") and Q.CODE_FILED in Q.skip_reason({}, "")
assert "not quality metrics" in Q.skip_reason(c, "| Quant | Size (GB) |\n| Q4_K_M | 4.2 |\n") 
for text in ("KLD vs bf16: 0.012", "Perplexity on wikitext-2", "PPL 5.23", "Top-1 agreement 98.7%", "MMLU 71.2", "accuracy 0.81",
             "benchmark results below", "gsm8k", "mean KL divergence", "calibration on c4"):
    assert Q.skip_reason(c, "| Size | 4.2 |\n" + text) is None, text
assert Q.skip_reason(c, "Decode speed 17.6 t/s on a Mac Studio, 4.2 GB download") is not None   # speed and size are not quality
m = Q.no_claims_map("org/repo", {"source_sha256": "a" * 64, "candidates_sha256": "b" * 64, "candidates": {}})
assert m["no_quality_claims"] and m["entries"] == [] and m["prose"] == [] and not re.search(r"\d", json.dumps({k: v for k, v in m.items() if "sha256" not in k}))
m["no_claims_reason"] = Q.skip_reason({}, "")
flag = Q.apply(m, {})["flags"][0]
assert flag.startswith("No quality claims for the quant (checked by code)"), flag
assert Q.apply({"repo": "x", "no_quality_claims": True, "no_claims_reason": "speed table only [L4]", "entries": []}, {})["flags"][0].startswith("No quality claims for the quant (card read in full)")

# 2. the queue builder on a synthetic model
cards = {
    "a/size-only": "# Quant\n\n| Quant | Size (GB) | BPW |\n|---|---|---|\n| Q4_K_M | 4.20 | 4.85 |\n| Q8_0 | 7.90 | 8.50 |\n\nDecode 17.6 t/s.\n",
    "b/no-numbers": "# Quant\n\nConverted with llama.cpp. No evaluation was run.\n",
    "c/quality": "# Quant\n\n| Quant | Size (GB) | PPL (wikitext-2) |\n|---|---|---|\n| Q4_K_M | 4.20 | 5.31 |\n",
}
with tempfile.TemporaryDirectory() as tmp:
    R = os.path.join(tmp, "model"); W = os.path.join(tmp, "work")
    os.makedirs(f"{R}/cache/cards"); os.makedirs(f"{R}/snapshots"); os.makedirs(f"{R}/audit")
    idx = {}
    for rp, text in cards.items():
        fn = f"cache/cards/{rp.replace('/', '__')}.md"; b = text.encode()
        open(f"{R}/{fn}", "wb").write(b); idx[rp] = {"file": fn, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    json.dump({"repos": idx, "model": "x/y"}, open(f"{R}/snapshots/index.json", "w"))
    json.dump({"todo": list(cards)}, open(f"{R}/audit/mapset.json", "w"))
    p = subprocess.run([sys.executable, f"{CODE}/pipeline/qcl_queue.py", W], env={**os.environ, "QCL_ROOT": R}, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-800:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["no_numbers"] == 2 and out["cards"] == 1, out
    queued = [l.strip() for q in glob.glob(f"{W}/pending/*.txt") for l in open(q) if l.strip()]
    assert queued == ["c__quality", "c__quality"], queued            # one A queue and one B queue, the quality card only
    for n, why in (("a__size-only", "not quality metrics"), ("b__no-numbers", "no numbers")):
        a, b = (json.load(open(f"{R}/maps/{r}/{n}.map.json")) for r in "AB")
        assert a == b and a["no_quality_claims"] and why in a["no_claims_reason"] and Q.CODE_FILED in a["no_claims_reason"], (n, a)
        cands = json.load(open(f"{R}/maps/cands/{n}.cands.json"))
        assert a["source_sha256"] == cands["source_sha256"] == idx[n.replace("__", "/")]["sha256"] and a["candidates_sha256"] == cands["candidates_sha256"]
    assert not os.path.exists(f"{R}/maps/A/c__quality.map.json")

# 3. recall on the committed data: no committed quality value sits on a card the vocabulary would skip
SPEED = re.compile(r"t/s|tok(?:ens?)?/s|speed|latency|throughput|accept", re.I)   # a speed figure filed under a benchmark key is not a quality claim
checked = spared = 0; missed = []
for mp in sorted(glob.glob(f"{CODE}/models/*/maps/A/*.map.json")):
    d = json.load(open(mp)); root = mp.split("/maps/")[0]
    try: idx = json.load(open(f"{root}/snapshots/index.json"))["repos"]; text = open(os.path.join(root, idx[d["repo"]]["file"]), encoding="utf-8", errors="replace").read()
    except (KeyError, OSError): continue
    quality = any(k for e in d.get("entries", []) for k in (e.get("metrics") or {}) if not SPEED.search(k)) or bool(d.get("prose"))
    skip = Q.skip_reason({"x": 1}, text) is not None   # candidates are not part of this check: only the vocabulary is
    checked += 1
    if quality and skip: missed.append(mp)
    if not quality and skip: spared += 1
assert checked > 1000 and not missed, missed[:10]
print(f"queue skip ok: {checked} committed run-A maps checked, every quality claim kept, {spared} no-claim cards would not have been read")
