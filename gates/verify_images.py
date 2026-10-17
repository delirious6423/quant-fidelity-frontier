#!/usr/bin/env python3
"""Gate 1 for image-sourced claims: re-digitize pinned chart images and check each transcribed label against
its marker position. Axis calibration is detected from gridlines in the image itself."""
import os, glob, math, sys
import numpy as np
from PIL import Image

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]   # sibling modules
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")          # data root: one folder per model (models/<slug>/); code stays here
IMG = {"kld": f"{ROOT}/snapshots/images/turboderp_qb_kld.png", "ppl": f"{ROOT}/snapshots/images/turboderp_qb_ppl.png"}

def gridlines(img, axis):
    g = img.mean(2)
    prof = np.median(g[200:1400:7, :], axis=0) if axis == "x" else np.median(g[:, 300:2100:7], axis=1)
    bg = np.median(prof[200:1400]); out = []
    for i in range(1, len(prof) - 1):
        if prof[i] > bg + 6 and prof[i] >= prof[i-1] and prof[i] >= prof[i+1]:
            if out and i - out[-1][-1] <= 2: out[-1].append(i)
            else: out.append([i])
    return [float(np.mean(o)) for o in out]

if not all(os.path.exists(p) for p in IMG.values()):        # this model has no image-sourced claims
    print("verify_images: no pinned chart images for this model; nothing to check"); sys.exit(0)
im = {k: np.asarray(Image.open(p).convert("RGB")).astype(float) for k, p in IMG.items()}
vx = gridlines(im["kld"], "x")                      # bpw 2..6
ky = gridlines(im["kld"], "y")                      # 1e-1, 1e-2
py = gridlines(im["ppl"], "y")                      # 1.52 .. 1.36 step 0.02
assert len(vx) == 5 and len(ky) == 2 and len(py) == 9, (vx, ky, py)
bpw = lambda x: 2 + (x - vx[0]) / ((vx[-1] - vx[0]) / 4)
kld = lambda y: 10 ** (-1 - (y - ky[0]) / (ky[1] - ky[0]))
ppl = lambda y: 1.52 - (y - py[0]) / ((py[-1] - py[0]) / 8) * 0.02

ns = {}; exec(open(f"{ROOT}/verified/part3_images.py").read(), ns)
fails = n = 0
for r in ns["R"]:
    for e in r["variants"] + r.get("cross", []):
        px = e["px"]; n += 1
        k_ok = abs(kld(px["kld_y"]) / float(e["m"]["mean_kld"]) - 1) < 0.03
        p_ok = abs(ppl(px["ppl_y"]) - float(e["m"]["ppl"])) < 0.0015
        b_ok = abs(bpw(px["x"]) - e["bpw"]) < 0.01
        if not (k_ok and p_ok and b_ok):
            fails += 1; print("FAIL", e["name"], kld(px["kld_y"]), ppl(px["ppl_y"]), bpw(px["x"]))
        # marker must actually be at px: sample a small window for a saturated marker colour
        win = im["kld"][int(px["kld_y"])-3:int(px["kld_y"])+4, int(px["x"])-3:int(px["x"])+4]
        if win.std(axis=2).mean() < 20: fails += 1; print("FAIL no marker at", e["name"])
# reference lines printed on the charts: BF16 PPL (ppl chart) and noise floor (kld chart)
def dotted_row(img, y_pred, win=6):
    best = None
    for y in range(int(y_pred) - win, int(y_pred) + win + 1):
        row = img[y, 300:2100].mean(1)
        light = (row > 110).astype(int)
        frac, flips = light.mean(), int(abs(light[1:] - light[:-1]).sum())
        if frac > 0.15 and flips > 80 and (best is None or flips > best[1]): best = (y, flips)
    return best
r0 = ns["R"][0]
checks = [("ppl", float(r0["baseline"]["ppl"]), lambda v: py[0] + (1.52 - v) / 0.02 * ((py[-1] - py[0]) / 8)),
          ("kld", float(r0["methodology"]["noise_floor"].split()[2]), lambda v: ky[0] + (-1 - math.log10(v)) * (ky[1] - ky[0]))]
for img_key, val, ypos in checks:
    n += 1
    hit = dotted_row(im[img_key], ypos(val))
    if not hit: fails += 1; print("FAIL reference line not found at transcribed value", img_key, val)
print(f"image claims checked: {n}, failures: {fails}")
sys.exit(1 if fails else 0)
