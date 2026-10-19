#!/usr/bin/env python3
"""Gate 1 for image-sourced claims: re-digitize pinned chart images and check each transcribed label against
its marker position. Axis calibration is detected from gridlines in the image itself."""
import hashlib, json, os, glob, math, sys
from pathlib import Path
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

import current as CUR
records = []
try:
    active_records = CUR.load(ROOT)
except (OSError, ValueError, KeyError, TypeError) as error:
    print("FAIL invalid image claim source:", error); sys.exit(1)
for record in active_records:
    points = [entry for kind in ("variants", "cross", "ladder") for entry in (record.get(kind) or [])
              if entry.get("src") == "chart_image"]
    references = [record[key] for key in ("baseline", "baseline2")
                  if isinstance(record.get(key), dict) and record[key].get("src") == "chart_image"]
    if points or references:
        records.append((record, points))
if not records:
    print("verify_images: no active image-sourced claims for this model; nothing to check"); sys.exit(0)
if any(record["repo"] != "turboderp/Qwen3.8-Flash-Next-exl3" for record, _ in records):
    print("FAIL image-sourced claims lack a supported chart calibration"); sys.exit(1)
for record, points in records:
    for point in points:
        if set(point.get("m") or {}) != {"mean_kld", "ppl"} or point.get("mem") is not None:
            print("FAIL unsupported image-sourced metric shape:", point.get("name") or point.get("variant")); sys.exit(1)
        if set(point.get("px") or {}) != {"x", "kld_y", "ppl_y"} or point.get("bpw") is None:
            print("FAIL unsupported image-sourced coordinate shape:", point.get("name") or point.get("variant")); sys.exit(1)
        if point.get("size") is not None and point.get("basis") != "hf_branch_total":
            print("FAIL image-sourced size lacks a checked HF branch listing"); sys.exit(1)
    for key in ("baseline", "baseline2"):
        reference = record.get(key)
        if isinstance(reference, dict) and reference.get("src") == "chart_image":
            metrics = set(reference) - {"src", "L", "L_extra"}
            if key != "baseline" or metrics != {"ppl"}:
                print("FAIL unsupported image-sourced reference shape:", key); sys.exit(1)
try:
    manifest = json.loads(Path(ROOT, "cache/manifest.json").read_text())["entries"]
    for path in IMG.values():
        file = Path(path)
        if not file.is_file():
            raise ValueError(f"required chart image is missing: {file.name}")
        relative = file.relative_to(Path(ROOT)).as_posix()
        pins = [entry for entry in manifest.values() if entry.get("path") == relative and entry.get("sha256")]
        raw = file.read_bytes()
        if not pins or any(hashlib.sha256(raw).hexdigest() != pin["sha256"] for pin in pins):
            raise ValueError(f"required chart image SHA-256 differs from its pin: {file.name}")
        if any("bytes" in pin and pin["bytes"] != len(raw) for pin in pins):
            raise ValueError(f"required chart image byte count differs from its pin: {file.name}")
except (OSError, ValueError, KeyError, TypeError) as error:
    print("FAIL", error); sys.exit(1)
im = {k: np.asarray(Image.open(p).convert("RGB")).astype(float) for k, p in IMG.items()}
vx = gridlines(im["kld"], "x")                      # bpw 2..6
ky = gridlines(im["kld"], "y")                      # 1e-1, 1e-2
py = gridlines(im["ppl"], "y")                      # 1.52 .. 1.36 step 0.02
if not (len(vx) == 5 and len(ky) == 2 and len(py) == 9):
    print("FAIL unsupported chart grid calibration", vx, ky, py); sys.exit(1)
bpw = lambda x: 2 + (x - vx[0]) / ((vx[-1] - vx[0]) / 4)
kld = lambda y: 10 ** (-1 - (y - ky[0]) / (ky[1] - ky[0]))
ppl = lambda y: 1.52 - (y - py[0]) / ((py[-1] - py[0]) / 8) * 0.02

fails = n = 0
for r, points in records:
    for e in points:
        px = e["px"]; n += 1
        valid_coordinates = all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                                for value in px.values())
        if valid_coordinates:
            valid_coordinates = all(3 <= px["x"] < chart.shape[1] - 3 and 3 <= px[key + "_y"] < chart.shape[0] - 3
                                    for key, chart in im.items())
        if not valid_coordinates:
            fails += 1; print("FAIL image coordinates must be finite and inside the image:", e["name"]); continue
        k_ok = abs(kld(px["kld_y"]) / float(e["m"]["mean_kld"]) - 1) < 0.03
        p_ok = abs(ppl(px["ppl_y"]) - float(e["m"]["ppl"])) < 0.0015
        b_ok = abs(bpw(px["x"]) - e["bpw"]) < 0.01
        if not (k_ok and p_ok and b_ok):
            fails += 1; print("FAIL", e["name"], kld(px["kld_y"]), ppl(px["ppl_y"]), bpw(px["x"]))
        # Each score needs its own marker. An empty or nonfinite sample never
        # establishes evidence (NaN comparisons otherwise accidentally pass).
        for key, chart in im.items():
            x, y = int(px["x"]), int(px[key + "_y"])
            win = chart[y-3:y+4, x-3:x+4]
            saturation = float(win.std(axis=2).mean()) if win.shape == (7, 7, 3) else float("nan")
            if not math.isfinite(saturation) or saturation < 20:
                fails += 1; print("FAIL no", key, "marker at", e["name"])
# reference lines printed on the charts: BF16 PPL (ppl chart) and noise floor (kld chart)
def dotted_row(img, y_pred, win=6):
    best = None
    for y in range(int(y_pred) - win, int(y_pred) + win + 1):
        row = img[y, 300:2100].mean(1)
        light = (row > 110).astype(int)
        frac, flips = light.mean(), int(abs(light[1:] - light[:-1]).sum())
        if frac > 0.15 and flips > 80 and (best is None or flips > best[1]): best = (y, flips)
    return best
for record, _ in records:
    checks = [("ppl", float(record["baseline"]["ppl"]), lambda v: py[0] + (1.52 - v) / 0.02 * ((py[-1] - py[0]) / 8)),
              ("kld", float(record["methodology"]["noise_floor"].split()[2]), lambda v: ky[0] + (-1 - math.log10(v)) * (ky[1] - ky[0]))]
    for img_key, val, ypos in checks:
        n += 1
        hit = dotted_row(im[img_key], ypos(val))
        if not hit: fails += 1; print("FAIL reference line not found at transcribed value", img_key, val)
print(f"image claims checked: {n}, failures: {fails}")
sys.exit(1 if fails else 0)
