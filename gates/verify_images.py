#!/usr/bin/env python3
"""Gate 1 for image-sourced claims: re-digitize pinned chart images and check each transcribed label against
its marker position. Axis calibration is detected from gridlines in the image itself."""
import colorsys, hashlib, json, os, glob, math, sys
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
        if set(point.get("px") or {}) != {"x", "kld_y", "ppl_x", "ppl_y"} or point.get("bpw") is None:
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
def evenly_spaced(lines, count):
    """The gridlines among the detected peaks: the one run of `count` lines with equal spacing (+-2 px). A dotted
    reference line (BF16 PPL) also peaks in the profile and is not a gridline; anything else is no supported chart."""
    runs = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            step = lines[j] - lines[i]
            if step < 100: continue
            run = [lines[i]]
            for k in range(1, count):
                near = [v for v in lines if abs(v - (lines[i] + k * step)) <= 2]
                if not near: break
                run.append(near[0])
            if len(run) == count: runs.append(run)
    return runs[0] if len(runs) == 1 else None

vx = gridlines(im["kld"], "x")                      # bpw 2..6
ky = gridlines(im["kld"], "y")                      # 1e-1, 1e-2
px_grid = gridlines(im["ppl"], "x")                 # weight size 20..90 GiB (PPL chart x; checked via markers only)
py = evenly_spaced(gridlines(im["ppl"], "y"), 6)    # 1.60 .. 1.35 step 0.05
if not (len(vx) == 5 and len(ky) == 2 and len(px_grid) == 8 and py):
    print("FAIL unsupported chart grid calibration", vx, ky, px_grid, py); sys.exit(1)
bpw = lambda x: 2 + (x - vx[0]) / ((vx[-1] - vx[0]) / 4)
kld = lambda y: 10 ** (-1 - (y - ky[0]) / (ky[1] - ky[0]))
ppl = lambda y: 1.60 - (y - py[0]) / ((py[-1] - py[0]) / 5) * 0.05

X = {"kld": "x", "ppl": "ppl_x"}                    # each chart has its own x axis: bpw (KLD chart), GiB (PPL chart)
gib = lambda x: 20 + (x - px_grid[0]) / ((px_grid[-1] - px_grid[0]) / 7) * 10

def hue(chart, x, y):
    """Hue of the saturated pixels around a marker (the series colour), or None when there are none."""
    win = chart[int(y)-3:int(y)+4, int(x)-3:int(x)+4].reshape(-1, 3)
    win = win[win.std(axis=1) > 20]
    if not len(win): return None
    return colorsys.rgb_to_hsv(*(win.mean(axis=0) / 255))[0] * 360

fails = n = 0
sizes = []   # (bpw, GiB at ppl_x): the two x axes must describe the same quant
for r, points in records:
    for e in points:
        px = e["px"]; n += 1
        valid_coordinates = all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                                for value in px.values())
        if valid_coordinates:
            valid_coordinates = all(3 <= px[X[key]] < chart.shape[1] - 3 and 3 <= px[key + "_y"] < chart.shape[0] - 3
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
            x, y = int(px[X[key]]), int(px[key + "_y"])
            win = chart[y-3:y+4, x-3:x+4]
            saturation = float(win.std(axis=2).mean()) if win.shape == (7, 7, 3) else float("nan")
            if not math.isfinite(saturation) or saturation < 20:
                fails += 1; print("FAIL no", key, "marker at", e["name"])
        # The two charts no longer share an x axis, so a PPL score could be read off another quant's marker. The PPL
        # marker must be the same series (colour) as the KLD marker, and sit where this quant's size puts it.
        if valid_coordinates:
            hk, hp = hue(im["kld"], px["x"], px["kld_y"]), hue(im["ppl"], px["ppl_x"], px["ppl_y"])
            if hk is None or hp is None or min(abs(hk - hp), 360 - abs(hk - hp)) > 10:
                fails += 1; print("FAIL marker series differs between the KLD and PPL charts:", e["name"], hk, hp)
            sizes.append((e["name"], e["bpw"], gib(px["ppl_x"])))
if len(sizes) >= 3:   # weight size grows linearly with bpw (same parameter count): a PPL marker of another quant falls off the line
    xs, ys = np.array([b for _, b, _ in sizes]), np.array([g for _, _, g in sizes])
    slope, intercept = np.polyfit(xs, ys, 1)
    for (name, b, g) in sizes:
        if abs(slope * b + intercept - g) > 1.5:
            fails += 1; print("FAIL PPL marker size does not match the quant's bpw:", name, round(g, 2), "GiB at", b, "bpw")
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
    checks = [("ppl", float(record["baseline"]["ppl"]), lambda v: py[0] + (1.60 - v) / 0.05 * ((py[-1] - py[0]) / 5)),
              ("kld", float(record["methodology"]["noise_floor"].split()[2]), lambda v: ky[0] + (-1 - math.log10(v)) * (ky[1] - ky[0]))]
    for img_key, val, ypos in checks:
        n += 1
        hit = dotted_row(im[img_key], ypos(val))
        if not hit: fails += 1; print("FAIL reference line not found at transcribed value", img_key, val)
print(f"image claims checked: {n}, failures: {fails}")
sys.exit(1 if fails else 0)
