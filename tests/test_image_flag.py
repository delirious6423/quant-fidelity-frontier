"""Cards whose images may hold quant results or details are flagged by code (pipeline/qcl_map.py image_flag)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
import qcl_map as Q

chart = ["# Quants", "![qb_kld](https://cdn-uploads.huggingface.co/production/uploads/x/a.png)", '<img src="https://host/b.jpg" alt="ppl">']
f = Q.image_flag(chart, False)
assert f and "No numbers in the card text" in f and "2 images" in f and "L2, L3" in f and "qb_kld" in f, f
g = Q.image_flag(chart, True)
assert g and g.startswith("Card embeds 2 images"), g
assert Q.image_flag(["![badge](https://img.shields.io/badge/x-y-green)", "![logo](logo.png)", "<img src='https://x/y.svg'>"], False) is None
assert Q.image_flag(["plain text, no images"], True) is None
assert [n for n, _, _ in Q.card_images(chart)] == [2, 3]
print("image flag ok")
