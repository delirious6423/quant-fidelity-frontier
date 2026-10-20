"""A highlight row that vanished (its card changed) must leave that table out, not stop the model's build."""
import html, os, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
src = open(os.path.join(CODE, "models", "qwen3.8-flash-next", "highlights.py")).read()
ns = {"rows": [], "repos": [], "E": html.escape, "fmtv": str, "BASIS_LABEL": {}}
exec(src, ns)
out = "".join(ns["render_highlights"]())   # no rows at all: every lookup comes back empty
assert "Same file, different numbers" in out and "<tbody></tbody>" in out
# one surviving claim is not a disagreement; two are
row = lambda by, v: {"repo": "u/x", "measured_by": by, "variant": "UD-Q2_K_XL", "kind": "own", "metrics": {"mean_kld": {"raw": v, "v": float(v)}},
                     "src": "https://huggingface.co/u/x", "src_label": "card"}
ns["rows"][:] = [row("u/x", "0.1")]
ns["find"]  # exists
D = "".join(ns["render_highlights"]())
assert D.count("<tr><td>") == 0, "a single claim must not be shown as a disagreement"
print("highlights soft ok")
