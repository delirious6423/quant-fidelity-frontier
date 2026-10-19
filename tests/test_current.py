"""A card that changes after review withdraws its old claims (listed as not yet reviewed) instead of failing the build."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gates"))
import current as C
idx = {"a/x": {"sha": "main"}, "b/y": {"sha": "abc12345" + "f" * 32}, "c/z": {"sha": "main"}}
R = [{"repo": "a/x", "sha": "1234567890", "variants": [{"m": {"kld": "0.1"}}], "fmt": "GGUF"},     # hand entry, card changed
     {"repo": "b/y", "sha": "abc12345", "variants": [{"m": {"kld": "0.2"}}]},                        # still current
     {"repo": "c/z", "sha": "deadbeef", "variants": [], "auto": True}]                               # unreviewed listing, card changed
out = {r["repo"]: r for r in C.current(R, idx)}
fails = [n for n, ok in [
    ("stale reviewed entry withdrawn", out["a/x"].get("auto") and out["a/x"]["variants"] == [] and out["a/x"]["flags"] == [C.STALE_FLAG]),
    ("stale entry re-pinned to the current card", out["a/x"]["sha"] == "main"),
    ("current entry kept", out["b/y"]["variants"][0]["m"]["kld"] == "0.2"),
    ("unreviewed listing follows the card", out["c/z"]["sha"] == "main" and out["c/z"].get("auto"))] if not ok]
for f in fails: print("[FAIL]", f)
print(f"{len(fails)} failed"); sys.exit(1 if fails else 0)
