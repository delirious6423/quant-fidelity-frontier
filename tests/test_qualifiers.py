"""Machine mapping keeps qualifiers: a value printed as "~5%", "about 357 GB" or "309.2 GB – 323.4 GB" is never
published as a bare point. Exit 1 on any failure."""
import os, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]
os.environ.setdefault("QCL_ROOT", os.path.join(CODE, "models", "qwen3.8-flash-next"))
import qcl_map as Q
fails = []
def check(c, name): print(("[PASS] " if c else "[FAIL] ") + name); c or fails.append(name)
for raw, line, want in [
    ("5%", "error reaches ~5% at the last layer", False), ("0.12", "rel_err≈0.12", False), ("357 GB", "download is about 357 GB", False),
    ("0.5", "KL <0.5 on wiki", False), ("2.5", "a 12.5 b", False), ("12", "a 12.5 b", False),
    ("91.035", "    <td>91.035</td>", True), ("415 GB", "**510 GB -> 415 GB (19% smaller).**", True), ("82.3%", "top-1 82.3% on code", True)]:
    check(Q.exact_in(raw, line) is want, f"exact_in({raw!r}, {line!r}) is {want}")
for raw, want in [("~462 GB", True), ("309.2 GB – 323.4 GB", True), ("about 426 GB", True), ("< 3 GB", True), ("462 GB", False), ("<td>4</td>", False), ("Q4_K-M 5", False)]:
    check(bool(Q.APPROX.search(raw)) is want, f"APPROX({raw!r}) is {want}")
print(f"{len(fails)} failed"); sys.exit(1 if fails else 0)
