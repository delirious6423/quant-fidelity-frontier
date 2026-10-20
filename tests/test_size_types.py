"""Model-level size-type summary: counts come from scanner-tagged bases; resident is never inferred from a download size."""
import json, os, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import size_types as S
rows = [dict(repo="a", size_gb=5, basis="disk"), dict(repo="a", size_gb=6, basis="disk"), dict(repo="b", size_gb=4, basis="resident"),
        dict(repo="c", size_gb=3, basis="excl_engram"), dict(repo="d", size_gb=None, basis="resident"), dict(repo="e", size_gb=2, basis=None)]
s = S.summarize(rows)
g = {x["label"]: x["repos"] for x in s["groups"]}
assert g == {"Resident / in-memory": 1, "Download / disk": 2, "Not stated": 1}, g   # a row without a size does not count; disk is not resident
assert s["ngram"] and s["resident"] == 1
assert not S.summarize([dict(repo="a", size_gb=5, basis="disk")])["ngram"]
assert S.summarize([dict(repo="a", size_gb=5, basis="disk")], "51B n-gram table kept on disk")["ngram"]
for slug in os.listdir(os.path.join(CODE, "models")):
    p = os.path.join(CODE, "models", slug, "site", "data.json")
    if os.path.exists(p):
        d = json.load(open(p))
        assert set(d["size_types"]) == {"groups", "ngram", "resident"}, slug
print("size types ok")
