"""Which verified entries still describe the card the index pins now.

A reviewed entry is pinned to the exact source content it was read from. Legacy
immutable commit prefixes remain supported; mutable references require an exact
content pin. When a scan saves a changed card, its old claims are withdrawn and
the repo is listed as not yet reviewed until a new mapping covers it."""
import glob, hashlib, json, os, re

STALE_FLAG = "Card changed after it was reviewed; its earlier claims are withdrawn until it is reviewed again"

def fresh(r, idx):
    m = idx.get(r["repo"])
    if not m: return False
    pin = r.get("source_sha256") if "source_sha256" in r else r.get("_review_source_sha256")
    if pin is not None:
        return isinstance(pin, str) and bool(re.fullmatch(r"[a-f0-9]{64}", pin)) and pin == m.get("sha256")
    # Only immutable commit prefixes can establish legacy freshness without a
    # content pin. Empty values and mutable branch names never establish it.
    revision, current_revision = r.get("sha"), m.get("sha")
    return (isinstance(revision, str) and bool(re.fullmatch(r"[a-fA-F0-9]{8,40}", revision))
            and isinstance(current_revision, str) and bool(re.fullmatch(r"[a-fA-F0-9]{40}", current_revision))
            and current_revision.lower().startswith(revision.lower()))

def entry_fingerprint(entry):
    """Bind a migration pin to the entire reviewed entry, not just its repo."""
    value = {key: item for key, item in entry.items() if key not in ("_part", "_review_source_sha256")}
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def current(R, idx):
    """R: entries from every verified/part*.py (each may carry _part). Returns the entries to use."""
    fresh_reviewed = {r["repo"] for r in R if not r.get("auto") and fresh(r, idx)}
    out, stale = [], {}
    for r in R:
        if r.get("auto"):   # no claims: an unreviewed listing simply follows the current card
            if r["repo"] not in fresh_reviewed: out.append(r if fresh(r, idx) or r["repo"] not in idx else {**r, "sha": idx[r["repo"]]["sha"]})
        elif fresh(r, idx): out.append(r)
        elif r["repo"] in idx: stale.setdefault(r["repo"], r)
    have = {r["repo"] for r in out}
    for rp, r in sorted(stale.items()):
        if rp in have: continue
        out.append({"repo": rp, "sha": idx[rp]["sha"], "fmt": r.get("fmt"), "runtime": r.get("runtime"), "weights_modified": r.get("weights_modified"),
                    "method": None, "methodology": None, "variants": [], "auto": True, "flags": [STALE_FLAG], "_part": "stale"})
    return out

def load_entries(root):
    """Load raw entries and attach only exact legacy-review pin matches."""
    pin_file = f"{root}/audit/review_pins.json"
    pins = json.load(open(pin_file)) if os.path.exists(pin_file) else {"version": 1, "entries": {}}
    if pins.get("version") != 1 or not isinstance(pins.get("entries"), dict):
        raise ValueError("Unsupported legacy review pin registry")
    R = []
    for p in sorted(glob.glob(f"{root}/verified/part*.py")):
        ns = {}; exec(open(p).read(), ns)
        for r in ns["R"]:
            r.pop("_review_source_sha256", None)  # this field is supplied only by the registry
            if not r.get("auto") and "source_sha256" not in r:
                pin = pins["entries"].get(entry_fingerprint(r))
                if isinstance(pin, dict) and pin.get("repo") == r.get("repo"):
                    r["_review_source_sha256"] = pin.get("source_sha256")
            r["_part"] = os.path.basename(p); R.append(r)
    return R

def load(root):
    idx = json.load(open(f"{root}/snapshots/index.json"))["repos"]
    return current(load_entries(root), idx)
