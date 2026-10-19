"""Creation-time source pins for candidate tasks and independent mappings.

Legacy maps can use their already-recorded source digest. That compatibility
path never advances a pin: changed cards require a new task and new mappings.
"""
import hashlib
import json
from pathlib import Path
import re


def candidate_values(document):
    if not isinstance(document, dict):
        raise ValueError("candidate file must be an object")
    values = document.get("candidates", document)
    if not isinstance(values, dict):
        raise ValueError("candidates must be an object")
    return values


def candidate_path(mapping_path):
    mapping_path = Path(mapping_path)
    name = mapping_path.name.removesuffix(".map.json") + ".cands.json"
    adjacent = mapping_path.with_name(name)
    return adjacent if adjacent.exists() else mapping_path.parent.parent / "cands" / name


def candidates_digest(values):
    raw = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def recorded_pins(root, run):
    if run not in ("A", "B", "C"):
        return {}
    path = Path(root) / "maps" / ("mapped_c.json" if run == "C" else "mapped.json")
    try:
        values = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return values if isinstance(values, dict) else {}


def mapping_source(mapping, candidates, index, legacy=None, *, path):
    """Return the source a map actually reviewed, or reject it as stale.

    A current card hash assigned while merging cannot prove when a mapping was
    made. New maps therefore echo the digest in their candidate task. Existing
    flat candidate files are supported only at their previously recorded pin.
    """
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be an object")
    candidate_values(candidates)
    repo = mapping.get("repo")
    if not isinstance(repo, str) or Path(path).name != repo.replace("/", "__") + ".map.json":
        raise ValueError("mapping filename does not match its repository")
    pin = index.get(repo, {}).get("sha256")
    if not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin):
        raise ValueError("current source has no valid SHA-256")
    if "source_sha256" in mapping:
        if mapping["source_sha256"] != pin:
            raise ValueError("mapping was made from a different source")
        if ("candidates" not in candidates or candidates.get("repo") != repo
                or candidates.get("source_sha256") != pin):
            raise ValueError("candidate task does not pin the mapped source")
        digest = candidates_digest(candidate_values(candidates))
        if candidates.get("candidates_sha256") != digest or mapping.get("candidates_sha256") != digest:
            raise ValueError("candidate IDs changed since the mapping task")
    elif "candidates" in candidates or (legacy or {}).get(repo) != pin:
        raise ValueError("legacy mapping has no matching recorded source pin; remap this card")
    return pin
