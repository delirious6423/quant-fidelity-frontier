#!/usr/bin/env python3
"""One-time migration of legacy mutable-reference reviews to exact content pins.

python3 pipeline/qcl_pin_reviews.py --initialize-baseline models/<slug>

Creates audit/review_pins.json without rewriting a claim or a source. Every pin
binds the full entry fingerprint and current source digest. All publication
gates run against a throwaway copy containing the candidate registry before it
can be installed. An existing registry cannot be expanded or repinned: later
reviews must use the normal digest-pinned mapping writer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "gates"))
import current
import gatecache


def baseline_registry(root):
    root = Path(root).resolve()
    index = json.loads((root / "snapshots/index.json").read_text())["repos"]
    entries, sources = {}, {}
    for file in sorted((root / "verified").glob("part*.py")):
        namespace = {}; exec(file.read_text(), namespace)
        for entry in namespace["R"]:
            groups = [item for kind in ("variants", "cross", "ladder") for item in (entry.get(kind) or [])]
            groups += [entry[key] for key in ("baseline", "baseline2") if isinstance(entry.get(key), dict)]
            if any(item.get("src") == "unsloth_docs" for item in groups):
                external = "snapshots/unsloth.ai__docs__qwen3.8-next.md"
                raw = (root / external).read_bytes()
                sources[external] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            if entry.get("auto") or "source_sha256" in entry or entry.get("sha") != "main": continue
            meta = index.get(entry["repo"], {})
            if meta.get("sha") != "main": continue
            path = (root / meta["file"]).resolve()
            if not path.is_relative_to(root): raise ValueError("source outside model directory")
            raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
            if digest != meta.get("sha256"): raise ValueError(f"source pin mismatch: {entry['repo']}")
            key = current.entry_fingerprint(entry)
            entries[key] = {"repo": entry["repo"], "source_sha256": digest}
    return {"version": 1, "entries": entries, "sources": sources}


def migrate(root):
    root = Path(root).resolve()
    output = root / "audit/review_pins.json"
    candidate = baseline_registry(root)
    if output.exists() and json.loads(output.read_text()) != candidate:
        raise ValueError("Existing baseline pins cannot be expanded or repinned; use a new source-reviewed mapping")
    gatecache.ROOT = str(root)
    before = {gate: gatecache.inputs_hash(gate) for gate in gatecache.GATES}
    with tempfile.TemporaryDirectory(prefix="qff-pin-baseline-") as folder:
        stage = Path(folder) / root.name
        shutil.copytree(root, stage, ignore=shutil.ignore_patterns("site", "site_dev", "maps", "work", "__pycache__", ".gatecache.json"))
        (stage / "audit").mkdir(exist_ok=True)
        (stage / "audit/review_pins.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
        for gate in gatecache.GATES:
            result = subprocess.run([sys.executable, str(CODE / "gates" / gate)],
                                    env={**os.environ, "QCL_ROOT": str(stage)}, capture_output=True, text=True)
            if result.returncode:
                raise ValueError(f"Baseline gate {gate} failed:\n{result.stdout}{result.stderr}")
    if any(gatecache.inputs_hash(gate) != before[gate] for gate in gatecache.GATES):
        raise ValueError("Baseline inputs changed during migration; no pins were installed")
    if not output.exists():
        # Exclusive creation makes it impossible to replace a concurrent review.
        with output.open("x") as handle:
            json.dump(candidate, handle, indent=2, sort_keys=True); handle.write("\n")
    return len(candidate["entries"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-baseline", action="store_true", required=True)
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps({"model": Path(args.root).name, "review_pins": migrate(args.root), "gates": "passed"}))
