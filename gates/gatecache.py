"""Cache passing publication gates only for an unchanged set of exact inputs.

The fingerprint includes gate helpers, claim files, source pins and source bytes.
Changing inputs while a gate is running fails the run instead of attaching an
older result to the new inputs.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]
ROOT = os.path.abspath(os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next"))
GATES = ("verify.py", "verify_columns.py", "verify_images.py")
CACHE = f"{ROOT}/audit/.gatecache.json"


def inputs_hash(gate):
    # Resolve the root once: glob results already include it, even when a caller
    # supplied QCL_ROOT=models/<slug>. Never prepend the root a second time.
    root = Path(ROOT).resolve()
    files = {Path(CODE) / "gates" / gate, Path(CODE) / "pipeline" / "autoextract.py"}
    # current.py controls which claims are checked; gatecache.py controls reuse.
    # Include every gate helper so refactoring validation cannot preserve a hit.
    files.update(Path(CODE, "gates").rglob("*.py"))
    files.update((root / "verified").glob("*.py"))
    files.update((root / "audit").glob("*exceptions*.json"))
    files.add(root / "audit" / "review_pins.json")
    files.update(path for path in (root / "snapshots").rglob("*") if path.is_file())
    files.update((root / "cache" / "cards").rglob("*.md"))
    files.update((root / "snapshots" / "index.json", root / "cache" / "manifest.json"))
    h = hashlib.sha256()
    h.update(sys.version.encode())
    for path in sorted(files):
        # Encode names and boundaries, including missing mandatory files.
        h.update(str(path).encode()); h.update(b"\0")
        if path.is_file():
            raw = path.read_bytes()
            h.update(str(len(raw)).encode()); h.update(b"\0"); h.update(raw)
        else:
            h.update(b"missing")
        h.update(b"\0")
    return h.hexdigest()


def run_all():
    try:
        with open(CACHE) as handle:
            cache = json.load(handle)
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}

    def one(gate):
        key = inputs_hash(gate)
        cached = cache.get(gate)
        if (isinstance(cached, dict) and cached.get("key") == key
                and cached.get("rc") == 0 and isinstance(cached.get("out"), str)):
            return gate, 0, cached["out"], True, key
        process = subprocess.run([sys.executable, f"{CODE}/gates/{gate}"], capture_output=True, text=True,
                                 env={**os.environ, "QCL_ROOT": str(Path(ROOT).resolve())})
        return gate, process.returncode, process.stdout + process.stderr, False, key

    with cf.ThreadPoolExecutor(len(GATES)) as executor:
        results = list(executor.map(one, GATES))
    checked = {}
    for gate, rc, out, hit, key in results:
        if inputs_hash(gate) != key:
            rc, hit = 1, False
            out += "\nFAIL publication inputs changed while gates were running; rerun against stable inputs\n"
        if rc == 0:
            cache[gate] = {"key": key, "rc": rc, "out": out}
        else:
            cache.pop(gate, None)
        checked[gate] = (rc, out, hit)
    # A cache is optional. Atomic replacement prevents a concurrent build or
    # interruption from leaving a partially written JSON file.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(CACHE), delete=False) as handle:
            temporary = handle.name
            json.dump(cache, handle)
        os.replace(temporary, CACHE)
    except OSError:
        pass
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return checked
