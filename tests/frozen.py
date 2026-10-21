"""Frozen Flash-Next fixtures for the regression suites.

The live model drifts every hour: when a publisher changes a card on the Hub, the claims reviewed against the old bytes
go stale, which is the gates' fail-closed behaviour and not a test subject. The suites that assert on hand-verified
Flash-Next claims therefore run on a throwaway copy in which every drifted repository is re-pinned to the card it was
reviewed against: tests/fixtures/qwen3.8-flash-next/index_entries.json holds those index entries, cards/ the bytes.
Nothing here touches models/. To add a repository after a new drift, copy its reviewed index entry and card bytes from
the last commit that carried them."""
import hashlib, json, os, shutil, tempfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
FLASH = CODE / "models/qwen3.8-flash-next"
FIXTURES = CODE / "tests/fixtures/qwen3.8-flash-next"


def freeze(root):
    """Re-pin every fixture repository inside the model root `root` (a copy). Returns the repositories re-pinned."""
    root = Path(root)
    spec = json.loads((FIXTURES / "index_entries.json").read_text())
    index_path = root / "snapshots/index.json"
    index = json.loads(index_path.read_text())
    repinned = []
    for repo, entry in spec["repos"].items():
        live = index["repos"].get(repo)
        if live is None or (live.get("sha256") == entry["sha256"] and live.get("sha") == entry["sha"] and live.get("file") == entry["file"]):
            continue
        card = FIXTURES / "cards" / os.path.basename(entry["file"])
        raw = card.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise AssertionError(f"fixture card for {repo} does not match its index entry")
        target = root / entry["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        index["repos"][repo] = dict(entry)
        repinned.append(repo)
    if repinned:
        index_path.write_text(json.dumps(index, indent=1))
    return repinned


def frozen_copy(dest=None):
    """Copy the live Flash-Next root to `dest` (or a new temporary directory) and freeze it. Returns the copy's path."""
    dest = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="qff-frozen-")) / "qwen3.8-flash-next"
    shutil.copytree(FLASH, dest, ignore=shutil.ignore_patterns("site", "site_dev", ".gatecache.json", "__pycache__"))
    freeze(dest)
    return dest
