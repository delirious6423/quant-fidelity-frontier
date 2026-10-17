"""Run the publication gates in parallel, with results cached by a content hash of every input.
A cached result is reused only if the gate script, every claims file, the snapshot index (which holds the SHA-256 of
every card), the exception files and the chart images are byte-identical to the run that produced it."""
import concurrent.futures as cf, glob, hashlib, json, os, subprocess, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(CODE, "pipeline"), os.path.join(CODE, "gates")]   # sibling modules
ROOT = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")          # data root: one folder per model (models/<slug>/); code stays here
GATES = ("verify.py", "verify_columns.py", "verify_images.py")
CACHE = f"{ROOT}/audit/.gatecache.json"
def inputs_hash(gate):
    h = hashlib.sha256()
    files = [f"{CODE}/gates/{gate}", f"{CODE}/pipeline/autoextract.py"] + sorted(glob.glob(f"{ROOT}/verified/*.py")) + [f"{ROOT}/snapshots/index.json"] \
          + sorted(glob.glob(f"{ROOT}/audit/*exceptions*.json")) + sorted(glob.glob(f"{ROOT}/snapshots/images/*")) \
          + sorted(glob.glob(f"{ROOT}/snapshots/*.json")) + sorted(glob.glob(f"{ROOT}/cache/manifest.json")) \
          + sorted(glob.glob(f"{ROOT}/snapshots/*.md")) + sorted(glob.glob(f"{ROOT}/cache/cards/*.md"))   # the card bytes themselves
    for f in files:
        p = f if os.path.isabs(f) else f"{ROOT}/{f}"
        if os.path.exists(p): h.update(p.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()
def run_all():
    try: cache = json.load(open(CACHE))
    except Exception: cache = {}
    def one(g):
        k = inputs_hash(g); c = cache.get(g)
        if c and c["key"] == k: return g, c["rc"], c["out"], True
        p = subprocess.run(["python3", f"{CODE}/gates/{g}"], capture_output=True, text=True)
        return g, p.returncode, p.stdout + p.stderr, False
    with cf.ThreadPoolExecutor(len(GATES)) as ex: res = list(ex.map(one, GATES))
    for g, rc, out, hit in res:
        if not hit and rc == 0: cache[g] = {"key": inputs_hash(g), "rc": rc, "out": out}   # only passing results are cached
        elif rc != 0: cache.pop(g, None)
    try: json.dump(cache, open(CACHE, "w"))
    except Exception: pass
    return {g: (rc, out, hit) for g, rc, out, hit in res}
