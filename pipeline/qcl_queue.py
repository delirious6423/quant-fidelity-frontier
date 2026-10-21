#!/usr/bin/env python3
"""Prepare mapping work for the cards in audit/mapset.json "todo": task prompts, candidate files, and queues for two
independent runs (A and B group the cards differently).

  QCL_ROOT=models/<slug> python3 pipeline/qcl_queue.py work/<slug> [KB_PER_QUEUE=120]

Writes WORK/tasks/*.prompt.txt, MODEL/maps/cands/*.cands.json (kept with the model, needed to re-apply maps), and
WORK/pending/A00.txt, B00.txt, ... Workers follow docs/runbooks/agents/MAP.md."""
import json, os, random, subprocess, sys
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")
W = sys.argv[1]; KB = int(sys.argv[2]) if len(sys.argv) > 2 else 120
todo = json.load(open(f"{R}/audit/mapset.json"))["todo"]
for d in ("tasks", "pending", "claimed"): os.makedirs(f"{W}/{d}", exist_ok=True)
os.makedirs(f"{R}/maps/cands", exist_ok=True); os.makedirs(f"{R}/maps/A", exist_ok=True); os.makedirs(f"{R}/maps/B", exist_ok=True)
if todo:
    subprocess.run([sys.executable, f"{CODE}/pipeline/qcl_map.py", "tasks", f"{W}/tasks", *todo], check=True, stdout=subprocess.DEVNULL)
names = [r.replace("/", "__") for r in todo]
for n in names: os.replace(f"{W}/tasks/{n}.cands.json", f"{R}/maps/cands/{n}.cands.json")
# a card with no numbers at all, or whose text never names a quality metric, has nothing for a worker to map: code writes
# its "no claims" map for both runs instead of queueing it (qcl_map.skip_reason; a worker cannot answer an empty prompt)
sys.path.insert(0, os.path.join(CODE, "pipeline")); import qcl_map as QM
idx = json.load(open(f"{R}/snapshots/index.json"))["repos"]
empty = []
for rp, n in zip(todo, names):
    cands = json.load(open(f"{R}/maps/cands/{n}.cands.json"))
    text = open(os.path.join(R, idx[rp]["file"]), encoding="utf-8", errors="replace").read()
    why = QM.skip_reason(cands.get("candidates") or {}, text)
    if why:
        m = QM.no_claims_map(rp, cands); m["no_claims_reason"] = why
        for run in ("A", "B"): json.dump(m, open(f"{R}/maps/{run}/{n}.map.json", "w"), indent=1)
        empty.append(n)
names = [n for n in names if n not in empty]
size = {n: os.path.getsize(f"{W}/tasks/{n}.prompt.txt") for n in names}
nq = 0
for run, seed in (("A", 1), ("B", 2)):
    order = sorted(names, key=lambda n: -size[n]); random.Random(seed).shuffle(order) if run == "B" else None
    k = max(1, -(-sum(size.values()) // (KB * 1000))); bins = [[0, []] for _ in range(k)]
    for n in order: b = min(bins, key=lambda b: b[0]); b[0] += size[n]; b[1].append(n)
    for i, (_, ns) in enumerate(b for b in bins if b[1]):
        open(f"{W}/pending/{run}{i:02d}.txt", "w").write("\n".join(ns) + "\n"); nq += 1
print(json.dumps({"cards": len(names), "queues": nq, "kb": round(sum(size.values()) / 1000), "no_numbers": len(empty)}))
