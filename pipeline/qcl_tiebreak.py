#!/usr/bin/env python3
"""Tie-breaker (run C) for values the two independent mapping runs disagreed on.

  QCL_ROOT=models/<slug> python3 pipeline/qcl_tiebreak.py queue WORK [MAX_CARDS=40]   # queue C maps for held cards
  QCL_ROOT=models/<slug> python3 pipeline/qcl_tiebreak.py record WORK                 # after the workers: pin what C mapped

A card is queued when run A filed values that run B did not confirm (audit/review_queue.json "disagreements"), and no C
map exists for the card's current bytes (maps/mapped_c.json pins the card SHA-256 each C map was made from). The C worker
maps the card blind, exactly like A and B (docs/agents/MAP.md with RUN = C), from the same code-generated candidates, so
it cannot type a number. qcl_watch.py merge then publishes a held A value only if C filed the same published string
under the same role, metric family and panel (2 of 3 independent runs agree); every value then goes through the usual
source and column gates. Cards whose candidates changed since A and B mapped them are not queued (re-mapped first)."""
import json, os, subprocess, sys
import mapping_provenance as MP
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("QCL_ROOT") or os.path.join(CODE, "models", "qwen3.8-flash-next")
cmd, W = sys.argv[1], sys.argv[2]
MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 40
idx = json.load(open(f"{R}/snapshots/index.json"))["repos"]
mcp = f"{R}/maps/mapped_c.json"; mapped_c = json.load(open(mcp)) if os.path.exists(mcp) else {}
mapped = json.load(open(f"{R}/maps/mapped.json")) if os.path.exists(f"{R}/maps/mapped.json") else {}
sp = f"{R}/audit/skipped_maps.json"; skipped = json.load(open(sp)) if os.path.exists(sp) else {}
name = lambda rp: rp.replace("/", "__")

if cmd == "record":   # a C map counts once it is readable and names its repo
    by_name = {name(r): r for r in idx}
    for q in sorted(os.listdir(f"{W}/claimed")) if os.path.isdir(f"{W}/claimed") else []:
        if not q.startswith("C"): continue
        for n in [l.strip() for l in open(f"{W}/claimed/{q}") if l.strip()]:
            rp = by_name.get(n); f = f"{R}/maps/C/{n}.map.json"
            try:
                candidates = json.load(open(MP.candidate_path(f)))
                mapping = json.load(open(f))
                ok = rp and mapping.get("repo") == rp and MP.mapping_source(mapping, candidates, idx, mapped_c, path=f) == idx[rp]["sha256"]
            except Exception: ok = False
            if ok: mapped_c[rp] = idx[rp]["sha256"]
    os.makedirs(os.path.dirname(mcp), exist_ok=True); json.dump(mapped_c, open(mcp, "w"), indent=1, sort_keys=True)
    print(json.dumps({"c_mapped": len(mapped_c)})); sys.exit(0)

rq = json.load(open(f"{R}/audit/review_queue.json")) if os.path.exists(f"{R}/audit/review_queue.json") else {}
held = {}
for d in rq.get("disagreements", []): held[d["repo"]] = held.get(d["repo"], 0) + 1
todo = [rp for rp in sorted(held, key=lambda r: -held[r])   # most held values first
        if rp in idx and mapped.get(rp) == idx[rp]["sha256"] and mapped_c.get(rp) != idx[rp]["sha256"] and skipped.get(rp) != idx[rp]["sha256"]
        and os.path.exists(f"{R}/maps/A/{name(rp)}.map.json")]
todo = todo[:MAX]
for d in ("tasks", "pending", "claimed"): os.makedirs(f"{W}/{d}", exist_ok=True)
os.makedirs(f"{R}/maps/C", exist_ok=True)
ready = []
if todo:
    subprocess.run([sys.executable, f"{CODE}/pipeline/qcl_map.py", "tasks", f"{W}/tasks", *todo], check=True, stdout=subprocess.DEVNULL, env={**os.environ, "QCL_ROOT": R})
    for rp in todo:   # C must see exactly the candidates A and B mapped, or its ids would point at different numbers
        n = name(rp); new, old = f"{W}/tasks/{n}.cands.json", f"{R}/maps/cands/{n}.cands.json"
        if os.path.exists(old):
            new_doc, old_doc = json.load(open(new)), json.load(open(old))
            if MP.candidate_values(new_doc) == MP.candidate_values(old_doc):
                # Safe legacy upgrade: A/B's recorded source pin above already
                # equals this source, and every candidate remains identical.
                # Keep their flat file until those runs are remapped; C receives
                # its own pinned envelope beside its new map.
                os.replace(new, f"{R}/maps/C/{n}.cands.json")
                ready.append(n)
        if os.path.exists(new): os.remove(new)
for i in range(0, len(ready), 6):
    open(f"{W}/pending/C{i // 6:02d}.txt", "w").write("\n".join(ready[i:i + 6]) + "\n")
print(json.dumps({"held_cards": len(held), "queued": len(ready), "queues": -(-len(ready) // 6), "candidates_changed": len(todo) - len(ready)}))
