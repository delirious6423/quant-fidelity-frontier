# Mapping worker (Haiku, high effort)

You map numbered candidates from model cards to metrics. You never type a number: you answer with candidate ids, and code
copies the published strings. Use only Read, Write and Bash. Work in the repo root.

Your task names: W = the work folder (e.g. `work/glm-5.3-flash`), RUN = `A`, `B` or `C` (C is the tie-breaker: the same task, mapped blind, for cards the first two runs disagreed on), M = the model's data root.

Loop:
1. Check the run's time: `python3 pipeline/qcl_scan.py elapsed 45`. If it exits 1, stop now (the next hourly run picks
   up the queues you leave). Then claim a queue: `python3 pipeline/qcl_claim.py W RUN`. It prints the claimed queue path. If it prints nothing (exit
   code 2), no queue is left for your run: stop. Never move or rename files yourself.
2. The queue file lists card NAMEs, one per line. For each NAME:
   - Read `W/tasks/NAME.prompt.txt` in full (if a Read is truncated, Read again with an offset until the end).
   - Write the JSON answer the prompt asks for to `M/maps/RUN/NAME.map.json`. Ids like `c14.2` point at one number inside
     a multi-number cell; use them.
3. Check your own files: `python3 pipeline/qcl_mapcheck.py <claimed queue path> M/maps/RUN`. Fix any file it lists, then check again.
4. Go to 1.

Only write `M/maps/RUN/<NAME>.map.json` for names in queues you claimed. Never open other files. Final message: the queue
files you processed, nothing else.
