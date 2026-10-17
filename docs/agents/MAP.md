# Mapping worker (Haiku, high effort)

You map numbered candidates from model cards to metrics. You never type a number: you answer with candidate ids, and code
copies the published strings. Use only Read, Write and Bash. Work in the repo root.

Your task names: W = the work folder (e.g. `work/glm-5.3-flash`), RUN = `A` or `B`, M = the model's data root.

Loop:
1. Claim a queue: `f=$(ls W/pending | grep "^RUN" | head -1) && mv "W/pending/$f" "W/claimed/$f" && echo "$f"`.
   If it prints nothing, stop. If `mv` fails, another worker took it: claim again.
2. The queue file lists card NAMEs, one per line. For each NAME:
   - Read `W/tasks/NAME.prompt.txt` in full (if a Read is truncated, Read again with an offset until the end).
   - Write the JSON answer the prompt asks for to `M/maps/RUN/NAME.map.json`. Ids like `c14.2` point at one number inside
     a multi-number cell; use them.
3. Check your own files: `python3 pipeline/qcl_mapcheck.py W/claimed/<queue file> M/maps/RUN`. Fix any file it lists.
4. Go to 1.

Only write `M/maps/RUN/<NAME>.map.json` for names in queues you claimed. Never open other files. Final message: the queue
files you processed, nothing else.
