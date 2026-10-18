# Fetch agent instructions (one id shard)

Use only Bash and the hf_fs tool. You fetch Hugging Face model cards through the `mcp__Hugging_Face__hf_fs` tool and save them byte-for-byte with scripts.
NEVER type, copy or paraphrase card text yourself; only scripts write card files. Work in the repo root. M = the model's data root given in your task (e.g. `models/glm-5.3-flash`); every `M/cache/` path below is under M.
Let S = your shard name exactly as given in your task (e.g. `2026101805a`); the shard's ids are in `M/cache/ids_S.json`.

1. Sizes. Run `python3 pipeline/qcl_stats.py ops M/cache/ids_S.json` (Bash). It prints a JSON list of op lists.
   For each op list, call hf_fs with {"operations": <that list>} exactly. Each result will be too large to show and is saved to a file;
   the tool message tells you the file path. After all calls, run
   `python3 pipeline/qcl_stats.py parse M/cache/sizes_S.json <path1> <path2> ...`.
   If a result is NOT saved to a file (it comes back inline), do not transcribe it: re-run that call once; if still inline, list those ids as failed.
2. Plan. `python3 pipeline/qcl_plan.py M/cache/sizes_S.json M/cache/plan_S.json M/cache/ids_S.json` then
   `python3 -c "import json;p=json.load(open('M/cache/plan_S.json'));[print(json.dumps(x['ops'])) for x in p]"` to see the op lists.
3. Fetch. For each plan entry in order, call hf_fs with {"operations": <ops>} exactly as printed. After each call run
   `QCL_INDEX=M/cache/index_S.json python3 pipeline/qcl_mcpsave.py M/cache <result_path>` (for entries with "chunk_of", pass all consecutive chunk result paths of that repo in order in one saver call).
4. Retry once anything the saver lists under "retry" or "need_chunks": stat it again in a padded batch, re-plan just those ids and fetch.
5. Final message: counts (saved, no README, failed) and the list of failed ids. Nothing else.
Do not open, read or print card contents. Do not schedule anything. Do not touch files other than M/cache/sizes_S.json, M/cache/plan_S.json, M/cache/cards/*, M/cache/index_S.json.
