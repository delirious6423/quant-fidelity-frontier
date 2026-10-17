# Fetch agent instructions (one id shard)

You fetch Hugging Face model cards through the `mcp__Hugging_Face__hf_fs` tool and save them byte-for-byte with scripts.
NEVER type, copy or paraphrase card text yourself; only scripts write card files. Work in `models/qwen3.8-27b/` of this repo.
Let S = your shard number (two digits), e.g. 03.

1. Sizes. Run `python3 ../../qcl_stats.py ops cache/ids_S.json` (Bash). It prints a JSON list of op lists.
   For each op list, call hf_fs with {"operations": <that list>} exactly. Each result will be too large to show and is saved to a file;
   the tool message tells you the file path. After all calls, run
   `python3 ../../qcl_stats.py parse cache/sizes_S.json <path1> <path2> ...`.
   If a result is NOT saved to a file (it comes back inline), do not transcribe it: re-run that call once; if still inline, list those ids as failed.
2. Plan. `python3 ../../qcl_plan.py cache/sizes_S.json cache/plan_S.json cache/ids_S.json` then
   `python3 -c "import json;p=json.load(open('cache/plan_S.json'));[print(json.dumps(x['ops'])) for x in p]"` to see the op lists.
3. Fetch. For each plan entry in order, call hf_fs with {"operations": <ops>} exactly as printed. After each call run
   `QCL_INDEX=cache/index_S.json python3 ../../qcl_mcpsave.py cache <result_path>` (for entries with "chunk_of", pass all consecutive chunk result paths of that repo in order in one saver call).
4. Retry once anything the saver lists under "retry" or "need_chunks": stat it again in a padded batch, re-plan just those ids and fetch.
5. Final message: counts (saved, no README, failed) and the list of failed ids. Nothing else.
Do not open, read or print card contents. Do not schedule anything. Do not touch files other than cache/sizes_S.json, cache/plan_S.json, cache/cards/*, cache/index_S.json.
