# pipeline/

Ingestion and scan scripts, indexed by stage. Each description is the first sentence of the script's module docstring; each command is the first entry command that docstring shows, with the `pipeline/` path added. Commands run from the repository root.

## Collection

| Script | What it does | Entry command |
|---|---|---|
| `qcl_fetch.py` | Fetch-once source ledger for one base model, straight from the Hugging Face API (no browser). | `python3 pipeline/qcl_fetch.py ORG/MODEL [--out DIR] [--workers 8] [--base https://huggingface.co]` |
| `qcl_catalogue.py` | Hub catalogue and delta for one model, from saved hf_fs search results (no model ever retypes them). | `python3 pipeline/qcl_catalogue.py models/<slug> RESULT_FILE...` |
| `qcl_stats.py` | Card-size lookups through the connector, padded so results are saved to disk (no model transcribes anything). | `python3 pipeline/qcl_stats.py ops IDS.json` |
| `qcl_plan.py` | Plan connector batches so every result is (a) within the connector's ~80 KB budget and (b) large enough (&gt;= 52 KB) to be saved to disk instead of returned inline. | `python3 pipeline/qcl_plan.py SIZES.json OUT_PLAN.json [IDS.json]` |
| `qcl_mcpsave.py` | Save model cards from a persisted Hugging Face connector result (hf_fs batch) byte-for-byte. | `python3 pipeline/qcl_mcpsave.py OUT_DIR RESULT_FILE [RESULT_FILE ...]` |
| `qcl_newmodel.py` | Turn a fetched card cache into a model data root the pipeline reads (QCL_ROOT=models/&lt;slug&gt;). | `python3 pipeline/qcl_newmodel.py models/<slug>` |
| `runner_fetch.py` | Read-only HTTP collection into a portable ledger and isolated shadow model root. |  |
| `qcl_discover.py` | Weekly discovery: find new base models, count their quants, admit the ones that pass the rules, re-rank the queue. | `python3 pipeline/qcl_discover.py` |
| `qcl_rank.py` | Order the queued models in watchlist.json by a weighted score: how much there is to compare, and how new it is. | `python3 pipeline/qcl_rank.py` |

## Mapping

| Script | What it does | Entry command |
|---|---|---|
| `autoextract.py` | Deterministic first-pass extractor + evaluation against the hand/agent-verified claims. | `python3 pipeline/autoextract.py eval` |
| `qcl_map.py` | Map-don't-type extraction. | `python3 pipeline/qcl_map.py tasks OUT_DIR REPO...` |
| `qcl_mapset.py` | Which cards still need an LLM mapping, after triage and copy detection. | `QCL_ROOT=models/<slug> python3 pipeline/qcl_mapset.py` |
| `qcl_queue.py` | Prepare mapping work for the cards in audit/mapset.json "todo": task prompts, candidate files, and queues for two independent runs (A and B group the cards differently). | `QCL_ROOT=models/<slug> python3 pipeline/qcl_queue.py work/<slug> [KB_PER_QUEUE=120]` |
| `qcl_claim.py` | Claim one mapping queue for a worker, atomically. | `python3 pipeline/qcl_claim.py WORK RUN` |
| `qcl_mapcheck.py` | Check one claimed mapping queue: every name has a parseable map in the run folder, naming the right repo, and using only candidate ids that exist for that card. | `python3 pipeline/qcl_mapcheck.py WORK/claimed/QUEUE.txt MODEL/maps/RUN` |
| `qcl_tiebreak.py` | Tie-breaker (run C) for values the two independent mapping runs disagreed on. | `QCL_ROOT=models/<slug> python3 pipeline/qcl_tiebreak.py queue WORK [MAX_CARDS=40]` |
| `qcl_adjudicate.py` | Compare a mapping run with the verified data and classify every disagreement mechanically. | `python3 pipeline/qcl_adjudicate.py MAPDIR [MAPDIR2]` |
| `qcl_rules.py` | Zero-LLM mapper: learned header dictionary + label rules, evaluated against the verified data. | `python3 pipeline/qcl_rules.py eval` |
| `detect_copies.py` | Mechanical copied-card detection: for every card, the share of its numeric claim lines that also appear verbatim in ONE card from a DIFFERENT owner whose repo was CREATED earlier. |  |
| `mapping_provenance.py` | Creation-time source pins for candidate tasks and independent mappings. |  |
| `gen_auto.py` | Generate verified/part9_auto.py: a catalogue entry for every snapshotted repo that no hand/agent extraction covers. |  |
| `qcl_fmt.py` | Format family of a repo from its id (and tags when known). |  |

## Merge and publish

| Script | What it does | Entry command |
|---|---|---|
| `qcl_watch.py` | Incremental ingestion: the work happens before anyone asks for the site. | `python3 pipeline/qcl_watch.py delta INDEX_NEW.json` |
| `qcl_live.py` | Print the data root of every model that is published (status "live" in watchlist.json; a model folder that is not in the watch list counts as live). | `for d in $(python3 pipeline/qcl_live.py); do QCL_ROOT=$d python3 web/build.py; done` |
| `readme_sync.py` | Keep README.md's facts generated, not typed. | `python3 pipeline/readme_sync.py` |
| `export_hf_dataset.py` | Export assembled public data to a reviewable HF dataset directory; never upload. |  |
| `deploy_state.py` | Fail-closed QA review and release receipts; no subprocess or network calls. |  |
| `qcl_benchinv.py` | Inventory of the task benchmarks that model cards report (machine-verified `bench:*` fields), grouped by canonical task. | `python3 pipeline/qcl_benchinv.py` |
| `qcl_pin_reviews.py` | One-time migration of legacy mutable-reference reviews to exact content pins. | `python3 pipeline/qcl_pin_reviews.py --initialize-baseline models/<slug>` |
| `merge_watchlist.py` | git merge driver for watchlist.json, so the live-update and new-model tasks can both commit it. | `git config merge.wljson.driver "python3 pipeline/merge_watchlist.py %O %A %B"` |

## Scheduled-task state machines

| Script | What it does | Entry command |
|---|---|---|
| `qcl_next.py` | The hourly scan as a state machine: the coordinating session only does what this prints, then runs it again. | `python3 pipeline/qcl_next.py` |
| `qcl_scan.py` | Deterministic steps of the hourly scan (docs/runbooks/HOURLY.md). | `python3 pipeline/qcl_scan.py lock` |
| `qcl_qa.py` | QA worker: the only job that publishes the site. | `python3 pipeline/qcl_qa.py` |

## Scripted runner

| Script | What it does | Entry command |
|---|---|---|
| `qcl_runner.py` | Scripted, local-only shadow pipeline. |  |
| `qcl_portable.py` | The QFF scan for any agent: a loop of plain commands and JSON files, no connector, subagent tool or provider needed. | `python3 pipeline/qcl_portable.py start --workspace DIR --models SLUG [SLUG ...] [--remap]` |
| `runner_state.py` | Local durable task journal. |  |
| `runner_workspace.py` | Confined local shadow workspace; no publication capability or remote exists here. |  |
| `runner_model.py` | Pure, vendor-neutral request/response boundary for the shadow runner. |  |
| `runner_replay.py` | Offline frozen-input replay and old/new extraction shadow comparison. |  |
| `runner_baseline.py` | Bounded replay of the *actual* legacy dispatch branches, never its entry point. |  |
| `runner_compare.py` | Value-level comparison of one model's merged claims in a shadow workspace against the committed claims of the same model. | `python3 pipeline/runner_compare.py WORKSPACE/models/SLUG models/SLUG` |
