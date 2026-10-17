# Hourly scan runbook

The scheduled task "QFF hourly scan" follows this file, top to bottom. It keeps every `live` model in `watchlist.json`
current, and adds at most one `queued` model per `min_hours_between_ingests` (5 h), in list order.

Division of labour:
- **You** (the coordinating session) run the `pipeline/qcl_scan.py` commands, make the Hub search calls and start agents.
  Every `qcl_scan.py` command prints one JSON line. **Exit code 0 means go on, 2 means nothing to do for this model, and
  1 means stop and report.**
- **Haiku agents** (model `haiku`, effort `high`) fetch cards and map them. Never use Sonnet or Opus for these.

Read `CLAUDE.md` first. These rules override everything below:
- Never type a number, a repo id or card text yourself. Scripts copy them.
- A gate failure means quarantine, never a hand edit.
- Commit only after `publish` exits 0.
- Never force-push.
- Never publish the Hugging Face MCP Space.

## Subagents per run

| When | Agents | Count | Job | Brief |
|---|---|---|---|---|
| Nothing changed (most hours) | none | 0 | — | — |
| Delta with changed cards | fetch | 1 per 100 changed cards (usually 1) | stat, plan, fetch and save cards byte-for-byte | `docs/agents/FETCH.md` |
| | mapping | 1 per queue, at most 8 at once: queues come in A/B pairs, so usually 2 | map candidates to metrics; two runs, A and B | `docs/agents/MAP.md` |
| Ingest a new model (about 100–250 quants) | fetch | 2–3 (one per 100 cards), all at once | as above | `docs/agents/FETCH.md` |
| | mapping | about 4–20 queues (half A, half B), at most 8 workers at once; each drains queues until none are left | as above | `docs/agents/MAP.md` |

There is no resolver and no reviewer agent in the hourly loop. A value that the two runs disagree on, or that fails a
gate, is held back (`models/<slug>/audit/review_queue.json`) and never published. Keep at most 8 agents running at once
(the session cap is 20). Start each agent with a prompt like the ones in step 4 or step 5.

## 1. Set up (every run)
```bash
# attach delirious6423/quant-fidelity-frontier with push access, then:
git clone --depth 50 https://github.com/delirious6423/quant-fidelity-frontier qff && cd qff
git config user.name Claude && git config user.email noreply@anthropic.com
pip install --break-system-packages pillow numpy "mcp>=1.9" playwright >/dev/null 2>&1 || pip install pillow numpy "mcp>=1.9" playwright
```
Playwright's Chromium is preinstalled in cloud sessions; never run `playwright install` there.

## 2. Lock and plan
```bash
python3 pipeline/qcl_scan.py lock      # exit 3 = another run started under 2 h ago: stop now, report "skipped (locked)"
python3 pipeline/qcl_scan.py plan
```
`plan` prints `delta` (the live slugs) and `ingest` (a slug, or null). For an ingest it creates `models/<slug>/model.json`.
Push the lock only once there is real work: when `plan` returns an `ingest`, or as soon as a `catalogue` step (step 3)
returns shards. Then run `git add scan.lock && git commit -qm "scan lock" && git push -q`, so a parallel run sees it. If
the push is rejected, run `git pull --rebase -q` and push again; if `scan.lock` came in from the remote with a fresh time,
stop. In an hour with nothing to do, nothing is committed: run `python3 pipeline/qcl_scan.py unlock` and stop.

## 3. Catalogue (each slug in `delta`, then the `ingest` slug)
1. `python3 pipeline/qcl_scan.py search-ops <slug>` (add `--full` for the ingest slug). It prints `op_lists`.
2. Call `hf_fs` once per op list, with the list **exactly** as printed. Each result is saved to a file, and the tool
   message gives its path. If a result comes back inline instead, run that call once more. Never copy inline output.
3. `python3 pipeline/qcl_scan.py catalogue <slug> [--full] <all saved result paths>`
   - exit 2: nothing changed for this slug, so go to the next one.
   - exit 1 with `page_full`: more than 80 quants changed since the last scan. Redo steps 1–3 with `--full`.
   - exit 0: note the `shards` list.

## 4. Fetch (each slug with shards)
Start one Haiku agent per shard, all at once:
> Follow docs/agents/FETCH.md in the repo at <absolute repo path>, with S = `<shard>` and M = `models/<slug>`.
> Use only Bash and the hf_fs tool.

Wait for every agent to finish. An agent's failed ids are either gated (`HF_FS_ACCESS_DENIED`: fine, they are recorded
as not accessible) or need one more agent run on the same shard.

## 5. Map (each slug that fetched cards)
1. `python3 pipeline/qcl_scan.py prepare <slug>`. Exit 2 means no cards with metrics changed, so go to step 6 for this
   slug.
2. Otherwise start Haiku workers, half with RUN = A and half with RUN = B, at most 8 at once:
   > Follow docs/agents/MAP.md in the repo at <absolute repo path>, with W = `work/<slug>`, RUN = `A`, M = `models/<slug>`.

   When a worker finishes and `work/<slug>/pending` still has files for its run, start another worker for that run.
3. `python3 pipeline/qcl_scan.py finish <slug>`
   - exit 1 with `bad`: those queues are back in `pending`. Run workers for them, then run `finish` again (at most twice).
     If they still fail, leave them. Those cards are listed as "not yet reviewed".
   - exit 0: `merge` reports the values published and the entries quarantined.

## 6. Publish
```bash
python3 pipeline/qcl_scan.py publish [--ingested <slug>]   # builds every live model, assembles, runs ALL tests
```
- **exit 1:** do not commit the scan. Discard it and release the lock:
  `git checkout -- . && git clean -fdq && python3 pipeline/qcl_scan.py unlock && git add -A scan.lock && git commit -qm "scan unlock (failed run)" && git push -q`.
  Then report the failing step and its `tail`. Nothing is lost: the next run sees the same delta. An ingested model that
  fails stays `queued`, and the next run retries it.
- **exit 0:**
  ```bash
  python3 pipeline/qcl_scan.py unlock
  git add -A && git commit -qm "Hourly scan <UTC time>: <progress_line>" && (git push -q || (git pull --rebase -q && git push -q))
  gh api -X POST repos/delirious6423/quant-fidelity-frontier/actions/workflows/pages.yml/dispatches -f ref=main
  ```
  `work/` and the built sites are gitignored and are never committed.

## 7. Report
For each model: cards fetched, values published (from `merge`), entries quarantined, and gated repos skipped. Then the
ingested model, if any, and whether the deploy was dispatched. If nothing changed anywhere, say "no changes", run `unlock`, and
commit nothing (the lock was never pushed).

## Rehearsal results (18 Oct 2026, two real cards, Haiku at high effort)
- **Fetch:** 2 of 2 cards saved in 31 s, byte-identical to the original capture, with no retries.
- **Mapping:**
  - The two Haiku runs agreed on 20 values, and all 20 were values the Sonnet runs had also found, so precision was good.
  - Recall was lower: Haiku found 20 values on one card where Sonnet found 35. Values that are missed stay unpublished;
    they are never guessed. Expect Haiku-mapped cards to show fewer values than Sonnet-mapped ones.
- **Bug found and fixed:** a shell `mv` with an empty queue name moved the whole `pending` folder. Workers now claim
  only through `pipeline/qcl_claim.py` (an atomic rename of one file).

## Known failure modes and what to do
| Symptom | Cause | Action |
|---|---|---|
| hf_fs result shown inline | output too small to be saved to a file | rerun the same call once; never transcribe it |
| `output_truncated` / budget error | more than ~80K chars in one call | use the op lists exactly as printed; they are sized to fit |
| more than 30 ops rejected | connector limit | the plans never exceed 30; do not merge op lists |
| `HF_FS_ACCESS_DENIED` | gated repo | expected; it is skipped and listed |
| saver lists `retry` / `need_chunks` | a truncated or split chunk | the fetch agent re-plans once (FETCH.md step 4) |
| agent wrote files to the wrong folder | worker error | `finish` catches missing maps and re-queues them |
| `pending` folder missing or moved | a worker moved files by hand | workers must use `pipeline/qcl_claim.py`; recreate `work/<slug>/pending`, move unprocessed queues back from `claimed/`, rerun |
| `REFUSING TO BUILD` | a gate failed | `merge` already quarantines failing entries; if a build still refuses, stop and report |
| UI test fails for a new model | the site is not ready | the model stays `queued`; report it |
| push rejected | another commit landed | `git pull --rebase` and push; never force |
