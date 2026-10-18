# Hourly scan runbook

The scheduled task "QFF hourly scan" follows this file, top to bottom. Each run has two parts, in this order:
- **Part A, live updates:** keep every `live` model in `watchlist.json` current. Small (usually 5–25 changed cards),
  published and committed on its own, so a slow ingest can never hold it back.
- **Part B, ingest (resumable):** work on at most one new model, in list order; a new one starts at most once per
  `min_hours_between_ingests` (5 h). An ingest is 100–250 cards and usually takes several runs: each run does what fits
  in its time budget, commits a checkpoint, and the next run continues from there.

**Time budget.** A run must end within ~55 minutes, before the next one starts. `python3 pipeline/qcl_scan.py elapsed`
says how long this run has held the lock. Start no new step once it passes 40 minutes; mapping workers check it
themselves (45). A lock older than 75 minutes is treated as left by a dead session and is taken over.

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
| Nothing changed and no ingest | none | 0 | — | — |
| Part A, changed cards | fetch | 1 per 100 changed cards (usually 1) | stat, plan, fetch and save cards byte-for-byte | `docs/agents/FETCH.md` |
| | mapping | 2 (one A, one B); up to 4 if more than 8 queues | map candidates to metrics; two runs, A and B | `docs/agents/MAP.md` |
| Part B, fetch phase | fetch | one per shard left, at most 3 at once | as above | `docs/agents/FETCH.md` |
| Part B, map phase | mapping | up to 8 (half A, half B); each drains queues until none are left or the time is up | as above | `docs/agents/MAP.md` |

There is no resolver and no reviewer agent in the hourly loop. A value that the two runs disagree on, or that fails a
gate, is held back (`models/<slug>/audit/review_queue.json`) and never published. Keep at most 8 agents running at once.

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
python3 pipeline/qcl_scan.py lock      # exit 3 = another run holds a lock under 75 min old: stop now, report "skipped (locked)"
python3 pipeline/qcl_scan.py plan
```
`plan` prints `delta` (the live slugs) and `ingest` (a slug or null; `resume: true` when it continues an earlier run).
Push the lock as soon as there is real work: when `plan` returns an `ingest`, or when a Part A `catalogue` returns shards.
Run `git add scan.lock && git commit -qm "scan lock" && git push -q`, so a parallel run sees it. If the push is rejected,
run `git pull --rebase -q` and push again; if `scan.lock` came in from the remote with a fresh time, stop. With nothing to
do, nothing is committed: run `python3 pipeline/qcl_scan.py unlock` and stop.

# Part A: live updates

## A1. Catalogue (each slug in `delta`)
1. `python3 pipeline/qcl_scan.py search-ops <slug>`. It prints `op_lists` (one list).
2. Call `hf_fs` once per op list, with the list **exactly** as printed. Each result is saved to a file, and the tool
   message gives its path. If a result comes back inline instead, run that call once more. Never copy inline output.
3. `python3 pipeline/qcl_scan.py catalogue <slug> <saved result paths>`
   - exit 2: nothing changed for this slug, so go to the next one.
   - exit 1 with `page_full`: more than 80 quants changed since the last scan. Redo steps 1–3 with `--full` on both
     commands (28 calls; only after a long outage).
   - exit 0: note the `shards` list.

## A2. Fetch (each slug with shards)
Start one Haiku agent per shard, all at once:
> Follow docs/agents/FETCH.md in the repo at <absolute repo path>, with S = `<shard>` and M = `models/<slug>`.
> Use only Bash and the hf_fs tool.

Wait for them. Failed ids are either gated (`HF_FS_ACCESS_DENIED`: fine) or get one more agent run on the same shard.

## A3. Map (each slug that fetched cards)
1. `python3 pipeline/qcl_scan.py prepare <slug>`. Exit 2: no changed card has metrics to map; skip to A4.
2. Otherwise start Haiku workers, one with RUN = A and one with RUN = B (two each if `queues` is more than 8):
   > Follow docs/agents/MAP.md in the repo at <absolute repo path>, with W = `work/<slug>`, RUN = `A`, M = `models/<slug>`.
3. `python3 pipeline/qcl_scan.py finish <slug>`
   - exit 1 with `bad` or `pending`: run one worker per run letter for those queues, then `finish` again (at most twice).
     If it still fails, go on: those cards stay "not yet reviewed" and the next run's `prepare` queues them again.
   - exit 0: `merge` reports the values published and the entries quarantined.

## A4. Publish Part A
Only if any slug fetched cards (otherwise go to Part B):
```bash
python3 pipeline/qcl_scan.py publish      # builds every live model, assembles, runs all tests (~2 min)
```
- **exit 0:** `git add -A -- . ':!scan.lock' && git commit -qm "Hourly scan <UTC time>: <progress_line>" && (git push -q || (git pull --rebase -q && git push -q))`,
  then dispatch the deploy:
  `gh api -X POST repos/delirious6423/quant-fidelity-frontier/actions/workflows/pages.yml/dispatches -f ref=main`.
- **exit 1:** do not commit. Discard Part A only: `git checkout -- models watchlist.json PROGRESS.md && git clean -fdq models`.
  Note the failing step and its `tail` for the report, and go on to Part B (the next run sees the same delta).

# Part B: ingest (only when `plan` returned an `ingest`)

Check the time first: `python3 pipeline/qcl_scan.py elapsed 30`. Exit 1: skip Part B this run (go to step 9). Run
`python3 pipeline/qcl_scan.py plan` once more (it recreates the new model's `model.json` if a Part A discard removed it),
then `python3 pipeline/qcl_scan.py ingest-state <slug>` says where the ingest stands; start at that phase.

## B1. `catalogue` phase
`search-ops <slug> --full` (28 op lists), call `hf_fs` once per list as in A1, then `catalogue <slug> --full <all paths>`.

## B2. `fetch` phase (`shards` lists what is left)
Start one Haiku fetch agent per shard left, at most 3 at once (prompt as in A2). When an agent finishes its shard, run
`python3 pipeline/qcl_scan.py fetched <slug> <shard>` (if it reports failed ids that are not gated, run one more agent
on that shard first; mark it fetched after that either way, so a bad id cannot block the ingest). Stop starting agents once `elapsed 40` exits 1; let running ones
finish, then go to B4.

## B3. `map` phase
1. `python3 pipeline/qcl_scan.py prepare <slug>` (maps made by earlier runs are counted, only the rest is queued).
   Exit 2: everything is mapped; go to step 3.
2. Start up to 8 Haiku workers, half RUN = A and half RUN = B (prompt as in A3). Workers stop by themselves at 45 minutes.
   While `elapsed 40` exits 0 and `work/<slug>/pending` still has files for a run, start another worker for it.
3. If `work/<slug>/pending` is empty: `python3 pipeline/qcl_scan.py finish <slug>` (handle `bad` as in A3), then
   `python3 pipeline/qcl_scan.py publish --ingested <slug>`.
   - exit 0: the model is live. `git add -A -- . ':!scan.lock'`, commit "Hourly scan <UTC time>: <progress_line>",
     push, and dispatch the deploy (as in A4). Done with Part B.
   - exit 1: the model stays unpublished; go to B4 (its data is kept, and the report names the failing step).
4. Otherwise (time is up with queues left): go to B4.

## B4. Checkpoint (an ingest that is not finished this run)
```bash
python3 pipeline/qcl_scan.py checkpoint <slug>      # status -> "ingesting": not built, not on the site, resumed next run
git checkout -- PROGRESS.md 2>/dev/null; git add models/<slug> watchlist.json && git commit -qm "Ingest checkpoint: <slug>" && (git push -q || (git pull --rebase -q && git push -q))
```
Only that model's folder and the watch list are committed. CI, Pages and `publish` build live models only
(`pipeline/qcl_live.py`), so a half-ingested model never reaches the site.

# End of every run

## 9. Unlock and report
```bash
python3 pipeline/qcl_scan.py unlock
git add -A scan.lock logs/scan && git commit -qm "scan unlock" && (git push -q || (git pull --rebase -q && git push -q))   # skip if the lock was never pushed
```
Every `qcl_scan.py` command logs its result line; `unlock` files the run's log as `logs/scan/<UTC>.jsonl`, committed with the
unlock, so a later check can see which steps ran and how they ended.

Report: for each live model, cards fetched, values published (from `merge`), entries quarantined, gated repos skipped;
for the ingest, the phase reached (catalogue / shards fetched / queues mapped / live) and whether a deploy was
dispatched; the run's minutes (`elapsed`). If anything failed, name the step and its `tail`. If nothing changed and
nothing was ingested, say "no changes".

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
| run takes most of an hour | an ingest was attempted in one go | Part B is time-boxed and resumable: checkpoint and let the next run continue |
| `scan.lock` pushed but never removed | the session died | the next run after 75 min takes the lock over; nothing to do |
| Part A `prepare` queues hundreds of cards | a model's cards were never recorded as mapped | for a hand-reviewed model run `QCL_ROOT=models/<slug> python3 pipeline/qcl_mapset.py --baseline` once (done for Flash-Next) and report it |
| UI test fails on load timeouts | external fonts/analytics unreachable | `tests/test_ui.py` blocks all external requests; if it still fails, report the `tail` |
| `implausible parsed values` on a new model | its BF16 files are larger than `max_size_gb` in model.json | `plan` sets it from the parameter count (2.5 GB per billion, else 1000); if a real BF16 size is still above it, report it (raising it is a config change, not a data edit) |
| `REFUSING TO BUILD` | a gate failed | `merge` already quarantines failing entries; if a build still refuses, stop and report |
| UI test fails for a new model | the site is not ready | the model stays `queued`; report it |
| push rejected | another commit landed | `git pull --rebase` and push; never force |
