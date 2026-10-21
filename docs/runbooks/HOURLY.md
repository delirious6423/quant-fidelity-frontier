# Hourly scan runbook

QFF uses separate collection and publication tasks. The scripts decide each step;
the coordinator (an agent or a person running the loop) performs the catalogue calls
and starts the mapping/review workers they request.
These are existing owner-approved schedules; running a script manually does not grant
permission for a new production deployment.

| Task | Schedule (UTC) | Entry point | Outcome |
|---|---|---|---|
| Collect cards | Hourly at :01 (GitHub Actions, `collect.yml`) | `qcl_collect.py --all-live` | Fetch every live model's changed cards over plain HTTPS; commit cache and index |
| Live updates | Hourly at :03 | `qcl_next.py --part A --collected` | Map, merge and gate changed cards; commit staged data |
| New model | Every 2 hours at :37 | `qcl_next.py --part B` | Resume the next ingest; checkpoint progress; hand completed models to QA as `review` |
| QA and publish | Hourly at :55 | `qcl_qa.py` | Review new values, run all tests, request Pages verification; record deployment only after exact-commit success |
| README check | Daily at 00:58 | `docs/runbooks/agents/README_CHECK.md` | Check generated counts, links and prose; no site deployment |

QA runs at :55 so that a live-updates run (workers stop at 45 minutes) has finished before QA synchronizes to main; an earlier :15 slot made QA race the scan's push. Only QA requests a site deployment. A scan's successful build or commit is not a
publication. Each task uses its own lock (`scan.lock`, `ingest.lock`, `qa.lock`) and
records step logs under `logs/scan/`.

## Collection without a connector

This is now the default path. `.github/workflows/collect.yml` runs `python3 pipeline/qcl_collect.py --all-live --workers 8`
every hour at :01 (and on `workflow_dispatch`): for every live model it makes one public catalogue call
(`base_model:quantized:ORG/MODEL`), fetches only the cards and file trees whose commit changed (`pipeline/qcl_fetch.py`,
resumable, pinned by commit sha and SHA-256), merges the cache into `snapshots/index.json` (`pipeline/qcl_newmodel.py`) and
prints one JSON line per model (catalogue size, cards new / changed / unchanged, unresolved, seconds). The workflow commits
`models/*/cache` and `models/*/snapshots/index.json` as `quant-claims-bot` only when a card changed (`Collect <UTC>: <summary>`);
an unchanged model changes no file: when no card or tree changed, the ledger and catalogue keep their previous bytes and the
merge is skipped, and the Hub's hourly metadata churn (downloads, likes) is taken over at most once a day (when the ledger's
`captured_at` is more than 24 hours old). `HF_TOKEN` (repository secret) raises the rate limit; a 429 is recorded in
`cache/rate_limit.json` for that model, the other models continue, and the next hour resumes. A direct record replaces an
older connector card of the same repo only when the Hub changed after the connector capture.

The live-updates task then runs `python3 pipeline/qcl_next.py --part A --collected` (or `QCL_COLLECTED=1`): the `start`
step makes no `hf_fs` call and starts no FETCH worker; every live model counts as freshly collected, is prepared
(`qcl_scan.py prepare`), gets mapping workers when it has queues, and is staged otherwise. Lock, unlock, commit and push
are the same as before. Without the flag the connector path below is unchanged.

## Start and follow the loop

Use a clean checkout with push access and enough history to contain the last
successful deployment. A full clone avoids silently losing that comparison point.

```bash
git clone https://github.com/delirious6423/quant-fidelity-frontier qff
cd qff
pip install pillow numpy "mcp==2.3.0" playwright
python3 pipeline/qcl_next.py --part A
```

Use `--part B` for ingestion, or `python3 pipeline/qcl_qa.py` for QA. Follow the
printed instruction, then invoke the same entry point again:

- `hf_fs`: call each printed op list exactly. Pass the saved result paths to
  `qcl_next.py --results <path1> <path2> ...` in order. A preview is not source data.
  The `hf_fs` connector tool is one way to make these catalogue calls. Without a
  connector, the same step can be done with the public HTTP collector
  (`python3 pipeline/qcl_runner.py --workspace DIR collect SLUG`, see
  `docs/runbooks/SCRIPTED_RUNNER.md` and `docs/runbooks/PORTABLE_SCAN.md`).
- `agents`: start workers with the printed prompts and concurrency limit. Mapping
  workers follow `docs/runbooks/agents/MAP.md`; QA workers follow `docs/runbooks/agents/REVIEW.md`.
  The scheduled tasks currently use a small fast model for these workers; any
  model or person can do the task.
- `error`: address the named failure, retaining the source files and checkpoints.
- `stop`: report the actual result. This can mean skipped, blocked, failed,
  dispatched, pending, or successfully reconciled; it never implies deployment by itself.

Do not bypass the state machine with a manual deploy or a blanket checkout/clean.
For an interrupted run use the matching entry point's `--abort`; QA retains
failed local work. Keep worker concurrency at most eight. Mapping workers claim
queues only through `pipeline/qcl_claim.py`, never by moving folders themselves.

## Collection and mapping

Live updates use `stage` for source checks. Ingestion fetches at most two shards
per run and checkpoints after catalogue, fetch and mapping rounds. Complete models
remain `review` until QA passes; `ingesting` models never enter the public assembly.

A and B map independently against the exact saved card. A third blind C run may
resolve disagreements when two runs agree on role, metric family and panel.
Source hashes must match throughout extraction, mapping and merge. A changed card
withdraws its prior claims until a fresh mapping passes; recording old map files
must never stamp them with a new card hash. Resolver overlays additionally pin the
mapping bytes. Values that remain disputed stay held, never guessed or edited.

See `docs/runbooks/FASTPATH.md` for individual ingestion commands and `docs/runbooks/FETCHING.md` for
rate limits, checkpoint recovery and authenticated fetching.

## Review and publication receipts

QA samples up to 20 new values. Every sampled item needs exactly one `ok` or
`wrong` verdict with a source explanation. Missing, duplicate, invalid or `unsure`
verdicts block publication and leave the review resumable. Wrong entries are held
in `audit/holds.json`; both merge and regeneration must succeed before proceeding.

After local verification, QA commits a pending candidate in
`audit/pending_release.json`. The commit that introduced that receipt establishes
its exact SHA, including after an interrupted run and fresh clone. Candidate pushes
are not automatically rebased: another commit requires synchronization and renewed
verification. Pages rejects a dispatch whose `expected_sha` differs from its actual
commit, then runs the complete reusable verification workflow and deploys that
exact artifact.

Dispatch success means only **pending**. On a subsequent run QA reconciles the
latest matching Pages workflow before checking whether there is new work. Only a
successful completed run for the exact candidate advances `audit/deployed.json`.
A failed run permits a new attempt; an unavailable or malformed status response
leaves the existing receipt unchanged. The report includes the workflow URL when
success is established. The public MCP Space is a separate, explicitly approved
deployment; publishing Pages does not publish the MCP server.

## Recovery

| Symptom | Response |
|---|---|
| Tool output says it was saved to a path | Use that complete file, never the preview |
| Gated repository / access denied | Preserve the status; do not invent an empty card |
| HTTP 429 | Stop that fetch run; retain completed checkpoints and `cache/rate_limit.json` |
| Missing or stale mapping | Queue a fresh mapping against the pinned source |
| Incomplete or uncertain QA verdict | Resolve the exact sample items and resume QA |
| Source, image, column or build gate failure | Keep the entry quarantined and report the failure |
| New model fails verification | Leave it in `review`; existing live models may proceed |
| Deployment base is unavailable in a shallow checkout | Run `git fetch --unshallow origin`, then restart QA |
| Candidate push rejected | Synchronize a clean checkout and rerun verification; never force-push |
| Pages run still queued or running | Leave deployment pending; reconcile on the next QA run |
| Workflow API unavailable | Report unavailable status; preserve the prior deployment receipt |

Report collection counts and held entries separately from publication status.
Include the failed step and its output when something blocks progress.

## Rehearsal results (18 Oct 2026, two real cards, small fast model at high effort)
- **Fetch:** 2 of 2 cards saved in 31 s, byte-identical to the original capture, with no retries.
- **Mapping:**
  - The two small-model runs agreed on 20 values, and all 20 were values the larger-model runs had also found, so precision was good.
  - Recall was lower: the small model found 20 values on one card where the larger model found 35. Values that are missed stay unpublished;
    they are never guessed. Expect cards mapped by the small model to show fewer values than cards mapped by the larger one.
- **Bug found and fixed:** a shell `mv` with an empty queue name moved the whole `pending` folder. Workers now claim
  only through `pipeline/qcl_claim.py` (an atomic rename of one file).

## Weekly discovery (queue maintenance)
`.github/workflows/discover.yml` runs `pipeline/qcl_discover.py --http --commit` every Sunday 02:11 UTC straight from the Hub API
(no connector; `docs/reference/DISCOVERY.md`). The fallback, `QFF weekly discovery`, runs `pipeline/qcl_discover.py` in a session
(same step-by-step protocol as `qcl_next.py`: do only what it prints).
It lists each lab's newest repos (labs and current-family prefixes in `discovery.json`), counts quant repos per candidate
(five search sorts, so counts pass the 80-per-search cap), admits models that pass the rules (>= 40 quant repos, released in the
last 6 months, current family), refreshes the counts of queued models, re-ranks the queue with `pipeline/qcl_rank.py`, and
commits `watchlist.json`, `audit/discovery.json` and `docs/QUEUE.md`. A new family from a listed lab is only reported
(owner adds its prefix to `discovery.json`). Models are ingested back to back: `min_hours_between_ingests` is 0.
