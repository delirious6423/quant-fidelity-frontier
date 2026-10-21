# Scripted runner: first migration, shadow only

This branch replaces coordinator-mediated collection, work dispatch and recovery
with local Python orchestration. It does **not** cut production over. Existing
`qcl_next.py`, `qcl_qa.py`, deployment receipts, schedules, sampling count, website
defaults and authoritative claims remain in place. No paid provider is configured.

## Commands

Run from the repository root. Python 3.10+ and the existing build dependencies are
sufficient; no server, database service or model SDK is added. Use an external,
local-disk directory, not a directory inside the repository. `/tmp` examples are
throwaway; choose a persistent volume for long-lived shadow operation.

```bash
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow init --models lfm2.5-24b-a2b
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow plan
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow prepare
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow status
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow run --responses /tmp/qff-responses --workers 8
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow resume --responses /tmp/qff-responses --workers 8
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow verify
```

`plan` calculates current triage/dedup/mapping eligibility in a disposable copy;
it performs no network or model request and leaves the workspace unchanged.
`prepare`, `run` and `resume` automatically execute safe deterministic stages and
export every eligible independent request to `requests/<task-id>.json`. `run`
without responses stops with those requests pending. The file adapter reads
`<task-id>.json` response objects from the specified directory; missing files do
not consume a lease or prevent later ready files from completing. It can finish
A/B, blind C and QA in one invocation when the responses are available.

Live collection and new-model ingestion are separate commands and workspaces:

```bash
# Read-only public HTTP, no connector coordinator; explicitly include known repos.
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow collect lfm2.5-24b-a2b --workers 8
# A genuinely new model need not have a repository directory yet.
python3 pipeline/qcl_runner.py --workspace /tmp/qff-new init-ingest --slug example-model --model Publisher/Model
python3 pipeline/qcl_runner.py --workspace /tmp/qff-new ingest example-model --include-repo Quantizer/Model-GGUF
```

Collection preserves all previously indexed/catalogued repositories as explicit
coverage, even after tags disappear. A missing repository, denied access, invalid
page or transient exhaustion blocks that model's freshness; other models can
continue. Collection does not declare old bytes current on a failed refresh.
It retains complete raw bytes and immutable revisions, follows all catalogue/tree
pages, validates both saved card and tree hashes, checkpoints cards before trees,
and respects saved `Retry-After` cooldowns. HTTP 200 with empty bytes differs from
404 missing README; neither is fabricated for a failed request. No token or HTTP
error body is written to diagnostic records.

The direct ledger is deliberately separate from connector `cache/index*.json`.
`qcl_newmodel.py` validates the direct import in a temporary root, then a short
installation lock switches the snapshot atomically. Old connector shards cannot
win by filename order. No `qcl_scan`/`qcl_next`/`qcl_qa` entry point is invoked.

## Model request contract

`runner_model.py` prepares JSON envelopes with the base model, repository, source
revision, SHA-256, entire decoded source with line numbers, all candidate objects,
candidate digest, generated mapping schema, code version, run identity and an
untrusted-source instruction. No map from A appears in B or C. A/B agreement,
blind C tie-breaking, resolver pins, code-copied candidate values, prose exceptions,
image flags and existing gates still come from the existing stage scripts.

The adapter receives one request only; it needs neither a shell nor repository
navigation. `task_id` is the stable provider idempotency key. A future adapter must
preserve provider request IDs and actual usage telemetry separately. No API-backed
adapter, batch API, live model quality evaluation or paid call is included here.
The synthetic `fake_response` in `runner_replay.py` is only a fixture adapter and
must never be connected to production acceptance.

External workers can claim and submit independently:

```bash
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow claim --workers 8 --lease 900 > /tmp/claim.json
# Submit claim.payload to an independently configured interpreter, then save its JSON.
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow complete /tmp/claim.json /tmp/response.json
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow resume
```

A null claim means no currently eligible work (including backoff/concurrency).
Responses are raw mapping objects, or one review verdict object with `id` equal
to the evidence fingerprint, `verdict` and a source explanation. Duplicate JSON
keys, nonfinite numbers, unknown fields/candidates, invalid source pins, malformed
prose and ambiguous/multiple verdicts fail closed. Identical redelivery of the
same completed lease is transport deduplication, not a second review. Conflicting
redelivery is rejected. `unsure` blocks in this runner, per this migration's
requirement; current production `deploy_state.assess_reviews` actually holds an
unsure item and may proceed. Production policy code was not changed.

QA fingerprints contain the complete entry interpretation, metric, role, panel,
units, attribution, evidence/line identity, model/repo/source hash and revision,
extractor/mapping code and review-policy versions. Only a byte-identical revision
update can reuse evidence across revision IDs; the original request retains its
reviewed revision and `review-reuse.json` records the current revision alias.
Neither numbers alone nor stripped panel/qualifier data establish reuse. Completed
individual verdicts survive missing/failed neighbors. Wrong entries go through
existing `audit/holds.json`, merge and regeneration, never number edits.

The fixed 20-value sample and seeded shuffle remain; changing selection from
raw-only to complete evidence is intentional. The selected sample remains frozen
when upstream revisions change but evidence does not, including >20 eligible
values and in-flight reviews. In shadow mode the comparison point and seed are
its frozen input commit, **not a production deployed receipt**. Production cutover
must explicitly wire the existing exact deployed comparison point and approve
that integration. This first-stage result is never a release approval.

## Durability, concurrency and recovery

SQLite uses local file locking, `BEGIN IMMEDIATE` atomic claims and FULL synchronous
commits. Every eligible task has a durable record; concurrency limits simultaneous
claims, never enqueue count. Least-recently-served model/stage groups prevent a
large model or A/B/C stage from starving another. Requests move through pending,
running, succeeded, failed and blocked. Leases carry unguessable tokens; expired
or superseded workers cannot install output. Completion persists the response in
the same transaction as succeeded status before acknowledgement. Only the
orchestrator materializes maps and runs merges under its local lock; workers
never write maps. No production lock is taken and HTTP/model waits do not hold
the orchestration lock. The collector has its own per-ledger lock.

A failed attempt backs off exponentially (capped at 60 seconds); three attempts
block the task with a reason. Expired leases are recovered on the next claim or
file-adapter resume, with the same bounded backoff. After the delay, resume again.
A deliberate operator retry is explicit and logged:

```bash
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow retry TASK_ID --reason 'provider repaired; prior malformed response replaced'
```

No new source-pinned skip is created for failures. Historical pinned skips remain
unchanged (the backlog is deferred). Source or code-version changes invalidate
runner-owned results, remove their installed source-only map pins and enqueue
new work. Existing legacy mappings are reused only under their original source
pins; the migration does not force remapping the complete reviewed catalogue.
Stage receipts avoid repeated successful extraction/merge on unchanged inputs.

SQLite is **not** a distributed queue: use a single local filesystem shared by
processes on one host, not NFS or independently writing replicas. A completed
provider result lost before it is durably delivered cannot be recovered magically;
persist response files/provider receipts, and reuse the stable task ID when retrying.

Explicit persistence across ephemeral sessions:

```bash
python3 pipeline/qcl_runner.py --workspace /tmp/qff-shadow backup /persistent/qff-checkpoint-001
# Transfer the complete checkpoint directory through approved storage to the next host.
python3 pipeline/qcl_runner.py --workspace /tmp/qff-restored restore /persistent/qff-checkpoint-001
python3 pipeline/qcl_runner.py --workspace /tmp/qff-restored resume
```

Backup uses SQLite's consistent backup API and a complete SHA-256 file manifest;
it does not copy a live database file. It refuses active collectors, symlinks,
existing destinations or repository paths. Restoring validates every file and
rejects extras/missing/changed bytes. Paths are portable. The caller must actually
retain the checkpoint on durable storage; `/tmp`, an unuploaded artifact or a
cloud session's disk is not a persistence guarantee. A provider worker may finish
after a backup; its receipt must also be transferred or redelivered to avoid
re-execution. Do not run both restored and original workspaces as writers.

## Verification and reproducible evidence

```bash
python3 pipeline/runner_baseline.py
python3 -m unittest discover -s tests -p 'test_runner*.py'
python3 pipeline/runner_replay.py --out /tmp/qff-replay-new-run
python3 tests/benchmark_runner.py --model qwen3.8-27b --out /tmp/qff-runner-bench
python3 -m unittest discover -s tests -p 'test_workflow_security.py'
git diff --exit-code -- models
```

`benchmark_runner.py` measures orchestration at real catalogue scale: it copies
one committed model into a new external workspace, forgets every mapping, and
times `prepare`, a `run` fed with synthetic fixture responses, and an unchanged
reconcile. It measures scheduling and bookkeeping only (no model calls, tokens
or cost) and is the harness behind the scale table in
[RESULTS.md](scripted-runner/RESULTS.md). Journal listings are filtered in SQL
and decode a payload only when it is needed; completion and slot lookups are
single-row queries; a reconcile reinstalls a map or re-exports a request only
when its bytes would change. A full re-map of a 1,253-repository model therefore
costs about a minute of orchestration per round rather than ten.

The replay output directory must be new. It contains old/new complete output JSON,
SQLite journals and measured results. The baseline executes AST-selected legacy
dispatch branches with inert scan/worker interfaces, never git-capable state
machines. Shadow extraction runs the existing safe stages in separate external
roots with no `.git` or remote. It compares full records/provenance, including
units, qualifiers, panels and image flags; no semantic normalization is applied.
The dispatch replay is separate from end-to-end extraction timing and actual
model quality. No tokens/cost figures are inferred from queues or tool calls.

Results: [baseline](scripted-runner/baseline.json),
[replay](scripted-runner/replay-results.json),
[existing regressions](scripted-runner/verification.md),
[real pinned input](scripted-runner/real-input-check.json),
[public HTTP smoke check](scripted-runner/public-check.json),
[independent review](scripted-runner/review.md).

Fault injection covers claim-process termination, result commit before lost
acknowledgement, duplicate/conflicting delivery, source/version changes in flight,
partial mapping and QA, rate limits/transients, portable restore, concurrent
claim contention, more work than concurrency, and attempted shadow publication.
No workflow, schedule or deployment can be invoked through the runner's command
surface. Registered model paths and symlinks are checked on load, not just creation.

`verify` rejects incomplete work/fetch/ingestion and runs gated builds after
regeneration; its receipt binds source signatures and artifact hashes. It is
**per-model shadow verification**, not the entire reusable release workflow and
not `audit/pending_release.json` or `audit/deployed.json`. Production must still
verify the exact candidate commit and deploy the exact tested artifact through
the existing release protections. There is deliberately no publish command.

## Migration and rollback

1. Review the branch and offline evidence in draft PR
   [#14](https://github.com/delirious6423/quant-fidelity-frontier/pull/14).
   The owner authorized pushing the branch and its automatic `pull_request`
   verification on 2026-10-21. Merge, deployment and production cutover remain
   separately authorized steps.
2. Approve a bounded shadow trial with explicit storage, representative source
   inclusions and an independently configured model adapter. Compare full accepted
   claims, held/missing claims, source freshness and provenance against the old path.
   Record real requests/tokens only when telemetry exists. No scheduled cutover yet.
3. Evaluate precision **and recall**, shared A/B omissions, images and difficult
   panels on independently labelled examples. Validate provider idempotency,
   independence, interruption/restore and exact deployed-baseline QA integration.
4. Only after review, authorize live collection/ingestion cutover separately from
   publication. Keep release verification, receipts, artifact and commit protections.
   Schedules and freshness cadence require their own approval.
5. Roll back by stopping the new invocation and retaining/exporting its journal;
   production continues its existing entry points. Discard only the isolated
   shadow workspace after preserving evidence. Never delete authoritative state,
   restore old mappings onto changed bytes, or bypass release verification.

Deferred: single-pass mapping, reduced review coverage, publisher parser auto-
acceptance, model-tier routing, historical held backlog, cadence/README schedule
prose, discovery and broad UI/model-data redesign. The fixed 20-value sample can
miss entire tables/panels and shared omissions; a future stratified table/panel
review requires its own labelled precision/recall evaluation and policy approval.

A synthetic small-model build exposed a pre-existing static-chart division by
zero below/equal 20 GB or on a single exact log decade. Two axis-degeneracy guards
and a focused regression fix that ingestion failure. All existing 40 model-site
files and 30 assembled artifacts remain byte-identical; no claim data was changed.

## PR #14 performance follow-up

The file adapter loads its available-response IDs once into a connection-local
SQLite temporary table. Each claim still commits independently; the temporary
table grants no lease and is reconstructed after restart. Selection considers the
oldest eligible task in each model/stage group and then the least recently served
group. Metadata-only queries exclude the evidence column from SQL entirely. Large
eligibility sets no longer depend on SQLite's bound-variable limit.

Direct collection stores each successful card/tree record in a small, atomic
`ledgers/<slug>/checkpoints/<identity-hash>.json` before proceeding. It compacts
these records into `index.json` once the worker round ends, including a round with
partial failures. The index is durably replaced before checkpoint records are
removed. Restart overlays any surviving records; both sides of an interrupted
compaction are safe to replay. A malformed ledger blocks freshness before it is
read. Existing index-only ledgers remain supported. Unchanged, validated records
need no checkpoint write. Cards, trees and checkpoint directories are created
and their parent entries synced before concurrent workers start.

Transfer the **whole** workspace using `backup`/`restore`, including checkpoint
records left by an interrupted collector. Copying only `index.json` can omit
successful partial work. Downgrading to the earlier runner requires letting a
collector finish/compact first or restoring a backup made by that version.
Do not delete checkpoint records to repair a malformed ledger; restore a verified
backup or explicitly refetch in a fresh shadow workspace.

Exported requests are checked against journal bytes and repaired if altered;
unchanged files are not rewritten. Installed map comparisons preserve JSON types
(`false` and `0` differ). Completion checks current base-model identity as well as
source bytes and interpretation-code version. Evidence preparation snapshots the
code version once per call, never across calls, and completion rechecks it.

Additional offline benchmark, including real local fsyncs and mocked HTTP:

```bash
python3 tests/benchmark_runner_io.py --out /tmp/qff-io-current
# Reproduce the reviewed PR baseline on the identical synthetic workload:
git worktree add --detach /tmp/qff-reviewed-code 4ad4b570a636b62bc9c3d267a21a319ae2f4734e
python3 tests/benchmark_runner_io.py --implementation /tmp/qff-reviewed-code --out /tmp/qff-io-reviewed
```

Use fresh external output directories. Run benchmarks sequentially on the same
host. `results.json` includes request counts, checkpoint bytes/writes, actual
runtime and a source-output digest. The full-model benchmark also saves complete
semantic `output.json` (records, holds, review queue, source pins, maps and task
results) for exact comparisons. These fixture responses do not evaluate model
precision/recall or real upstream latency. See
[PERFORMANCE_FOLLOWUP.md](scripted-runner/PERFORMANCE_FOLLOWUP.md) for measured
results, independent review findings and scope limits.

The independent three-repetition evidence is now complete and locally rechecked:
[INDEPENDENT_VALIDATION.md](scripted-runner/INDEPENDENT_VALIDATION.md). Use its
medians and ranges for performance claims; the earlier single-run measurements
remain historical. The correction moves helper evidence export after both timed
sides, and all three repetitions were rerun with fresh output directories.
