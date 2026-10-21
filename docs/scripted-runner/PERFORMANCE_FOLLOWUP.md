# PR #14 follow-up: review and measurements

Reviewed upstream commit: `4ad4b570a636b62bc9c3d267a21a319ae2f4734e`.
Integrated locally with current main (`77c157c`) and the prior review branch.
The remote review contains one substantive owner review and no inline comments.
The 55 tests on the updated PR passed before follow-up runtime changes (69.827 s).
The earlier review's 569-second preparation / interrupted 35-minute response run
are historical reviewer measurements; they were not repeated in this session.

A 100× runtime stretch target was recorded before optimization, without reducing
source validation, model independence, QA sample size, or result durability.
No model/provider calls or live upstream requests were used in this follow-up.
No token or monetary savings are measured or claimed.

## Changes and review findings

- SQL listings omit unused evidence; an index avoids scanning the full journal to
  allocate each task's creation sequence. Fair selection reads one eligible
  candidate per model/stage. The file adapter loads IDs into a temporary table
  once, avoiding repeated whole-list SQL parameters and SQLite variable limits.
  Every individual claim and completion still commits durably.
- Collection replaces growing whole-index writes with small atomic records and
  an index compaction. It keeps the card-before-tree checkpoint guarantee, hash
  validation, rate limits and retained catalogue coverage. This avoids nearly
  a gigabyte of metadata serialization per 1,253-repo collection in the fixture.
- Profiling 4,956 existing Qwen review bundles found 9,912 code-version scans
  (7.876 seconds of 14.935 profiled seconds). Each preparation now snapshots the
  version once; completion still checks current code and source identity.
- The request-export optimization from the reviewed commit trusted file existence.
  Altered exports are now repaired from the journal while intact exports retain
  their modification time. UTF-8 is explicit.
- Independent review reproduced stale freshness after malformed ledger recovery,
  Python equality treating `0` and `false` as equal installed map values, and
  acceptance of delayed work after base-model identity changed. All three are
  fixed with counterfactual regressions; no published claim data was changed.
- Checkpoint directories are created and synced before concurrent fetch writers.
  Recovery tests cover abrupt process death before compaction, fresh-workspace
  transfer and interruption after installing the compact index.

The fresh reviewer found no remaining blocker for the local shadow scope after
these fixes. It independently ran 16 state tests and 18 fetch tests, then verified
the model-identity and process-death/restore counterfactuals after final fixes.
Production entry points, schedules, deployment workflows and publication capability
remain unchanged. This review does not establish live-provider quality or approve
production cutover.

## Full-model orchestration

Frozen Qwen3.8-27B inputs: 1,253 repositories, 1,190 mapping requests and 20 QA
requests, synthetic responses, all 1,210 tasks succeeded. Actual model launches: 0.
The same input model root and seed were used in both workspaces. These are single
local observations; focused tests/I/O benchmarks overlapped parts of the runs,
so the byte/operation reductions are firmer evidence than small timing differences.

| Phase | Reviewed PR | Follow-up |
|---|---:|---:|
| Prepare | 61.20 s | 35.88 s |
| Mapping response round | 76.35 s | 54.10 s |
| Remaining QA round | 8.73 s | 5.25 s |
| Unchanged reconcile | 4.55 s | 2.95 s |
| Status | 0.99 s | 0.40 s |

Complete comparison is exactly equal for all 614 generated records, holds, review
queue, source identities, maps/candidates and 1,210 task outcomes. Only JSON object
key ordering is canonicalized for hashing; no semantic field is removed.
Output digest: `cc76388616fa44d19bc4204f64cc2915d5c89c4333624ffb5d9157929c43e1f9`.
These synthetic full-remap records are isolated benchmark outputs, never changes
to the authoritative model data or accepted live claims.

## Collection and scheduler workload

The first measured pair (same 1,253 synthetic repositories, actual disk syncs)
reduced collection from 167.14 to 4.20 seconds; unchanged collection from 94.28 to
4.14 seconds. Initial checkpoint serialization fell from 999,725,851 bytes to
2,161,029 bytes; unchanged collection fell from 1,000,550,298 bytes to zero.
HTTP request counts were unchanged: 2,509 initial, 2 unchanged. Each successful
new card and tree still has a synced record; this is not time-based checkpoint
coalescing. A sequential repeat and final validation are recorded below.

Sequential repeat (baseline code from a detached `4ad4b57` checkout; identical
synthetic inputs and eight workers, baseline followed by candidate):

| Operation | Reviewed PR | Follow-up | Ratio |
|---|---:|---:|---:|
| Enqueue 1,190 tasks | 1.944 s | 0.105 s | 18.6× |
| Claim and complete 1,190 tasks | 8.062 s | 5.358 s | 1.5× |
| Initial 1,253-repo collection | 31.453 s | 5.863 s | 5.4× |
| Unchanged collection | 49.888 s | 3.534 s | 14.1× |
| Initial checkpoint bytes | 999,767,442 | 2,161,029 | 462.6× fewer bytes |
| Unchanged checkpoint bytes | 1,000,550,298 | 0 | no checkpoint serialization |

The 100× **runtime** stretch was not achieved. The large variance between first
and sequential measurements is retained; do not treat the faster ratios as an
end-to-end production estimate. Checkpoint write count remains 2,507→2,508 on the
initial collection (one extra final compaction), demonstrating that per-success
syncs were preserved. The unchanged collection performs no ledger writes.

Every base/repo source record is exactly equal between implementations. Full
snapshot-file hashes differ solely because `captured_at` and `updated_at` record
the real times of the two runs. Their actual before/after values, all raw timing
summaries and full-model output hashes are retained in
[performance-followup.json](performance-followup.json); no source revision, hash,
metric, qualifier or attribution difference was removed.

## Reproduction and handoff

Run the commands in `docs/SCRIPTED_RUNNER.md`, using new external directories for
every repetition. `tests/benchmark_runner.py` exports semantic outputs for direct
comparison; `benchmark_runner_io.py --implementation <checkout>` runs the same
synthetic workload against either implementation. These harnesses perform no
network calls or model execution.

The owner requested independent measurement in another session. A patch and
self-contained prompt were prepared, pinned to `4ad4b57`, with exact SHA-256,
frozen input commit, three sequential alternating repetitions, full output
comparison requirements and prohibitions on remote writes/provider calls. This
session cannot provision another machine: it has no cloud-provisioning API or
cloud identity, and no Incus/QEMU/KVM. Local correctness suites run in two isolated
processes; independent-machine measurements remain pending the separate session.

## Final local verification

All 66 runner tests passed; the 18 fetch tests were rerun after the last directory
sync change. All nine live builds, 886 UI checks, assembly, gates, claims, MCP,
discovery and the fetch/source/provenance/QA-release/assembly/workflow safety
regressions passed. Raw command outcomes are in
[followup-validation.json](followup-validation.json). The legacy/new fixture replay
remains exactly equal with zero repeated successful stages after restart;
[followup-replay.json](followup-replay.json) retains the actual outputs/metrics.
The published model inputs, watchlist, README and release state match current main
exactly. Only isolated synthetic outputs were changed by the performance runs.

No remote branch, PR comment, Actions run, production schedule or deployment was
changed in this follow-up. The earlier automatic-review rejection of a push that
would trigger PR Actions remains unresolved; no attempt was made to bypass it.
Independent-machine repetitions, live-provider quality/idempotency and persistent
hosting remain outstanding. Production cutover still needs separate authorization.
