# Measured results and limits

Latest follow-up: [independent three-repetition validation](INDEPENDENT_VALIDATION.md)
confirms 1.33x median Qwen measured-stage improvement over reviewed PR commit
`4ad4b57`, with exact output preservation. Earlier measurements below retain their
original baselines and should not be conflated with that comparison.

Starting default commit: `670acdfb0c84e4ebf609d656ac06912dfaba8f2a`.
No production entry point, model API or remote write was used for baseline replay.
The 100× runtime stretch target was recorded before implementation. It was **not
achieved**; there is no measured token or monetary saving.

| Fixed offline workload | Legacy replay | Scripted replay |
|---|---:|---:|
| Eligible mapping tasks | 24 | 24 |
| Logical mapping requests | 24 | 24 |
| Simulated coordinator handoffs | 2 | 0 |
| Completed mapping tasks | 24 | 24 |
| Actual model/agent launches in replay | 0 | 0 |
| Completed work repeated | 0 | 0 |
| Dispatch runtime | 0.0062 s | 0.0681 s |

Legacy's first dispatch starts eight worker prompts, leaving 16 until the next
handoff. The new path records every request before execution and enforces eight
simultaneous leases. This is a scheduling/recovery result, not a model-efficiency
claim. SQLite durability adds local overhead relative to inert baseline stubs.

The full frozen-card extraction replay preserves complete accepted records,
source/candidate provenance, flags and holds exactly. It produces four metric
claims plus their sizes, resolves one A/B disagreement through blind C, and
finishes with zero unresolved tasks. The new path uses 17 fixture responses
(mapping and QA separately represented in its journal). Repeated unchanged
reconciliation performs **zero stage subprocesses**. Both complete output JSON
files and their exact digest are retained alongside these results.

Old safe extraction stages took **2.7560 s**; the new durable runner including
its semantic-review fixture path took **4.4694 s**. These timings exclude real
provider/network latency and do not measure the whole legacy scheduled system.
The old extraction timing does not include a legacy coordinator or live QA model.
Do not use the ratio as an estimate of production latency or savings.

Partial QA replay: current `assess_reviews` blocks a two-item sample after one
verdict; current `qcl_qa` then ends the run and a fresh frozen session would repeat
the completed item (one repeated logical review, inferred from that control flow).
The journal instead persists the first result and claims only the second.
The current raw-value selector misses an unchanged-number/new-panel counterfactual;
the new complete-evidence identity detects it. Revision-only updates with identical
bytes generate no mapping or QA requests, including an over-20 sample and in-flight
review. These are tested counterfactuals, not labelled model-quality measurements.

Validation:

- **51 focused tests passed** (`unittest discover`, 69.237 s on the final full run),
  covering integration, faults, persistence, fetch and model contracts.
- A subsequent focused fetch check confirms complete imports also clear an old
  `not_fetched` marker; **14 fetch tests pass**.
- **9 workflow-security tests pass** after adding test invocations only; triggers
  and publication jobs are unchanged.
- All eight existing live builds and UI suites pass (786 UI checks), along with
  existing source/gate/claim/provenance/release/MCP/discovery suites. All 40 model-site
  and 30 assembled files remain byte-identical after the degenerate-axis fix.
- Real pinned LFM2.5 input: zero model requests, all 335 authoritative files unchanged,
  and exact new-versus-existing-stage equality. Both current paths regenerate 19
  image-warning flags missing from committed output; these explicit differences
  are retained in `real-input-check.json`, not normalized away or applied live.
- Public read-only check: two real HF GETs, HTTP 200, pinned README 8,092 bytes.
  This is a transport/source-integrity smoke check, not a full live catalogue or
  provider validation.

Historical logs are reported separately in `baseline.json`: 48/33/30/11 records,
with 17/6/3/1 finish calls respectively. Logs expose requested connector op-list
counts and stage durations, not actual coordinator sessions, billed requests,
tokens or money. The documented schedule arithmetic is 24 + 12 + 24 + 1 = 61
coordinating sessions/day plus weekly discovery; it is an estimate, not billing.
Scheduler-side prompt definitions are still unavailable in the repository.

## Orchestration at catalogue scale (2026-10-21)

The fixture replays above hold seven cards. `tests/benchmark_runner.py` repeats
the same offline measurement on committed model roots with every mapping
forgotten, so each unique card is queued for runs A and B (synthetic fixture
responses; no model calls, tokens or cost). Before this change the journal was
listed and every payload (a whole numbered card) decoded once per task, on each
completion and on each enqueue, so orchestration grew quadratically. A full
re-map of the 1,253-repository Qwen3.8-27B root (1,190 mapping tasks) decoded
722,555 payloads during `prepare` alone. Filtered SQL listings, single-row
completion and slot lookups, a byte-keyed index cache, one decoded source per
card, enqueue bursts committed as one durable transaction, requests exported
from the journal's canonical text and install-only-if-changed outputs remove that:

| Offline workload (4 cores, local disk) | Before | After |
|---|---:|---:|
| LFM2.5-24B-A2B, 75 repos, 106 mapping tasks: `prepare` | 7.05 s | 3.98 s |
| LFM2.5-24B-A2B: `run` consuming 106 responses, then blind C and QA round | 12.58 s | 3.89 s + 0.54 s |
| LFM2.5-24B-A2B: unchanged reconcile / `status` | 0.72 s / 0.20 s | 0.29 s / 0.11 s |
| Qwen3.8-27B, 1,253 repos, 1,190 mapping tasks: `prepare` | 569.1 s | 52.2 s |
| Qwen3.8-27B: `run` consuming 1,190 responses, then blind C and QA round | stopped after 35 min, 573 of 1,190 done (> 70 min projected) | 60.3 s + 4.7 s |
| Qwen3.8-27B: unchanged reconcile / `status` | not measured | 2.3 s / 0.36 s |

What remains at Qwen scale is dominated by the existing stage subprocesses
(extraction, merge and regeneration, about 15 s per reconcile), one fsync per
exported request on first export, and one durable transaction per completed
task. These are the durability contract, not bookkeeping. Timings are single
local runs, not provider latency; results, holds and provenance are unchanged
(`test_runner*.py`, 55 tests, and the old/new replay still agree).

## Stage scripts (2026-10-21, second review)

After the orchestration fixes, a Qwen reconcile is dominated by the stage
subprocesses it wraps, above all `qcl_watch.py merge`, which production's hourly
`finish` also runs for every model on every call. Two changes, both with
byte-identical gate output on all nine live models and identical merged claims:

- `gates/verify_columns.py` memoizes its pure label helpers (`_strip_md`,
  `_ltoks`, `compact`, `_lab`, `_shape`): 2.4 million regex substitutions per
  Qwen run became a dictionary lookup. Qwen3.8-27B 3.69 s -> 1.62 s,
  Qwen3.8-Flash-Next 2.24 s -> 1.04 s.
- `qcl_watch.py merge` runs its source/column check through `gates/gatecache.py`
  (three gates in parallel, reused only for byte-identical inputs). Before,
  every merge pass ran the gates serially and then `build.py` ran the same gates
  again, cold, on the same claims file. Qwen3.8-27B merge 23.6 s -> 11.2 s
  (three gate passes), with `verified/part8_machine.py` and `maps/mapped.json`
  byte-identical and `audit/review_queue.json` equal as multisets.
- The merge's quarantine order was the iteration order of a set of tuples, so
  `review_queue.json` changed with `PYTHONHASHSEED` on every run (one source of
  the 900-line audit diffs in every hourly commit). It is now sorted; a repeated
  merge is byte-identical, and the runner's stage receipt no longer changes for
  an unchanged input.

`tests/benchmark_runner.py` does not show these gains: its synthetic responses
map one value per card, so its merge passes have almost nothing to gate (Qwen
prepare 26.1 s, response round 42.5 s + 2.5 s, unchanged reconcile 1.2 s on an
idle machine, within noise of the previous head). The merge figures above come
from the committed Qwen3.8-27B maps (616 real A/B maps, 4,020 values), which is
what production's `finish` and a real shadow reconcile process.

Remaining adoption work: configure and validate a real interpretation adapter,
independent precision/recall including shared omissions, persistent storage,
production deployed-baseline integration, and exact-commit/full-artifact release
verification. The current verify command is deliberately limited to gated shadow
model builds. No production schedule/cadence, publication, remote branch or PR changed.
