# Measured results and limits

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

Remaining adoption work: configure and validate a real interpretation adapter,
independent precision/recall including shared omissions, persistent storage,
production deployed-baseline integration, and exact-commit/full-artifact release
verification. The current verify command is deliberately limited to gated shadow
model builds. No production schedule/cadence, publication, remote branch or PR changed.
