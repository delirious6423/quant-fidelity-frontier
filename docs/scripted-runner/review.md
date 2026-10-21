# Independent implementation review

A fresh reviewer agent inspected the orchestration, SQLite journal, fetch adapter,
model boundary, workspace confinement and replay implementation. Review used only
local code and isolated offline tests; no production entry point, remote write,
schedule or deployment was invoked.

## Findings and changes

| Finding | Outcome |
|---|---|
| Missing response file could abandon an already-claimed batch and starve later ready responses | Fixed: atomic claims filter eligible response IDs; submitted results are persisted. Integration regression passes. |
| `resume` did not recover expired `running` records because it never called claim for them | Fixed: running records with responses reach atomic lease recovery and bounded backoff. Integration regression passes. |
| Absolute model slug in loaded `runner.json` escaped the shadow root | Fixed: validate loaded model names and enforce model-directory containment. Independently reproduced rejection; regression passes. |
| Task-version invalidation left installed maps and source-only mapped pins active | Fixed: remove invalidated journal-owned map outputs and pins before planning replacement work. Integration regression passes. |
| One failed fetch prevented independent later models from progressing | Fixed: retain explicit fetch blocker and continue independent model work. Verification stays blocked. Regression passes. |
| Global orchestration lock covered HTTP retries, preventing timely worker completion | Fixed: collection retains its own ledger lock; only installation takes the orchestration lock. Code inspected. |
| Plan command returned only existing task counts | Fixed: compute actual eligibility in a disposable isolated copy; original workspace remains byte-identical. Regression passes. |
| Ingest command required an existing repository model folder | Fixed: `init-ingest` creates a separate blocked new-model workspace, requiring successful collection before processing. Regression passes. |
| No separate gated shadow verification command | Fixed: per-model gated builds produce artifact hashes and a deliberately nonpublication receipt; full release verification remains separate. Regression passes. |
| Small fixture triggered existing static-chart zero-width axes | Focused `build.py` denominator guards; regression covers a model below 20 GB and a single exact logarithmic decade. No numeric claim changes. |

## Independently executed checks

- `python3 -m unittest discover -s tests -p 'test_runner*.py'`: initial 36 component tests passed (38.373 s).
- `python3 -m unittest discover -s tests -p 'test_runner.py'`: initial 11 integration regressions passed (37.165 s).
- Read-only confinement probe: an absolute authoritative model path in loaded
  configuration now fails with `invalid registered model slug`.

The component tests alone did not expose the initial orchestration defects; the
integration regressions were added in response to this review. Test timing is
local wall time, not model/provider latency or billing telemetry.

## Final follow-up

The review found and closed two additional recovery cases:

- A revision-only change could resample the next 20 previously unsampled values,
  or invalidate an in-flight review, despite byte-identical evidence. The sample
  generation now uses source-content hashes and mapping bytes. Request receipts
  retain their original revision, and reuse compares the complete semantic bundle
  while excluding only that revision and the derived envelope fingerprint.
  Explicit reuse records identify the current revision and original reviewed task.
  A 30-item fixture verifies that the same 20 items remain selected and an already
  running review can complete after the revision changes.
- A crash between saving the sample and enqueueing its tasks could lose required
  reviews after a revision-only update. The sample now durably stores complete
  original request bundles, allowing exact re-enqueue on resume. Status and verify
  block absent samples, missing sampled verdicts and malformed sample membership.
  A regression deletes the review journal rows to simulate that interruption,
  changes the revision, and confirms original requests return pending and verify
  refuses to proceed.

Verification now reconciles persisted holds before building, checks task/sample/
fetch blockers under the orchestration lock, and rechecks them after building.
Inherited `QCL_*` settings cannot disable gates. Backup also refuses an active
collector instead of taking a mixed checkpoint. Conflicting same-lease duplicate
results persist a blocking state; identical delivery remains idempotent.

Additional independently executed checks:

- The expanded 14 integration tests passed (43.425 s).
- Four targeted final regressions passed (29.356 s): sample-before-enqueue crash,
  over-20/in-flight revision reuse, completed revision reuse, and the gated shadow
  receipt's publication boundary.
- Final `python3 tests/test_runner_state.py`: 10 passed (0.286 s).
- Final `python3 tests/test_runner_fetch.py`: 14 passed (1.153 s).

No remaining blocking finding was identified in the reviewed local shadow scope.
This review is not production-adoption approval. It does not establish live model
quality, provider request/idempotency behavior, or a production release receipt.
Full exact-commit/exact-artifact release verification and an authorized provider
trial remain necessary before cutover. Existing regression, real frozen-source
and public read-only evidence are reported separately in `verification.md` and
its linked records.

## PR #14 performance follow-up (2026-10-21)

A fresh reviewer examined the incoming `4ad4b57` changes and follow-up SQL,
checkpoint/recovery, export and source-binding changes. It reproduced and helped
close three correctness findings: stale complete freshness after malformed ledger
recovery; Python equality hiding a malformed `0`/`false` map field; delayed task
acceptance after base-model identity changes. Export bytes, explicit UTF-8 and
parent-directory syncs were also checked. Focused regressions cover each finding.

Independent checks: 16 state tests and 18 fetch tests passed. After final fixes,
the reviewer reran the changed-model delayed-result and process-death/restored
checkpoint counterfactuals and checked the diff. Final finding: no remaining
blocker for local shadow operation. Fair scheduling, independent durable claims,
source validation and publication boundaries remain intact. No workflow,
deployment, qcl_next or qcl_qa modification was introduced in this follow-up.

Live-provider quality, hosted persistence and production cutover were not tested
or approved by this review. Independent-machine performance repetitions are being
handed off separately; reported local results are not substitutes for them.
