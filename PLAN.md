# Holistic hardening review

Baseline `35d7c50`; branch `review/holistic-hardening`.

- [x] Inspect current source and fixed five-model baseline.
- [x] Delegate independent source integrity, UI, system, and benchmark reviews.
- [x] Reproduce and repair defects with counterfactual regression tests.
- [x] Measure browser payload performance before/after on the same workload.
- [x] Verify all live models and all 36 regression suites.
- [x] Close final independent mapping review and MCP latency measurement.
- [x] Prepare review branch and report; production requires owner approval.

---

# Explicit comparison classifications

Branch: `feat/evidence-comparison-classifications`. Owner explicitly approved merging PR #8 and deploying GitHub Pages on 2026-10-19 after the production approval question.

- [x] Inspect build, automatic sets, chart ranking controls, MCP and pinned measurement sources.
- [x] Add evidence-required reusable pair/group/per-metric classifier and validated source bindings.
- [x] Apply shared results to build output, website ranking/UX and MCP.
- [x] Preserve published values, set IDs/order, defaults and chart geometry; compare against baseline exports.
- [x] Run source gates, regression suites, responsive checks and independent review.
- [x] Document audit, evidence scope, checks and limitations; prepare the draft PR for owner review.

Independent work: pinned-source evidence audit, classifier/tests, MCP, UI. Integration and baseline preservation checked centrally.

## Approved release
- [x] Integrate main `ca6748a` to preserve the stacked benchmark chart design.
- [x] Validate integrated layout and obtain fresh independent release review.
- [ ] Publish updated PR branch; require green CI before merge.
- [ ] Merge PR #8 and deploy GitHub Pages.
- [ ] Verify live data preservation, classification behavior and chart defaults.

## Scripted runner — first migration (workflow/scripted-runner)
Starting default-branch commit: `670acdfb0c84e4ebf609d656ac06912dfaba8f2a`.

- [x] Inspect current source, runbooks, workflow triggers and historical logs.
- [x] Record bounded legacy dispatch baseline before runtime work; 100× stretch target.
- [x] Durable fair scheduler, confined shadow workspace, one-shot/resume commands.
- [x] Direct-fetch contract, complete model evidence boundary and persisted QA.
- [x] Frozen-input shadow comparisons, fault injection and existing regressions.
- [x] Independent review; migration/rollback/runbook and local PR description.

No production commands, schedules, data edits, paid calls, merge or deployment.
A draft PR triggers `verify.yml` (`pull_request` has no filter); retain local
review branch until opening a PR and its Actions run are explicitly authorized.

## PR #14 follow-up review and performance (2026-10-21)
- [x] Read owner review and updated commit `4ad4b57`; integrate into local branch without discarding prior main integration.
- [x] Reproduce focused tests and fixed catalogue-scale benchmark before optimization.
- [x] Measure and reduce remaining task-selection/payload I/O and collection checkpoint costs without relaxing durability or validation.
- [x] Add fault/scale regressions, compare exact claims/holds/provenance, run required local checks and obtain independent review.
- [x] Record measurements, review findings and remaining provider/cutover limits.

Runtime stretch target remains 100× on identical frozen inputs, recorded before this round of optimization. Report actual improvements; do not trade away source validation, independent requests, review coverage or per-result durability. Production state machines, schedules and publication remain unchanged. The previous integration push was blocked by automatic review; this turn performs local review/testing and improvements.

## Independent measurement evidence (2026-10-21)
- [x] Retrieve supplied Drive archive and verify archive/compact file hashes.
- [x] Match patch, 93 candidate code files and 3,790 frozen input files to this branch.
- [x] Recompute medians from all 18 corrected runs; verify six exact output/task pairs and retained IO clock differences.
- [x] Update primary results and local PR description, preserving superseded/earlier measurements and scope limits.

Documentation/evidence-only update. Existing complete local regression results
remain applicable to the identical implementation; no production or remote action.
