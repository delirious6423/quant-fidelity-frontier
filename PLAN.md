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
