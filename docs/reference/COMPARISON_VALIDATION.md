# Comparison classification validation

Feature branch: `feat/evidence-comparison-classifications`. Integrated with production base `ca6748a` (stacked benchmark charts and collapsed quant/evidence sections). Owner explicitly approved merge and GitHub Pages deployment; the integrated release is awaiting green CI.

## Preservation check

Before changing implementation, every live model was built and its JSON exported. An independent reviewer compared every original row field and every original point field against the rebuilt output, including raw/parsed claims, source/methodology, coordinates, axis definitions, references and metric/set ordering. The latest production update changed layout only; its source data matches that baseline.

| Model | Claim rows retained | Original sets retained | Set points retained |
|---|---:|---:|---:|
| Qwen3.8-27B | 1,346 | 93 | 1,296 |
| Qwen3.8-Flash-Next | 1,098 | 36 | 399 |
| GLM-5.3-Flash | 199 | 11 | 128 |
| DeepSeek-V4.1-Flash | 283 | 10 | 76 |
| DeepSeek-V4-Flash-0731 | 160 | 7 | 67 |
| **Total** | **3,086** | **157** | **1,966** |

Result: all original numerical claims, sources, points, set IDs/order, metrics/order and axes retained exactly. Initial overview selection, saved/query/hash precedence, metric and log/linear defaults are covered by browser tests. AtomicChat retains all 27 existing chart points and supports KLD/top-1 comparison across quant publishers; its unrelated metrics are not implicitly verified.

## Actual test output

The following commands completed successfully after building all five live models and running `assemble.py`:

```text
python3 -m unittest discover -s tests -p test_comparison.py
Ran 43 tests ... OK
python3 -m unittest discover -s tests -p test_protocol_metadata.py
Ran 10 tests ... OK
python3 -m unittest discover -s tests -p test_verified_protocol_sets.py
Ran 15 tests ... OK
python3 -m unittest discover -s tests -p test_mcp_comparison.py
Ran 19 tests ... OK
python3 tests/test_mcp.py
33 passed, 0 failed
```

The 87 focused regression tests include all requested same/different/unknown publisher/protocol/base-model scenarios; missing references/datasets; contradictory metadata; copies/quotes; shared evaluators; per-metric corpora; stale source hashes/altered quotes; scoped equivalence endpoints; and non-transitive grouping. Eighty seeded randomized cases independently compare memoized grouping to naive all-pairs classification. MCP smoke tests use actual AtomicChat claims and cross-model records, and a real subprocess/stdio client.

Existing gate/transcription checks passed: `test_gates.py` (15 checks), `test_claims.py` (67 checks), `test_model_panels.py` (9), `test_mapping_panels.py` (22), `test_combined_columns.py` (8), `test_column_caption.py` (8), `test_readme.py` (4), `test_benchmarks.py` (5), plus qualifiers, current-source freshness and discovery/export checks. The export preserves 3,086 claims. `readme_sync.py --check --structure`, Python compilation, JS syntax and `git diff --check` also passed. Every live build passed its source/column/image gates.

Browser suites:

| Suite | Actual result |
|---|---|
| Existing explorer tests, all five models, desktop and mobile | 502 passed, 0 failed |
| Model layout/default/deep-link tests | Five pages at 320/390/1366px passed |
| Homepage | 360/390/768/1366px plus JavaScript-disabled navigation passed |
| Benchmark tabs/pilot | 320/390/1366px passed |
| Comparison-specific UI | 320/390/1366px passed |

The comparison UI suite checks metric-specific eligibility, disabled unverified sorts/frontiers, distinct publishers, explicit unknown original evaluator, source details, all points retained, cross-view measurement focus despite differing quant IDs, and no horizontal overflow. Screenshots are [linked here](../history/previews/comparisons/README.md).

The assembled deliverable remains 21 static files, approximately 27.4 MB, below GitHub Pages limits. No database or site-serving backend was introduced.

## Independent adversarial review

A reviewer who did not author the implementation reproduced and then independently verified fixes for four classes of failure: empty metadata masquerading as agreement; unscoped equivalence evidence unlocking unrelated experiments; MCP prefix selection losing corpus overrides; and quoted/copied cards acquiring evaluator ownership. The reviewer also checked explicit-null ownership in generated cross-publisher views and measurement focus in destination views. Final review found no remaining blocking findings.

Compatibility is based on published evidence, not independent reproduction. Currently two real experiments have reviewed protocol bindings; other measurement sets remain visible and unverified. Absolute checkpoint/dataset/token hashes, evaluator revisions and some KL definition details remain unavailable in those sources and are explicitly limited to the shared experiment.

## Bounded group-classification benchmark

The first unoptimized classifier required 24.540644497 seconds for the fixed actual Qwen overview KLD population: 540 measurements and 145,530 pairs. The target was at least 100× improvement (245.406 ms). After exact-profile memoization and the final correctness fixes, the median of nine runs was 10.069 ms: approximately 2,437× faster than that original run. The entire structured output matched the naive baseline exactly, including all pair counts, 354 compatible pairs, evidence/reasons and first failure examples. Workload SHA-256: `50a3a80dfc886ae071026ef066a1a6da8783975f3ec23bef159bc362b0fced0b`.

This measures classification of that fixed population, not page-load or model-inference speed. No frontend performance claim is made.

## CI SDK compatibility

GitHub CI installed MCP SDK 2.3.0, which exposes error results as `is_error`; SDK 1.30.0 uses `isError`. The black-box tests accept both spellings. Both SDK versions passed all 33 MCP smoke checks and all 19 MCP comparison regressions, including accepted verified rankings and rejected unverified rankings.

## Final release integration

Main advanced to `ca6748a` during approval. The merge preserves its stacked benchmark charts and removes the obsolete tab styles/scripts while keeping comparison disclosures. Rebuilt GLM and Qwen3.8-27B with all source gates; benchmark and comparison browser suites passed at 320/390/1366px. A fresh independent release reviewer reran all 87 classification, adapter, grouping and MCP regressions successfully and found no blocking code issues.
