# QFF holistic review

Baseline: `35d7c508613f19e37053380915b153dfd1c10c55`, the published cross-model benchmark release.
Review branch: `review/holistic-hardening`. Production publication is not part of this review.

## Outcome and scope

The review covers source fidelity, comparison semantics, model and benchmark
interactions, accessibility, performance, the read-only MCP server, collection,
assembly and release safety. Methods combine provenance auditing, fault injection,
protocol conformance testing and reproducible performance measurements.

The fixed workload contains five live models, 2,191 repositories, 3,086 exported
rows, 157 measurement sets and 113 benchmark reports from seven sources. Every
model data export, the shared benchmark export and the model manifest remain
byte-identical to the baseline. No source cards, published values or verified
claim files are rewritten. The new review-pin registries bind 876 legacy reviews
to their exact entry and source hashes; one external documentation source is also pinned.

Design contracts retained: initial measurement sets and chart defaults, collapsed
evidence and quant controls, benchmarks after the fidelity chart, direct benchmark
selection with all sources, and the mobile size/score slider pattern.

## Findings and repairs

| Area | Reproduced problem | Result |
|---|---|---|
| Gate caching | A relative data root could exclude actual source/claim bytes from cache keys; changed inputs could inherit an earlier passing run | Resolve paths once; include gate helpers, claims, pins and source bytes; compare fingerprints again after execution; replace caches atomically |
| Source fidelity | Recorded SHA-256 metadata was not enough to detect edited card bytes; mutable branch names could retain a stale review | Check exact bytes; pin legacy reviews to whole-entry fingerprints; withdraw changed sources until reviewed again |
| Mapping lifecycle | `--record` could assign the current card digest to existing old maps, and merge then published stale methodology under the new pin | Bind new candidate tasks and maps at creation; legacy maps retain only their previously recorded pin; changed cards require fresh work |
| Image evidence | Missing/changed images, unsupported metric shapes, empty/nonfinite pixel windows and a missing PPL marker could evade checks | Require image pins and supported shapes, finite in-bounds coordinates, and independently validate both score markers and reference lines |
| Units and signs | A source GB listing could be relabelled GiB; a Unicode negative sign could disappear | Verify source units and signs before publication; retain the raw published strings |
| Benchmark attribution | A number on the cited line could belong to a different benchmark, variant or unit | Validate supported source/table schemas and row/column identity; unknown schemas fail closed; all 113 existing reports preserved |
| Collection | Credentials could follow an off-origin redirect; retries continued after rate limiting; a reused output could mix model ledgers | Limit authentication to the source origin, reject HTTPS downgrade, stop new requests after 429, retain completed checkpoints, validate destination identity before writes/network |
| Assembly | Missing live outputs or unsafe destination paths could cause incomplete output or data loss | Preflight all live models and static outputs; stage the whole assembly; retain/restore the previous complete output on failure |
| QA | Missing, uncertain or duplicate verdicts could proceed; Git errors could look like no changes; lock commits could absorb staged work | Require one decisive explained verdict per sample; check every Git result and publication dependency; preserve dirty work and unpublished commits |
| Deployment | A dispatch was recorded as deployed before external success; a branch could advance between tests and dispatch | Track a pending exact commit, reject branch races, deploy the reusable workflow's tested artifact, and advance the receipt only after a matching successful workflow |
| Explorer | Fourteen real zero-KLD claims disappeared on linear plots; exact negative thresholds could exclude boundary points; share links lost filters and precision | Retain zero values on linear, correct inclusive negative boundaries, serialize complete chart state and restore focus during redraws |
| Keyboard controls | Rerenders could lose focus and radio controls lacked expected keyboard movement | Restore focused controls, implement arrow/Home/End movement, expose actual slider values, and return focus when the benchmark picker closes |
| Browser payload | Protocol objects and aggregate diagnostics were duplicated in inline chart JSON | Project only browser-used fields; keep full source/protocol evidence in the unchanged public data export |

## MCP alignment and current SDK

The server now exposes nine read-only tools. `list_benchmarks` and
`search_benchmarks` bring the latest website benchmark coverage to agents. They
support exact benchmark/unit groups, source and publisher filters, size/score
limits and bounded pagination. They retain raw values and source provenance;
matching benchmark names never imply compatible protocols or a blended ranking.
The existing search, repository, measurement comparison, frontier and budget tools
continue to enforce per-metric ranking permission.

Other repairs include a bounded refresh interval, invalidation even when model
counts do not change, serialized cache updates, explicit refresh failures,
manifest/data schema and identity validation, path/redirect containment, response
size limits, finite positive budgets and actionable protocol errors.

The default dependency is pinned to **MCP Python SDK 2.3.0**, the current stable
release checked on 2026-10-19. The v1.30 adapter remains covered by transport tests.
Official references: [2.3.0 release](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.3.0)
and [SDK documentation](https://py.sdk.modelcontextprotocol.io/).

Real HTTP testing exposed two integration defects beyond ordinary function tests:

- Public hosting must explicitly allow the configured hostname while rejecting
  foreign Host and Origin values. Tests expect HTTP 421 and 403 respectively.
- Qwen3.8-27B responses exceed the current SDK's default 1 MiB SSE event limit.
  QFF now uses JSON HTTP responses for its request/response tools. The normal
  client receives 3,108,224 bytes for 93 sets and 1,815,929 bytes for 200 search
  rows without increasing its limits. These byte counts measure serialized text
  content, not necessarily total wire traffic.

Tests cover initialization, tool discovery, empty resource/prompt listings, all
nine tools over stdio and HTTP, modern protocol discovery, failures, cache refresh,
comparison eligibility and all 28 benchmark/unit groups with all 113 reports.
The repository supplies a placeholder Space URL; local integration success does
not establish public endpoint health. No Space or Pages deployment was performed.

## Performance target and method

The target, recorded before optimization, is **100×** for each measured dimension
without changing claims, source evidence, comparison eligibility or approved
defaults. Correctness and accessibility are not described as numerical speedups.

`tests/benchmark_payload.py` measures the exact same five-model snapshot before
and after using Node/V8, ten warmups and nine batches of thirty JSON parses. Each
number below is the median batch mean, with disk I/O outside the timed section.
The reports retain min/max batch times and data-export hashes. This measures JSON
parsing only, not whole-page rendering, user interaction or network latency.

| Model | Chart JSON bytes before → after | Parse ms before → after | Measured speedup | 100× target ms |
|---|---:|---:|---:|---:|
| deepseek-v4-flash-0731 | 198,912 → 92,812 | 0.603 → 0.318 | 1.90× | 0.0060 |
| deepseek-v4.1-flash | 368,283 → 168,896 | 1.144 → 0.543 | 2.10× | 0.0114 |
| glm-5.3-flash | 357,329 → 163,505 | 1.066 → 0.579 | 1.84× | 0.0107 |
| qwen3.8-27b | 4,889,322 → 1,576,902 | 12.802 → 5.228 | 2.45× | 0.1280 |
| qwen3.8-flash-next | 1,173,017 → 573,856 | 4.095 → 1.549 | 2.64× | 0.0410 |

Runtime: `v24.19.0`. Raw reports: [before](review/performance-before.json), [after](review/performance-after.json).

The 100× target was **not reached**. Summed parse time across the five pages is
19.709 → 8.218 ms, or 2.40×. Qwen3.8-27B HTML falls from 7,146,224 to 3,838,296
bytes (46.3% smaller); its chart JSON falls by 67.7%. Total assembled output
falls from 28,372,375 to 23,984,020 bytes (15.5% smaller). Gzipped HTML across
the five pages falls only 984,739 → 911,684 bytes (7.4%); repeated evidence
already compressed well, so uncompressed reductions cannot be called network speedups.

Remaining limits and quantified bounds:

- Unchanged public data exports occupy 14,750,779 bytes, 61.5% of the final
  static output. Even removing everything else would yield only 1.92× against
  the original artifact size. A 100× artifact reduction requires changing the
  delivery contract or compression representation, not dropping required evidence.
- The largest page still contains 1,576,902 bytes of chart JSON. Reaching its
  100× parse target requires another 40.8× reduction in measured parsing time.
  Loading just the selected set and hydrating others on demand could move work
  off the initial path, but would need separate cold-network, offline and share-state
  benchmarks; no unmeasured speedup is claimed.
- The largest HTML document has 2,261,394 bytes outside chart JSON. Even eliminating
  its inline payload entirely buys at most another 1.70× reduction in its current
  uncompressed HTML. Further gains require deferred ledger rendering or pagination,
  with explicit accessibility, no-JavaScript and source-discovery validation.
- Full source hashing and gate execution remain intentional build costs.
  Incremental source verification must preserve the changed-input counterfactuals;
  trusting mutable revisions or reusing stale approvals is not a performance fix.


Cold builds (gate caches cleared) took 11.89 → 11.53 seconds across all five
models; assembly took 0.53 → 0.40 seconds. These are single timing runs,
not a statistically established speedup. The build target is 0.119 seconds;
no material build-speed improvement is claimed. Final assembly deliberately does
more preflight validation than the original.

### MCP latency experiment

A separate [actual-transport benchmark](../tests/benchmark_mcp.py) runs ten requests
per round across all five models: all set classifications and best-under-100-GB
queries. One warmup and five timed rounds exclude startup/import/source loading.
The 100× target was recorded before experimenting. Profiling found repeated
classification work, so an isolated bounded cache keyed by complete current
protocol evidence was tested; response hashes stayed identical.

The repeated paired run measured stdio 365.5 → 317.5 ms (1.15×) and HTTP
261.0 → 228.6 ms (1.14×). Timing ranges overlap. Each round returns 6,947,489
bytes of evidence text; hashing, copying and serialization remain significant.
A 100× target requires roughly 3.66 ms for that stdio workload and 2.61 ms for
HTTP, versus more than 200 ms in the prototype. The prototype was **rejected**:
no production cache or retained MCP speedup is claimed. A smaller summary-first
response contract is the next architectural lever, with full evidence fetched
on demand; that needs an explicit API design and fresh workload measurements.

Both paired runs, profiles, raw timings, unchanged-response hashes and the
isolated experimental patch are preserved in
[review/mcp-performance.json](review/mcp-performance.json).

## Verification and review

The final verification record is saved in [review/verification.json](review/verification.json).
It includes each regression suite, the five gated builds, per-model UI runs,
MCP checks on both SDK generations, and the data-preservation hashes.

The checks exercise real source snapshots and generated pages, plus deliberately
corrupted inputs: altered values/units/signs, changed source/image bytes, missing
or duplicated QA verdicts, failed Git/API responses, hostile redirects, unsafe
assembly destinations, stale mappings, and oversized MCP responses. Browser
coverage includes 320/390/768/1366px layouts as applicable, native disclosures
without JavaScript, share-state round trips, focus, chart controls and every live
model. The final build exports are compared byte-for-byte with the saved published
baseline, not just compared by counts.

Three independent reviewers examined integrity, build/release safety and MCP/UI
behavior. Their counterexamples found the stale-map approval path, false Git
skip, unrelated staged-file capture and SDK2 large-response failure. Fixes were
reviewed by someone other than their author and added to regression coverage.
Every test suite is required by CI; the local QA publisher discovers them all.
The v1.30 transport compatibility check runs in its own environment.

## Boundaries and rollout

- This remains a claims-only site, not a new model evaluation. Passing gates do
  not prove publisher methodology is complete or that all benchmarks are comparable.
- Image validation is calibrated marker/axis verification of supported pinned
  charts, not general-purpose OCR or a statistical re-evaluation of their measurements.
- Source-pin registries are an audited one-time compatibility migration. Their
  installer refuses changed existing registries and runs source gates before installing.
  Future mappings carry explicit source/candidate pins; never rebaseline stale claims.
- Full review uses the fixed source snapshot. Newly fetched cards still need the
  same independent mapping and QA workflow.
- HTTP tests use local real servers and production-sized data. A real hosted MCP
  URL and deployment approval are still needed for a public endpoint smoke test.
- Review and merge this branch before any owner-approved production deployment.
  Pages verifies and deploys the same commit/artifact. The MCP Space requires a
  separate explicit deployment and post-deployment check.
