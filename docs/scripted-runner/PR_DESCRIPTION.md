Replace coordinator-mediated shadow runs with a durable scripted pipeline

The current collection loop delegates mechanical fetch/dispatch work to sessions,
initially slices mapping dispatch to eight prompts, and restarts completed QA after
partial review. This first migration adds a local shadow runner that enqueues all
eligible work and persists individual results before acknowledgement.

The runner reuses existing extraction, A/B/blind-C agreement, source pins, holds
and gates. It adds SQLite leases/fair scheduling, portable checked backups, direct
HTTP ledger integration, complete vendor-neutral interpretation requests, semantic
QA reuse and a separately gated shadow verification receipt. Production schedules,
state machines, release receipts and deployment interfaces remain unchanged.

Validation includes 51 focused offline/fault-injection tests, old/new frozen-input
claim/provenance equality, unchanged/revision-only no-request checks, real pinned
repository inputs, bounded public HTTP reads, all nine current live builds and UI suites,
and existing source/release/MCP/discovery regressions. A fresh independent reviewer
challenged recovery and publication boundaries; identified blockers were fixed and
regressed. Evidence and exact commands are in `docs/SCRIPTED_RUNNER.md`.

A small-model fixture exposed existing static-chart zero-width/log-decade axes;
a separate focused fix leaves all 40 model-site and 30 assembled files byte-identical.
No published claims were edited. The 100× runtime stretch target was not achieved;
no token or cost savings are claimed.

This is review-ready shadow infrastructure, not production adoption. Live provider
execution/quality, deployed-baseline QA integration, persistent hosting and an
authorized shadow trial remain outstanding. The owner authorized draft PR #14 and its automatic verification on 2026-10-21.
Current main (`77c157c`) was integrated into the review branch and required local
checks were rerun successfully; its QA/data changes were preserved exactly.
Merge to main, deployment and production cutover remain outside that approval.
