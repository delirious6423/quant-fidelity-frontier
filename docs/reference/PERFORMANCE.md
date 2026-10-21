# End-to-end performance: baseline, target, result

Every number here is a measurement on a fixed workload, named so the same measurement can be
repeated. "Before" is the state of `main` at commit `83a9d19` (21 Oct 2026). The target was
written down before the work started: 100× on everything code controls. Where a stage is
bounded by something code does not control (a model interpreting a card, a browser rendering
a page), the floor is stated instead of lowered.

## Workloads

| Id | Workload | How to repeat |
|---|---|---|
| W1 | Hourly live-updates cycle with a real delta: 6 models changed, 14 mapping prompts, 5 tie-break models | `logs/scan/20261020T1412Z-live.jsonl` (timestamps), scheduled task "QFF live updates" |
| W2 | Hourly cycle with nothing changed on any live model | `logs/scan/20261019T0204Z-live.jsonl`, `20261019T0706Z-live.jsonl` |
| W3 | Deterministic publish path on all 9 live models, locally, one process at a time: build, assemble, README sync, gates and claims suites, MCP and discovery suites | `scratchpad/baseline_e2e.sh` (the commands of `docs/reference/DEVELOPING.md`) |
| W4 | CI `verify-claims` on one pull request | GitHub Actions run 39550495008 |
| W5 | A full new-model ingest: LFM2.5-24B-A2B, 75 repositories, 53 unique cards mapped from scratch (A, B, C and QA) | `docs/history/scripted-runner/SHADOW_TRIAL.md` |

## Before (measured)

### W1: live-updates cycle with a delta, 35.8 min

| Stage | Minutes | Who does the work |
|---|---:|---|
| lock, plan, search-ops | < 0.1 | script |
| catalogue: 8 connector calls by the coordinating agent, then `catalogue` | 0.65 | agent + connector |
| fetch: 6 FETCH worker agents, one per shard | 2.9 | worker agents + connector |
| prepare | 0.1 | script |
| mapping round 1 (8 of 14 prompts) | 10.4 | worker agents |
| mapping round 2 (the 6 prompts round 1 never started) | 6.5 | worker agents |
| blind tie-breaker run C | 13.7 | worker agents |
| stage, commit, push, unlock | 0.6 | script |

Of 35.8 minutes, 33.5 are interpretation (worker agents) and about 2.3 are code and
collection. The three worker rounds run one after another.

### W2: cycle with nothing changed

Script time between lock and unlock: 0.2 to 0.3 min (catalogue calls and their parsing). The
agent session around it (clone, dependency install, connector calls, reporting) is not in the
logs; the QA task's skip run on 20 Oct took 28 s end to end as recorded by the scheduler.

### W3: deterministic publish path, local

| Step | Seconds |
|---|---:|
| build, 9 models, cold caches | 13.0 |
| build, 9 models, warm caches | 13.1 |
| assemble `_site` | 4.9 |
| README sync check | 0.3 |
| `tests/test_gates.py` | 20.8 |
| `tests/test_claims.py` | 0.1 |
| `tests/test_ui.py`, per model (browser) | 14 to 19 |
| `tests/test_mcp.py` | 4.8 |
| `tests/test_discovery.py` | 1.4 |

Build is already cheap; the gate cache saves 2 s on the largest model. The browser suites are
the cost: 9 models × about 16 s.

### W4: CI verify-claims, 7 min 25 s

| Step | Seconds |
|---|---:|
| checkout, Python, pip, Playwright install | 31 |
| assembly and workflow safety suites | 55 |
| gates + build + UI tests, 9 models in sequence | 171 |
| benchmarks and responsive suites | 40 |
| gate regression, claims, panel and column suites | 27 |
| benchmark view and extracted data | 21 |
| homepage, model page, explorer | 52 |
| comparison engine and MCP suites | 39 |
| the rest | 9 |

### W5: full new-model ingest

Policy today: one queued model per 5 hours, mapped in 40-minute slices every 2 hours by worker
agents. The shadow trial mapped the same model end to end with 18 parallel workers in about 7
min of interpreter wall clock and 15 s of orchestration.

## Target (written before the work)

| Workload | Before | Target | What code controls |
|---|---:|---:|---|
| W2 no-change cycle | 2 to 3 min of agent session (estimate), 0.25 min script | 2 s | everything: a scheduled HTTP catalogue with nothing to do |
| W1 code and collection share | 2.3 min | 1.4 s | catalogue and fetch over HTTP in-process, no connector, no FETCH agents |
| W1 interpretation share | 33.5 min in 3 serial rounds | floor: one round of interpretation at the interpreter's own latency, all prompts at once | rounds, not latency |
| W3 deterministic path | 45 s without browser suites | 0.45 s | only what changed is rebuilt and retested |
| W4 CI wall time | 7 min 25 s | 4.4 s is not reachable: a browser suite alone takes 14 s; stretch 2.5 min critical path | parallel jobs, cached setup |
| W5 full ingest | 5 h policy, 40-min slices | interpreter-bound; orchestration 15 s | the loop runs all prompts at once through any interpreter |

## After

Filled from measurements on the merged changes.
