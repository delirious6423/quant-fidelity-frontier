# Fast path for the next model

## Baseline
The first build (Qwen3.8-Flash-Next) took about **11.5 h wall clock**, of which about 5.5 h was active work:

| Where the time went | Time |
|---|---|
| Fetching through the browser | about 2 h |
| LLM agents typing every value as Python, in 3 waves (44 cards by hand, 152, then 257 re-read) | about 2 h |
| Rework after two external reviews found whole classes of errors | about 3 h |
| UI (now reusable) | about 1 h |

## Target
**At least 100× faster:** under 7 minutes from a model id to a site that has passed its gates and is ready to preview. Stretch goal: under 3.5 minutes.

## Measured stages (this repo, 17–18 Oct 2026)

| Stage | Before | Now | How measured |
|---|---|---|---|
| Fetch 453 cards at their pinned sha, plus file trees | ~2 h (browser, 429s, login) | **5.3 s** against a mock HF API with 40 ms latency and a 429 on every 7th request. All 453 cards byte-identical; a resume run takes 0.17 s | `qcl_fetch.py`; mock in scratchpad |
| Deterministic extraction of every table cell, prose claim and bpw label | n/a | **1.3 s** for 453 cards. 92% of verified values land on a parsed line, and 83–86% of core metrics get the right metric family without any LLM | `python3 autoextract.py eval` |
| Triage: cards that need no LLM | 0 | 64 of 453 skip the LLM; **0 cards with claims were wrongly skipped** | same |
| LLM extraction | ~9 min per 51 cards per agent, typing values | **Map, don't type**: 24 cards in **41 s** wall clock (8 Sonnet agents × 3 cards, 22–55 s each). Code copies every value, so values are verbatim by construction | `qcl_map.py` |
| Accuracy against the 3,075 hand-verified values (24-card stratified sample, 218 core values) | — | One run: 83–92% recall. Two independent runs combined: **92%** recall. Both runs agreed on 213 values, 180 of them in the ground truth. A hand-check of 12 of the other 33 found **all 12 correct**: claims the ground truth omits, such as per-corpus KL columns and noise floors | `qcl_adjudicate.py` |
| Gates, build and all test suites | ~50 s | ~50 s (of which UI tests take 27 s) | timed |

Haiku was also tried as the mapper. It was 2–3× slower than Sonnet here (109–154 s per queue), and one Haiku agent overwrote its siblings' files. Use Sonnet.

## Pipeline
Each step is one command, and every step can be resumed.

1. **Fetch.** `python3 qcl_fetch.py ORG/MODEL --out models/<slug>/cache --workers 16`. This needs huggingface.co to be reachable: add it to the allowed domains. Otherwise run `gh workflow run fetch.yml -f model=ORG/MODEL -f slug=<slug>`, where a GitHub runner fetches the ledger and commits it.
2. **Extract.** `autoextract.py` lists every table cell, prose claim and bpw label as a candidate, with its line number.
3. **Map, as two independent runs in parallel.**
   - `qcl_map.py tasks` writes one compact task per card.
   - Run N Sonnet agents with 3 cards each, twice, with different card orderings.
   - Agents answer with JSON mappings only.
4. **Adjudicate.** Accept the values both runs agree on. Only the disagreements, about 35% of values, go to one resolver pass, which sees both mappings and the cited lines.
5. **Apply.** `qcl_map.py apply` writes `verified/part*.py`. Every value is copied from its source cell, so the verbatim, sign, unit and column gates hold by construction.
6. **Build and run the gates:** `build.py` (fails closed), `test_gates.py`, `test_claims.py`, `test_ui.py`, then `make_pack.py`.
7. **Cold review by a different model.** Give it the pack plus the adjudication report, scoped to this model's data. The earlier review findings are already encoded as tests.

## Projected wall time for the next model (≈450 cards)

| Stage | Time |
|---|---|
| Fetch | under 1 min direct; ~2 min via GitHub Actions, including runner start |
| Extract and triage | 2 s |
| Map, 2 runs: ~390 cards → 260 agent tasks of 3 cards | ~2 min at 64 concurrent agents; ~8 min at 16 |
| Resolve disagreements | ~1 min |
| Build, gates, tests and pack | ~1 min |
| **Total to preview-ready** | **≈ 5–6 min, ≈ 115–140× faster than 11.5 h** |
| Cold review by a separate model, plus fixes | 10–20 min; now the bottleneck |
| **Total including review** | **≈ 15–25 min, ≈ 30–45×** |

## What still limits it

**Review latency.** The cold review took 24 min last time and is now the largest single stage. Two levers:
- Give the reviewer the mechanical adjudication report, the disagreements and the new flags, instead of the whole site. Estimated 5–8 min.
- Run two reviewers on halves of the data in parallel.

**Mapping recall of ~92%.** About 8% of values are missed by both runs: values stated only in prose, sizes printed away from the metric table, and results that exist only as images. Two levers:
- A third mapper run adds about 1 min.
- A per-publisher template memory should raise recall from the second model on, since publishers such as unsloth, bartowski and turboderp release quants for many models.

**Concurrency.** 64 parallel agents assumes enough plan headroom. At 16 agents the mapping stage takes about 8 min, and the total to preview-ready is about 12 min (≈ 57×).

**One-time access.** Direct fetching needs huggingface.co on the allowed domains. Otherwise use the GitHub Actions route, which needs an `HF_TOKEN` repo secret.

---

# Stage 2: another 100× (measured 18 Oct 2026)

**Starting point (stage 1):** 5–6 min to a preview and 15–25 min to a reviewed site.
**Target:** a preview in ≤ 3.5 s and a reviewed site in ≤ 15 s.

That target rules out any LLM on the request path. So the LLM, mapping and review all move to **ingestion**, which runs before anyone asks. A request then becomes a build over data that is already mapped and reviewed.

## Measured

| Piece | Before | After | Check |
|---|---|---|---|
| Gates + build, nothing changed | 7.3 s (3 gates in series, duplicate verify run, indented JSON) | **0.42 s** (parallel gates, content-hash cache, compact JSON) | timed |
| Gates + build, cold | 7.3 s | **2.09 s** | timed |
| Cache safety | — | the cache keys on the bytes of every gate script, claims file, card, index, exceptions file and image. A test edits one card with a warm cache present, and the build must refuse | `test_gates.py` "gate cache never masks an edited card" (15/15 pass) |
| Machine-mapped output vs gates | — | after three mechanical fixes to `apply` (prose line refs, row labels, unit captions above or below a table), **316 of 319 values pass every gate untouched**. The 3 that fail are routed for review | scratch copy with 24 repos replaced |
| Merge two runs, quarantine failures, rewrite claims | — | **2.8 s** for 24 repos: 230 values agreed, 2 entries quarantined, 20 disagreements queued for the resolver. Gates pass after the merge | `qcl_watch.py merge` |
| Accuracy of what ships from the machine path | — | 219 core values published, 84.9% of them in the hand-verified data. A hand-check found the rest correct (claims our data had omitted). This covers 83% of the hand-verified values; the resolver and fallback reads cover the rest | scored |
| Zero-LLM rules mapper (learned header dictionary + label rules) | — | 4 ms per card, 78% recall, but only 60% role agreement. **Not good enough to publish alone**; usable as an instant draft and as a third voter | `qcl_rules.py eval` (leave-one-repo-out) |

## Design: ingest continuously, serve instantly

1. **Watch list.** Every base model with at least N quantized derivatives on Hugging Face, plus any model you name.
2. **Hourly scheduled task** (the shortest interval scheduled tasks allow). For each watched model:
   1. `qcl_fetch.py` resumes and fetches only new or changed cards (about 0.3 s per card).
   2. `qcl_watch.py delta` lists those cards.
   3. `autoextract.py` turns them into candidates (milliseconds).
   4. Two independent mapping runs: one Sonnet agent per 3 cards, all in parallel, about 40 s.
   5. `qcl_watch.py merge` keeps the values both runs agree on, quarantines anything that fails a gate, and queues the disagreements (about 3 s).
   6. A resolver agent plus a cold review **of the delta only** (about 1 min).
   7. Build, gates and tests, then commit to the repo (a few seconds).
3. **Request = build.** About 0.4 s when nothing has changed, and about 2 s cold. The site is always the last reviewed snapshot. Anything still in the review queue is shown as "pending review", never as a verified claim.

## Achieved vs target

| | Stage 1 | Stage 2 | Speed-up | Target |
|---|---|---|---|---|
| Request → preview, watched model | 5–6 min | **0.42 s** warm / **2.09 s** cold | **~160–800×** | ≤ 3.5 s ✔ |
| Request → reviewed site, watched model | 15–25 min | **0.42–2.1 s** (review already done during ingestion) | **~430–3,500×** | ≤ 15 s ✔ |
| New quant uploaded → on the site | n/a | ≤ 1 h staleness (hourly schedule) + ~2 min of processing | — | — |
| A model **not** on the watch list (cold start) | 5–6 min | still ~5–6 min for the first ingestion | 1× | not met |

## What still limits it
- **Cold start for unwatched models.** The fix is breadth: put every base model with quants above a threshold on the watch list, so the first ingestion has already happened. The cost is about 6 agent-minutes per model, once, and then only the deltas.
- **Freshness.** Bounded by the hourly minimum for scheduled tasks. Per-card processing is about 2 minutes.
- **Machine-path coverage.** It reaches 83% of the hand-verified values today. The rest needs the resolver or a fallback read. The first ingestion of each model should still get a full cold review.
- **Access.** Fetching needs huggingface.co (allowed domains, or the GitHub Actions route). Hourly results are committed to the public repo and deployed by the `deploy-pages` workflow.
