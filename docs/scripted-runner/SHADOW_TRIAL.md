# Bounded shadow trial: LFM2.5-24B-A2B, 2026-10-21

The first end-to-end run of the scripted runner with real interpretation,
not synthetic responses, on one live model. It answers the question the PR
left open: does the per-card request boundary produce claims of production
quality, and what does a full re-map cost in interpreter work.

Scope and rules: shadow workspace from main at `28e08005` with every mapping
forgotten, so all 53 unique cards (75 repositories) were mapped from scratch;
runs A and B by separate workers; blind C by workers that saw no A or B map;
the fixed 20-value QA sample reviewed by workers that saw only the claim and
its card. Interpretation was done by Claude Code subagents (Sonnet) through
the file-adapter contract (`requests/<id>.json` in, `responses/<id>.json`
out), which is byte-for-byte what `runner_adapter.py` sends to the Messages
API. No authoritative data was touched; nothing was published.

## What the runner did

| Stage | Tasks | Outcome |
|---|---:|---|
| Mapping A + B | 106 | 105 validated; 1 rejected fail-closed (a candidate id that does not exist), retried once with a fresh worker and accepted |
| Blind C | 16 | 16 validated; 33 held values confirmed by the third run |
| QA sample | 20 | 20 `ok`, 0 `wrong`, 0 `unsure` |
| Shadow verify | | gated build passed; receipt in `shadow-trial/shadow-verification.json` |

Orchestration time end to end was about 15 s (prepare 7.0 s, four response
rounds 3.7 / 3.5 / 2.9 / 2.4 s, verify 1.2 s). Interpreter wall clock was about
3 min for the mapping round, 3 min for blind C and 1 min for QA, at up to 18
workers in parallel; roughly 2.85 M subagent tokens in total, dominated by the
workers reading request files through tools. (A direct API call sends the
same bundle once and spends none of that overhead.)

## Claims versus the committed data

Compared with the committed claims for the same 75 repositories (Haiku-mapped
by the hourly pipeline), at the number level (`shadow-trial/numbers.txt`;
entry labels, corpus spelling and unit punctuation ignored):

| | Count |
|---|---:|
| Numbers the trial publishes | 625 |
| Numbers the committed data publishes | 485 |
| In both | 432 |
| Recall of committed numbers | 89.1% |
| Precision against committed numbers | 69.1% |

The 193 trial-only numbers concentrate on cards where the committed data has
almost nothing (primitive-ai NVFP4A16: +34 over 6; OsaurusAI JANGH2: +27 over
3; GestaltLabs NVFP4: +22 over 1): they are the recall the FASTPATH rehearsal
predicted Haiku would miss, not errors, and the 20-item QA sample found no
wrong value among them. The 53 committed numbers the trial misses are mostly
two EXL3 ladder tables whose rows A and B named differently and which C did
not fully resolve (held, never guessed), plus one repository held entirely.

Exact-key agreement (`shadow-trial/exact-keys.txt`) is lower because the two
paths spell the same thing differently: entry "2.5" versus "EXL3 2.5 bpw",
`kld:wikitext` versus `kld:wikitext-2`, "78.9" versus "78.9 %". None of these
is a numerical difference.

## Defect found and fixed

Four prose values were mapped as a sentence, "perplexity is 21.00 on the
wikitext rows", for a `ppl:wikitext` key. Every gate passed because the text
is verbatim on the cited line. `runner_model.validate_mapping` now requires a
prose value to be one number with at most a qualifier before it and a unit
after it (`test_runner_model.test_prose_value_is_one_number_not_a_sentence`).
The committed data carries 23 values of a similar shape from table cells (for
example "87.20% (−1.21, p = 0.79)"); those are whole-cell candidates, not
prose, and are left as they are.

## What this does and does not show

It shows the runner's contract works end to end on real cards with real
interpretation: strict validation catches invented ids, the retry path is
explicit, blind C resolves a third of the disagreements, QA and verify run
unattended, and the result is at least as complete as the committed data.
It does not measure a provider's latency or cost, and the interpreter here
was a coding agent reading files, not the API adapter. The next step is the
same trial through `runner_adapter.py` with a key and a spend cap, which
records tokens per task in the journal.
