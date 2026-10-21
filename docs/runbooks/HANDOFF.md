# Handoff (21 Oct 2026, 08:10 UTC)

Written for the next session that takes this work over (any agent, any machine). Everything below is on `main`
or in the two open pull requests; nothing lives only in a session. Read `AGENTS.md` first, then this file, then
`PROGRESS.md` for the longer history.

## State of `main`

- `main` is **red** since the first collection backfill (commits 923dec4 and fea47ca, 07:47 UTC): `collect-cards`
  installed cards for repositories the connector catalogue had never listed, and `gates/verify.py` refused every
  model that has such a repository (`MISSING: [...]`, run 39564738422). The fix is in PR #23 (below): a collected
  but unmapped repository is `PENDING MAPPING`, left off the site, and the next live-updates cycle maps it.
- Hourly `collect-cards` (`.github/workflows/collect.yml`, :01 UTC) works against the live Hub: the backfill of
  2,946 cards took 14.5 min (12 of them waiting out anonymous 429s), and the run with nothing changed takes about
  half a second per model and commits nothing. Card images: 814 of 836 URLs pinned; the 22 others are 10 oversize
  decorative GIFs, 10 dead links (404) and 2 gated repositories (403). PR #24 classifies them so only a retryable
  failure turns the step red.
- `HF_TOKEN` is **not set** as a repository secret (the run logs show it empty). Without it every collect run is
  at the anonymous rate limit. Settings → Secrets and variables → Actions.
- The three scheduled tasks (live updates :03, new model :37 every 2 h, QA :55; `docs/runbooks/HOURLY.md`) are
  enabled and still run the connector path. The live-updates task should move to
  `python3 pipeline/qcl_next.py --part A --collected` now that the backfill is in; that drops the 2.35 min of
  connector catalogue and FETCH workers from the cycle. Not done yet.
- GitHub Actions ran every workflow today (this is a public repository, standard runners); if the account shows a
  billing hold, the local runner below replaces CI entirely.

## Open pull requests

| PR | Branch | What | State |
|---|---|---|---|
| #23 | `auto/ci-local` | `pipeline/ci_local.py` (runs `verify.yml` locally from the workflow file), `$RUNNER_TEMP` for the workflow's scratch files, `pyyaml` in the safety job, the `PENDING MAPPING` gate, tests and docs | local battery through the runner: 35 steps, all green except the site job still finishing at the time of writing; Actions re-running after the pyyaml fix (427b2b1) |
| #24 | `auto/images-classify` | unpinned image URLs classified as `unresolved` / `gone` / `refused`; strict fails only on `unresolved`; 9 image tests | carries #23 (merged in as 7e3cbfd); Actions re-running |

Merge order: #23 first (it turns `main` green), then #24. Merge method `merge`, as for every PR so far. After #23
merges, the next `collect-cards` run and the next live-updates cycle map the pending repositories; `main`'s
verify goes green on the merge commit.

## Run CI on any machine

```bash
git pull
python3 pipeline/ci_local.py                      # everything in verify.yml, in the workflow's order; exit 1 on any failure
python3 pipeline/ci_local.py --list               # what would run
python3 pipeline/ci_local.py --only safety        # or --models SLUG ..., --only site --from "MCP"
python3 pipeline/ci_local.py --receipt ci.json    # JSON receipt: commit, steps, outcomes, seconds
```
Needs Python 3.11+, `pip install pyyaml pillow numpy playwright mcp` and `python -m playwright install chromium`
(`--pip` runs the workflow's own install lines). Per-step logs go to a temporary directory named at the start.
The runner runs `safety` before the model jobs because they share one checkout here (some safety suites assert
the checkout stays untouched while builds write `audit/verify_counts.json`); discard that audit change afterwards
with `git checkout -- models` unless a QA publish is meant to commit it.

## Measured numbers (for `docs/reference/PERFORMANCE.md`, "After" section, still to be filled)

- Live-updates cycle after PR #15 (all prompts in round one): 29.2 min on the 06:34 UTC run of 21 Oct
  (`logs/scan/20261021T0634Z-live.jsonl`): catalogue 0.35, fetch 2.0, prepare 0.3, one mapping round 15.3
  (20 prompts, 9 models), tie-break C 10.2, stage 0.3. Before: 35.8 min (`20261020T1412Z-live.jsonl`).
- Collect with nothing changed: 0.5 s for the 467 cards of Flash-Next (run 39562915089); backfill 14.5 min
  (run 39562759606). Images: 0.4 s per model when nothing changed.
- QA at :55: 2.0 min (`20261021T0603Z-qa.jsonl`), no collision; before 3.1 min and a skipped run at 15:08.
- CI: 4 min 23 s with the four-job workflow (run 39555698237); 7 min 25 s before.
- Local runner on a 4-core container: model jobs 19 to 43 s each, four at a time; safety 106 s.
- The before/after statechart with every phase is published at https://claude.ai/artifact/36iaiAnzoPJUN3VCV7PNoE
  (private artifact of the owner; the numbers above are its source).

## Plan, in order

1. Merge #23 when its battery is green (local receipt or Actions), then #24.
2. Point the "QFF live updates" scheduled task at `--part A --collected` (`docs/runbooks/HOURLY.md` has the
   prompt); watch the next cycle: it should map the pending repositories and the cycle should lose the catalogue
   and fetch minutes.
3. Fill the "After" rows of `docs/reference/PERFORMANCE.md` and the measured table of `docs/reference/CI.md` from
   the numbers above and the next runs; one PR.
4. Decide the image policy for the 22 unpinned URLs: keep `refused` for files over 8 MB (they are decorative GIFs,
   not charts) or raise `--max-bytes`; `gone` entries can only change when the publishers fix their cards.
5. `tests/test_images.py` failed once on this container while the full battery was loading the CPU, and passed
   on every other run (local and Actions); the failure output was not captured. If it fails again, read which
   case and look at the retry timing in the loopback CDN first.
6. Next performance levers, both deterministic: the `site` job's browser suites (2 min 46 s of the 4 min 23 s)
   and the QA task's tests that repeat them; and folding run C into the mapping round in the portable loop.
7. Weekly discovery (`.github/workflows/discover.yml`) has not run on schedule yet; watch its first run.
