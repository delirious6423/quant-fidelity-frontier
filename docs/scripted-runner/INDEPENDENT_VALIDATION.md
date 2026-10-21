# Independent validation of the performance follow-up

The independent session completed three sequential repetitions on a separate
managed environment, alternating baseline/candidate order. Median Qwen measured
stage time decreased **60.53 -> 45.56 seconds (1.33x; 24.7% less time)**. All six
model pairs preserve exact claims, provenance and task outcomes. The 100x runtime
stretch target was not met. These corrected repeated measurements supersede the
earlier single-run figures as the primary performance evidence.

| Workload / metric | Baseline median [min, max], seconds | Candidate median [min, max], seconds | Ratio |
|---|---:|---:|---:|
| Qwen measured stages | 60.530 [57.950, 63.530] | 45.560 [45.380, 46.560] | 1.33x |
| Qwen preparation | 25.400 [23.650, 26.010] | 15.270 [14.900, 15.940] | 1.66x |
| Qwen response rounds | 33.650 [32.840, 35.860] | 29.390 [29.220, 29.480] | 1.14x |
| LFM measured stages | 4.100 [3.860, 4.560] | 3.740 [3.640, 4.480] | 1.10x |
| Initial collection, 1,253 repos | 19.720 [19.583, 20.401] | 9.013 [7.961, 9.173] | 2.19x |
| Unchanged collection | 15.034 [14.884, 16.095] | 5.881 [5.482, 5.909] | 2.56x |
| Enqueue 1,190 tasks | 0.649 [0.590, 0.684] | 0.083 [0.080, 0.095] | 7.78x |
| Claim/complete 1,190 tasks | 6.557 [6.448, 6.687] | 6.356 [6.088, 6.760] | 1.03x |

Measured stages sum preparation, response rounds, unchanged reconciliation and
status. Initialization, fixture generation and evidence export are excluded.
There are only three observations per side; overlapping ranges, especially for
claim/complete and the small model, do not establish a statistically significant
gain. No median regression occurred in the corrected set, but individual runs can
regress. The separate small legacy/scripted replay remains slower for the scripted
path (0.816 -> 1.557 seconds); that is a different comparison from this PR baseline.

All collection runs made 2,509 initial and two unchanged **mocked** HTTP requests.
Initial checkpoint JSON payloads fell from about 1 GB to 2,161,029 bytes, with
2,507 -> 2,508 writes because the candidate adds one final compact index. Unchanged
candidate checkpoint writes and payload bytes were zero. Counts exclude filesystem
metadata/device amplification. Per-card/tree fsyncs and per-task commits remain.
Actual model/provider requests and worker-agent launches were zero. Tokens/cost
were not measured; synthetic task counts are not provider requests.

## Evidence identity and local recheck

- Baseline code: `4ad4b570a636b62bc9c3d267a21a319ae2f4734e`.
- Frozen model inputs: `77c157c39d6db6e026dae1891510c758acbef1a8`.
- Candidate: baseline plus handoff patch SHA-256
  `69d54bcbd6317f4365f28fc487e0fe5b842821ae4cd92f82ad0c9da1523d36d1`.
- Received `qff-validation-evidence.tar.gz`: 23,793,179 bytes; SHA-256
  `69ca02a1c9a6318e24ae919eb401b3fdbe2d30502738351de148c1dcdb7bc882`.
- All 343 compact archive-manifest entries verify. All 93 candidate code-file hashes
  match the local implementation; all 3,790 frozen input-file hashes match locally
  and remain identical between the external before/after manifests.
- All 18 raw result records match their saved per-run JSON. Medians were recomputed
  locally and match the supplied summary. All six full-model output files and
  supplemental task-evidence files are byte-identical between paired sides.

For each repetition, LFM has 61 output records and 108 succeeded tasks; Qwen has
614 records and 1,210 succeeded tasks. Complete output includes holds, review
queues, source identities, maps/candidates and results. No units, qualifiers,
panels, attribution, roles or provenance were normalized away. Supplemental task
files preserve requests, IDs/generations, results, attempts, active flags and
creation order. The large full-workspace archive/raw SQLite journals were not
needed for this artifact recheck and were not downloaded.

IO snapshots are **not** byte-identical: each pair differs only in the real
`captured_at` and `updated_at` clocks. All other fields and complete base/repository
records match. Those exact clock differences and differing file hashes remain in
[independent-validation.json](independent-validation.json), alongside every raw
run result, recomputed medians and output/task hashes. The supplied complete metric
ranges are retained in [independent-summary.csv](independent-summary.csv).

## Correction and validation scope

The first measurement set exported candidate-helper evidence between paired sides.
The external coordinator was corrected to export helper evidence after both timed
sides, and all three repetitions were rerun in fresh directories. No candidate
code, inputs, workers, fixtures or timing functions changed. No initial result was
selected into the final summary. The superseded set remains in the user's separate
archive; it included a claim/complete median regression (7.209 -> 7.998 seconds).
Named stage timings exclude the candidate harness's own final evidence export;
coordinator wall-clock totals are not used for performance ratios.

External logs confirm 55 baseline / 66 candidate runner tests and nine workflow
security tests on each side. Replay preserves exact output with zero unresolved
work or repeated successful stage after restart. Two extra fault probes interrupted
collection before compact-index installation and killed a process after the first
checkpoint unlink. Both resumed with only two metadata requests and no repeated
successful card/tree request. Their complete result records are retained locally.

The external host reported Python 3.12.14, SQLite 3.53.1, four-CPU quota and 16 GiB
RAM. Timed outputs used overlayfs; code/input roots used tmpfs. No competing
benchmark/test was launched during timings; other host activity is unknown.
Filesystem sync calls were real, but physical power-loss durability was not tested.

This session verified the received artifacts, implementation/input identity and
calculations; it did not rerun the external experiments. This documentation-only
update reuses the prior full local build/UI/source/MCP verification because the
implementation hashes are unchanged. Live provider latency, mapping quality,
provider retry/idempotency, persistent hosting and production cutover remain
unvalidated. No remote push, PR write, Actions run, deployment or schedule change
was performed. The next authorization remains the review-branch push and its
automatic PR verification, followed separately by a bounded provider/storage trial.
