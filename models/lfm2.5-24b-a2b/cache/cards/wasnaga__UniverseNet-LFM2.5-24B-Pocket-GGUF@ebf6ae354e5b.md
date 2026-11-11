---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - lfm2
  - lfm2moe
  - liquid-ai
  - moe
  - llama-cpp
  - ios
  - mobile
  - on-device
  - tool-calling
  - agent
  - universe-network
---

# UniverseNet LFM2.5-24B — current v16 release

Pocket, Edge and Pro reduce expert width and quantize weights. Workstation retains all 1536 expert orbitals: it is a full-width low-bit control, not a parameter-removal result. Filenames retain the internal `v16` build tag for provenance. Sizes below are decimal GB, not runtime RAM requirements.

| Edition | GB | Total / active parameters (B) | Validation PPL | PPL / teacher | Top-1 agreement | Top-5 recall |
|---|---:|---|---:|---:|---:|---:|
| Pocket | 3.996 | 12.68 / 1.62 | 38.17 | 1.375× | 71.7% | 91.15% |
| Edge | 5.65 | 16.51 / 1.86 | 32.95 | 1.187× | 75.98% | 93.07% |
| Pro | 7.306 | 20.33 / 2.1 | 30.69 | 1.106× | 79.27% | 94.56% |
| Workstation | 8.711 | 24.16 / 2.34 | 29.01 | 1.045× | 81.42% | 95.17% |

Teacher validation PPL: 27.75. These are teacher-forced validation diagnostics from the condensation outputs, not task accuracy, general fidelity, or an independent benchmark. Probe traces and calibration were used in model development; no fresh held-out generalization claim is made. Sample composition/counts are not included in the summary JSONs. Top-1 means token agreement with the teacher, not the six-skill index from v15.

| Edition | Tool-call Top-1 probe | Plan explore-minus-repeat (nats) | User-correction pivot (nats) |
|---|---:|---:|---:|
| Pocket | 100% | 6 | 14.12 |
| Edge | 100% | 5.875 | 13.25 |
| Pro | 100% | 5.188 | 15.5 |
| Workstation | 100% | 3.875 | 15.72 |

The tool-call diagnostic is not BFCL or end-to-end agent success. Margins measure specific continuation preferences; they are not success rates. All four JSONs are in `evals/v16/` in the model repository.

## What changed

- H5A two-path core and multi-orbital cross-bond coupling (`W_prior + ΔW`): top co-activating dropped orbitals feed retained valence `W_up`; prior-centered residual ridge solves `W_down`.
- Importance-matrix floor and per-column covariant norm guard address low-bit dead-column scale singularities.
- Multi-turn tool-response and user-correction calibration, with stronger transition-boundary weights; removal of the late-layer closing-thought drift impulse.

Implementation: `exp12_lfm25_24b_pocket_v16_m768_iq2.py`. Mechanism diagnostics are development evidence, not independent generalization tests.

## Live application report — not a standardized benchmark

The release operator reports all tiers tested with Corporate AI Machine and Claude Code 2.1.309. Workstation reportedly used full GPU MoE offload on RTX 4080 (`--n-cpu-moe 0`) at 131072‡ configured context: 13296‡ / 16376‡ MiB device usage, 9079‡–11251‡ tok/s prefill on approximately 51500‡ tokens and 158‡–211‡ tok/s decode at 35000‡–38000‡ context depth. These observations are **user-reported and provisional**; raw timing/configuration logs have not been supplied with this release. No controlled speedup or teacher comparison is asserted here. Source: `benchmarks/results/lfm/v16_live_user_report.json` (handoff transcription, not a measured log).

## Six measured task subsets — v16 versus v15

All four editions completed the v15-sized protocol on the local RTX 4080: greedy decoding, one slot and 4096-token output budget. Non-math uses the same fast questions (seed 42); math replays the exact v15 60-item IDs (seed 1234), not a newly sampled set. All 744 responses are saved; no request errors were recorded. These are **small, development-exposed subsets, not full splits or independent confirmation**. The earlier full-split attempt was stopped and is excluded from these tables. The tool suite is 10 custom AST-style tasks, **not full BFCL**. Math retains the existing automatic grader, including format-equivalence limitations; no manual score corrections were applied.

| Edition | Tools /10 | Instructions /20 | Science /20 | Code /16 | GSM8K /60 | MATH-500 /60 |
|---|---:|---:|---:|---:|---:|---:|
| Pocket | 10 | 18 | 12 | 9 | 50 | 20 |
| Edge | 10 | 19 | 18 | 13 | 53 | 35 |
| Pro | 9 | 20 | 19 | 15 | 57 | 42 |
| Workstation | 10 | 20 | 20 | 15 | 53 | 41 |

**Not a uniform improvement.** Pocket's science falls 16 → 12, code 10 → 9, GSM8K 51 → 50 and MATH 22 → 20 despite better validation PPL. Edge improves instructions and MATH but loses three GSM8K items. Pro improves code 13 → 15 and MATH 38 → 42, while losing one tool and one science item. Workstation improves instructions, retains MATH, and loses two GSM8K items. Differences of a few questions do not establish broad superiority.

Explainer bars show these six subset percentages against original-model references. Chips show **percent of original — six-suite subset index**: sum of the six student percentage scores divided by the sum of the six parent reference percentages, with equal suite weight. This descriptive mixed-subset comparison is not a guarantee of overall capability. No speed benchmark or new phone measurements were run. Sources: `evals/v16/official_eval_results_lfm25_v16_4suites.json` and `evals/v16/heldout60_v16_matched_v15.json`, plus their `.responses.jsonl` records. Math sets and calibration probes informed development; this is a matched regression comparison, not fresh generalization evidence.


## Coverage and historical evidence

The six v16 task subsets are now measured above. Repeated llama-bench speed measurements, iPhone tests and filled-context sweeps remain unavailable for v16. The v15 six-skill and phone tables remain **historical v15-only evidence**; identical file sizes do not transfer measurements. Qwen releases are unchanged.

## Current downloads

- [Pocket](https://huggingface.co/wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF/resolve/main/UniverseNet-LFM2.5-24B-Pocket-12B-v16-IQ2.gguf?download=true)
- [Edge](https://huggingface.co/wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF/resolve/main/UniverseNet-LFM2.5-24B-Edge-16.5B-v16-IQ2-IQ3.gguf?download=true)
- [Pro](https://huggingface.co/wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF/resolve/main/UniverseNet-LFM2.5-24B-Pro-20.3B-v16-IQ2-IQ3.gguf?download=true)
- [Workstation](https://huggingface.co/wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF/resolve/main/UniverseNet-LFM2.5-24B-Workstation-24B-v16-IQ2-IQ3.gguf?download=true)

---

# Historical v15 release — evidence archive

**Four GGUF editions, zero gradient retraining.** Pocket, Edge and Pro remove expert orbitals and use low-bit quantization; Workstation retains the original expert width and is a quantization-only control.

The current release provides **Pocket, Edge, Pro and Workstation**. The `v15` filename tag identifies the exact research build used for the published results; it is not a requirement to count through previous releases. The four v15 files are superseded by v16; this section records v15 evidence only. Historical public-suite scores and phone speeds are retained as evidence, not as scores for the current files.

## Choose a v15 edition

| Edition | File (GB) | Orbitals / expert | Total / active parameters (B) | GSM8K / 60 | MATH-500 / 60 | RTX 4080 tok/s |
|---|---|---|---|---|---|---|
| [Pocket v15](UniverseNet-LFM2.5-24B-Pocket-12B-v15-IQ2.gguf) | 3.996 | 768 | 12.68 / 1.62 | 51 | 22 | 310.8 |
| [Edge v15](UniverseNet-LFM2.5-24B-Edge-16.5B-v15-IQ2-IQ3.gguf) | 5.65 | 1024 | 16.51 / 1.86 | 56 | 32 | 281.3 |
| [Pro v15](UniverseNet-LFM2.5-24B-Pro-20.3B-v15-IQ2-IQ3.gguf) | 7.306 | 1280 | 20.33 / 2.1 | 57 | 38 | 266.3 |
| [Workstation v15](UniverseNet-LFM2.5-24B-Workstation-24B-v15-IQ2-IQ3.gguf) | 8.711 | 1536 | 24.16 / 2.34 | 55 | 41 | 265.1 |

All four files were measured on the **same 60 GSM8K and 60 MATH-500 questions** (seed 1234; 4096-token output budget). These are not full dataset splits. Original Q4_K_M: 56 / 60 and 43 / 60. Scores differ by a few questions at the larger tiers: neither overlap nor a one-question lead establishes teacher parity.

![LFM v15 math results, size and measured RTX 4080 speed](assets/lfm_v15_overview.png)

### Which one should I use?

- **Pocket v15:** smallest download; more math problems solved than the same-sized Pocket v7.1 in the paired test (73 vs 51 / 120).
- **Edge v15:** more expert width at nearly the old Edge file budget. It solves 88 / 120 on this pass.
- **Pro v15:** 95 / 120 at 7.306 GB, retaining more width than Pro v6.
- **Workstation v15:** 96 / 120 at 8.711 GB. Full expert width, compressed storage; do not count it as a parameter-removal result.

**Coverage matters:** the six skill categories are now measured on subsets; phone throughput and touched-cache capacity are now measured below; Babbo and long-context answer quality remain unmeasured. See [LFM_V15_RESULTS.md](LFM_V15_RESULTS.md) for token caps, provenance and limits. Questions were excluded from calibration, but diagnostic results on these questions also informed model design.

## Six measured skill areas — LFM v15

All four v15 editions now have measured results in every skill category. The four newly run suites use the historical fast questions (seed 42, 66 items total); the two math suites use the separate seed-1234 held-out subsets (120 items). Every model sees the same questions within each suite. Greedy decoding, 4096-token answer budget, local RTX 4080, one inference slot. **These are small subsets, not full splits.** BFCL is the harness's 10 hand-written AST-style function-calling tasks, not the full official BFCL leaderboard.

| Edition | Tools / 10 | Instructions / 20 | Science / 20 | Code / 16 | GSM8K / 60 | MATH-500 / 60 |
|---|---|---|---|---|---|---|
| Pocket v15 | 9 | 18 | 16 | 10 | 51 | 22 |
| Edge v15 | 10 | 18 | 18 | 13 | 56 | 32 |
| Pro v15 | 10 | 19 | 20 | 13 | 57 | 38 |
| Workstation v15 | 10 | 19 | 20 | 15 | 55 | 41 |

Tools = AST-style function calls; instructions = IFEval strict prompt pass; science = ARC-Challenge; code = HumanEval pass@1. Four-suite totals are 53, 59, 62, and 64 / 66 respectively. These totals must not be called scores out of the historical 106-item protocol.

Explainer chips use an **equal-suite-weight index**: sum of these six percentage scores divided by the sum of the original's six reference percentages. The four non-math references are historical original-model fast scores; math references use the matched 60-item sets. This descriptive mixed-subset index is not a statistical fidelity estimate or proof of teacher parity. Raw new run: `evals/v15/official_eval_results_lfm25_v15_4suites.json`.

Still unmeasured: v15 Babbo journeys, long-context answer quality, full splits and fresh independent confirmation. Math-set selection effects and token caps remain as documented below.


## Speed and hardware — current v15 files

| Edition | RTX 4080 prefill PP512 tok/s | RTX 4080 decode TG128 tok/s | iPhone / Mac / GX10 speed |
|---|---|---|---|
| Pocket v15 | 12,278.5 | 310.8 | iPhone 18 Pro: 70 tok/s |
| Edge v15 | 10,155.1 | 281.3 | iPhone 18 Pro: 65.4 tok/s |
| Pro v15 | 8,207 | 266.3 | iPhone 18 Pro: 64.5 tok/s |
| Workstation v15 | 7,357.9 | 265.1 | iPhone 18 Pro: Metal OOM |


Stock llama-bench: `-p 512 -n 128 -r 5 -fa 1 -b 512 -ub 512 -ngl 99`. All four loaded and ran on the RTX 4080 (16 GB). File size alone is not a runtime-RAM requirement: leave memory for KV cache, scratch buffers and the OS. The iPhone 18 Pro measurements below establish short-context operation for Pocket, Edge and Pro; Workstation fails there.

## iPhone 16 Pro — Pocket context comparison

Pocket completed 32768 context with both f16 and q8_0 on the 8 GB iPhone 16 Pro (`iPhone17,1`), at 20.3 / 22.3 tok/s. Both runs ended at fair thermal state. Neither 65536 run completed: f16 hit Metal OOM during fill; q8_0 failed during decode with a GPU recovery/discard error (not independently proof of OOM). At 131072 q8_0, Metal OOM was followed by a confirmed vm-pageshortage Jetsam kill; 131072 f16 was not run. Largest tested completed context is 32768, versus 131072 on the 18 Pro—not a proven maximum. Other editions and five-repetition short-context speed were not measured on the 16 Pro in this sweep. Metal's recommendation is not a proven hard limit. Source: `evals/v15/iphone16pro_ctxsweep_pocket_v15.json`; hardware ID corrects its stale raw marketing-name field.

## Current release on iPhone 18 Pro — measured

SHA-256-verified files; iPhone19,2, iOS 27.2, 11.36 GiB physical RAM. Stock llama.cpp `52c9bad`, Metal, flash attention, full GPU offload, 2048 context, batch/ubatch 512, four threads. Warmup followed by five PP512/TG128 repetitions. The phone was unplugged and nominal thermal during the short benchmarks.

| Edition | Prompt processing tok/s | Decode tok/s | Load seconds | Metal allocation after context creation (GiB) |
|---|---|---|---|---|
| Pocket | 1,666.4 ± 9.5 | 70 ± 0.2 | 4.3 | 4.02 |
| Edge | 1,416.9 ± 7.4 | 65.4 ± 0.1 | 6.5 | 5.56 |
| Pro | 1,187 ± 8.6 | 64.5 ± 0.7 | 8 | 7.3 |
| Workstation | failed during decode | explicit Metal OOM | — | 8.61 |

Metal allocation is **not total system RAM usage**. CPU and GPU share physical memory on this phone; there is no separate VRAM pool to add to RAM. Mapped weights are largely absent from the app's footprint accounting. The approximately 190 MiB footprint counters therefore must not be described as total model memory. Pro completed at short context despite its larger file; Workstation did not complete. Its later signal 9 was deliberate cleanup, not a Jetsam event.

### Real-filled context sweep: decode tok/s at depth

Each fresh process fills approximately 90% of the configured context with real token-zero decoding, then generates 128 tokens without resetting the cache. This measures touched-cache capacity and throughput, **not retrieval or long-context answer quality**.

| Edition / KV type | 8192 context | 32768 | 65536 | 131072 |
|---|---|---|---|---|
| Pocket / f16 | 65.8 | 51.3 | 33.5 | 16.9 |
| Pocket / q8_0 | 64.9 | 50.5 | 28.9 | 15.7 |
| Edge / f16 | 50 | 35.5 | 28.9 | Metal OOM |
| Edge / q8_0 | 60.6 | 48.1 | 27.5 | 13.5 |

Pocket completed every tested configuration. Edge completed 65536 f16 and 131072 q8_0; its 131072 f16 fill produced an explicit Metal insufficient-memory error. These are largest **tested completed configurations**, not a proven maximum between the tested sizes. Pro was not context-swept. Long fills reached fair/serious thermal states—including Edge's f16 sweep—so f16/q8_0 speed differences are not thermally controlled comparisons. Battery fell from 65% to 20% over the overall session.

Workstation and Edge-f16 failures demonstrate GPU allocation/working-set pressure; the app headroom counter still reported substantial room and did not predict those errors. Metal's recommended working set is a guideline, not a guaranteed hard threshold. No Jetsam kill was established in these current-release runs.

Sources: `benchmarks/results/lfm/iphone18pro_bench_*_v15.json`, `iphone18pro_ctxsweep_*_v15.json`, and raw device logs under `demo/ios-device-bench/evidence/v15/`. Historical v6/v7.1/v10 phone measurements remain separate.


## Download and run

Download a file from the table above, or use:

```bash
hf download wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF UniverseNet-LFM2.5-24B-Pocket-12B-v15-IQ2.gguf --local-dir .
llama-server -m UniverseNet-LFM2.5-24B-Pocket-12B-v15-IQ2.gguf -ngl 99 -fa on -c 8192 --jinja --port 8080
```

Use a current llama.cpp build supporting the file's IQ2/IQ3 types. Replace the filename to select another edition. `--jinja` enables the supplied tool-call template; the Pocket smoke test passed a reminder tool call, but this does not establish general tool accuracy. Do not pass `--reasoning-format none` for that workflow.

## Files and evidence

```text
UniverseNet-LFM2.5-24B-Pocket-12B-v15-IQ2.gguf
UniverseNet-LFM2.5-24B-Edge-16.5B-v15-IQ2-IQ3.gguf
UniverseNet-LFM2.5-24B-Pro-20.3B-v15-IQ2-IQ3.gguf
UniverseNet-LFM2.5-24B-Workstation-24B-v15-IQ2-IQ3.gguf
evals/v15/                                     per-question results, speed and condensation JSONs
MANIFEST.md                                    exact sizes and SHA-256 hashes
LFM_V15_RESULTS.md                              current results and limitations
```

## How it was made

Every expert and the original router remain. The v15 recipe uses open-loop Teacher-trajectory calibration, two-bond orbital selection, covalent residual absorption, an envelope clamp and coupled Gauss–Seidel updates. Per-expert activation importance and covariance pre-deformation guide single-pass IQ2/IQ3 quantization. Non-expert architecture is preserved. Workstation retains all expert orbitals.

Condenser: `exp12_lfm25_24b_pocket_v15_m768_iq2.py`. The versioned logs record widths and runtime for each edition. [Paper §§6.6–6.7](https://wasnaga-universe-network.static.hf.space/paper.html) · [visual explainer](https://huggingface.co/spaces/wasnaga/universe-network).

## Earlier releases — historical measurements only

The demo and all tables in this section belong to the versions named here. They do not fill missing v15 measurements. All historical weights are retired from the current download list; their saved results remain as historical evidence.

<video src="https://huggingface.co/wasnaga/UniverseNet-LFM2.5-24B-Pocket-GGUF/resolve/main/assets/demo_iphone_airplane.mp4" controls width="360"></video>

*Legacy Pocket v7.1 demo on iPhone 16 Pro in airplane mode. Quantitative iPhone 16 speed measurements used the Pocket v6 file.*

<details>
<summary>Expand v6 / v7.1 / v10 benchmark and device tables</summary>

### Historical: Hardware compatibility

✅ measured on that exact device · 🟢 fits by memory, not yet tested · ❌ does not fit

| Device (memory) | Pocket v6/v7.1 | Edge v6 | Pro v6 | Original |
|---|---|---|---|---|
| iPhone 16 Pro / 17 Pro (8 GB) | ✅ 43.12 tok/s (v6/v7.1; current release not tested on these 8 GB phones) | ❌ | ❌ | ❌ |
| iPhone 15 Pro, iPad (8 GB) | 🟢 | ❌ | ❌ | ❌ |
| iPhones with 6 GB or less | ❌ | ❌ | ❌ | ❌ |
| Mac mini / MacBook M1–M4 (8 GB) | ✅ 37.36 tok/s (M1) | ❌ | ❌ | ❌ |
| Mac with 16 GB (M1–M4), iPad Pro 16 GB | 🟢 | 🟢 | 🟢 | ❌ |
| MacBook Air M5 (24 GB) | ✅ 106.22 tok/s | ✅ 98.98 | ✅ 92.85 | ✅ 72.07 |
| 8 GB GPU (RTX 3060 Ti / 4060 / 3070) | 🟢 | 🟢 (short context) | ❌ | ❌ |
| 12 GB GPU (RTX 3060 / 4070) | 🟢 | 🟢 | 🟢 | ❌ |
| 16 GB GPU (RTX 4080 / 4070 Ti S / 5080) | ✅ 304.1 tok/s (4080) | ✅ 290.6 | ✅ 272.1 | partial CPU offload: 43.5 |
| 24 GB+ GPU (RTX 3090 / 4090 / 5090) | 🟢 | 🟢 | 🟢 | 🟢 |
| NVIDIA DGX Spark / GX10 (128 GB) | ✅ 174.8 tok/s | ✅ 163.5 | ✅ 151 | ✅ 119 |

Decode speed in tokens/s from `llama-bench` tg128, fully offloaded (`-ngl 99`). Leave about 1–2 GB above the file size for the KV cache and the OS. iOS and Metal impose device-dependent memory budgets; the historical phone measurements in this section are not current-release measurements; current iPhone 18 Pro results appear above.

### Historical: Public benchmarks (same harness for Teacher and students)

Greedy, `llama-server`, `max_tokens 4096`, `fast` subsets: n=20 per suite (HumanEval 16, BFCL 10). Small samples: differences under ~15 points are not significant.

| Suite | Original | Pro v6 | Edge v6 | **Pocket v7.1** |
|---|---|---|---|---|
| IFEval (prompt strict) | 95 | 95 | 90 | 70 |
| MATH-500 | 65 | 75 | 45 | 20 |
| GSM8K | 95 | 95 | 80 | 65 |
| HumanEval pass@1 | 93.75 | 93.75 | 62.5 | 6.25 |
| ARC-Challenge | 100 | 95 | 70 | 50 |
| BFCL-AST (tool calls) | 100 | 100 | 90 | 80 |
| `</think>` closure / loop-free | 92.45 / 96.23 | 92.45 / 92.45 | 86.79 / 89.62 | 72.64 / 70.75 |
| held-out PPL (× Teacher 28.31) | 1.00 | 1.116 | 1.209 | 1.476 |

These small-sample results describe Pro v6, Edge v6 and Pocket v7.1 only. They are retained for reproducibility, not assigned to the v15 files.

![Capability vs orbital budget](assets/lfm25_tiers.png)

### Historical: Speed on real devices

`llama-bench -p 512 -n 128 -r 5 -fa 1 -b 512 -ub 512 -ngl 99`, decode tokens/s. The iPhone runs the same loop in a minimal llama.cpp app, fully offline.

| Device | Original | Pro v6 | Edge v6 | Pocket |
|---|---|---|---|---|
| iPhone 16 Pro (8 GB) | does not fit | does not fit | does not fit | **43.12** (peak RSS 188.6 MB) |
| iPhone 18 Pro (12 GB) | does not fit | not tested | **74.1** (v6 file) | **80.6** (v7.1 file) |
| Mac mini M1 (8 GB) | does not fit | does not fit | does not fit | **37.36** |
| MacBook Air M5 (24 GB) | 72.07 | 92.85 | 98.98 | **106.22** |
| RTX 4080 (16 GB) | 43.5 (20 MoE layers on CPU) | 272.1 | 290.6 | **304.1** |
| NVIDIA GX10 (128 GB) | 119 | 151 | 163.5 | **174.8** |

Pocket rows for the iPhone, M1, M5 and GX10 were measured with the v6 file. v7.1 has identical tensor shapes (RTX 4080: v6 303 vs v7.1 304.1 tok/s). Where the Teacher fits in fast memory the condensed tiers are ~1.3–1.5× faster. Where it does not (16 GB GPU, 8 GB devices) they are 6–7× faster or the only option.

### Historical: Long context on a phone (iPhone 18 Pro, 12 GB)

Each cell is a fresh process that allocates the context, **really decodes about 90% of it**, then times 128 decode steps at that depth (decode tok/s; llama.cpp `52c9bad`, Metal, flash attention, charging). The 18 Pro has 11.4 GiB of RAM and about 8 GiB of that is Metal's recommended working set. Only 10 of the 40 layers keep a KV cache, so the cache is small: 20 KiB per token at f16.

| model, KV cache | 2k | 8k | 32k | 64k | 128k (full native context) |
|---|---|---|---|---|---|
| Pocket v7.1, f16 | 69.5 | 72.9 | 56.7 | 34.4 | 18.5 |
| Pocket v7.1, q8_0 | 69.2 | 65.3 | 55.3 | 29.1 | not run |
| Edge (v10 file, f16) | 75.4 | 67 | 52.5 | 27.8 | **failed** |
| Edge (v10 file, q8_0) | 68.5 | 64.9 | 49.5 | 31 | 14.1 |

- **Pocket fits the full 128k context at f16** with room to spare (6781 MiB allocated). **Edge needs a q8_0 KV cache for 128k**: at f16 its 128k run allocated 8515 MiB, over the 8192 MiB working-set recommendation, and failed after the first 512 tokens (the log does not state a cause). Use `-ctk q8_0 -ctv q8_0`.
- Speed falls with depth because every new token attends over the whole cache. This measures throughput on a filled cache, **not** whether the model uses 128k tokens well, which we have not tested.
- Edge was measured with the Edge v10 file (160 MB larger than the Edge v6 offered here), so v6 should have at least as much headroom. That is an inference.

### Historical: Agentic test (Babbo family assistant, 11 multi-turn journeys)

Reminders with ISO-8601 dates across summer/winter time, shared memories, clarification turns, refusing unknown people.

| | Original | Pro v6 | Edge v6 | Pocket v7.1 |
|---|---|---|---|---|
| T=0, seed 42: summer / winter | 9/11 / 10/11 | 6/11 / 8/11 | 8/11 / 9/11 | 8/11 / 7/11 |
| T=0.3, 3 seeds: summer / winter (of 33) | 10/11 / 9/11 (1 seed, of 11) | 24/33 / 24/33 | 25/33 / 27/33 | 24/33 / 23/33 |

**Caveat:** the condensation calibration set includes traces from these journeys, so read this as a "seen-task" score, not general agent ability. Single-seed differences of 2–3 journeys are within seed noise.

### Historical: Pocket versions

| | v6 | v7.0 | **v7.1 (legacy phone-tested)** |
|---|---|---|---|
| IFEval / MATH-500 / GSM8K | 60 / 10 / 60 | 80 / 15 / 55 | 70 / 20 / 65 |
| HumanEval / ARC-C / BFCL | 18.75 / 40 / 50 | 12.5 / 30 / 50 | 6.25 / 50 / 80 |
| Babbo T=0.3, 3 seeds (summer + winter, of 66) | 25/33 + 23/33 | 22/33 + 19/33 | 24/33 + 23/33 |
| held-out PPL | 41.9 | 41.62 | 41.79 |

v7 added a "two-bond" constraint that kept each condensed readout pointing where the Teacher's did (cosine 0.9555 vs 0.9467‡ for v6). These files are retired from current downloads; the current release uses the v15 build.


</details>

## Limitations

- Current v15 accuracy covers six skill subsets with different sample sizes, not general ability or full splits.
- The fixed answer budget truncates many competition-math generations; counts are in the results document.
- No fresh independent test set was used after selecting the v15 design. Paired p-values are exploratory under that selection.
- New and historical evaluations differ in question sets and device conditions. Do not combine their totals.
- The source is already Q4_K_M; the process operates on dequantized source weights, not a BF16 reference.
- Workstation demonstrates full-width quantization, not orbital condensation.

## License

Derived from LiquidAI/LFM2.5-24B-A2B under the LFM Open License v1.0. See `license_link`.
