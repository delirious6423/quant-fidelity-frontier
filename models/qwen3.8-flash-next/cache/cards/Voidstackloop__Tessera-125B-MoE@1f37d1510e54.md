---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
tags:
- gguf
- moe
- quantized
- experimental
---
# Tessera-125B-MoE, 448 experts (EXPERIMENTAL)

Quantized, expert-pruned GGUF of Qwen/Qwen3.8-Flash-Next (architecture qwen4exp: 48 layers, 512 routed experts per layer, 10 active).

* **Experts:** 448 of 512 kept per layer, chosen by routed share on English, code, math and reasoning calibration text. The pruning is a byte slice
  of the finished quantized tensors (kept experts are bit-identical to the 512-expert file, no requantization); router rows were removed the same way.
  `expert_count` = 448, `n_expert_used` = 10.
* **Quantization** (`llama-quantize --pure` with a domain importance matrix): gate/up experts IQ2_S, down experts IQ4_NL, body Q6_K,
  per-layer n-gram table Q4_0 (loaded lazily with `-lzm on`), shared experts / hc / ple Q8_0.
* `Qwen3.8-Flash-Next-k448-IQ2_S.gguf` is 70.5 GiB; `mtp-Qwen3.8-Flash-Next-balanced.gguf` is the multi-token-prediction draft head (not pruned), for `--spec-type draft-mtp`.
* `SHA256SUMS` lists the checksums of both GGUF files.
* Needs a llama.cpp build with qwen4exp support (commit 7894e8b or newer).

## Evaluation (2026-10-26)

Everything below was run by the author on this exact file with llama.cpp. Each row names the setting; "best" means the best setting we tried for that
benchmark, not a cherry-picked run, and every number comes from a single run. Samples are small (60 tasks per benchmark), so a 95% interval is
roughly +/-12 points. Per-task tables are in `benchmarks/results/`, the scripts in `benchmarks/scripts/`, the full write-up in `benchmarks/REPORT.md`.

| Benchmark | Best result | Setting | Notes |
|---|---|---|---|
| **LiveCodeBench v6** (60 of 175 problems, 2025-01 to 2025-04), best | **54/60 = 90.0%** | thinking, 16,384-token budget + up to 2 repair turns | **Chained estimate, not one clean run**: 41 passes from the 4k run + 9 rescued by rerunning the 21 failures at 16k + 4 rescued by repair turns. Repair turns show the model the first failing visible sample test (hidden tests only score the result), so this is an agent-style loop, not single-shot. |
| LiveCodeBench v6, clean single attempt | 49/60 = 81.7% | thinking, 16,384-token budget, one attempt, all 60 rerun | The comparable number for single-shot leaderboards. 4k budget: 41/60 (68.3%); no-think: 38/60 (63.3%). 16k vs 4k paired p = 0.021. |
| CRUXEval (60 tasks) | **58/60 = 96.7%** | any setting | At its ceiling in every run. |
| Coding total (LiveCodeBench + CRUXEval, 120 tasks) | **112/120 = 93.3%** (estimate, with repair turns); clean single attempt 107/120 = 89.2% | thinking, 16k budget (+ repair turns for the 112) | 4k: 99/120; no-think: 96/120. |
| HumanEval+ / MBPP (60 tasks) | 55/60 = 91.7% | no-think, greedy | HumanEval+ 22/24, MBPP 33/36. |
| MATH-500 (45 tasks) | 39/45 = 86.7% | no-think, greedy, 2,048-token cap | AIME 0/15 in the same run: every answer hit the token cap, so it is not a measure of ability. Thinking mode on math was only tested on a few failures. |
| MMLU-Pro (40) + BBH (20) | 50/60 = 83.3% | no-think | BBH 20/20, MMLU-Pro 30/40. Thinking with a 4k budget rescued 5 of the 10 failures (about 55/60 as an estimate); a 16k budget rescued 3 of 10. |
| English (RACE, CoLA, rule-following; 60 tasks) | 52/60 = 86.7% | no-think | Quick set; not compared with other models. |


### Against the unsloth UD-IQ4_XS quant (87 GiB)
No-think greedy on a 120-task suite (HumanEval+, MATH-500, MMLU-Pro; 40 each): Q4 103/120, this model 101/120, paired sign test p = 0.754.
**It matches Q4 within noise; it does not beat it.** Same suite on the same machine: 11.7 vs 8.2 tokens/s, and the file is 17 GiB smaller.
The full-precision parent was not run by us; vendor-reported LiveCodeBench v6 for the parent is 91.9.

### Speed on an RTX 4060 Laptop (8 GB) with 96 GB RAM, native Windows llama.cpp with a GPU expert cache
About 22 tokens/s decoding short code answers (MTP draft head, or a large GPU expert cache without MTP); 18 to 19 tokens/s with a 128k context
window configured; 14 to 16 tokens/s for prose. Prefill about 125 to 165 tokens/s. 

### Caveats
* One run per setting, 60 tasks per benchmark, a few problems of run-to-run noise.
* The thinking-budget results are fixed-budget runs; the hard LiveCodeBench problems use the whole budget.
* Competitor numbers are not included because we did not run other models ourselves.
* This is an experimental, pruned, 2-bit-class quantization: expect weaker results than the full-precision parent on hard tasks.
