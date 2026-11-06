---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
language: [en]
tags: [qwen3.8, moe, trit, ternary, quantized, tr-quantisation, mixed-precision, vllm, mtp]
datasets: [nvidia/OpenCodeInstruct, EleutherAI/hendrycks_math, openai/gsm8k, OpenAssistant/oasst2]
inference: false
---

![TR models](images/hero_1200x600.png)

# Qwen3.8-Flash-Next TR2.57-G (vLLM, one DGX Spark)

**Qwen3.8-Flash-Next translated with TR Quantisation**: its weights are stored as trits (ternary digits), on average 2.57 trits per
weight, each weight a small integer code on a symmetric grid of 9 or 27 levels with one 16-bit scale per group of weights. Routed
experts are kept at two depths: experts the router uses often keep 3 trits, the rest 2. No retraining and no fine-tuning: the
architecture, tokenizer and chat template are Qwen's. Served by vLLM 0.32.0 with the `tritq-qwen38` plugin, which adds the `trit`
quantization method.

- **Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision `de4b8e4d`), Qwen Community
  License 1.0, by the Qwen team. About 180B parameters in total; in Qwen's breakdown, 125B with 6B activated, plus 51B n-gram
  embedding and 4B MTP. Built from Qwen's BF16 weights; not derived from any NVIDIA checkpoint.
- **What is in this repo:** the text model (vision encoder removed) as packed trit tensors in safetensors; `ngram_tq27.bin`, the
  n-gram embedding table as packed 3-trit rows in one file, which the runtime reads from the SSD so it never has to fit in memory;
  routers, hyper-connection weights, norms and the multi-token-prediction (MTP) head as in Qwen's BF16 checkpoint; Qwen's tokenizer,
  chat template and configuration.
- **Runtime:** `tritq-qwen38` (vLLM plugin + Docker image), [github.com/ternii-ai/tritq-vllm](https://github.com/ternii-ai/tritq-vllm).
  Mac and llama.cpp: see `Ternii/Qwen3.8-Flash-Next-TR3-G-GGUF` (a different, all-3-trit file; experimental).
- **Name:** TR2.57-G = TR Quantisation at 2.57 trits per weight on average, codes calibrated with GPTQ (-G).
- **Publisher:** Wai Chung.

## At a glance: TR2.57-G vs NVIDIA's NVFP4 checkpoint and Qwen's FP8 release

Smaller and faster to decode than NVIDIA's NVFP4 checkpoint and a community-tuned NVFP4 recipe, with no significant accuracy
difference from NVIDIA's NVFP4 or from Qwen's own FP8 release on the suites below. It is behind NVFP4 on some of them (nominally,
not significantly), prompt-processing speed was not compared, and speed was measured on English prompts only. One DGX Spark per
server, text only, thinking off unless stated. Details and caveats under Results.

| | TR2.57-G | NVIDIA NVFP4 | Qwen FP8 (original) | |
|---|---|---|---|---|
| **Size:** weights in GPU memory, with the MTP draft head | 63.84 GiB | 75.63 GiB | — | **16 % smaller** than NVFP4; a community NVFP4 recipe: 73.51 GiB |
| ... download (files in `SHA256SUMS`) | 104.35 GB | 132.73 GB | — | 21 % smaller |
| **Speed:** decode tokens/s, 1 user (community benchmark, median prompt; each engine at its own settings) | **84.3** | 40.5 | not run | **x2.08** vs NVIDIA's checkpoint; **x1.17** vs the community recipe (71.9) |
| ... 6 users, total tokens/s | 95.2 | 62.0 | | x1.54; x1.12 vs the recipe (85.2) |
| ... prompt processing | not compared | | | see Limitations |
| **Accuracy:** GSM8K (1,319, 8 requests in flight) | 92.11 | 91.85 | 91.96 | no significant difference |
| HumanEval+ (plus) | 92.07 | 93.90 | 92.68 | no significant difference (NVFP4 nominally ahead) |
| MBPP+ (plus) | 79.10 | 78.84 | 79.10 | no significant difference |
| IFEval (prompt strict) | 83.73 | 84.84 | not run | no significant difference (NVFP4 nominally ahead) |
| MGSM, multilingual maths (11 languages) | 89.96 | 89.13 | not run | no significant difference |
| BFCL v4 function calling | 74.04 | 72.20 | not run | TR nominally ahead, not significant after correction |
| LiveCodeBench, thinking, 32k tokens (454) | 64.54 | 61.45 | 64.98 | TR vs original: no significant difference; NVFP4 vs original: depends on the test family (see Results) |

- Accuracy rows give the mean of two servers where two were run (GSM8K, HumanEval+, MBPP+ for TR and NVFP4); the original ran one.
- Speed was measured on English prompts, with 2 comparator servers per comparator (no p values). NVIDIA's checkpoint ran untuned, with
  its n-gram table read through our disk-backed lookup (not a stock setup).
- "No significant difference" is not a claim of equality: at these test sizes a 2-3 point difference on HumanEval+ or LiveCodeBench
  would not have been detectable.

## Quick start (NVIDIA DGX Spark, 128 GB)

```bash
docker pull ghcr.io/ternii-ai/tritq-qwen38:0.3 && docker tag ghcr.io/ternii-ai/tritq-qwen38:0.3 tritq-qwen38:0.3
# or build it: git clone https://github.com/ternii-ai/tritq-vllm && cd tritq-vllm/runtime &&
#              docker build -f docker/Dockerfile.qwen38-vllm032 -t tritq-qwen38:0.3 .
docker run --rm --gpus all --ipc=host --network host --ulimit memlock=-1 --shm-size 16g \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  tritq-qwen38:0.3 Ternii/Qwen3.8-Flash-Next-TR2.57-G --served-model-name qwen38-tr \
  --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 8 --gpu-memory-utilization 0.85 \
  --kv-cache-memory-bytes 10737418240 --language-model-only --moe-backend triton \
  --speculative-config '{"method":"mtp","num_speculative_tokens":6,"draft_confidence_threshold":0.7,"draft_confidence_fallback_depth":3}'
```

Then use any OpenAI-compatible client against `http://localhost:8000/v1`. Our benchmarks ran with thinking off
(`"chat_template_kwargs": {"enable_thinking": false}`) unless stated. Keep the model on a local NVMe drive: the n-gram table is read
from it while the model runs. The first start takes about 11-12 minutes (kernel compilation) unless a cache directory is mounted.
`--kv-cache-memory-bytes 10737418240` (10 GiB, 50,202 tokens) left at least 30 GB of host memory free in our runs on a 128 GB Spark; raise it only if
nothing else runs on the machine.

What the image turns on (release configuration):

| setting | default | how to change |
|---|---|---|
| MTP speculative decoding with a **confidence stop** (vLLM PR #60068, applied in the image) | on via `--speculative-config` above: with one request decoding, draft up to 6 tokens and stop when the drafter's top-1 probability falls below 0.7; with two or more, draft 3 | fixed depth: `'{"method":"mtp","num_speculative_tokens":3}'` |
| draft head | **sparse, all languages** (`TRITQ_DRAFT_HEAD=sparse`) | `-e TRITQ_DRAFT_HEAD=english` (first 100,000 vocabulary rows: about 1 % faster on English, slower on other languages; the default head is 13-30 % faster than it on Chinese, Japanese, Russian and Korean) or `full` |
| FP8 MTP draft head (the draft head quantised to FP8 at load) | on | `-e TRITQ_MTP_FP8=0` for the stored BF16 draft head (66.24 GiB in GPU memory instead of 63.84, measured) |
| v2 kernels | on | `-e TRITQ_KERNELS_V2=0` |
| 8-bit hyper-connections (quantised at load) | on | `-e TRITQ_HC_INT8=0` for the stored BF16 values |
| n-gram page read-ahead | on | `-e TRITQ_NGRAM_PREFETCH=0` |

The full model verifies every draft with the full vocabulary, so the draft head is chosen for speed. It is not output-neutral, though:
under the confidence stop the drafts decide how many tokens are verified per step, which changes floating-point rounding, so the draft
head and the FP8 draft head can change the greedy text on some prompts (accuracy measured below, in this configuration). The v2 kernels
alone keep the text unchanged at one request.

## Hardware and memory

| | measured | notes |
|---|---|---|
| Weights in GPU memory (vLLM, text model + MTP draft head, release configuration) | **63.84 GiB** | NVIDIA's NVFP4 checkpoint: 75.63 GiB; a community NVFP4 recipe: 73.51 GiB; with the BF16 draft head: 66.24 GiB |
| KV cache | 10 GiB = 50,202 tokens | fixed by the quick-start flag; the n-gram table is not resident (read through the page cache) |
| Free host memory while serving | at least 30 GB in our benchmark runs | 128 GB Spark, 8 sequences, 8,192-token context |
| Download (the files listed in `SHA256SUMS`) | 104.35 GB (104,345,619,728 bytes) | NVIDIA's NVFP4 repo: 132.73 GB. Part of the gap is the n-gram file: ours 3-trit packed, 33.9 GB; NVIDIA's FP8 file holding the n-gram table and MTP head, 53.72 GB |
| Tested on | one NVIDIA DGX Spark (GB10, 128 GB unified memory), vLLM 0.32.0, CUDA 13.0 | other GPUs, smaller machines and multi-GPU setups untested |

## Results (released checkpoint, release configuration)

One DGX Spark per server, vLLM 0.32.0. Accuracy was measured on the released weights with our image `tritq-qwen38:0.3` at its default
settings. The speed and multilingual rows were measured on the development build of the same runtime with the same settings; the 0.3
image then gave 84.9 tokens/s at 1 user (85.1 / 84.9 / 84.9, one server on a different Spark; identical text on 40 of 40 prompts). The released `config.json`
drops bookkeeping keys the runtime does not read. Drift was scored separately (2026-10-24). Every number was recomputed from the raw
files by an independent reviewer.

**Speed (English prompts).** Decode tokens/s. "Median prompt" is the community single-Spark benchmark (tonyd2wild
`bench_categories.py` @6ad1c8f, unchanged: 40 short English prompts in 8 categories, one user, greedy, thinking off; figure = decode
rate of the median prompt, median of 3 runs); "long answers" is our benchmark (8 English maths, code and prose prompts x 2 repeats, up
to 512 tokens). All servers interleaved on one DGX Spark in one session: 4 of ours, 2 of each comparator; **each engine with its own
serving settings**.

| | median prompt, 1 user | mean over prompts, 1 user | 6 users, total | long answers, 1 / 4 / 8 users, per stream | long answers, 8 requests, total |
|---|---|---|---|---|---|
| TR2.57-G, release default (all-language draft head) | **84.3** | 77.3 | 95.2 | 74.7 / 34.5 / 22.8 | 126.0 |
| A community NVFP4 recipe (salient-data qwen38-next-dgx-recipe, its published image and settings) | 71.9 | 66.8 | 85.2 | 62.5 / 30.9 / 24.3* | 107.2* |
| NVIDIA NVFP4 checkpoint on vLLM 0.32.0, MTP-3, untuned | 40.5 | 35.7 | 62.0 | 35.0 / 21.6 / 15.4 | 87.1 |
| ratio, TR / community recipe | **x1.17** | x1.16 | x1.12 | x1.20 / x1.11 / not comparable* | x1.18* |
| ratio, TR / NVIDIA checkpoint | **x2.08** | x2.17 | x1.54 | x2.14 / x1.60 / x1.48 | x1.45 |

\* The recipe runs at most 6 sequences, so at 8 requests 2 of them wait: its per-stream 24.3 is per running stream (ahead of our 22.8)
and its total is at its own 6-sequence limit.

- Ratios are of arm means with **2 comparator servers** each, so no p value is given for them. Every TR server was faster than both
  servers of each comparator on every statistic in this table except the recipe's per-stream figure at 8 requests.
- **English prompts only; no NVFP4 comparison exists off English.** On Chinese, Japanese, Russian and Korean the default (all-language)
  draft head is faster than the English-and-code head by 16 % (zh), 13 % (ja), 30 % (ru) and 19 % (ko), and 1 % slower on English
  (8 long-answer prompts per language, 1 user, 2 + 2 servers, faster on 8 of 8 prompts in each of those four languages; measured with
  the FP8 draft head, without the v2 kernels). TR vs TR only.
- Serving settings: ours max-model-len 8,192, 10 GiB KV cache, up to 8 sequences; the recipe 262,144 context, FP8 KV cache, prefix
  caching, up to 6 sequences; NVIDIA's checkpoint GPU memory 0.70, its n-gram table read through our plugin's disk-backed lookup.
- **Prompt processing and long-context speed were not compared:** our run against NVIDIA's checkpoint at 2k-16k context was unstable on
  the NVIDIA side (2-5x swings within one server), so we quote no ratio.
- No energy or efficiency claim.

**Accuracy vs NVIDIA NVFP4** (full test sets, greedy, thinking off, max 1,024 new tokens, IFEval 1,280, evalplus its default; paired exact McNemar test item
by item on one server pair, Holm-corrected over the 15 primary comparisons; TR and NVFP4 ran on two identical DGX Sparks):

| | GSM8K (1,319), 8 in flight | HumanEval+ base / plus (164) | MBPP+ base / plus (378) | IFEval prompt strict / loose (541) | MGSM (11 x 250) | BFCL v4 (3,691) |
|---|---|---|---|---|---|---|
| TR2.57-G, server 1 / server 2 | 92.34 / 91.89 | 94.51 / 92.07 (both) | 92.06 / 79.10 (both) | 83.73 / 87.62 (server 1) | 89.96 (server 1) | 74.04 (server 1) |
| NVIDIA NVFP4 checkpoint, server 1 / server 2 | 91.81 / 91.89 | 95.73 / 94.51; 94.51 / 93.29 | 92.33 / 78.31; 92.86 / 79.37 | 84.84 / 88.17 (server 1) | 89.13 (server 1) | 72.20 (server 1) |
| exact McNemar, server 1 (raw p) | 0.28 | plus 0.22 | plus 0.58 | 0.45 | 0.067 | 0.0065 |

- **No significant difference from NVIDIA's NVFP4 on any primary comparison after correction** (smallest Holm-adjusted p 0.098, BFCL).
  Where NVFP4 is nominally ahead: HumanEval+ (plus 92.07 vs 93.90, mean of two NVFP4 servers), MBPP+ base, IFEval. Where TR is
  nominally ahead: GSM8K, MBPP+ plus, MGSM, BFCL.
- GSM8K: lm-eval `gsm8k_cot_llama`, 0-shot chat, strict "The final answer is" match, 8 requests in flight on the confidence-stop path
  (at one request in flight TR scores 92.72 on both servers; 8 in flight is nominally 0.4-0.8 points lower, not significant after
  correction). HumanEval+ / MBPP+: evalplus 0.3.1. IFEval and MGSM: lm-eval 0.4.13.
- **MGSM** (8-shot native chain of thought, flexible extraction): no language differs significantly. With the strict extractor TR scores
  63.2 vs 75.6 on Telugu (Holm-corrected p 0.0002 over all 67 comparisons): TR ends Telugu answers in a form the strict pattern does not
  read; flexible extraction gives 87.2 vs 87.6. Read it as an answer-format difference. MGSM was not run on Qwen's original.
- **BFCL v4** (bfcl-eval 2026.3.23, 12 categories, near-greedy): the pooled lead comes from declining irrelevant calls (irrelevance 67.08
  vs 55.00, the one category that survives correction over all 67 comparisons) and the parallel / parallel_multiple categories. TR is nominally behind on
  simple_python (93.25 vs 95.00), multiple (92.00 vs 92.50), live_simple (76.36 vs 78.68), live_multiple (75.31 vs 76.83), live_parallel
  (56.25 vs 81.25, 16 items), live_parallel_multiple (54.17 vs 58.33, 24 items) and live_relevance (81.25 vs 93.75, 16 items). One server per arm; NVFP4 had 3 request errors, scored as failures (if all 3 had been
  correct the verdict would not change). Read it as that, not as a general function-calling advantage.
- **Reproducibility:** at one request in flight TR's served text was identical across two server starts (GSM8K 1,319 of 1,319 replies;
  evalplus 0 scoring flips). With 8 requests in flight the text depends on batch composition (666 of 1,319 replies identical across
  starts). NVIDIA's checkpoint varied between starts even at one request.

**Accuracy vs Qwen's own FP8 release** (`Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf2`, served on two Sparks with tensor parallelism 2;
the BF16 original does not fit two Sparks). Paired exact McNemar, Holm over the full 29-comparison family:

| | Qwen FP8 (original) | TR2.57-G | NVIDIA NVFP4 |
|---|---|---|---|
| LiveCodeBench, thinking on, 32,768-token budget (454 problems) | **64.98** | **64.54** | 61.45 |
| ... easy (110) / medium (141) / hard (203) | 99.09 / 85.82 / 32.02 | 99.09 / 82.27 / 33.50 | 98.18 / 80.85 / 28.08 |
| ... answers cut at the budget | 143 | 146 | 159 |
| GSM8K strict / flexible (1,319, 8 in flight, thinking off) | 91.96 / 92.12 | 92.34 / 92.65 (server 1); 91.89 / 92.04 (server 2) | 91.81 / 91.89; 91.89 / 92.12 |
| HumanEval+ base / plus (thinking off) | 93.90 / 92.68 | 94.51 / 92.07 | 95.73 / 94.51; 94.51 / 93.29 |
| MBPP+ base / plus (thinking off) | 92.59 / 79.10 | 92.06 / 79.10 | 92.33 / 78.31; 92.86 / 79.37 |

- **TR shows no significant difference from Qwen's FP8 release** on any of these (Holm p 1.00 everywhere; LiveCodeBench raw p 0.87).
  This is not equivalence: the LiveCodeBench gap is 0.44 points with 38 discordant problems, an approximate 95 % range of +-2.7 points,
  so a loss as large as NVFP4's cannot be ruled out.
- **NVIDIA NVFP4 is 3.5 points below the original on LiveCodeBench with thinking** (raw p 0.005). Whether that is significant depends on
  the test family: Holm over the full 29-comparison family gives 0.151 (a conservative family, padded with near-duplicate rows); over the
  3 thinking-mode LiveCodeBench tests 0.016; over the 5 LiveCodeBench rows 0.026. NVFP4's deficit of 16 problems matches its 16 extra
  answers cut at 32k; on answers that finished, pass rates are within 0.3 points across the three (94.9 / 95.1 / 94.6 %), which is
  consistent with NVFP4 reasoning longer and hitting the budget.
- LiveCodeBench: official lcb_runner prompt, extraction and tests, dataset `code_generation_lite` window 2024-08-01..2025-05-01, Qwen's
  thinking sampling (temperature 1.0, top_p 0.95, top_k 20), seed 0, **one sample per problem per arm**, so part of each gap is
  sampling noise. Not comparable with published LiveCodeBench figures. A thinking-off run at a 4,096-token budget gave 49.56 / 50.00 /
  49.34 (original / TR / NVFP4, no significant difference), but cut ~46 % of answers on every arm, so read it as a paired retention
  check only.
- Different sessions and serving setups: the original ran at tensor parallelism 2 over two Sparks, TR and NVFP4 at 1; one server per
  arm except where two are shown. English and Python only.

**Drift** (next-token probabilities against the original BF16 model): on GSM8K answers TR is closer to the original than NVIDIA's
NVFP4 (ratio 0.91) and on HumanEval answers closer (0.76); on general text it is further (1.30).

**Comparator and method.** NVIDIA's checkpoint: `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fc694b54`, weights unmodified, vLLM 0.32.0, MTP-3,
GPU memory 0.70, its n-gram table read from the SSD through the same disk-backed lookup as ours (stock vLLM keeps it in pinned host
memory). Community recipe: `salient-data/qwen38-next-dgx-recipe` @f309019, its published image by digest, its own settings. Harness,
server flags and scripts: [github.com/ternii-ai/tritq-vllm/tree/main/runtime/bench](https://github.com/ternii-ai/tritq-vllm/tree/main/runtime/bench).

## Limitations

- **Speed is measured on English prompts only**; no NVFP4 comparison exists off English. Ratios against each comparator rest on 2
  comparator servers, and each engine ran at its own serving settings.
- **Prompt processing and long contexts were not compared** with NVFP4 (see Results). Accuracy at long context was not measured; the
  quick start caps the context at 8,192 tokens.
- **Not ahead everywhere:** NVFP4 is nominally ahead on HumanEval+, MBPP+ base and IFEval, and on seven BFCL categories; the community
  recipe is ahead per running stream at 8 requests. None of the accuracy gaps is significant.
- **Accuracy differences of a few points are not resolved.** "No significant difference" is not equivalence; a 2-3 point gap on
  HumanEval+ or LiveCodeBench would not have been detected.
- **The draft heads change the served text** through the confidence stop (accuracy above is measured with them). Text identity of the
  v2 kernels with several requests decoding was not checked; with 8 requests in flight GSM8K was nominally 0.4-0.8 points lower than
  with one (not significant).
- **The confidence stop is a single-user setting.** With two or more requests it falls back to 3 drafts.
- **Drift:** closer to the original than NVFP4 on answers, further on general text.
- **Not measured:** multilingual accuracy other than MGSM; MGSM, IFEval and BFCL against the original; energy.
- One DGX Spark, vLLM 0.32.0 only; other GPUs and multi-GPU setups are untested.
- Per bit, a well-calibrated NVFP4 is at least as accurate as a trit grid of the same size. The speed and size advantage comes from how
  the bits are allocated and how the kernels read them, not from the number format itself.
- Same safety properties and limitations as the base model; no additional alignment or filtering was applied.

## How it was made (conceptual)

The publisher's BF16 weights are translated, not retrained: choose how many trits each part of the model needs from measured
sensitivity (with more trits for routed experts that the router uses often); round every weight to a k-trit code with a calibrated
method (GPTQ); pack the codes; then gate the result (every packed matrix read back bit for bit, drift against the original, benchmarks
against the original and NVFP4). Routers, norms, hyper-connection weights and the MTP head keep their original precision in the files.
The calibration procedure and the allocation procedure are not published.

## Licence (Qwen Community License 1.0)

Derived from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision `de4b8e4d`, Copyright (c) 2026 Qwen)
and distributed under the **Qwen Community License 1.0** (`LICENSE`, unmodified; our changes are listed in `NOTICE`). Two conditions
of that licence pass to you:

1. If you use this model in a commercial product or service with more than 100,000,000 monthly active users or US$20,000,000 monthly
   revenue, the model name must be displayed prominently in its user interface.
2. If you or your affiliates run a Model-as-a-Service or AI Work Assistant business (as defined in the licence), you need a separate
   licence from Qwen before commercial use.

Questions about the licence: model-business@notice.qwencloud.com. Built from Qwen's BF16 weights; not derived from any NVIDIA
checkpoint. The weight files have been modified by Wai Chung; no training was performed. Calibration prompts were drawn from NVIDIA's
OpenCodeInstruct (CC BY 4.0), the MATH and GSM8K training sets (MIT) and OpenAssistant oasst2 (Apache-2.0); see `NOTICE`. This is not
an official Qwen release and is not endorsed by Alibaba Cloud.

The runtime credits vLLM pull request #60068 (confidence stop, Apache-2.0, by @stecasta, via salient-data/qwen38-next-dgx-recipe patch
0010) and #59740 (sparse draft head algorithm, reimplemented), and that recipe's patch 0011 (skinny-GEMM rows, Apache-2.0); see the
runtime's NOTICE.

**Trademarks.** Qwen is a trademark of Alibaba Cloud / Alibaba Group. NVIDIA, DGX, DGX Spark and CUDA are trademarks and/or registered
trademarks of NVIDIA Corporation in the U.S. and other countries. Mac, MacBook Pro and Metal are trademarks of Apple Inc., registered in
the U.S. and other countries and regions. Other names are the property of their respective owners. They are used only to identify the
products we compare with or run on. This project is independent and is not affiliated with, sponsored or endorsed by Alibaba, NVIDIA,
Apple or Hugging Face. "NVFP4" names NVIDIA's 4-bit format and NVIDIA's published checkpoints in it.

## Contact

hello@ternii.com (questions, bug reports, licence and takedown requests).

## Files and checksums

| file | size (bytes) | sha256 |
|---|---|---|
| `model-00001.safetensors` … `model-00049.safetensors` (49 shards) and `model.safetensors.index.json` | | one line per file in `SHA256SUMS` |
| `ngram_tq27.bin` (n-gram table, packed 3-trit rows) | 33,920,162,816 | `b3a3176d2a828f622083dd75f81bbf6988c4db2be2ebd70e0a50f9e45f1f8277` |
| `config.json` | | listed in `SHA256SUMS` |
| tokenizer, chat template and processor files | as in the base model | copied unchanged from `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d`; listed in `SHA256SUMS` |
| all files in `SHA256SUMS` | 104,345,619,728 | |

Check a download with `sha256sum -c SHA256SUMS` from inside the folder.
