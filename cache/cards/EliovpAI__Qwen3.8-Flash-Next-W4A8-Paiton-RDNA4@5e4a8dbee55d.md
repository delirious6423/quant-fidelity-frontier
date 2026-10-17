---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: paiton
pipeline_tag: text-generation
inference: false
tags: [paiton, vllm, amd, rocm, rdna4, gfx1201, int4, 4-bit, w4a8, gptq, hadamard, moe, mtp, speculative-decoding, expert-offload]
---


# Qwen3.8-Flash-Next 4-bit (W4A8) for Paiton on two RDNA4 GPUs

**The accuracy of the 16-bit model, on two R9700: the same GSM8K and HumanEval scores, and MMLU-Pro within one point.** The
16-bit model is 335 GiB (360 GB) and can't run on this hardware at all.

| Task benchmark | Questions | 16-bit | **This release (4-bit)** |
|---|---:|---:|---:|
| GSM8K (grade-school math) | 1,319 | 96.9 % | **96.8 %** |
| HumanEval (Python, pass@1) | 164 | 96.3 % | **96.3 %** |
| MMLU-Pro (100 per category × 14) | 1,400 | 84.7 % | **83.7 %** |

Same prompts, sampling seeds and thinking-on chat template for both. GSM8K and HumanEval are within the 16-bit model's own
run-to-run variation (0.0 and 1.2 points; paired McNemar p = 1.00). The MMLU-Pro gap of 1.0 point is not statistically
significant (p = 0.16) and comes from replies that reach the 16,384-token limit: the 4-bit model thinks about 12 % longer; on
the 1,271 questions where neither model reached the limit it scores 91.1 % against 91.0 %
([method](#task-benchmarks)). Perplexity is within 0.9 % of the 16-bit model on wikitext and within 0.4 % on unseen text
([Perplexity](#perplexity)); on long documents, needles and full-vocabulary KL see [below](#long-documents-needles-and-kl).

## Accuracy

### Task benchmarks

Both models answer the same questions with the same chat template (thinking on), the same sampling (temperature 1.0,
top-p 0.95, top-k 20) and the same per-question seed; replies are capped at 16,384 tokens and a capped reply counts as
wrong. The 16-bit reference is the original checkpoint served with stock vLLM; this release is served through the launcher
exactly as shipped (`--mode 4bit`). GSM8K is scored 0-shot on the `\boxed{}` answer, HumanEval by running the tests in a
sandbox, MMLU-Pro on a fixed 1,400-question subset (100 per category, all 14 categories).

| Task | Questions | 16-bit | This release | Difference (paired 95 % CI) | 16-bit vs itself, second seed set |
|---|---:|---:|---:|---:|---:|
| GSM8K | 1,319 | 96.9 % | **96.8 %** | -0.1 points (-0.8 … +0.7) | 0.0 points |
| HumanEval pass@1 | 164 | 96.3 % | **96.3 %** | 0.0 points (-2.9 … +2.9) | 1.2 points |
| MMLU-Pro | 1,400 | 84.7 % | **83.7 %** | -1.0 points (-2.3 … +0.3) | 0.9 points (on 916 questions) |

Questions answered differently: GSM8K 14 right only in 16-bit and 13 right only in 4-bit, HumanEval 3 and 3 (exact McNemar
p = 1.00 for both), MMLU-Pro 49 and 35 (p = 0.16). Mean reply length: GSM8K 565 tokens (16-bit 560), HumanEval 1,510
(16-bit 1,388), MMLU-Pro 3,240 (16-bit 2,902).

MMLU-Pro replies that reached the 16,384-token limit (counted as wrong): 104 for this release, 79 for the 16-bit model. Of the
questions only one model got right, 30 are 16-bit-right answers that this release did not finish within the limit and 14 the
other way round. On the 1,271 questions where neither model reached the limit, the scores are 91.1 % (this release) and
91.0 % (16-bit), with 21 and 19 questions right in only one of them. Per category (16-bit / this release, 100 questions
each): biology 93/92, business 88/86, chemistry 91/91, computer science 86/84, economics 92/93, engineering 75/73, health
83/79, history 77/78, law 75/76, math 96/93, other 81/80, philosophy 76/74, physics 87/88, psychology 86/85.

### Perplexity

Measured the llama.cpp way: raw text, no chat template, consecutive chunks of the context length, each chunk scored on its
second half, and the full vocabulary compared against bf16 on the same tokens. The quantised model runs the emulation of the
served arithmetic described below.

| Text, context length | bf16 | **this release, as served** | change |
|---|---:|---:|---:|
| wikitext-2 (test), 512 | 4.589 | **4.629** | **+0.87 %** |
| wikitext-2 (test), 4096 | 3.929 | **3.961** | **+0.80 %** |
| unseen text, 512 | 6.938 | **6.954** | **+0.23 %** |
| unseen text, 4096 | 6.163 | **6.187** | **+0.38 %** |

*Unseen text* is 22 long fineweb-edu documents (262,144 tokens) that were never used in calibration or anywhere else in
making this quant. It sits next to wikitext on purpose: this model's n-gram embedding table has partly memorised wikitext,
which inflates differences there about 2x. The most likely next token agrees with bf16 at **91.0-92.7 %** of the scored
positions; two correct bf16 implementations agree at 95.9-96.7 %, which is the ceiling.

### Long documents, needles and KL

| Measure | bf16 | this release, as served |
|---|---:|---:|
| NLL on long documents (44 docs, 8K-32K tokens), document level | 1.7574 | **1.7603** (+0.0028, within 0.16 %) |
| NLL on long documents, positions ≥ 8K | 1.7858 | **1.7887** (+0.0028) |
| Needle retrieval at 131,072 tokens (40 needles, exact match) | 40 / 40 | **40 / 40**, 419 of 419 answer tokens identical |
| Top-1 agreement, evaluation corpus (170,884 text+assistant positions) | 96.77 % (two bf16 implementations) | **93.36 %** |
| KL t+a, evaluation corpus, free / routing locked | 0.0077 (two bf16 implementations) | **0.0306 / 0.0166** |

Everything here is measured against the bf16 model with an emulation of the served arithmetic: the same expert, trunk,
attention and head arithmetic the runtime executes, in the configuration the launcher serves (`--mode 4bit`: INT6 trunk,
int8 prompt activations, fp8 KV cache). KL t+a is the average full-vocabulary Kullback-Leibler divergence from bf16's
next-token distribution to this model's at text and assistant positions (zero = identical). "Free" lets the quantised model
choose its own experts; "routing locked" forces bf16's expert choices and isolates the arithmetic error. Top-1 is how often
the most likely next token agrees with bf16. NLL is the negative log-likelihood in nats per token on assistant text.
`--mode 4bit-200k` serves the same experts and fp8 KV cache with the 8-bit trunk; it was not measured separately.

### Agreement with bf16 by configuration (evaluation corpus, full set)

| Weights / configuration | bits / weight (routed experts) | KL t+a, free | KL t+a, routing locked | top-1 agreement |
|---|---:|---:|---:|---:|
| bf16 (two correct implementations against each other: the floor) | 16 | 0.0077 | n/a | 96.77 % |
| INT3 ([W3A8 repository](https://huggingface.co/EliovpAI/Qwen3.8-Flash-Next-W3A8-Paiton-RDNA4)) | 3.125 | 0.0604 | 0.0413 | 90.74 % |
| all-INT4 (same recipe, every expert INT4, 8-bit trunk, exact arithmetic) | 4.125 | 0.0288 | 0.0135 | 93.51 % |
| INT4 / INT3 hybrid (this repository's experts, 8-bit trunk) | 3.926 | 0.0293 | 0.0152 | 93.49 % |
| **this repository as served** (hybrid experts, INT6 trunk, int8 prompt rows, fp8 KV cache) | 3.926 | **0.0306** | **0.0166** | **93.36 %** |
| standard MXFP4, round-to-nearest, measured by us on the same experts | 4.25 | 0.0515 | 0.0320 | 91.69 % |

The standard MXFP4 row is our own measurement: it applies the standard OCP format (FP4 E2M1, E8M0 scale per 32 weights, no calibration) to the same experts with
the same trunk and arithmetic, so it isolates the weight format: the served 4-bit stack has 41 % lower KL than standard MXFP4 (free routing).

### Long documents (long-v1: 44 held-out documents of 8K-32K tokens, 524,288 positions)

NLL on assistant text (nats per token; difference to bf16 and its 95 % interval):

| Weights | < 2K | 2-4K | 4-8K | ≥ 8K | document level |
|---|---|---|---|---|---|
| bf16 reference | 1.8416 | 1.7573 | 1.6911 | 1.7858 | 1.7574 |
| INT3 (3-bit release) | 1.8498 (+0.0082) | 1.7682 (+0.0108) | 1.7020 (+0.0110) | 1.7942 (+0.0083) | 1.7671 (+0.0097) |
| **this repository as served** | **1.8475 (+0.0059 [+0.0028, +0.0088])** | **1.7619 (+0.0046 [+0.0016, +0.0072])** | **1.6915 (+0.0004 [−0.0134, +0.0082])** | **1.7887 (+0.0028 [+0.0006, +0.0070])** | **1.7603 (+0.0028)** |

The 4-8K interval is wide because one agentic document's free-routing trajectory diverges from bf16's in an assistant
segment (a few hundred positions); with the bf16 routing the same positions stay close.

KL over all positions per position bucket, free / routing locked [top-1 on assistant text]:

| Weights | < 2K | 2-4K | 4-8K | ≥ 8K | document level t+a |
|---|---|---|---|---|---|
| INT3 (3-bit release) | 0.1554 / 0.1043 [89.46 %] | 0.6300 / 0.2931 [89.66 %] | 0.6408 / 0.2984 [90.06 %] | 0.4333 / 0.2150 [91.65 %] | 0.0561 / 0.0408 / 90.39 % |
| **this repository as served** | **0.0955 / 0.0489 [91.99 %]** | **0.4595 / 0.1573 [92.28 %]** | **0.4978 / 0.1711 [92.58 %]** | **0.3419 / 0.1282 [93.60 %]** | **0.0307 / 0.0180 / 92.75 %** |
| standard MXFP4, round-to-nearest, measured by us on the same experts (8-bit trunk, bf16 KV cache) | 0.1313 / 0.0756 [90.84 %] | 0.5866 / 0.2618 [91.24 %] | 0.6108 / 0.2794 [91.64 %] | 0.4178 / 0.1979 [92.55 %] | 0.0395 / 0.0266 / 91.72 % |

**Needle retrieval** at 131,072 tokens (20 single needles at depths 2.5-97.5 % and 5 multi-key prompts with 4 needles
each, 40 needles; exact match, paired against bf16's own answers): 40 of 40 (bf16: 40 of 40), 20 of 20 single and 20 of 20
multi-key, every bf16 answer reproduced token for token (419 of 419 answer tokens, 25 of 25 prompts), with the fp8 KV cache.

### Speculative decoding

The MTP layer's experts are INT4 (GPTQ on the quantised target). Expected accepted drafts per step at depth 3
(teacher-forced, greedy, assistant text): 1.782 with the exact head, 1.773 with the shipped int2 draft head, against
1.789 for the bf16 MTP layer on the bf16 model.

## The weights

These are 4-bit weights for [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision
`de4b8e4d`), quantised with our own GPTQ run and served by Paiton's native RDNA4 runtime on two AMD Radeon AI PRO R9700
cards.

**GPUs:** 2 × Radeon AI PRO R9700 (`gfx1201`, 32 GB each) · **System RAM:** 256 GB for the benchmarked configuration, 128 GB and 96 GB supported
(see [Hardware](#hardware-requirements)) · **Size on disk:** 123.9 GiB · **Context:** 262,144 tokens natively; served at
98,304 (speculative decoding on) and 200,000 (off) · **KV cache:** fp8, about 305,000 tokens
(`4bit`) / 247,000 tokens (`4bit-200k`) · **Text weights, MTP layer and draft heads; bf16 vision tensors included**

## Speed

It is also fast: through the OpenAI-compatible server a single request decodes at **~190 tok/s** and a
64K-token prompt prefills at about **7,000 tok/s** (BetterBench 64K row 6,991), and an exact 64,000-token prompt at
**7,162 tok/s**, with part of the experts served from pinned system RAM
because the 4-bit model does not fit two 32 GB cards next to its KV cache.

BetterBench 0.6.0 standard profile through the vLLM OpenAI-compatible server on two R9700 (tensor parallel 2), exact
wire, the benchmark's defaults (temperature 0.7, top-p 0.95, top-k 20, 20 measured passes per category after 3 warm-ups,
a unique nonce per request so the prefix cache is always cold, 48 completed requests per concurrency level, 8 runs per
prefill depth). Decode, latency and concurrency come from the `4bit` mode (MTP depth 3, 98,304-token window); prefill
from the `4bit-200k` mode (MTP off, 200,000-token window).

| Line | This release |
|---|---:|
| Decode, 1 request, weighted over 8 categories (median of 3 runs) | **189.8 tok/s** |
| Prefill, 64K-token prompt (BetterBench 64K row) | **6,991 tok/s** |
| Prefill, exact 64,000-token prompt, cold cache | **7,162 tok/s** |
| Time to first token, short prompts (p50) | **121 ms** |
| Aggregate throughput, 8 concurrent requests | **420.7 tok/s** (48 of 48) |
| Update interval p99 | **17.7 ms** per stream update |

Decode per category (tok/s): chat 171.4, code 187.1, file edit 207.5, JSON 217.5, math 217.2, prose 170.4,
reasoning 170.0, summarization 206.1. Prefill by depth (tok/s): 2K 5,564 · 8K 6,771 · 16K 6,641 · 32K 7,044 ·
64K 6,991 · 128K 6,857; exactly 64,000 tokens with a cold cache: 7,162 tok/s.

**KV cache.** Both modes keep the KV cache in fp8 (E4M3, one scale per token). `--mode 4bit` holds 305,338 tokens
(enough for three requests at the full 98,304-token window, or eight shorter
ones); `--mode 4bit-200k` holds 247,058 tokens (one request at the full 200,000-token window). Both run with prefix
caching off, as measured. For agents and coding tools there is an opt-in cache mode, `--mode 4bit-cache`
(`run-flashnext-4bit-cache.sh`): prefix caching plus an 8 GiB RAM prefix tier. Repeated long prompts reach their first token in about 1 s
instead of 6-16 s (64K tokens: 1.1 s instead of 12.3 s), with the same output from the RAM tier as from a cold start. The cost: about 22 % lower multi-user throughput than `--mode 4bit` (8 concurrent requests 321.4 tok/s; single-stream 146.7) and about 0.1-0.2 s slower first token on new prompts. It needs about 8 GiB more pinned host memory (about 108 GiB of free GTT on the first GPU, in practice a 256 GB host).

MTP note: with speculative decoding the server emits several accepted tokens per stream update, so the update p99 is
the gap between updates, not between tokens.

Peak VRAM (`mem_info_vram_used`, per GPU, on the release image under load): 30.8 / 30.7 GiB in the `4bit` mode, 30.6 / 30.6 GiB in
`4bit-200k`. Pinned host memory (GTT, first GPU): 95.5 GiB and 93.0 GiB.

## How to run

The weights are served by the Paiton container through the
[public Paiton plugin repository](https://github.com/Eliovp-BV/paiton-vllm-plugin), which pins the runtime image and
provides the launcher under `models/Qwen3.8-Flash-Next`. Transformers, stock vLLM and llama.cpp cannot load this format.

```bash
pip install huggingface_hub
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit.sh        # 98,304-token window, speculative decoding on
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-200k.sh   # 200,000-token window, speculative decoding off
bash models/Qwen3.8-Flash-Next/run-flashnext-4bit-cache.sh  # opt-in cache mode (agents, coding tools)
python3 models/Qwen3.8-Flash-Next/launch-flashnext.py --mode 4bit --dry-run   # checks RAM, GTT size, GPU links, disk
```

The first start downloads and verifies the weights (SHA256SUMS). Image: `ghcr.io/eliovp/paiton-vllm-plugin:qwen38-flashnext-rocm10-vllm029-20261011-r3` (digest `sha256:0862ca1c999f12834b6169e84bdf594eb5f8e4db4efbd79a909e32efdf16a8c0`).

## Hardware requirements

| Component | Requirement |
|---|---|
| GPUs | Two AMD Radeon AI PRO R9700 (`gfx1201`, 32 GB each) in one host, ideally with direct PCIe peer-to-peer between them (the launcher warns if the kernel reports no peer link) |
| System RAM | **256 GB as benchmarked; 128 GB and 96 GB supported** (`--mode 4bit`, `4bit-cache`, `4bit-long`). The launcher picks the memory layout from the RAM and the GPU's GTT limit (half the RAM by default) and prints it at start; same output in every layout. 256 GB: the n-gram (PLE) table, the other host tables and the host-resident experts pinned in GTT memory, about 109 GiB of host RAM in use while serving. 128 GB: the PLE table (about 49 GiB) in pageable memory, about 110 GiB in use. 96 GB: the PLE table read through the page cache from two raw files laid out once next to the weights (49 GiB of extra disk, checked before the start), about 61 GiB in use plus page cache. Slower on 128 GB and 96 GB hosts (numbers in the GitHub README). 64 GB hosts are not supported yet (the launcher stops and says why). `--mode 4bit-xl` and `4bit-200k` keep the pinned layout and fall back to ROCm's user-pointer memory with a warning where GTT is short (see Limitations); raising the GTT limit is a system setting for the host's administrator. |
| Disk | 123.9 GiB for this repository; `contract-probe/` (2.2 GiB) is optional; 96 GB hosts: 49 GiB more for the raw PLE files |
| Host | Linux x86-64, ROCm 10 host driver, Docker with `/dev/kfd` and `/dev/dri` access |

## Format

- **Routed experts.** Per layer the 410 most-used experts (by the routing statistics in `common/`) are INT4, levels
  −8..7, one signed bf16 scale per (row, 128 weights) in a block-diagonal Hadamard-128 basis; the 102 least-used are a
  3-bit codebook (CB3E64), the format of our 3-bit release. `expert-format-map.json` lists the format of every expert.
  Expert inputs are quantised to fp8 (E4M3) per token after the online Walsh-Hadamard transform. Calibration: sequential
  GPTQ on 2 million tokens with per-expert down-projection Hessians.
- **Trunk.** W8A16: E4M3 per output row with bf16 activations. Optional INT6 g128 trunk (`trunk-int6/`, applied over the
  W8 trunk by tensor name), used by the `4bit` mode.
- **Output head.** int8 per (row, 128) with a signed bf16 scale and fp32 logits.
- **n-gram (PLE) table.** E4M3 with fp32 row scales in pinned system RAM, never on the GPU.
- **MTP layer, draft heads, vision.** INT4 MTP experts, W8 drafter trunk, int2 draft head; vision encoder bf16 as in the
  checkpoint.
- `FORMAT.md` is the byte-level specification, `QUALITY.md` the accuracy record, `manifest.json` the per-tensor index.

## Limitations

- Even with the default GTT memory the GPUs can pause briefly while the host does heavy file I/O (the kernel compacts memory
  under page-cache pressure, which moves pages the GPUs map). The launcher waits for the host to settle after the start before
  it reports READY; heavy disk activity on the host during serving can still cause short pauses.
- The weights are lossy (tables above). Task benchmarks (math, code, knowledge), multilingual and tool-calling quality
  were not measured for this release.
- About 41 GiB of experts live in system RAM, so speed depends on the host's memory bandwidth and PCIe links; the numbers
  above were measured on a PCIe 4.0 x16 host.
- `--host-pools userptr` (ROCm's default user-pointer host memory) works on smaller hosts but is slower (186.0 tok/s
  decode, 388.9 tok/s at 8 requests in our validation run) and can pause the GPU queues for several seconds when the
  kernel compacts host memory.
- Two R9700 cards only; tied to checkpoint revision `de4b8e4d`.

## License and attribution

The base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is released by Qwen under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE); these weights are a derivative and are distributed under the same license. The
quantisation method builds on GPTQ. Performance measured with [BetterBench](https://github.com/GGZ14/BetterBench) 0.6.0 by
GGZ14.

[Paiton](https://eliovp.com/products/paiton) · [Public Paiton plugin](https://github.com/Eliovp-BV/paiton-vllm-plugin)
