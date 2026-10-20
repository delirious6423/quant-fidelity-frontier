---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
tags:
- liquid
- lfm2.5
- moe
- tenstorrent
- blackhole
- p150
- tt-metal
- vllm
- bfp4
- bfp8
- gptq
- speculative-decoding
---

# LFM2.5-24B-A2B — TT-native mixed BFP4/BFP8 (GPTQ experts), single Tenstorrent P150

A community, experimental build of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
for one **Tenstorrent Blackhole P150** card, served through vLLM with an OpenAI-compatible API.

> **Not an official Liquid AI or Tenstorrent release.** This is a modified derivative of LFM2.5-24B-A2B made by
> Lottolabs. The checkpoint stores Tenstorrent block-floating-point codes (`lfmq-1` format) that only the bundled
> runtime reads: it is **not** GGUF, AWQ, GPTQ-for-CUDA, or a standard Transformers checkpoint, and it does not load
> in stock Transformers, llama.cpp or upstream vLLM. Text only.

## Revisions

| revision | runtime image | what changed | how to get it |
|---|---|---|---|
| **`main` = tag `m12`** (`m12-20261014`) | `lottolabs/lfm2.5-24b-a2b-tt-p150:m12-20261014`, ID `sha256:b598ea4b18f50d467cfb0adaebb672a30aeb485a8dd07d80624c7542ae641450` | m9: decode runs at the width of the active users (1/2/4/8/16), with a rewritten batch-invariant SDPA decode. m10/m11: exact on-device sampling for any temperature/top-k/top-p/min-p/penalties/seed, plus default-sampling presets. m12: prompt-lookup speculative decoding for a single user. | `launch.py` from `main` |
| tag `m8` (`m8-20261013`, commit `4944258f9090bb59aa1732c2912d4997eb4bf553`) | `lottolabs/lfm2.5-24b-a2b-tt-p150:m8-20261013`, ID `sha256:46a994bce04d6e8f8264b824ec8ffd9bd1b34d8eff6f832b9c5af95df8ef72d2` | first release: fused decode/prefill kernels, device sampling for top-k ≤ 64 | `curl -LO …/resolve/m8/launch.py` and run it with `--revision m8` |

The checkpoint is the same in both revisions (manifest SHA256 `cbac0e2e…3817`). Each `launch.py` serves only its own
release. Images m9–m11 were internal steps and are not published separately; m12 contains all of their changes.

## What is in this release (`m12-20261014`)

| path | what |
|---|---|
| `checkpoint/` | quantized weights (`experts-L02..L39`, `layer-L00..L39`, `global` safetensors, 14,543,258,903 bytes), `manifest.json` (per-tensor format/shape/SHA256, per-file SHA256, GPTQ settings), the original `config.json`, `generation_config.json`, tokenizer files and `chat_template.jinja` (unmodified), `LICENSE`, and the reader `lfmq/` (`reader.py`, `codec.py`: torch + safetensors) |
| `runtime-image.tar.gz` | `docker save` of the serving image `lottolabs/lfm2.5-24b-a2b-tt-p150:m12-20261014` (tt-metal + vLLM TT plugin + the LFM2 runtime overlay + three source patches, see `runtime/`) |
| `launch.py` | standard-library launcher: resolve revision → download → verify every byte → load image (ID-checked) → serve |
| `runtime-release.json`, `SHA256SUMS` | size + SHA256 of every file; launch.py pins the release ID, image ID and checkpoint-manifest SHA256 |
| `runtime/lfm25-runtime-src.tar.gz` | the runtime source baked into the image: the `lfm2` model package incl. kernels, vLLM plugin bundle + parsers, Dockerfile and the patches it applies (`plugin-exact-sampling.patch`, `plugin-speculate.patch`, `vllm-lookup-params.patch`), quantization/GPTQ scripts, serve script, design notes |
| `evidence/` | the result files every number below is taken from |
| `NOTICE-original-model-card.md` | Liquid AI's original model card (retained attribution) |

## Precision as served

| weights | format | notes |
|---|---|---|
| routed experts (38 MoE layers × 64 experts × w1/w3/w2; 22.9 B params) | **BFP4** (TT block geometry), **GPTQ** | 4-bit mantissa + shared 8-bit exponent per 16 values |
| attention q/k/v/out, conv in/out projections, dense MLP (layers 0–1), LM head | **BFP8**, round-to-nearest of the original | |
| router, expert_bias, norms, conv kernels, embedding table | BF16 / FP32 (expert_bias) | unchanged from the original |
| KV cache | BFP8 | |
| serving logits | BF16 | as the BF16 reference's lm_head |

Everything (14.5 GB) is resident in the P150's 32 GB DRAM: no host/CPU experts.

## Quality

All quality numbers are teacher-forced against the original BF16 model run on CPU (Transformers 5.12.1, eager
attention, FP32 logits) on the same 38 records / **57,876 scored tokens** (chat 16,758, instruct 1,216, code 23,522,
book 16,380). Measured 2026-10-13 with the m8 runtime, using the same weights as this release. Later images were
gated against m8 instead: greedy output equal to the m8 model-level reference, device sampler equal to vLLM's
host sampler, and speculative output equal to plain decoding (see [Verification](#verification)).

### TT P150 vs CPU BF16

`evidence/quality/tt-m8-vs-cpu-bf16.score.json`

| corpus | tokens | PPL BF16 → TT | ΔPPL | top-1 agreement | KL mean | KL p99 |
|---|---|---|---|---|---|---|
| chat | 16,758 | 2.9590 → 2.9386 | −0.69% | 0.9347 | 0.0260 | 0.229 |
| instruct | 1,216 | 2.8325 → 2.9147 | +2.90% | 0.9424 | 0.0328 | 0.412 |
| code | 23,522 | 3.7181 → 3.7374 | +0.52% | 0.9233 | 0.0466 | 0.530 |
| book | 16,380 | 10.4953 → 10.5780 | +0.79% | 0.8804 | 0.0658 | 0.444 |
| **all** | **57,876** | **4.6416 → 4.6552** | **+0.29%** | **0.9149** | **0.0458** | 0.427 |

Instruct has only 1,216 tokens (dNLL standard error ±0.0097): its ΔPPL is within noise.

### Noise floor and quantization ablations (CPU, same harness)

Fake-quant runs read the same tensors through `lfmq` on CPU. The noise floor is the BF16 model itself, run with
512-token chunked prefill and compared against one whole-sequence pass. Files: `evidence/quality/*.json`.

| run | ΔPPL all | top-1 all | KL all |
|---|---|---|---|
| BF16 noise floor (chunked vs whole) | −0.02% | 0.975 | 0.0045 |
| BF16 experts, BFP8 dense/head only | +0.37% | 0.959 | 0.0114 |
| **this checkpoint (GPTQ-BFP4 experts + BFP8), fake-quant** | +0.57% | 0.915 | **0.0459** |
| RTN-BFP4 experts + BFP8 (no GPTQ) | +1.07% | 0.906 | 0.0550 |
| this checkpoint on the P150 (table above) | +0.29% | 0.915 | 0.0458 |

GPTQ lowers KL by 17% vs round-to-nearest BFP4 experts. The device result matches the CPU fake-quant of the same
weights (KL 0.0458 vs 0.0459).

### GSM8K (first 50 test questions, thinking on, max 4,096 new tokens; `evidence/gsm8k/`)

| run | image | score | notes |
|---|---|---|---|
| CPU BF16 reference (original weights), greedy | – | 49/50 | misses item 12 (pred 12, gold 13) |
| single user, greedy | m8 | 50/50 | item 12 hit the cap after writing `#### 13` |
| 16-slot server, 16 concurrent, greedy | m9 | 50/50 | 1 truncated |
| 16-slot, 16 concurrent, **liquid** preset, seeds 1 / 2 / 3 | m11 | 50/50 · 50/50 · 49/50 | 0 truncated; mean 1,127 / 1,021 / 1,038 new tokens |
| 16-slot, 16 concurrent, **agentic** preset, seeds 1 / 2 / 3 | m11 | 50/50 · 49/50 · 49/50 | 0 truncated; mean 1,176 / 1,272 / 1,233 new tokens |
| single user, **speculation on**, liquid, seed 1 | m12 | 50/50 | 0 truncated, 303 s |
| 16-slot, 16 concurrent, liquid, seed 1 | m12 | 50/50 | 0 truncated |

The three seed-1 liquid runs produced the same 50 texts byte for byte: m11 (16 concurrent), m12 (16 concurrent) and
m12 (single user with speculation). Seeded sampling therefore depends neither on the batch nor on speculation.
Fifty questions is a sanity check, not a benchmark.

## Sampling

The device sampler reproduces vLLM's sampling exactly for any combination of temperature, top-k (including
disabled), top-p, min-p, presence/frequency/repetition penalties and seed. Seeded requests give the same tokens
whatever else is in the batch. The details are in `runtime/…/NOTES-runtime.md` section 17.

| request | decode path |
|---|---|
| temperature 0, no penalties | device argmax |
| temperature 0 + presence / frequency / repetition penalty | exact device sampler |
| temperature ≥ 0.01 (vLLM raises 0 < T < 0.01 to 0.01) with top-k 1–64, top-p, min-p, any penalties, seed | exact device sampler (top 72 candidates per row); the host redraws one token if a top-k tie group may continue past the candidates |
| top-k disabled or > 64, with or without top-p / min-p / penalties | exact device sampler plus the full-vocabulary probability mass; the host redraws one token when the draw or the nucleus/min-p cut falls beyond the candidates |
| different settings per request in one batch | handled per row (a row draws the same token as when run alone) |
| logprobs / prompt_logprobs, logit_bias, allowed_token_ids, bad_words, min_tokens not yet met, structured outputs | the whole step runs on vLLM's host sampler |

Host redraws per 64-token request: 0–1 for the liquid/agentic/reasoning presets, 5 of 53 tokens at T 1, 61 of 64 at
T 2 (decode then drops to 69 tok/s), 63 of 64 for top_k 500 without top_p (107 tok/s)
(`evidence/sampling/m11_matrix_b1_liquid.json`).

**Default sampling presets.** These apply to requests that set no sampling fields; any field a request sets
overrides its default. Choose a preset with `launch.py --sampling-preset`, or pass your own JSON with
`--default-sampling '{"temperature":0.7,"top_p":0.9}'` (keys temperature, top_k, top_p, min_p, repetition_penalty,
max_new_tokens). A field the defaults leave unset takes vLLM's neutral value: top-k disabled, top-p 1, min-p 0, no
repetition penalty, temperature 1.

| preset | default sampling | source |
|---|---|---|
| `liquid` (default) | temperature 0.1, top-k 50, repetition penalty 1.05 | the model's `generation_config.json` |
| `agentic` | temperature 0.6, top-p 0.95 | Liquid's model card, agentic use |
| `reasoning` | temperature 0.6 | Liquid's model card, other evaluations |
| `greedy` | temperature 0 | |

Evidence (`evidence/sampling/`, image m11; the sampler is unchanged in m12):

- **Kernel tests.** 178 real rows and 2 synthetic rows. Every unflagged row keeps exactly the reference's token set.
  Maximum probability error against a float64 reference: liquid 3.8e-6 (from vLLM's float32 penalty arithmetic),
  agentic 7.8e-9, reasoning 1.3e-8, T 1 1.1e-8.
- **Frequencies.** 100,000 device draws per setting, compared with the reference by chi-square: liquid p 0.35,
  agentic 0.27, reasoning 0.50, T 1 0.87, min-p + frequency + repetition 0.91, top-k 100 + top-p 0.9 0.49. A host
  replay of 2,000 draws per setting gave 0 mismatches.
- **Device vs vLLM's host sampler over HTTP.** 250 seeds × 2 prompts × 3 presets, 30 two-sample chi-square tests:
  minimum p 0.11.
- **Seeds.** Seeded requests are reproducible, and 16 concurrent requests equal the same requests run alone.
- **HTTP matrix.** 44 cases pass per preset.

## Speculative decoding (prompt lookup)

With one user (`--max-num-seqs 1`, the default), the server drafts up to 7 tokens per step from n-gram matches in the
request's own prompt and output (n = 4 down to 2) and verifies them in one traced device step. A step that accepts `a`
drafts emits `a + 1` tokens. A policy chooses the draft count from the measured acceptance and the verify cost, and
probes again after misses. The output distribution is unchanged:

- greedy output is token-for-token equal to plain decoding;
- a sampled draft is accepted only when the row's own draw (same Philox uniform per position) equals it, so seeded
  output is identical to the non-speculative path.

It speeds up copy-heavy work (code edits, document QA, rewrites, agent loops) and costs nothing elsewhere. Turn it off
with `--speculate off`; tune it with `--spec-tokens 1-7` and `--lookup-min` / `--lookup-max`. With `--max-num-seqs` > 1
it is always off.

Benchmark (image m12, `evidence/speculation/bench_on.json` / `bench_off.json`): liquid default sampling, max_tokens
1,536, medians over seeds 1–3. Both runs produced the same text for every seed and workload.

| workload | prompt tokens | off tok/s | **on tok/s** | speedup | acceptance | tokens / step | TTFT off / on |
|---|---|---|---|---|---|---|---|
| code_edit (copy) | 3,303 | 167.9 | **230.8** | 1.37× | 0.596 | 1.76 | 0.191 / 0.194 s |
| doc_qa (copy) | 2,741 | 166.1 | **279.2** | 1.68× | 0.688 | 2.38 | 0.127 / 0.129 s |
| rewrite (copy) | 1,730 | 166.7 | **216.0** | 1.30× | 0.472 | 1.65 | 0.176 / 0.177 s |
| agentic (copy) | 2,329 | 163.4 | **233.1** | 1.43× | 0.493 | 2.07 | 0.099 / 0.101 s |
| chat | 32 | 166.6 | 177.0 | 1.06× | 0.423 | 1.13 | 0.065 / 0.067 s |
| gsm8k | 73 | 167.0 | 172.8 | 1.03× | 0.381 | 1.11 | 0.070 / 0.070 s |
| prose | 32 | 166.6 | 169.4 | 1.02× | 0.352 | 1.07 | 0.066 / 0.066 s |

Host load average was 1.0–1.5 for the on run. The off run started at load 10 because of another tenant; an earlier
seed-1 off run at load ~1 gave prose 169.3 tok/s (`try_off3.json`).

Equivalence evidence (`evidence/speculation/`):

- **Greedy.** Speculation on equals speculation off token for token: chat, code and prose 64/64 each; a 3,303-token
  code edit 640/640; a 2,741-token document QA 640/640.
- **Device test.** Eager verify rows are bitwise equal to the sequential 1-row steps (K = 1/2/3/7, including rejected
  drafts), and so is the committed conv state. Through the traced adapter, speculative output equals plain output
  65/65 tokens for each of greedy, liquid, agentic and reasoning (`test_speculate_t2.json`).
- **Sampled, same seeds.** 1,500 requests: the on and off result files are byte-identical
  (`collect_100_on_vs_off.sha256`).
- **Sampled, different seeds.** 30 chi-square tests, minimum p 0.054.

## Speed

Measured over the OpenAI API on 127.0.0.1, 2026-10-13/14. Each row names the image that produced it and its host
load average; a figure is reused from an older image only where later images did not change that path.

### Single user (max-num-seqs 1, max-model-len 131,072)

Plain decode, speculation off, 512 output tokens (ignore_eos), median of 3. Image m12, load 1.1–1.6
(`evidence/speculation/off_http_checks.json`):

| prompt | greedy | greedy + repetition 1.05 | liquid default sampling | TTFT |
|---|---|---|---|---|
| 128 | 169.4 tok/s | 168.3 | 166.6 | 0.061 s |
| 2,048 | **169.1 tok/s** | 167.9 | 166.7 | 0.169 s |
| 32,768 | 148.8 tok/s | – | 146.6 | 3.33 s |

Speculation (the default) adds 2–68% depending on how much the output copies from the context (table above).

Per preset, plain decode. Image m11, load 2–9 from other jobs (`evidence/sampling/m11_bench_b1.json`):

| prompt | greedy | liquid | agentic | reasoning |
|---|---|---|---|---|
| 128 | 172.1 | 167.2 | 166.0 | 164.5 |
| 2,048 | 169.2 | 166.6 | 164.5 | 165.1 |
| 32,768 | 149.3 | 146.8 | 145.2 | 145.2 |

### Time to first token and prefix cache

Prefill is unchanged since m8. Single-user TTFT, median of 3, fresh windows. Image m8, load 1.0–1.5
(`evidence/serving/http_prefill_m8_b1.json`):

| prompt | TTFT | prefill |
|---|---|---|
| 2,048 | 0.170 s | 12.0K tok/s |
| 8,192 | 0.544 s | 15.0K tok/s |
| 32,768 | 3.325 s | 9.9K tok/s |
| 126,976 | 21.83 s | 5.8K tok/s |

Exact prefix caching, image m12 with speculation on (`evidence/speculation/on_http_checks.json`): repeating a
2,112-token prompt cuts TTFT from 0.273 s cold to 0.064–0.072 s; a 32,832-token prompt goes from 3.43 s to 0.101 s.
The resumed output equals the cold output (64/64 greedy tokens).

### Concurrent users (`--max-num-seqs` 2–16)

Since m9, the decode step runs at the smallest width (1/2/4/8/16) that holds the active users, and the SDPA decode
gives each row a partition that depends only on its own position. A single user on a 16-slot server therefore runs as
fast as on a 1-slot server.

Traced device step in ms, model level, n users active at the given position, before (m8) → after (m9)
(`evidence/batch-decode/m8_time.json`, `m9_time.json`; load 0.1–1.5):

| position | n=1 | n=2 | n=4 | n=8 | n=16 |
|---|---|---|---|---|---|
| 0 | 6.22 → 5.29 | 7.68 → 6.32 | 10.04 → 8.11 | 12.85 → 10.79 | 16.24 → 15.03 |
| 16,384 | 25.23 → 5.66 | 26.68 → 7.17 | 28.91 → 9.65 | 31.50 → 13.62 | 35.35 → 21.72 |
| 65,536 | 82.26 → 6.94 | 83.75 → 9.85 | 86.07 → 14.77 | 88.73 → 23.52 | 92.63 → 41.65 |
| 122,880 | 148.82 → 8.39 | 150.28 → 12.75 | 152.66 → 20.50 | 156.06 → 34.46 | 160.21 → 65.53 |

Inference matrix (LLM Inference Benchmark, "localmaxxing" inference matrix): 16-slot server, max-model-len
131,072, 1,048,576-token KV pool, 30 s per cell, max_tokens 8,192 with ignore_eos. Values are aggregate decode tok/s
over C concurrent streams. m9 used liquid defaults (`evidence/matrix/matrix_m9_b16.txt`, `lmx-matrix-m9-b16-liquid.bench.json`);
m11 used the agentic preset, since the bench sends no sampling fields (`matrix_m11_b16_agentic.txt`,
`lmx-matrix-m11-b16-agentic.bench.json`). Each cell reads m9 / m11:

| context | C=1 | C=2 | C=4 | C=8 | C=16 |
|---|---|---|---|---|---|
| 0 | 162.5 / 162.7 | 257.3 / 261.2 | 387.9 / 392.2 | 566.3 / 576.0 | 821.5 / 832.8 |
| 4k | 160.1 / 161.2 | 251.1 / 254.9 | 379.3 / 375.1 | 531.4 / 526.3 | 754.9 / 740.9 |
| 16k | 152.1 / 152.7 | 236.1 / 233.4 | loop flag / 334.7 | 455.7 / 454.9 | 608.1 / 607.5 |
| 32k | 143.1 / loop flag | 213.7 / loop flag | loop flag / 296.7 | 378.7 / 380.0 | 487.2 / 486.9 |
| 64k | 127.2 / 127.4 | 177.9 / 181.3 | loop flag / 238.5 | 283.0 / 284.7 | skipped (KV pool) |
| 120k | 107.4 / loop flag | 141.9 / 141.5 | 178.3 / 177.8 | 202.6 / 202.3 | skipped (KV pool) |

For comparison, the m8 image on the same matrix (`matrix-m8-b16.txt`) gave 102.5 / 37.4 / 6.8 tok/s at C=1 for
contexts 0 / 16k / 120k, and 788.8 tok/s at C=16, context 0.

**About the "loop flag" cells.** The bench's exact-loop guard voids a cell when one stream's text repeats a fixed
period. Each run had three flagged cells, in different places. In every flagged stream, the prompt asked for a
mathematics article from an architecture document; the model refused and ended its answer. The bench then forced
generation past EOS (ignore_eos, 8,192 tokens), and the model restarted the same refusal again and again (periods of
357–1,473 characters, starting at character 1,856–7,393). This comes from forcing generation past the end, not from
the sampler: the device and host sampling distributions match (above), and the flags move between cells from run to
run.

Per-user and aggregate decode with 2K prompts, 512 tokens per stream. Image m11 on a 16-slot server, load 4–7
(`evidence/sampling/m11_bench_b16_u4.json`, `_u16.json`):

| users | greedy | liquid | agentic | reasoning |
|---|---|---|---|---|
| 4 | 97.9 per user (348 aggregate) | 98.5 (348) | 96.2 (342) | 96.7 (343) |
| 16 | 49.3 (629) | 49.2 (626) | 47.7 (611) | 48.5 (618) |

Concurrent TTFT with 2K prompts on a 16-slot server (image m8, `evidence/serving/http_prefill_dev-m8_b16.json`): 4 users
mean / max 0.544 / 0.671 s; 16 users 2.487 / 2.651 s. Prompts are prefilled one at a time.

## Fused kernels

The single-user decode step is one traced program sequence: 5.47 ms/token on device at position 2,000, against
17.5 ms for the stock ttnn step (`runtime/…/NOTES-fast-kernels.md`).

- **MoE decode** (`MoEDecodeV2` / `MoEDecodeRows`): one launch runs the router (sigmoid + expert-bias top-4,
  normalized) and the four experts' BFP4 gate/up + SwiGLU + down, streamed straight from DRAM. The multi-row version
  streams the union of the rows' experts.
- **Residual add + RMSNorm fused into the following matmul**: operator norm + conv in_proj or attention qkv; ffn norm
  + dense w1|w3. Layer 0 also folds in the embedding gather.
- **Gated short-conv op** in one program: in_proj → B·x → 3-tap causal conv with state shift → C· → out_proj.
- **Attention front**: qkv + per-head q/k RMSNorm + RoPE + paged BFP8 KV write.
- **SDPA decode** (rewritten in m9): 1–16 rows. Each row's partitions depend only on its own position, which makes it
  batch-invariant. All live rows' partitions are spread over every core, idle rows cost nothing, and it covers the
  full 131K table.
- **DRAM-streaming BFP8 decode matmuls** for the projections and the 128K-vocab LM head.
- **Decode widths** 1/2/4/8/16, each with its own captured traces (argmax, logits, exact sampler).
- **Sampler** (m10/m11): a per-worker exact top-72 scan plus the full-vocabulary mass in fixed point, a row-maximum
  pre-pass, then a per-row exact merge using vLLM's arithmetic and a Philox draw.
- **Speculative verify** (m12): K + 1 rows of one sequence in one traced step (widths 2/4/8). Conv and attention
  verify modes keep the state untouched, and a commit kernel then updates the conv state, the sampler history and
  the accepted count.
- **Prefill**: an all-core MoE routing/prefix kernel, grouped expert matmuls that stream each expert block once per
  chunk, a single-program conv prefill, 2D-multicast matmul configs, and HiFi4 SDPA.

## Chat template, parsers

- The model always reasons: the chat template's generation prompt ends with `<think>`. The `lfm2` reasoning parser
  splits `reasoning_content` (before `</think>`) from `content`.
- Tool calls: the model emits pythonic calls between `<|tool_call_start|>` and `<|tool_call_end|>`
  (e.g. `[get_weather(city="Paris", unit="celsius")]`); the `lfm2` tool parser returns OpenAI `tool_calls` (JSON
  arguments, `finish_reason: tool_calls`), streamed and non-streamed. Send tools in the standard `tools` field.
- Structured outputs apply after the reasoning ends; a prompt still inside `<think>` is not constrained.
- For bit-exact greedy output pass `temperature: 0` and `repetition_penalty: 1.0`.

## Requirements

- One Tenstorrent **Blackhole P150** (32 GB), tt-kmd driver, `/dev/tenstorrent`, 1 GiB hugepages
  (`/dev/hugepages-1G`) as for any tt-metal workload; the card must be idle and owned by you.
- Docker, Python ≥ 3.11 (standard library only), ~42 GB disk for the download (14.5 GB checkpoint + 5.5 GB archive,
  plus ~22 GB once the image is loaded), ~15 GB for the converted-tensor cache, ~1 GB of kernel cache.
- The first start converts the weights, compiles kernels and captures the decode, sampler and speculative-verify
  traces: 8 min on the release host. Later starts reuse both caches: 51 s from launch to ready, including the
  launcher's checksum pass.

## Download and serve

```bash
curl -LO https://huggingface.co/Lottolabs/LFM2.5-24B-A2B-TT-Mixed-BFP4-BFP8-P150/resolve/main/launch.py

# Download and verify only (no Docker or device needed)
python3 launch.py --download-only --cache-root /large-disk/lfm25

# One user, 131,072-token context, prompt-lookup speculation and prefix caching on, liquid default sampling
python3 launch.py --cache-root /large-disk/lfm25 --device-ownership-confirmed

# 16 concurrent users, 131,072 tokens each, sharing a 1,048,576-token KV pool; also 2 / 4 / 8
python3 launch.py --cache-root /large-disk/lfm25 --max-num-seqs 16 --device-ownership-confirmed

# Agentic defaults, or your own
python3 launch.py --cache-root /large-disk/lfm25 --sampling-preset agentic --device-ownership-confirmed
python3 launch.py --cache-root /large-disk/lfm25 --default-sampling '{"temperature":0.7,"top_p":0.9}' --device-ownership-confirmed

# Print the exact docker command. Other options: --speculate off, --spec-tokens N, --lookup-min/--lookup-max N,
# --max-model-len N, --kv-pool-tokens N, --prefix-caching off, --port P, --revision REV, --release-dir DIR
python3 launch.py --cache-root /large-disk/lfm25 --print-command
```

The KV pool defaults to max-model-len × max-num-seqs and is capped at 1,048,576 tokens, the largest pool validated
(16-slot server, 131,072 per sequence). `--release-dir DIR` serves an already downloaded release directory in place;
every byte is still verified. The server listens on `http://127.0.0.1:8000/v1`, model id `LiquidAI/LFM2.5-24B-A2B`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "LiquidAI/LFM2.5-24B-A2B", "max_tokens": 1024,
  "messages": [{"role": "user", "content": "What is 17 * 23?"}]}'
```

Stop the server with Ctrl-C or `docker stop lfm25-24b-a2b-api`. The launcher never stops containers, resets the card
or takes the device from other processes.

**Cache safety.** The tensor cache is bound to the checkpoint-manifest SHA256 and the image ID through
`lfm25-cache-identity.json`. A cache built for anything else, or a non-empty directory without that file, is refused.
Caches from m8 are therefore not reused by m12.

## Verification

Gates passed by the m12 image before release (`evidence/speculation/`, `evidence/batch-decode/`,
`evidence/sampling/`, `evidence/gates/`):

- **Adapter.** Traced decode equals eager decode (logits identical); the exact sampler equals a host replica.
- **Batch.** Batch-vs-single with the golden chat prompt in the last slot: 64/64 at B = 2/4/8/16, both against the
  B = 1 run and against m8's model-level greedy.
- **1-slot server.** HTTP greedy equals the model-level greedy (64/64); reasoning and tool parsing OK; prefix resume
  equals cold prefill (64/64 at 2,112 and 32,832 tokens).
- **Sampling.** The 44-case HTTP matrix and the seed tests pass for each preset, and the speculation equivalence tests
  above pass.
- **16-slot server** (agentic): HTTP greedy 64/64; matrix and 16-way seed tests pass; GSM8K-50 50/50.
- **Checkpoint.** 771 tensors were re-checked against the original, and all 84 file hashes match.

Release check (2026-10-14, `evidence/release-verification/m12-local-staging-serve.json`, load 0.7–2.7):

- `launch.py` verified every file of the staged release.
- Its Docker command is argument-for-argument the reference `serve_lfm2.sh` in 6 configurations: default; speculation
  off + agentic; custom JSON + spec tuning + prefix caching off; 4 sequences with speculation requested (ignored);
  16 sequences with a 1,048,576-token pool; 8 sequences with custom JSON.
- The server it started with empty caches (defaults: one user, liquid, speculation on) answered:
  - greedy chat "391" (17 × 23) with the reasoning split off;
  - a default-sampling chat;
  - the tool call `get_weather({"city": "Paris", "unit": "celsius"})` with `finish_reason: tool_calls`;
  - GSM8K first 10: 10/10;
  - decode for a 2,019-token chat prompt asking for an original essay (greedy, 512 tokens): 178.8 tok/s (median of 3),
    TTFT 0.170 s.

A raw greedy continuation of a book window with ignore_eos degenerates into repeated text, which prompt lookup copies
(340.7 tok/s), so it is recorded but not used as a decode figure.

Published-file check (`evidence/release-verification/m12-hub-download-serve.json`):

- `launch.py` fetched from this repository downloaded commit `cebd423356a7c4bd8a2b5379987970ef894798d9` anonymously
  into an empty directory: 195 files, 20,073,798,951 bytes, every SHA256 matched; 42 min at ~8 MB/s.
- Served from that download (load 0.3–1.0), it answered:
  - greedy chat "391";
  - the tool call `get_weather` (`finish_reason: tool_calls`);
  - GSM8K first 10: 10/10;
  - 2K chat decode 178.9 tok/s, TTFT 0.17 s.

Later commits only add this record to the card and `evidence/`.

## Limitations

- Text only. Single P150 only (no multi-card / tensor parallel). Max context 131,072 tokens per sequence.
- Quantized: expect small differences from BF16 (table above). Long greedy reasoning traces amplify a single flipped
  near-tie, so outputs can differ from the BF16 text even when the answers agree.
- Greedy output can differ from a model-level eager reference at exact BF16 near-ties: a top-1/top-2 margin of 0 or
  1 BF16 step, which the eager-prefill and serving paths round differently. Serving with speculation on and off gives
  the same tokens.
- Speculative decoding works for one user only. Prefill is not chunked across requests: one prompt at a time, so a
  long prompt delays other users' first tokens. Prefix caching resumes at 2,048-token boundaries.
- Sampling settings the device cannot resolve fall back to vLLM's host sampler for that token or step, and are
  slower. Examples: T ≥ 2 with top-k disabled, top-k > 64 without top-p, logprobs, logit_bias, structured outputs.
- Experimental software on a shared research host. The numbers come from one P150 on 2026-10-13/14 and vary with host
  load.

## License and attribution

The model weights are a derivative of **LFM2.5-24B-A2B by Liquid AI**, licensed under the **LFM Open License v1.0**
(full text: [`LICENSE`](LICENSE), also in `checkpoint/LICENSE`). Liquid AI's original model card is retained as
`NOTICE-original-model-card.md`.

- **Commercial use limit (Section 5):** commercial use is licensed only for entities with annual revenue below
  **US $10,000,000** (the "Threshold"); commercial use by an entity at or above the Threshold is not licensed (the
  Threshold does not apply to qualified non-profits for non-commercial or research purposes). Read the License before
  using this release commercially.
- **Modifications notice (Section 4(b)):** Lottolabs modified the model weights:
  - routed-expert weights were quantized with GPTQ to Tenstorrent BFP4;
  - attention, conv, dense-MLP and LM-head weights were rounded to BFP8;
  - all tensors were re-packed into the `lfmq-1` safetensors layout (`checkpoint/experts-*`, `checkpoint/layer-*`,
    `checkpoint/global.safetensors`, `checkpoint/manifest.json`).

  `config.json`, `generation_config.json`, the tokenizer files and the chat template are unmodified copies.
- The runtime image bundles tt-metal, vLLM and the vLLM TT plugin under their own licenses (inside the image). The
  LFM2 runtime overlay and the patches applied to vLLM and the plugin are in `runtime/lfm25-runtime-src.tar.gz`.
- "LFM" and "Liquid" are Liquid AI's names, used here only to describe the origin of the work.

## Provenance

- Base model: `LiquidAI/LFM2.5-24B-A2B`, revision `1769516ac588c4c558089c94f27d96b0dc67b183`.
- GPTQ calibration: 128 × 2,048 tokens (seed 1) plus 8 held-out sequences, rendered with the LFM chat template:
  - 48 UltraChat `train_sft` chats;
  - 32 Python source files (site-packages modules);
  - 16 Dolly and 16 Alpaca instruction chats;
  - 16 Project Gutenberg prose chunks.

  The calibration set was checked disjoint from the evaluation texts (no shared 32-token window).
- GPTQ: per-expert, routing-weighted Hessians (w2 weighting, 128-token shrinkage toward the pooled routed Hessian),
  act-order, block 128, gate/up damping 0.01, down damping 1.0; 0 RTN fallbacks. Settings and per-layer metric hashes
  are in `checkpoint/manifest.json`.
- Evaluation texts: chat (UltraChat test), instruct (code/multilingual), code (Python files) and book (Opticks)
  records. GSM8K: the first 50 questions of the test split.
- Runtime image: built FROM a tt-metal + vLLM (TT plugin) P150 image, with the overlay and patches in `runtime/`.
