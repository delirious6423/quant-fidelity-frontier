---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- vllm
- deepseek
- deepseek-v4.1-flash
- deepseek-v4.1-flash-1017
- mixture-of-experts
- 3-bit
- rtx-pro-6000
- blackwell
- sm120
---

# DeepSeek-V4.1-Flash-1017 EXL3 3-bit for 2× RTX PRO 6000 Blackwell

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), ready to serve on two RTX PRO 6000
Blackwell cards (sm_120, 96 GB each) with vLLM at tensor-parallel 2. The routed experts are our own calibrated EXL3 3-bit
quantization of the 1017 weights, made with coolbho3k's method. The repo also has:

- int4 Engram tables;
- a 3-bit DSpark drafter;
- a serving stack that keeps a third of the routed experts (the ones agentic coding uses least) in pinned host RAM, with
  a VRAM cache for the host experts that recent decode steps used most;
- optionally, a CPU runtime that computes those host experts on a large AVX-512 host CPU instead of reading them over PCIe;
- custom MoE kernels for sm_120, including a decode-once prefill kernel with int8 tensor-core math;
- an 8-bit all-reduce for prefill chunks between the two cards;
- vLLM patches that render and tokenize chat prompts and parse tool calls the way the official DeepSeek API does, and that
  skip decoder-replay work that no later token reads.

The `recipe/` folder has the image build, the sm_120 patches, the kernels, the CPU runtime, and the serving, benchmark, KL
and quantization scripts.

> **This repo holds DeepSeek-V4.1-Flash-1017** since revision `bdc01239` (2026-10-20). Before that it held
> DeepSeek-V4.1-Flash with coolbho3k's 3.0 bpw experts (revision `6062be5f`), and before that a 2.0 bpw pack (revision
> `28b7ab71`); both are still in the repo history. On the same two cards and the same agentic benchmark, one session at a
> time, the default configuration of this release decodes 28% faster than the V4.1-Flash release (224 vs
> 176 tok/s) and replays the 13 sessions in 229 s instead of 283 s. With the optional CPU runtime it decodes
> 49% faster (262 tok/s, 209 s).

## What changed in this update

This update changes only the serving stack; the weights are the same as in revision `bdc01239`. Rebuild the image and the
kernels (`docker/build.sh`, `kernels/build.sh`). `serve/serve.sh` prints a warning when the image predates the new vLLM
patches.

- **Prompt encoding and tool-call parsing now follow the official DeepSeek API. This is a correctness fix.**
  - In the previous release, the tool-call parser accepted a plain `<parameter name="x">` in a reply or in its reasoning as a
    tool-call opener. This came from vLLM PR #61142, which tolerates drifted DSML openers. A reply that only showed XML
    became a call to a tool named `x`, and the rest of the reply was lost. Tool calls now count only where the model
    emitted the DSML token, as in the official API's output parser (`VLLM_DS41_DSML_STRICT`). #61142's drifted openers are
    still accepted after a real DSML token.
  - Client text that spelled a control token (`<｜User｜>`, `</think>`, the end-of-sentence token) used to be tokenized as
    that control token. Control tokens now come only from the chat template, and all message text is encoded as plain
    text, as the official API does (`VLLM_DS41_SEGMENT_ENCODE`). Rendering and tokenizing a long conversation is also
    about 3-7× faster.
  - `reasoning_effort` `minimal`, `medium` and `ultra` returned HTTP 400. They now map to `low`, `high` and `max`.
  - Several other requests now render as the API renders them:
    - `tool_choice: "none"` no longer renders the tools;
    - `response_format` (`json_object` or `json_schema`) is now rendered;
    - developer messages are no longer dropped in thinking mode;
    - tool schemas render only name, description and parameters;
    - tools go to the leading system message;
    - a system-only prompt gets the assistant header.

  Each change has its own `VLLM_DS41_*` switch. All are on by default; `=0` restores the previous behavior
  (`recipe/README.md`).
- **8-bit prefill all-reduce** (`DS41_FP8AR=1`). The all-reduces between the two cards for prefill chunks of 256 rows or
  more send int8 values with a bf16 scale per 32 elements. Decode keeps the exact all-reduce.
- **Int8 decode-once prefill kernel** (`DO_INT8=1`). The VRAM experts of each prefill chunk run on int8 tensor cores
  instead of fp16. `DO_INT8=0` restores the fp16 kernel.
- **Exact decoder-replay trim** (`VLLM_DS41_REPLAY_FINAL_WINDOW=1`). Each prefill chunk replays its last 128 rows through
  the sliding-window layers. For chunks far from the end of the prompt, no later token reads that replay, so those chunks
  now keep a single row. Every row that is read is computed exactly as before.

The 8-bit all-reduce and the int8 kernel are lossy. In same-day KL pairs each stayed within run-to-run noise of the exact
version (see Fidelity). All four together, on our development build with the CPU runtime, against a same-day control:

- 1-session agentic wall clock: 230.8 → 202.9 s (-12%);
- 4-session agentic wall clock: 182.5 → 163.4 s (-10%);
- fresh 40K-token prefill: +14-17%;
- decode speed: unchanged;
- KL: 0.0520 / 0.0309 / 0.0112 and 0.0339 on short100, against 0.0516 / 0.0307 / 0.0111 and 0.0346 for the control (at most +0.9% on any corpus).

## Fidelity

KL to the original DeepSeek-V4.1-Flash-1017, measured two ways (`recipe/kl/`):

- **AtomicChat's corpora, our 1017 reference.** AtomicChat's V4.1-Flash protocol: three corpora (neutral text, code, agentic
  transcripts), 24 windows of 4,096 tokens each, the second half of each window scored (49,152 positions per corpus), top-512
  logprobs. There is no published 1017 reference for it, so we ran the original 1017 checkpoint ourselves (vLLM, exact FP4
  expert weights) twice: once as the reference, once as the noise floor. The neutral corpus is multilingual (about half Latin
  script, the rest spread over some 20 others); the code and agentic corpora are mostly English.
- **kyuz0's short100 panel.** An independent reference: full-vocabulary FP32 logits of the original 1017 from DeepSeek's
  reference code ([kyuz0/deepseek-v4.1-flash-1017-qa](https://huggingface.co/datasets/kyuz0/deepseek-v4.1-flash-1017-qa)),
  100 cases, 6,159 scored positions.

Each cell shows the mean KL lower bound in nats, then top-1 agreement with the original. Lower KL is better.

| Model | Neutral text | Code | Agentic | short100 |
|---|---:|---:|---:|---:|
| Original 1017, second run (noise floor) | 0.0082 · 97.1% | 0.0050 · 98.2% | 0.0047 · 98.8% | — |
| **This pack, as served (default)** | **0.0522 · 92.8%** | **0.0306 · 96.0%** | **0.0111 · 98.2%** | **0.0338 · 95.4%** |
| This pack, as served with the CPU runtime (`CPU_EXPERTS=1`) | 0.0522 · 92.8% | 0.0306 · 96.0% | 0.0109 · 98.2% | 0.0348 · 95.5% |
| The first 1017 release (`bdc01239`), as served | 0.0522 · 92.8% | 0.0308 · 96.0% | 0.0112 · 98.3% | 0.0346 · 95.1% |
| The first 1017 release, CPU runtime | 0.0521 · 92.9% | 0.0310 · 95.9% | 0.0107 · 98.2% | 0.0339 · 95.2% |
| This pack on the previous release's serving stack | 0.0518 · 92.9% | 0.0309 · 95.9% | 0.0109 · 98.3% | 0.0336 · 95.5% |

"As served" means `recipe/serve/serve.sh` with its defaults (GPU-only host experts, the 8-bit prefill all-reduce, the int8
decode-once kernel, FP8 lm_head, the VRAM cache, CUDA graphs and every VRAM trim), with speculative decoding off, an 8K
context window and no prefix cache for the logprob capture. Our gate for a serving change is a same-day KL pair on our
development build, with every corpus within +3%. Each new change passed it:

| Change | Neutral | Code | Agentic | short100 |
|---|---:|---:|---:|---:|
| 8-bit prefill all-reduce | 0.0517 → 0.0519 | 0.0309 → 0.0309 | 0.0110 → 0.0109 | 0.0345 → 0.0341 |
| Int8 decode-once kernel | 0.0513 → 0.0519 | 0.0309 → 0.0311 | 0.0111 → 0.0110 | 0.0348 → 0.0344 |
| All four changes | 0.0516 → 0.0520 | 0.0307 → 0.0309 | 0.0111 → 0.0112 | 0.0346 → 0.0339 |

The KL capture sends token IDs and asks for prompt logprobs, so it does not go through the prompt encoder. Requests with
prompt logprobs also keep every replay row. The replay trim was therefore checked separately, for identical outputs and
teacher-forced agreement (`recipe/results/RESULTS.md`). An FP8 (e4m3) version of the int8 kernel failed the gate (+5.1% on
neutral text) and is not shipped.
The previous release's pack scored 0.0545 / 0.0321 / 0.0101 against the original V4.1-Flash on AtomicChat's own reference.
Those numbers are not directly comparable: a different model, and a reference with a higher noise floor (0.0158 / 0.0100 /
0.0055), measured on a different engine.

Quality probes with thinking off and greedy decoding:

| | GSM8K-200 | HumanEval |
|---|---:|---:|
| This pack, default configuration | 98.5% | 97.0% |
| The first 1017 release (default; same with the CPU runtime) | 98.5% | 97.0% |
| Previous release (V4.1-Flash) | 98.0% | 95.7% |

Text and image input both work. The model's own vision encoder is loaded by default, and images go through the OpenAI
`image_url` content type (`VISION=0` serves text only). Prompts are rendered exactly like DeepSeek's 1017 reference encoder:
vLLM PR #62196 gave identical prompts, or the same error for invalid input, on all 40 cases we checked. They are also
tokenized the way the official API tokenizes them (see above). A needle in a fresh 1.04M-token prompt is found at the
default 1M context window (110.4 s of prefill).

## Speed

Test setup:

- Two RTX PRO 6000 Blackwell Max-Q at their 300 W cap, on PCIe (no NVLink), tensor-parallel 2.
- Host: AMD Threadripper PRO 9995WX (96 cores), 8-channel DDR5-4800. The host CPU matters only for the CPU runtime.
- A 1M context window with a 4.25 GiB fp8 KV cache (1.07M tokens).
- CUDA graphs and DSpark speculative decoding with 5 draft tokens.
- A ~2.5 GB/GPU side process stayed resident, and no other clients were connected.
- The server was `recipe/serve/serve.sh` with its defaults, serving the files in this repo, built from `recipe/` exactly
  as in the quick start below:
  - GPU-only host experts;
  - the 8-bit prefill all-reduce;
  - the int8 decode-once kernel;
  - the replay trim;
  - API prompt encoding.
- "CPU runtime" rows add `CPU_EXPERTS=1`.
- "First 1017 release" rows are the same test of revision `bdc01239`.
- "Development build" rows are same-day A/Bs of the same code on our development paths, with the CPU runtime, run before
  the recipe was assembled.

Throughput is in tokens per second.

**Agentic coding sessions.** 13 held-out multi-turn agentic coding sessions with tool definitions and tool calls in the
history (104 turns, 8.9M prompt tokens in total), replayed turn by turn with streaming, temperature 0 and up to 1,024 output
tokens per turn, each configuration on a fresh server. We don't publish the session set.

| Configuration | 1 session: decode | wall clock | time to first token (median / mean) | 4 sessions: decode per stream | aggregate decode | wall clock |
|---|---:|---:|---:|---:|---:|---:|
| **This release, default** | **224.1** | **229 s** | **0.50 s / 1.35 s** | **41.2** | **110.4** | **179 s** |
| **This release, CPU runtime** | **261.8** | **209 s** | **0.47 s / 1.28 s** | **50.6** | **127.8** | **161 s** |
| First 1017 release, default | 223.5 | 255 s | 0.63 s / 1.63 s | 42.9 | 106.6 | 202 s |
| First 1017 release, CPU runtime | 261.8 | 235 s | 0.57 s / 1.54 s | 45.7 | 107.9 | 178 s |
| All four changes, CPU runtime (development build) | 273.4 | 203 s | 0.46 s / 1.29 s | 48.7 | 125.9 | 163 s |
| None of them, CPU runtime (development build, same day) | 270.6 | 231 s | 0.57 s / 1.53 s | 48.5 | 110.2 | 183 s |
| Previous release, same harness | 175.7 | 283 s | 0.57 s / 1.53 s | 39.3 | 105.2 | 214 s |

Decode is per stream, weighted by tokens. Aggregate decode counts the tokens of all streams over the time any stream was
decoding (2.3-2.6 of the 4 streams on average). The 1-session numbers repeat within about 1%. The 4-session numbers vary by
up to ~15% between boots. Agentic replies are mostly tool calls, which DSpark predicts well.

The four changes speed up prefill and time to first token; decode speed stays the same. Each change on the development
build, against a same-day control without it:

- **8-bit all-reduce:** 4-session wall clock -11% (197 → 175 s), 1-session -3%.
- **Int8 kernel:** 4-session wall clock -8% (186 → 171 s), 1-session -3% (231 → 224 s), fresh 40K-token prefill +6%.
- **Replay trim:** fresh multi-chunk prefill 3.7-4.6% faster (34K-130K tokens); agentic wall clock within noise.
- **New encoder:** renders and tokenizes the 104 session prompts in 3.8-4.8 s instead of 12-14 s (four CPU cores). Clients
  wait for this before their first token.

**Fixed prompts** (`recipe/bench/bench-fixed.py`: code prompts, thinking off, the same tokens and speculative acceptance on
every run; decode speed is the difference between a 400-token and a 16-token reply on a warm prefix cache):

| Benchmark | This release, default | This release, CPU runtime | First 1017 release, default | First 1017 release, CPU runtime | All four changes, CPU runtime (development build) | Previous release |
|---|---:|---:|---:|---:|---:|---:|
| Prefill, fresh 40K-token prompts (two prompts) | 11,157-11,467 | 11,624-12,182 | 9,347-9,794 | 10,372-10,561 | 11,853-12,278 | 9,031-9,125 |
| Decode, 2K prompt, 1 stream | 156.9 | 179.0 | 158.0 | 191.2 | 192.4 | 129.7 |
| Decode, 46K prompt, 1 stream (median of 8) | 172.8 | 210.1 | 169.0 | 202.8 | 196.7 | 114.4 |
| Decode, 46K prompts, 4 streams (aggregate) | 362.9 | 406.2 | 361.9 | 405.5 | 412.1 | 215.9 |

The same-day development control without the four changes gave 10,384-10,510 / 205.8 / 202.2 / 420.2. The 46K
single-stream decode medians vary by about 5% between runs.

The first large prefill after a fresh install compiles about 60 kernels once and caches them in the recipe's cache
directory: on a cold cache the first 40K-token prompt prefilled at 7,700-8,000 tok/s, and every later prompt runs at the
speeds above. The default column was measured on a warm cache (the fp16 decode-once kernel on the same cache: 10,389-10,536).

**Benchmarks from earlier releases** (`recipe/bench/`, CPU runtime, 1M context; measured on the first 1017 release's
development build, before this update; the previous release's numbers are from its 2026-10-15 validation, also at 1M):

| Benchmark | First 1017 release | Previous release |
|---|---|---|
| 46K synthetic prompt, 1 stream (`bench-sbs.py`) | 10,179 tok/s prefill (4.9 s to first token), 95.4 tok/s decode | 10,451-10,486 (4.8 s), 63.0-69.4 |
| Agent turn: 34K cached + 6K new prompt tokens, 300 output tokens, 1 stream (`hit_bench.py`) | 0.76 s to first token, 253 tok/s decode | 0.91 s, 114 tok/s |
| Same, 4 concurrent sessions | 2.36 s to first token (median), 421 tok/s aggregate | 2.64 s, 216 tok/s |
| Fresh 1.04M-token prompt with a needle | found, 122.9 s (8,463 tok/s) | found, 120.5 s |

With the CPU runtime, fresh prefill on the synthetic 46K prompt was about 3% slower than in the previous release, while the
40K code prompts above prefilled 14-15% faster. This plan likely keeps more experts in host RAM, and the synthetic prompt
routes to them more often. At its peak (the 1.04M-token prompt), nvidia-smi showed 96,655 and 97,029 MiB of 97,887 MiB in
use (previous release: 95,771 and 96,279), including the side process and the vision encoder. With this release's default
configuration, the same 1.04M-token needle was found in 110.4 s, at a peak of 96,221 and 96,595 MiB. That includes the
8-bit all-reduce's 85 MiB receive buffer; the int8 kernel and the replay trim use no extra VRAM.

**Power** (measured on the first 1017 release). Estimated box power while requests ran (CPU package from RAPL, both GPUs,
plus 100 W for the rest): about 900 W with the CPU runtime and about 815 W on the GPU-only path. The GPUs drew 576-593 W
together either way; the CPU package went from about 137 W to 221 W. In the same-day development A/B the four changes moved
the 1-session estimate from 898 to 912 W.

## How it fits in 2× 96 GB

At 3 bits the routed experts alone take about 204 GB, more than the two cards hold. The stack keeps the rest of the model
in VRAM and moves the coldest data to pinned host RAM:

- **Cold experts in host RAM.** 5,296 of 15,360 routed experts stay in pinned host RAM: the ones that long multi-turn agentic
  coding sessions use least (`recipe/plans/`). All of the drafter's experts stay in VRAM.
- **A VRAM cache for them.** 512 slots per GPU hold copies of the host experts that recent decode steps used most; once warm
  it served 50-80% of the cold decode routings on the fixed-prompt bench and 26-37% in the agentic sessions.
- **Two ways to run the rest.** By default a second stream with 32 SMs (64 during prefill) runs the cold experts, reading
  their weights over PCIe while the VRAM experts run on the main stream. With `CPU_EXPERTS=1`, a pool of 42 CPU cores per GPU
  computes them next to their weights instead (AVX-512 VNNI kernels for the EXL3 3-bit codebook, inside the CUDA graphs), so
  only activations and results cross PCIe. The CPU runtime needs a large AVX-512 CPU with high memory bandwidth; on another
  host, `recipe/cpuexp/check_host.sh` tells you whether it can run, and `serve.sh` falls back to the GPU-only path if not.
- **Engram and embeddings in host RAM.** The int4 Engram tables and the embedding table are pinned in host RAM and read
  over UVA, inside CUDA graphs.
- **VRAM trims, about 9 GiB per GPU in total**, mostly spent on keeping more experts in VRAM: one FP8 copy of each large dense
  layer instead of FP8 plus a Marlin repack (2.7 GiB), a smaller indexer prefill workspace (2.0 GiB) and decode block table
  (1.5 GiB), 3-bit EXL3 drafter experts instead of MXFP4 (1.0 GiB), attention `wo_a` held in FP8, re-quantized exactly
  (0.65 GiB), the embedding table in host RAM (0.62 GiB), an FP8 lm_head (0.33 GiB), and a rope table sized to the context
  window (0.25 GiB).
- **Custom kernels.**
  - An MoE kernel rewritten for sm_120 (`recipe/kernels/xmoe`). It has a flat decode scheduler with a hand-tuned inner
    loop, and a prefill path that reads each host expert over PCIe once per chunk instead of once per 32 rows.
  - A decode-once prefill kernel for the VRAM experts (`recipe/kernels/xmoe_do`).
  - Its int8 tensor-core build (`recipe/kernels/xmoe_do_i8`, the default). It is restructured around persistent blocks,
    one barrier per K slab, and decode work interleaved with the MMAs. Activations are int8 with one scale per row and
    128-wide slab, and the dequantization per slab is exact.
  - Split-K FP8 GEMV kernels for the dense layers at decode sizes, and fused decode kernels that cut the launches per step.
- **8-bit prefill all-reduce** (`recipe/kernels/fp8ar`, `DS41_FP8AR=1`). The tensor-parallel all-reduces of prefill chunks
  (256 rows or more) send int8 values with a bf16 scale per 32 elements instead of bf16: each card pushes its partial straight
  into the other card's receive buffer over PCIe P2P, and both cards add the same bytes, so their results stay bit-identical.
  A 16K-token all-reduce takes 1.8 ms instead of 4.0 ms. Decode keeps its exact all-reduce inside the CUDA graphs, and the
  embedding all-reduce stays exact.
- **Replay trim** (`recipe/patches/vllm/v41-replay-final-window.diff`). Layers 20-39 hold only 128-token sliding-window KV,
  and for each prefill chunk vLLM replays the chunk's last 128 rows through them. Only the end of the prompt and the decode
  steps read those rows. A chunk that ends more than 128 tokens before the end of its prompt now keeps one placeholder row:
  on a 34K-token prompt (chunks of 16K, 16K and 1.2K tokens), the first two chunks replay 1 row each instead of 128.

The serving container is capped at 260 GiB of host RAM, most of it pinned.

## What's in this repo

| Files | Contents |
|---|---|
| `model-00003` to `model-00042` | Layers 0-39. Routed experts are our calibrated EXL3 3-bit quantization (MUL1 codebook); the other tensors are DeepSeek's FP8, unchanged. |
| `model-00044` to `model-00046` | The DSpark draft layers. We re-encoded their routed experts from MXFP4 to EXL3 3-bit (MCG codebook, uncalibrated); the other tensors are DeepSeek's. |
| `model-00047`, `model-00048` | The Engram tables of layers 1 and 14 in int4 with fp16 per-32 scales: 111 GB, down from DeepSeek's 203 GB of FP8. |
| `model-00001`, `model-00002`, `model-00043` | DeepSeek's shards, unchanged: the vision tower, the embedding and image tokens, the final norm and lm_head. |
| `config.json` | DeepSeek's 1017 config with an EXL3 `quantization_config` (`bits` 3, `codebook` `mul1`, `mtp_experts` `exl3`, `engram_embed_format` `int4_fp16s32`, the FP8 settings of the non-routed weights). |
| `recipe/` | Image build, patches, kernels, CPU runtime, host-expert plans, and serving, benchmark, KL and quantization scripts. |

The source is `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e10`, and the shard layout follows it (layer L in
shard L+3). The tokenizer, `encoding/`, `inference/` and `LICENSE` come from DeepSeek's release. The download is 332 GB.

How the experts were made (`recipe/quantize/README.md`): we served the original checkpoint, recorded each layer's expert
inputs and routing on 320 windows of 4,096 tokens of AtomicChat's DeepSeek calibration corpus (`AtomicChat/calib-corpora`), and quantized every routed
expert with ExLlamaV3's LDLQ at 3 bits. Gate and up use the Hessian of the rows routed to that expert; down uses the Hessian
of its actual inputs, rebuilt from the quantized gate and up with the model's SwiGLU clamp and weighted by the router.

## Quick start

You need:

- two sm_120 GPUs with 96 GB each and a CUDA 13 driver;
- Docker with the NVIDIA container toolkit;
- about 260 GiB of free host RAM;
- 340 GB of disk;
- for the optional CPU runtime: an x86_64 CPU with AVX-512 (VNNI, VBMI, VBMI2) and many cores and memory channels (see
  `recipe/cpuexp/README.md`).

```bash
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir ~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw
cd ~/models/DeepSeek-V4.1-Flash-1017-EXL3-3bpw/recipe
bash docker/build.sh                      # serving image dsv41-flash-1017-exl3-3bpw-sm120
bash kernels/build.sh                     # the xmoe, xmoe_do, xmoe_do_i8 and fp8ar kernels, about 3 minutes
API_KEY=change-me bash serve/serve.sh     # OpenAI-compatible API on :8000, model deepseek-v4.1-flash
```

With a suitable host CPU, build and check the CPU runtime on the serving machine and turn it on:

```bash
bash cpuexp/build.sh && bash cpuexp/check_host.sh
API_KEY=change-me CPU_EXPERTS=1 bash serve/serve.sh
```

Check the running server with:

```bash
export API_KEY=change-me
python3 bench/needle-test.py 200 && python3 bench/test-toolcall.py && python3 bench/test-image.py
bash bench/fetch-evaldata.sh && python3 bench/make_corpus.py && bash bench/validate.sh    # the synthetic suite above
```

If you served revision `bdc01239`, the first 1017 release, rebuild the image and the kernels. The image now applies three
more vLLM patches (`patches/vllm/v41-*.diff`), and `kernels/build.sh` builds a fourth kernel. The image tag is unchanged, so
an old image would serve without the encoder and parser fixes; `serve/serve.sh` warns when the image lacks them.

If you used the V4.1-Flash release before that, the image tag, the vLLM patches, the plugin patch and the kernels all
changed. The runtime hooks' environment variables now use the `DS41_` prefix (renamed from that release's prefix).
`patches/vllm/sm120-fixes.diff`, `patches/vllm-exl3/exl3-sm120.patch` and `kernels/sm120_dense*` are the renamed patch and
kernel files, so update any scripts that set the old names.

`recipe/serve/serve.sh` documents every setting, and `recipe/README.md` explains the stack.

## Limitations

- The CPU runtime was tested by us on one CPU, an AMD Threadripper PRO 9995WX, and independently reproduced by a user on a
  32-core Threadripper PRO 9975WX (discussion #5). The core pools are now derived from the host's L3 layout
  (`recipe/cpuexp/README.md`), so other CPUs need no setting; `EXL3_CPU_EXPERTS_CPUS` still overrides them. With fewer
  cores or memory channels it can be slower than the GPU-only path. The library is compiled for the serving host's CPU and
  must be built there.
- The host-expert plan is sized for two 96 GB cards with ~2.5 GB per card used by another process. At the 1M default the
  peak leaves about 1 GB per card (85 MiB less with the 8-bit all-reduce's receive buffer). With more free VRAM, a smaller
  plan is faster. With less, raise the plan size or lower `KV_MEM`.
- The default plan is tuned for long agentic coding sessions. Traffic that routes differently uses host experts more often;
  `recipe/plans/README.md` shows how to build a plan from your own routing counts.
- Image input is checked with a small probe (`recipe/bench/test-image.py`), not an image benchmark suite.
- The quality evidence is the KL table, GSM8K-200, HumanEval and needle tests, not a full benchmark suite.
- Two prefill changes are lossy, and KL to the original model stayed within run-to-run noise for each:
  - the 8-bit prefill all-reduce: about 0.6% relative RMS error per all-reduce against the exact one (`DS41_FP8AR=0` serves
    with the exact NCCL all-reduce);
  - the int8 decode-once kernel: about 1% relative error per expert GEMM, against 0.02% for the fp16 kernel (`DO_INT8=0`
    serves with the fp16 kernel).
- Prompt encoding changed for these requests (the `VLLM_DS41_*` switches restore the previous behavior one by one):
  - message text that spells a control token;
  - developer messages;
  - `response_format`;
  - `tool_choice: "none"`;
  - tools with extra or null fields;
  - conversations whose first message is not a system message.
- Some official-API behaviors are not ported:
  - `tool_choice` `required` or a named function (vLLM uses guided decoding instead);
  - prefix continuation of the last assistant message;
  - the API's strict-schema normalization.
- The stack is pinned to vLLM `nightly-5555a2d8` with PRs #59585, #61039, #61142, #60792 and #62196 and our patches in
  `recipe/patches/vllm/`, ExLlamaV3 `5be88657`, vllm-exl3 `d3cfd394` and the sfxnz recipe commit in
  `recipe/docker/build.sh`.

## Credits and license

Built on the work of:

- DeepSeek;
- coolbho3k (the calibration method);
- sfxnz (the V4.1 EXL3 recipe);
- turboderp (ExLlamaV3);
- vcruz305 and Mia's AI Lab (vllm-exl3);
- AtomicChat (the KL protocol and calibration corpus) and kyuz0 (the short100 reference panel);
- the vLLM and FlashInfer teams.

The weights keep DeepSeek's MIT license (`LICENSE`). The code in `recipe/` is MIT, except `recipe/patches/vllm-exl3/`, which
is AGPL-3.0-only like the plugin it patches. `recipe/THIRD_PARTY_NOTICES.md` lists every upstream component.
