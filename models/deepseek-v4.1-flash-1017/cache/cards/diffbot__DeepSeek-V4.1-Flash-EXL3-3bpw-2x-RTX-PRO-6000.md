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
- custom MoE kernels for sm_120.

The `recipe/` folder has the image build, the sm_120 patches, the kernels, the CPU runtime, and the serving, benchmark, KL
and quantization scripts.

> **This repo now holds DeepSeek-V4.1-Flash-1017.** Until this update it held DeepSeek-V4.1-Flash with coolbho3k's
> 3.0 bpw experts (revision `6062be5f`), and before that a 2.0 bpw pack (revision `28b7ab71`); both are still in the repo
> history. On the same two cards and the same agentic benchmark, this release decodes 54% faster than the previous one with
> the CPU runtime (271 vs 176 tok/s, one session at a time), and 21% faster on the GPU-only path (213 tok/s, measured on an
> earlier build of it).

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
| **This pack, as served** | **0.0521 · 92.9%** | **0.0310 · 95.9%** | **0.0107 · 98.2%** | **0.0339 · 95.2%** |
| This pack on the previous release's serving stack | 0.0518 · 92.9% | 0.0309 · 95.9% | 0.0109 · 98.3% | 0.0336 · 95.5% |

"As served" means the serving configuration below, including FP8 lm_head, host-RAM experts, the VRAM cache, the CPU
runtime, CUDA graphs and every VRAM trim, with speculative decoding off, an 8K context window and no prefix cache for the
logprob capture; the serving changes since the previous release left KL unchanged within noise.
The previous release's pack scored 0.0545 / 0.0321 / 0.0101 against the original V4.1-Flash on AtomicChat's own reference.
Those numbers are not directly comparable: a different model, and a reference with a higher noise floor (0.0158 / 0.0100 /
0.0055), measured on a different engine.

Quality probes with thinking off and greedy decoding:

| | GSM8K-200 | HumanEval |
|---|---:|---:|
| This pack | 98.5% | 97.0% |
| Previous release (V4.1-Flash) | 98.0% | 95.7% |

Text and image input both work: the model's own vision encoder is loaded by default and images go through the OpenAI
`image_url` content type (`VISION=0` serves text only). Prompts are rendered exactly like DeepSeek's 1017 reference encoder
(vLLM PR #62196: identical prompts, or the same error for invalid input, on all 40 cases we checked). A needle in a fresh 1.04M-token prompt is found at the default
1M context window (123 s of prefill).

## Speed

Test setup:

- Two RTX PRO 6000 Blackwell Max-Q at their 300 W cap, on PCIe (no NVLink), tensor-parallel 2.
- Host: AMD Threadripper PRO 9995WX (96 cores), 8-channel DDR5-4800. The host CPU matters only for the CPU runtime.
- A 1M context window with a 4.25 GiB fp8 KV cache (1.07M tokens).
- CUDA graphs and DSpark speculative decoding with 5 draft tokens.
- A ~2.5 GB/GPU side process stayed resident, and no other clients were connected.
- The server was `recipe/serve/serve.sh` with its defaults, serving the files in this repo; "CPU runtime" rows add
  `CPU_EXPERTS=1`.

Throughput is in tokens per second.

**Agentic coding sessions.** 13 held-out multi-turn agentic coding sessions with tool definitions and tool calls in the
history (104 turns, 8.9M prompt tokens in total), replayed turn by turn with streaming, temperature 0 and up to 1,024 output
tokens per turn, each configuration on a fresh server. We don't publish the session set.

| Configuration | 1 session: decode | wall clock | time to first token (median / mean) | 4 sessions: decode per stream | aggregate decode | wall clock |
|---|---:|---:|---:|---:|---:|---:|
| **This release, CPU runtime** | **271.1** | **229 s** | **0.57 s / 1.52 s** | **52.4** | **129.0** | **178 s** |
| This release, GPU-only (earlier build*) | 213.2 | 261 s | 0.61 s / 1.62 s | 37.9 | 99.2 | 203 s |
| Previous release, same harness | 175.7 | 283 s | 0.57 s / 1.53 s | 39.3 | 105.2 | 214 s |

\* The GPU-only row is from an earlier build of this stack: without the four decode fusions (+6% at one session in a
same-day A/B: 221.1 vs 208.3 tok/s), the hand-tuned decode loop (+3%) and the prefill read-once kernel, and with the drafter's
rarely used experts in host RAM. The released GPU-only configuration has all of them but was not run through this
benchmark.

Decode is per stream, weighted by tokens; aggregate decode counts the tokens of all streams over the time any stream was
decoding (2.5 of the 4 streams on average). The 1-session numbers repeat within about 1%; the 4-session numbers vary by
up to ~15% between boots (a near-identical arm gave 43.8 per stream and 191 s). Agentic replies are mostly tool calls,
which DSpark predicts well.

**Fixed prompts** (`recipe/bench/bench-fixed.py`: code prompts, thinking off, the same tokens and speculative acceptance on
every run; decode speed is the difference between a 400-token and a 16-token reply on a warm prefix cache):

| Benchmark | This release, CPU runtime | This release, GPU-only (earlier build*) | Previous release |
|---|---:|---:|---:|
| Prefill, fresh 40K-token prompts (two prompts) | 10,297-10,535 | 8,875-9,224 | 9,031-9,125 |
| Decode, 2K prompt, 1 stream | 178.6 | 144.5 | 129.7 |
| Decode, 46K prompt, 1 stream (median of 8) | 201.2 | 169.4 | 114.4 |
| Decode, 46K prompts, 4 streams (aggregate) | 427.1 | 332.8 | 215.9 |

**Benchmarks from earlier releases** (`recipe/bench/`, CPU runtime, 1M context; the previous release's numbers are from its
2026-10-15 validation, also at 1M):

| Benchmark | This release | Previous release |
|---|---|---|
| 46K synthetic prompt, 1 stream (`bench-sbs.py`) | 10,179 tok/s prefill (4.9 s to first token), 95.4 tok/s decode | 10,451-10,486 (4.8 s), 63.0-69.4 |
| Agent turn: 34K cached + 6K new prompt tokens, 300 output tokens, 1 stream (`hit_bench.py`) | 0.76 s to first token, 253 tok/s decode | 0.91 s, 114 tok/s |
| Same, 4 concurrent sessions | 2.36 s to first token (median), 421 tok/s aggregate | 2.64 s, 216 tok/s |
| Fresh 1.04M-token prompt with a needle | found, 122.9 s (8,463 tok/s) | found, 120.5 s |

Fresh prefill on the synthetic 46K prompt is about 3% slower than before, while the 40K code prompts above prefill 14-15%
faster, likely because this plan keeps more experts in host RAM and the synthetic prompt routes to them more often. At its peak (the 1.04M-token
prompt), nvidia-smi showed 96,655 and 97,029 MiB of 97,887 MiB in use (previous release: 95,771 and 96,279), including the
side process and the vision encoder.

**Power.** Estimated box power while requests ran (CPU package from RAPL, both GPUs, plus 100 W for the rest): about 900 W
with the CPU runtime and about 815 W on the GPU-only path. The GPUs drew 576-593 W together either way; the CPU package went
from about 137 W to 221 W.

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
- **Custom kernels.** An MoE kernel rewritten for sm_120 (`recipe/kernels/xmoe`): a flat decode scheduler with a hand-tuned
  inner loop, and a prefill path that reads each host expert over PCIe once per chunk instead of once per 32 rows. A
  decode-once prefill kernel for the VRAM experts (`recipe/kernels/xmoe_do`), split-K FP8 GEMV kernels for the dense layers at
  decode sizes, and fused decode kernels that cut the launches per step.

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
bash kernels/build.sh                     # the xmoe and xmoe_do MoE kernels, about 2 minutes
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

If you used the previous release, rebuild the image and the kernels: the image tag, the vLLM patches, the plugin patch and
the kernels all changed. The runtime hooks' environment variables now use the `DS41_` prefix (renamed from the previous
release's prefix), and `patches/vllm/sm120-fixes.diff`, `patches/vllm-exl3/exl3-sm120.patch` and `kernels/sm120_dense*` are
the renamed patch and kernel files, so update any scripts that set the old names. `recipe/serve/serve.sh` documents every setting. `recipe/README.md` explains the stack.

## Limitations

- The CPU runtime was tested on one CPU, an AMD Threadripper PRO 9995WX. Its default core pools assume that CPU's layout;
  on another CPU set `EXL3_CPU_EXPERTS_CPUS` (the check script suggests a value). With fewer cores or memory channels it
  can be slower than the GPU-only path. The library is compiled for the serving host's CPU and must be built there.
- The GPU-only agentic numbers come from an earlier build of this stack (see the footnote above).
- The host-expert plan is sized for two 96 GB cards with ~2.5 GB per card used by another process; at the 1M default the peak
  leaves about 1 GB per card. With more free VRAM, a smaller plan is faster. With less, raise the plan size or lower
  `KV_MEM`.
- The default plan is tuned for long agentic coding sessions. Traffic that routes differently uses host experts more often;
  `recipe/plans/README.md` shows how to build a plan from your own routing counts.
- Image input is checked with a small probe (`recipe/bench/test-image.py`), not an image benchmark suite.
- The quality evidence is the KL table, GSM8K-200, HumanEval and needle tests, not a full benchmark suite.
- The stack is pinned to vLLM `nightly-5555a2d8` with PRs #59585, #61039, #61142, #60792 and #62196, ExLlamaV3 `5be88657`,
  vllm-exl3 `d3cfd394` and the sfxnz recipe commit in `recipe/docker/build.sh`.

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
