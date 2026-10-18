---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- vllm
- deepseek
- deepseek-v4.1-flash
- mixture-of-experts
- 3-bit
- rtx-pro-6000
- blackwell
- sm120
---

# DeepSeek-V4.1-Flash EXL3 3.0 bpw for 2× RTX PRO 6000 Blackwell

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), ready to serve on two RTX PRO 6000
Blackwell cards (sm_120, 96 GB each) with vLLM at tensor-parallel 2. The routed experts are
[coolbho3k's calibrated EXL3 3.0 bpw quantization](https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw).
Our additions are:

- int4 Engram tables;
- a 3-bit DSpark drafter;
- a serving stack that keeps 30% of the routed experts (the ones agentic coding uses least) in pinned host RAM and runs
  them next to the VRAM experts;
- a decode-once prefill kernel for the 3-bit experts.

The `recipe/` folder has the image build, the sm_120 patches, the MoE kernel, and the serving, benchmark, KL and
quantization scripts.

> **This repo used to hold a 2.0 bpw pack.** On the same two cards, the 3.0 bpw stack has about 7× lower KL to the
> original model. It is faster on every benchmark shape we publish but one, so it replaced the 2.0 bpw pack here. The
> old files are still in the repo history at revision `28b7ab71`.

## Fidelity

We measured KL against the original model with
[AtomicChat's V4.1-Flash protocol](https://huggingface.co/datasets/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-metrics). It
uses the original model's top-512 logprobs on three corpora, 49,152 scored positions each. The neutral corpus is
multilingual (about half Latin script, the rest spread over some 20 others, from Arabic and Cyrillic to CJK and Hangul);
the code and agentic corpora are mostly English. Each cell shows the mean KL
lower bound in nats, then top-1 agreement with the original. Lower KL is better. `recipe/kl/` has the client and the
steps.

| Model | Neutral text | Code | Agentic |
|---|---:|---:|---:|
| Original model, second run (noise floor) | 0.0158 · 96.1% | 0.0100 · 97.7% | 0.0055 · 98.6% |
| NVIDIA's NVFP4 checkpoint (4-bit, for reference) | 0.0352 · 94.1% | 0.0197 · 96.8% | 0.0089 · 98.3% |
| **This pack, as served** | **0.0545 · 92.9%** | **0.0321 · 96.0%** | **0.0101 · 98.2%** |
| Previous 2.0 bpw pack from this repo | 0.3613 · 80.4% | 0.2147 · 88.5% | 0.0766 · 94.7% |

"As served" means the exact serving configuration below: FP8 lm_head, host-RAM experts, CUDA graphs, and every VRAM
trim. For comparison, coolbho3k's pack on our earlier, simpler stack scored 0.0529 / 0.0310 / 0.0096. The serving changes
cost almost nothing measurable. The numbers above are from the 2026-10-15 rebase onto vLLM main; the previous
release scored 0.0541 / 0.0324 / 0.0099.

Quality probes with thinking off and greedy decoding:

| | GSM8K-200 | HumanEval |
|---|---:|---:|
| This pack | 98.0% | 95.7% |
| Previous 2.0 bpw pack | 97.5% | 91.5% |

Text and image input both work: the model's own vision encoder is loaded by default and images go through the OpenAI
`image_url` content type (`VISION=0` serves text only). A needle planted in prompts of 101K, 250K and 448K tokens is found
at every length, and so is one in a 1.04M-token prompt at the default 1M context window (about 2 minutes of prefill; peak 95.8 / 96.3 GB
per GPU including a ~2.5 GB side process). A Korean spot check (8 prompts, 3
samples each, thinking on and off) answered in Korean every time, with no drift into Chinese; see `recipe/results/RESULTS.md`.

## Speed

Updated 2026-10-01: a decode-once prefill kernel (18% faster prefill) and a host-expert plan tuned for agentic coding.
2026-10-02: image input on by default; the vision encoder adds about 1.2 GB of VRAM per GPU and no measurable speed cost.
2026-10-04: vLLM PR #59894 (the decoder replay skips work whose output it never used): a fresh 1M-token prompt prefills in
123 s instead of 145 s, 46K prefill is 4–5% faster, decode and KL are unchanged (`recipe/results/RESULTS.md`). The tables below
were measured before it.
2026-10-05: vLLM PR #58560 (merged upstream) keeps warmup writes out of the null KV block, a latent bug that upstream saw
crash DSpark on sm_120; on our cards the first request was already clean, and speed and KL are unchanged.
2026-10-15: rebased onto vLLM main (`nightly-5555a2d8`, the v0.32.0rc0 code), where PRs #58132, #59894 and #58560 are
upstream, plus PRs #61039, #61142 and #60792 and a one-line fix that frees the engram layers' post-mapped stream early
(without it the new base needed about 1.1 GB more VRAM per GPU on a 1M-token prompt). Speed, KL and quality are unchanged
within noise: 46K prefill 10.5K tok/s at 1 stream and 19.0K at 4, agent turns 0.91 / 2.64 s to first token, 20-agent
fan-out 116 s, GSM8K 98.0%, HumanEval 95.7% (`recipe/results/RESULTS.md`).

Test setup:

- Two RTX PRO 6000 Blackwell Max-Q at their 300 W cap, on PCIe (no NVLink), tensor-parallel 2.
- A 512K context window with a 4 GiB fp8 KV cache (814K tokens), the default when these numbers were taken. The default
  is now a 1M window with a 4.25 GiB cache (1.07M tokens); that changes only buffer sizes, and 46K-prompt and agent-turn
  speeds matched within noise (see `recipe/results/RESULTS.md`).
- CUDA graphs and DSpark speculative decoding with 5 draft tokens.
- A ~2.5 GB/GPU side process stayed resident, and no other clients were connected.
- The server was `recipe/serve/serve.sh` with its defaults, serving the files in this repo. `recipe/bench/validate.sh`
  produced every number here except the agentic coding sessions, whose session set we don't publish.

Throughput is in tokens per second.

**Agentic coding sessions.** 16 multi-turn agentic coding sessions, 8 consecutive turns each, with tool definitions and
tool calls in the history, prompts up to ~350K tokens, 3 sessions at a time:

| Host-expert plan | Decode per stream (median) | Wall clock, 128 turns | Time to first token (median / p90) |
|---|---:|---:|---:|
| `plan-agent30-dh.json` (this release's default) | 73.9 | 380 s | 0.77 s / 12.4 s |
| `plan-gen30-dh.json` (the 2026-09-30 release's plan) | 47.9 | 503 s | 0.91 s / 12.2 s |

One session at a time (16 other sessions, 8 turns each, prompts up to ~175K tokens, median ~34K) with the default plan:
**159 tok/s decode** weighted by tokens (177 tok/s median per turn) and 0.43 s median time to first token (2.2 s p90).
The same sessions 4 at a time: **232 tok/s aggregate decode** (67 tok/s per stream; on average 3.5 of the 4 streams were
decoding at any moment). Agentic replies are mostly tool calls, which DSpark predicts well.

**Synthetic prompts** (`recipe/bench/bench-sbs.py`: non-streaming requests with a synthetic prompt; decode speed is the
difference between a 400-token and a 16-token reply on a warm prefix cache; aggregated across streams):

| Benchmark | This release | 2026-09-30 release | Previous 2.0 bpw |
|---|---:|---:|---:|
| Decode, 2K prompt, 1 stream | 74.4 | 123.8 | 113.3 |
| Decode, 2K prompt, 4 streams | 130.6 | 248.4 | 277.2 |
| Prefill, 46K prompt, 1 stream | 10,008–10,114 (5.0 s to first token) | 8,851–8,903 (5.7 s) | 4,044 |
| Decode, 46K prompt, 1 stream | 66.4–70.7 | 118.2–119.3 | 116.6 |
| Prefill, 46K prompts, 4 streams | 18,105 | 15,911 | 8,085 |
| Decode, 46K prompts, 4 streams | 129.4 | 268.9 | — |

Decode on these synthetic prompts is about half the 2026-09-30 release's, because the default plan now keeps the experts
that agentic coding uses in VRAM and the synthetic prompts use others. `PLAN=plan-gen30-dh.json` gives the previous decode
speed with the new prefill kernel (46K prompt, 1 stream: about 10,450 tok/s prefill and 130 tok/s decode), and
`PLAN=plan-blend30-dh.json` sits in between; `recipe/plans/README.md` compares the three.

Agent-shaped synthetic benchmarks:

| Benchmark | This release | 2026-09-30 release |
|---|---|---|
| Agent turn: 34K cached + 6K new prompt tokens, 300 output tokens, 1 stream | 0.94 s to first token, 122 tok/s decode | 1.11 s, 139 tok/s |
| Same, 4 concurrent sessions | 2.79 s to first token (median), 215 tok/s aggregate | 3.44 s, 250 tok/s |
| 20 agents × 3 turns each, 10K-token shared prefix + 30K private tokens per agent | 119 s wall clock, no errors, 74% prefix-cache hits | 116 s |

DSpark accepted 4.1 tokens per verification step on average across the run. At its peak, nvidia-smi showed 94,725 and
95,093 MiB of 97,887 MiB in use, including the side process and the vision encoder.

## How it fits in 2× 96 GB

At 3.0 bpw the routed experts alone take 204.5 GB, more than the two cards hold. The stack keeps the rest of the model
in VRAM and moves the coldest data to pinned host RAM:

- **Cold experts in host RAM.** 4,608 of 15,360 routed experts and 176 of the drafter's 384 experts stay in pinned host
  RAM: the ones that long multi-turn agentic coding sessions use least. `recipe/plans/` explains how they were chosen and
  has alternative plans for other traffic.
- **Parallel launch.** Each MoE layer runs its VRAM experts on the main stream. At the same time, a second stream with 32
  SMs (64 during prefill) runs the cold experts, which read their weights over PCIe. In agentic coding sessions, 2.4% of
  decode routings land on a cold expert.
- **Engram and embeddings in host RAM.** The int4 Engram tables and the embedding table are pinned in host RAM and read
  over UVA, inside CUDA graphs.
- **VRAM trims, about 9 GiB per GPU in total.** Most of the freed memory lets more experts stay in VRAM:
  - one FP8 copy of each large dense layer instead of FP8 plus a Marlin repack (2.7 GiB);
  - a smaller indexer prefill workspace (2.0 GiB);
  - a smaller indexer decode block table (1.5 GiB);
  - 3-bit EXL3 drafter experts instead of MXFP4 (1.0 GiB);
  - attention `wo_a` held in FP8, re-quantized exactly (0.65 GiB);
  - the embedding table in host RAM (0.62 GiB);
  - an FP8 lm_head (0.33 GiB);
  - a rope table sized to the context window (0.25 GiB).
- **Custom kernels.** An MoE kernel rewritten for sm_120 (`recipe/kernels/xmoe`); a decode-once prefill kernel
  (`recipe/kernels/xmoe_do`) that dequantizes each expert weight tile once per 128 rows instead of once per 32, for 18%
  faster prefill; split-K FP8 GEMV kernels for the dense layers at decode sizes; and vLLM's decoder-side sliding-window
  replay (PR #58132). Together with the move to a vLLM nightly, the replay made prefill 1.7× faster than our earlier
  stack.

The serving container uses about 230 GiB of host RAM, most of it pinned.

## What's in this repo

| Files | Contents |
|---|---|
| `model-00001` to `model-00051` | coolbho3k's shards at revision `650cae2c`, unchanged. Routed experts are EXL3 3.0 bpw with the MUL1 codebook, calibrated. Non-routed weights are DeepSeek's FP8. The vision tower is included but not used. |
| `model-00052` to `model-00054` | The DSpark draft layers. We re-encoded their routed experts from MXFP4 to EXL3 3-bit (MCG codebook, uncalibrated); the other tensors are DeepSeek's. |
| `model-00055`, `model-00056` | The Engram tables of layers 1 and 14 in int4 with fp16 per-32 scales: 111 GB, down from DeepSeek's 203 GB of FP8. |
| `config.json` | coolbho3k's `quantization_config`, plus `mtp_experts: "exl3"`, `layer_bits` 40–42 = 3 (the drafter) and `engram_embed_format: "int4_fp16s32"`. |
| `recipe/` | Image build, patches, kernels, host-expert plan, and serving, benchmark, KL and quantization scripts. |

The tokenizer, `encoding/`, `inference/`, `evaluation/`, `assets/`, the tech report and `LICENSE` come from DeepSeek's
release. The download is 332 GB.

## Quick start

You need:

- two sm_120 GPUs with 96 GB each and a CUDA 13 driver;
- Docker with the NVIDIA container toolkit;
- about 250 GiB of free host RAM;
- 340 GB of disk.

```bash
hf download diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000 --local-dir ~/models/DeepSeek-V4.1-Flash-EXL3-3bpw
cd ~/models/DeepSeek-V4.1-Flash-EXL3-3bpw/recipe
bash docker/build.sh                      # serving image dsv41-flash-exl3-3bpw-sm120
bash kernels/build.sh                     # the xmoe and xmoe_do MoE kernels, about a minute
API_KEY=change-me bash serve/serve.sh     # OpenAI-compatible API on :8000, model deepseek-v4.1-flash
```

Check the running server with:

```bash
export API_KEY=change-me
python3 bench/needle-test.py 200 && python3 bench/test-toolcall.py && python3 bench/test-image.py
bash bench/fetch-evaldata.sh && python3 bench/make_corpus.py && bash bench/validate.sh    # the suite behind the numbers above
```

`recipe/serve/serve.sh` documents every setting. `recipe/README.md` explains the stack.
`recipe/quantize/README.md` shows how the pack was assembled.

## Limitations

- Image input is checked with a small probe (`recipe/bench/test-image.py`), not an image benchmark suite.
- The host-expert plan is sized for two 96 GB cards with ~2.5 GB per card used by another process. With more free
  VRAM, a smaller plan is faster. With less, raise the plan size or lower `KV_MEM`.
- The default plan is tuned for long agentic coding sessions. Short synthetic prompts route more often to host-RAM
  experts under it, so decode on the synthetic benchmarks above is slower than with the previous plan.
  `PLAN=plan-blend30-dh.json` balances both; `recipe/plans/README.md` compares the plans.
- The quality evidence is the KL table, GSM8K-200, HumanEval and needle tests, not a full benchmark suite.
- vLLM's custom all-reduce stays disabled because it fails CUDA graph capture on this setup.
- The stack is pinned to vLLM `nightly-5555a2d8` with PRs #59585, #61039, #61142 and #60792, ExLlamaV3 `5be88657`, vllm-exl3 `d3cfd394` and the sfxnz
  recipe commit in `recipe/docker/build.sh`.

## Credits and license

Built on the work of:

- coolbho3k (the 3.0 bpw quantization);
- DeepSeek;
- sfxnz (the V4.1 EXL3 recipe);
- turboderp (ExLlamaV3);
- vcruz305 and Mia's AI Lab (vllm-exl3);
- AtomicChat (the KL reference);
- the vLLM and FlashInfer teams.

The weights keep the MIT license of DeepSeek and coolbho3k (`LICENSE`). The code in `recipe/` is MIT, except
`recipe/patches/vllm-exl3/`, which is AGPL-3.0-only like the plugin it patches. `recipe/THIRD_PARTY_NOTICES.md` lists
every upstream component.
