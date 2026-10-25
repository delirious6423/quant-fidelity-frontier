---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: vllm
tags:
  - exl3
  - exllamav3
  - quantized
  - moe
  - deepseek
  - v100
  - sm70
  - 1cat-vllm
  - speculative-decoding
---

# DeepSeek-V4.1-Flash-1017 · EXL3 · franken-v100 (~2.8 bpw routed experts)

An EXL3 expert pack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e10`) sized for **8× Tesla V100-SXM2-32GB** (256 GB of Volta HBM2, NVLink), served by the SM70 port of
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) at the model's native **1,048,576-token context** with **DSpark speculative
decoding**. It does for 8×V100 owners what the DGX-Spark packs do for GB10 owners.

This project started from [0xSero](https://huggingface.co/0xSero)'s
[DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw): his pack was
the first that ran on 8×V100 and the baseline every engine change here was measured against. See *Acknowledgements*.

Nothing here was re-quantized. The pack is a per-expert selection across two public EXL3 packs (see *Recipe*), chosen so
that the whole model, the DSpark drafter and a 1M-token KV cache fit the 32 GB cards; the dense, attention, Engram and MTP
tensors are the official FP8 / BF16 / MXFP4 weights.

## Current build (2026-10-25)

The engine has moved on since this card was first written. The build serving today adds, on top of PR #1243: decoder
replay in prefill (layers 21–39 run only each request's last 3,072 rows, exact), fused WMMA sparse attention for prefill,
a tiled EXL3 prefill MoE kernel, RAM-resident Engram tables (hugepages; the gather is a memcpy), layer-sliced prefill
between decode steps, and fused indexer kernels. Every addition sits behind a `VLLM_DS41_*` knob and was gated against
the fp32 oracle before it went live. None of it is in PR #1243 or in `recipe/` yet.

| current build | |
|---|---|
| KL vs the fp32 oracle | 0.0358, 95.79 % top-1 |
| Prefill, 1 stream | 2,985–3,231 tok/s from 8K to 128K context |
| Decode, DSpark k=5, 1 stream (replay bench) | 69.2 tok/s |
| Context | 1,048,576 tokens; 64K / 256K / 512K needles pass |

**DSpark fp16 tap fix (gated, rolling out with the next build).** The drafter reads hidden-state taps from the target
model. Those taps were captured in fp16, and on the attention-sink tokens at the start of a chat they overflowed to
inf, which left NaN rows in the drafter's window. For roughly the first 100 steps of every chat the drafts were all
zeros, 42 % of steps overall. Completion-style benchmarks never saw it because they have no BOS. Scaling the taps down
before the fp16 cast fixes it. Gate results, single stream, before → after:

| | before | after |
|---|---|---|
| Steps with NaN drafts | 42 % | 0 |
| First-position acceptance | 0.458 | 0.745 |
| Code | 42.6 tok/s | 59.2 tok/s |
| Math | 42.0 tok/s | 58.5 tok/s |
| Structured JSON | 45.2 tok/s | 56.6 tok/s |
| Multilingual | 32.0 tok/s | 43.4 tok/s |
| Prose | 25.9 tok/s | 29.9 tok/s |
| Chat | 28.9 tok/s | 30.3 tok/s |

`recipe/serve_gpubox_exl3.sh` is the 2026-10-21 launcher that matches PR #1243 and the table below. It still serves the
pack; the current build's launcher and engine source aren't published yet.

## Measured on the 2026-10-21 build (8× V100, PP2×TP4, 1Cat-vLLM SM70 port, PR #1243)

| | |
|---|---|
| Decode, plain, 1 stream | 30.5 tok/s |
| Decode, plain, 4 streams | 63–74 tok/s aggregate |
| Decode, DSpark k=5, 1 stream: code / structured JSON / prose / chat | 66–68 / 71 / 33–40 / 31 tok/s |
| Decode, DSpark k=5, 2 streams | 76–79 tok/s aggregate |
| Prefill | ~900 tok/s plain, ~560–600 tok/s with DSpark |
| Context | 1,048,576 tokens native; 64K and 256K needle-in-a-haystack PASS; prefix cache hit 16K prompt 16 s → 1 s |
| KV pool | 1.29–1.39M tokens at `--max-num-batched-tokens 1024` |

Short replies with low draft acceptance decode at 20–23 tok/s under DSpark (below plain); code and long structured output
nearly double. Prompt logprobs on long prompts are not available under the 1M DSpark configuration (no memory slack for
full-vocabulary chunk logits).

**Quality vs the native model.** The reference is DeepSeek's own `inference/model.py` run on CPU in fp32 with every
tilelang kernel replaced by exact torch math (our oracle), on 3,134 positions over 8 domains, top-20 support:

| pack | routed bpw | KL(native ‖ pack) | top-1 agreement | NLL gap |
|---|---|---|---|---|
| 0xSero 2.7 bpw | 2.70 | 0.066 | 94.2 % | +0.052 |
| HeNryous 2.75 bpw | 2.75 | 0.042 | 95.5 % | +0.028 |
| **franken-v100 (this pack)** | **~2.8** | **0.0365** | **95.6 %** | **+0.019** |

Per domain: code 0.012 · math 0.017 · agentic 0.028 · knowledge 0.041 · multilingual 0.047 · chat 0.060 · writing 0.111.
This panel is small and writing-heavy and reads about 2× the numbers the pack cards report on their own panels; compare
packs with each other. Greedy outputs are bit-identical between single-request and batched decode (2 and 3 concurrent
copies), and step-by-step decode agrees with one-pass prefill of the same tokens (self-consistency KL ≤ 0.05 mean).

## What is in this repository

* `model-experts-LNN.safetensors` (40 files, ~4.7 GiB each): the EXL3 routed experts of layer NN — tensors
  `layers.NN.ffn.experts.E.w{1,2,3}.{trellis,suh,svh,mul1}`, codebook `mul1`, exllamav3 tile layout, per-expert K in
  `exl3_moe_plan.json` (`layers[L]["k"]`, 384 digits). Copied byte-exactly from the donor packs.
* `model-dense-0000N.safetensors` (~17 GiB total): every official non-expert tensor (attention, dense FFN, norms,
  hyper-connection weights, embeddings, head, vision, aligner, Engram projection weights) and the 3 MTP / DSpark draft
  blocks including their MXFP4 experts, byte-exact from the official checkpoint.
* `model.safetensors.index.json`, `exl3_moe_plan.json`, `config.json`, `tokenizer.json`, `tokenizer_config.json`,
  `encoding/`, `LICENSE` (MIT, DeepSeek).
* `recipe/`: the scripts that built, served and gated the pack (see below).

**Not in this repository: the Engram embedding tables** (189 GB, unmodified). The index points at the official shard
files `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors` of
`deepseek-ai/DeepSeek-V4.1-Flash-1017`; download those two and place (or symlink) them next to the other files.
Serve them from a dataset with a small record size (we use ZFS `recordsize=16K`): Engram reads 256-byte rows at random
and a 1 MB record size amplifies the reads ~500×. The official MXFP4 routed experts of the 40 main layers are not included
either (the EXL3 experts replace them).

## Recipe

**Pack composition** (`recipe/frankenplan.py`, `recipe/make_franken_hybrid.py`):

1. Base: [HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark)
   (sudoingX lineage: sensitivity-informed per-expert bit allocation, native-1017 calibration). It measured best of the
   public packs on our oracle panel.
2. Upgrades from [Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw):
   experts are upgraded to the next wider K available (K2→K3) in descending routing frequency (routing histograms from a
   mixed workload, `recipe/route_workload.py`) until each pipeline stage's byte budget is used. Budgets per GPU under
   PP2×TP4 with expert parallelism over the 4 TP ranks: **stage 1 (layers 0–19) 24.37 GiB, stage 2 (layers 20–39)
   22.0 GiB**. The stage-2 budget is what leaves room on the last stage for the DSpark drafter plus a 1M-token KV cache;
   a 23.0 GiB variant (plain decode only) scores the same KL (0.036).
3. `make_franken_hybrid.py` merges the selection into one index with the official dense tensors; `repack_upload.py` wrote
   the self-contained shards in this repository byte-exactly.

**Engine**: 1Cat-vLLM SM70 port, [PR #1243](https://github.com/1CatAI/1Cat-vLLM/pull/1243) (stacked on #1031): EXL3
mixed-K trellis decode inside the MoE GEMV, per-layer CUDA-graph decode attention over a packed FP4 compressed KV, fused
hyper-connection kernels, Engram rows served from host RAM (deferred zero-copy path), DSpark drafter in per-batch-size
CUDA graphs, batch-invariant dense GEMMs. CUDA 12.8; Python 3.12; torch 2.10 cu128. Host: 64 cores, 372 GB RAM (the
Engram tables and their row index live in host memory), GPU clocks locked at 1530 MHz.

**Serving** (`recipe/serve_gpubox_exl3.sh`, `recipe/_ds41_common.sh`; paths are our box's, adjust `DS41_*`):
PP2×TP4 cut at layer 20, `--max-model-len 1048576 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.943`,
`VLLM_DS41_MOE_EXL3_PLAN=<dir>/exl3_moe_plan.json`, `VLLM_DS41_MOE_EXL3_GROUP_MIN_TOKENS=2`,
`VLLM_DS41_BATCH_INVARIANT_GEMM=1`, `VLLM_DS41_ATTN_GRAPH=1`, `VLLM_DS41_BUILDER_GRAPHS=1`, `VLLM_DS41_HC_FUSED=1`,
`VLLM_DS41_HC_NORM_EXACT=1`, and for DSpark
`--speculative-config '{"method":"dspark","num_speculative_tokens":5}'` with the hf-overrides in the launcher,
`--no-async-scheduling`, `--max-num-seqs 2`. Memory table for this pack: weights 27.77 / 25.29 GiB per GPU (stage 1 / 2).

**Gates** (`recipe/`): `kl_panel.py`, `oracle_cpu.py` + `oracle_kl_compare.py` (native reference), `divergence*.py`
(batched vs single exactness), `kl_selfconsistency.py` (decode vs prefill), `prefix_cache_gate.py`, `needle.py`,
`bench_decode.py`, `bench_streams.py`, `dspark_bench.sh`.

## Caveats

* No memory slack at the 1M DSpark configuration (31.6–32.1 GB of 32.8 GB after a 256K prompt). Lower the context or use
  the plain-decode mode if you add anything.
* The pack targets V100 (SM70) through the 1Cat-vLLM port above. It is a standard EXL3 checkpoint otherwise, but the
  stage budgets and the plan are V100-specific; GB10 users are better served by the donor packs directly.
* Quantization credit belongs to the donor packs' authors (HeNryous / sudoingX lineage, Mia-AiLab). This repository adds
  the selection, the engine work and the measurements.

## Acknowledgements

**0xSero.** This work exists because of
[0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw). It was
the first DeepSeek-V4.1 pack that ran on 8×V100 and it served as the baseline for two days of engine work: the 1M-context
bring-up, the packed FP4 KV cache, the graph-captured attention, the EXL3 MoE kernels and the first DSpark runs were all
measured on it, and the first franken plan borrowed 204 of its K4 experts. His experts are not in this particular pack
only because the plan's donor order filled the budget from HeNryous and Mia first; the 0xSero row in the quality table is
his pack measured on our panel.

**HeNryous / sudoingX** for the base pack and its sensitivity-informed bit allocation, and **Mia-AiLab** for the EAQ pack
the upgrades come from. The donor packs contributed quantized weights; the pack selection is ours.

**The V100 engine work is ours and 1Cat's.** [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) is the SM70 vLLM fork that
makes Volta a first-class target (its kernels, build and the DeepSeek-V4 groundwork), and PLI Labs' #1031 brought
DeepSeek-V4.1 up on it. Everything this pack needs on top (the EXL3 mixed-K MoE kernels, graph-captured decode attention
over packed FP4 KV, the fused hyper-connection kernels, the Engram host service, DSpark speculative decoding with
multi-sequence drafter graphs, batch-invariant GEMMs, the 1M-context fit, the gates and the measurements) was done by us
on the 8×V100 box and is submitted upstream as PR #1243.
