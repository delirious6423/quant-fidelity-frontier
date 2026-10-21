---
language:
- en
- zh
library_name: exllamav3
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3
base_model: zai-org/GLM-5.3
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- exl3
- glm_moe_dsa
- mixed-precision
- dgx-spark
- gb10
- vllm
- mtp
- tensorfold
---

# keys-GLM-5.3-EXL3-2.75BPW

Full **GLM-5.3** (753B, `glm_moe_dsa`) quantized to EXL3 with a **per-expert mixed bit width averaging 2.75 bpw** on
the routed experts, sized to serve on **four NVIDIA DGX Sparks** (GB10, 128 GB unified memory each) with room for a
200K-token KV pool (or 1M tokens with decode-context-parallel 4).

Unmodified weights of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (no fine-tuning, no edits),
quantized from the BF16 checkpoint. For the earlier uniform-width quant see
[keys-GLM-5.3-EXL3](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3).

## Run it: TensorFold on four DGX Sparks (recommended)

The fastest way to serve this checkpoint is [TensorFold](https://github.com/ashhart/TensorFold)'s native `glm_moe_dsa`
engine — tensor parallel 4, one rank per Spark, no vLLM in the serving path; a drafted reply equals a serial one.
Recipe, one-shot launcher, benches and an [AGENTS.md](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark/blob/main/AGENTS.md)
for coding agents: [drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark).
Code: [drowzeys/TensorFold:glm53-tp4-2026-10-05](https://github.com/drowzeys/TensorFold/tree/glm53-tp4-2026-10-05). **The default
needs no extra draft model** (GLM-5.3's own MTP layer drafts); DFlash2 is optional.

### Performance (prebuilt image `2026-10-05`, thinking on)

Four DGX Sparks, the published image through `one-shot.sh`, default settings, **thinking on** (GLM-5.3 is a reasoning
model: measured the way it is meant to be used; no thinking-off numbers are published).

| Summary | Speed |
|---|---|
| **Prose** (`./one-shot.sh bench`, greedy, 300 tokens, whole request) | **40.2 tok/s** |
| **Code** (same) | **39.0 tok/s** |
| **25K-token prompt** (prefill + answer, needle found) | **25.3 s** |
| **Prefill** at 128K tokens | **1,038 tok/s** (TTFT 125 s) |
| **Turn 2 of a 93K-token conversation** (prompt reuse, on by default) | **0.85 s** to first token (85.7 s without) |
| **4 streams, aggregate** (greedy chat, thinking on): prose / code | **78.7 / 100.2 tok/s** |
| **1M context**: needles at 115K / 456K / 905K | **all PASS**; decode 37.0 / 35.7 / 33.7 tok/s at those depths |

| Single stream, thinking on | MTP = 2 (default, no extra model) | DFlash2 (optional draft) |
|---|---|---|
| Short context, greedy: prose / code | 37.7 / 41.4 tok/s | 36.1 / **47.7** tok/s |
| 32K context, sampled (T = 1.0): prose / code | 31.7 / 34.9 tok/s | **33.4 / 39.4** tok/s |

Prompt reuse: an identical resend 0.15 s, a new conversation with the same 14.7K-token system prompt 0.38 s (13.5 s
without); at 4 x 32K turn 2 in 0.69 s. Agent frameworks (Hermes, OpenClaw): `PARALLEL=4 CONTEXT=32768`, agent context
64K, compression under 32K: each agent step starts in ~0.15-1 s instead of ~16 s.

Memory safety on GB10 is part of the recipe (`vm.swappiness=1`, shards dropped from the page cache as they load,
`one-shot.sh` drops page cache and waits for memory to come back between lanes, and reports each node's headroom):
~9-15 GB free on every Spark while serving, no swap. Details in the recipe README.

Bertholomus' four `tf_greedy.py` prompts, thinking on: MTP 41.1 / 40.8 / 43.1 / 42.6, DFlash2 48.6 / 52.0 / 50.4 / 53.6
tok/s (their published MTP-3 numbers, thinking off, on their 3.0 bpw quant: 34.6-39.6 / 40.6-48.1 / 42.1-48.4 /
38.1-43.6). Optional DFlash2: [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) (CC BY-NC-ND 4.0,
download it yourself).

### Quick start

```sh
# every Spark, after each boot (node/gb10-node-settings.sh in the recipe repo)
sudo sysctl -w vm.compaction_proactiveness=0 vm.swappiness=1
# the checkpoint at the same path on all four (an NFS export works; each rank reads only its share)
hf download drowzeys/keys-GLM-5.3-EXL3-2.75BPW --local-dir /models/GLM-5.3-EXL3-2.75BPW
# from any machine that can ssh to the four Sparks (one-shot.sh from the recipe repo)
export NODES="spark1 spark2 spark3 spark4" MODEL=/models/GLM-5.3-EXL3-2.75BPW
./one-shot.sh check && ./one-shot.sh up && ./one-shot.sh wait && ./one-shot.sh bench
```

Image `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-05` (also `:latest`; CUDA 13 / GB10, extensions
prebuilt, fastest settings as defaults). The launcher finds each Spark's RoCE rails and GIDs itself; an
OpenAI-compatible server comes up on rank 0, port 8890. With `DRAFT=<DFlash2 dir>` requests can pick
`"tf_mtp": "dflash"` (fastest on code) or `"auto"`; 4 streams: `PARALLEL=4 CONTEXT=32768`.

### Build your own draft (optional)

An on-policy fine-tune of the DFlash2 draft on this checkpoint's outputs gave +10.5 % prose / +7.8 % code at 32K on
our `2026-10-03` build. It cannot be published (incoai's CC BY-NC-ND-4.0 license);
[draft-finetune/](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark/tree/main/draft-finetune)
builds your own for non-commercial use.

## What is in it

| Part | Format | Bits |
|---|---|---|
| Routed experts (layers 3-77, 256 each) | EXL3 trellis, `mul1` codebook, **a width per expert** | 2-bit: 5,436 experts · 3-bit: 13,128 · 4-bit: 636 → **mean 2.75** |
| MTP layer (78) experts | EXL3 `mul1` | 8 |
| Attention, shared experts, dense MLPs (layers 0-2), indexer `wq_b` | EXL3 `mul1` (`exl3_dense`) | 5 |
| `kv_b`, indexer `wk` / `weights_proj`, router, norms, embeddings, `lm_head` | BF16 | 16 |

- 81 safetensors shards, 258 GB. `kv_a_proj_with_mqa` is stored 640 wide (576 real outputs + zero padding to the
  128-column tile).
- Per-expert widths are listed in `quantization_config.json` (`expert_bits`, format `keys35-v1`); the allocation gives
  more bits to experts whose quantization error hurt the layer output most (measured on calibration activations).
- `KEYS35_ASSEMBLY_REPORT.json` records the assembly checks.

## Quality

| Metric | Value |
|---|---|
| Perplexity, prose (12,223 tokens) | 3.560 |
| Perplexity, code (3,882 tokens) | 2.913 |
| Same with 5-bit non-expert layers replaced by FP8 | 3.567 / 2.911 (a tie: EXL3 is kept, it is faster) |
| KL divergence vs BF16 (exllamav3 `model_diff`, 32 × 2048 = 65,536 tokens) | **0.124 nats** (reverse 0.146; per-token median 0.017, p90 0.30) |
| Top-1 agreement with BF16 | **89.6 %** (top-2 62.7 %) |
| Perplexity on the same tokens | 2.938 (BF16 2.750) |
| Label in top-5 | 91.2 % (BF16 91.6 %) |

## Alternative: vLLM, tensor-parallel 4 across four DGX Sparks (batched serving)

This checkpoint needs EXL3 support that stock vLLM does not have yet. Our serving image (not yet public) is vLLM with:

- **exllamav3 1.5.0** kernels; a patched cuda-exl3 `exl3_moe_gemm` that accumulates into its output (`out=`);
- an `exl3_mixedk` MoE method: experts grouped by width for prompt chunks, and for decode windows (≤ 32 rows)
  [TensorFold](https://github.com/ashhart/TensorFold)'s universal EXL3 experts kernel (one launch per projection,
  a width per expert; vendored at v0.3.6.3 with a row-stride patch);
- an `exl3_dense` linear method for the 5-bit non-expert layers;
- MTP speculative decoding (k = 2), FULL CUDA graphs, fused all-reduce + RMSNorm, and a RoCE one-shot all-reduce for
  decode-sized messages (`VLLM_ROCE_ALLGATHER_MAX_SIZE=4MB`; larger messages stay on NCCL).

Key settings (TP = 4): `--kv-cache-dtype fp8 --max-model-len 200000 --max-num-seqs 8
--kv-cache-memory-bytes 14092861440 --async-scheduling --compilation-config.pass_config.fuse_allreduce_rms=true
--cudagraph-capture-sizes 1 2 3 4 6 8 12 24 --speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
Weights take 67.5 GiB per rank.

### Measured on 4 × DGX Spark (single stream, 32K-token context, temperature 1.0, top-p 0.95, 512 tokens)

| Profile | Prose | Code | Mixed | Tokens / pass | Pass |
|---|---|---|---|---|---|
| 200K context, RoCE all-reduce (standing) | **23.8 tok/s** | **30.9 tok/s** | 26.1 | 1.88 | 79 ms |
| 1M context (`--max-model-len 1000000 --decode-context-parallel-size 4`, 1.23M-token pool) | 16.6 | 21.1 | 18.1 | 1.82-2.28 | 108-110 ms |

Prompt processing (200K profile): 749 tok/s at 4K, 733 at 32K, 668 at 128K tokens (TTFT 195 s at 128K). The 1M
profile passed a needle at 128K (409 tok/s prefill at 32K). Prefill is the open problem: we expect it can reach
~2× this (rank skew in the all-reduce and the MoE prefill kernel are the two measured causes).

## 6-bit output head (optional, added 2026-10-12)

`lm_head-6bit/lm_head-6bit.safetensors` (714 MB) is `lm_head.weight` quantized to EXL3 at 6.0 bits with exllamav3
(tensors `lm_head.trellis`, `lm_head.suh`, `lm_head.svh`, `lm_head.mul1`). The checkpoint's own `lm_head.weight` stays
bf16 and nothing else in the repository changed.

The `2026-10-12-mlx-serve` image of the recipe reads the head from this file when it is in the checkpoint folder
(`hf download` puts it there), and from the bf16 tensor when it is not or with `HEAD_Q=off`. Measured on four DGX
Sparks, as served, against the BF16 checkpoint over 137,490 positions:

| | bf16 head | 6-bit head |
|---|---|---|
| Head launch on a Spark | 2.16 ms | 0.79 ms |
| KL, all sets | 0.175612 | 0.175724 |
| KL, tool calls | 0.559064 | 0.559098 |
| Top-1 agreement | 89.64% | 89.60% |
| Decode, 32K x 4, one stream, thinking on | 38.1 prose / 56.3 code tok/s | 39.6 / 57.7 |

Other engines ignore the folder.

## Credits

- [Z.ai](https://huggingface.co/zai-org) — GLM-5.3.
- [Ash Hart / TensorFold](https://github.com/ashhart/TensorFold) and the TensorFold contributors (Apache-2.0; earlier
  code MIT) — the engine, its EXL3 kernels and the `glm5_next` family the serving engine builds on.
- [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) (Apache-2.0) — GLM prompt-kernel
  designs and the decode, sampling, stop, rail, rank-check and prompt-reuse patches adapted here.
- [Jay Leaton](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache-2.0) — the L2 prefetch and trellis-load
  work behind MiaAI-Lab's 0046 / 0047.
- [BertholomusAI (Albert Lee)](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10) (Apache-2.0) — decode side
  stream, MTP index reuse, concurrent-stream draft cut and warm-up capture (ideas re-implemented), head-to-head prompts.
- [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT) — EXL3; vcruz305 — the per-expert
  mixed-width EXL3 work this quantization builds on; cuda-exl3 — the grouped prompt GEMM.
- [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0) — RoCE one-shot collectives.
- [incoai](https://huggingface.co/incoai) — the optional DFlash2 draft (CC BY-NC-ND 4.0, not redistributed).
- NVIDIA — DGX Spark, CUDA, NCCL; Anthropic's Claude (Claude Code) — engineering assistance.

Use is subject to the GLM-5.3 license of the base model.
