---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- autoround
- int4
- w4a16
- gptq
- qwen3_5
- intel
- xpu
---

# Qwen3.8-27B-Int4-AutoRound

INT4 (W4A16) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
produced with [Intel AutoRound](https://github.com/intel/auto-round) (0.15.0.dev, git main),
exported in the `auto_round` format (GPTQ-compatible packing, `quant_method: auto-round`).

**19 GB** on disk vs 55.6 GB BF16 (2.9× smaller). Tuned and validated on
2× Intel Arc B70 (Battlemage, 32 GB) with vLLM's XPU backend, TP=2.

## Quantization recipe

- `bits=4, group_size=128, sym=True`, `iters=200`, `nsamples=128`, `seqlen=2048`,
  calibration dataset `NeelNanda/pile-10k`, signed-rounding tuning (AutoRound default).
- **Kept in BF16 (unquantized):**
  - vision tower (`model.visual.*`) — weights are bit-identical to the base model
  - `lm_head` (untied, 1.27B params)
  - GDN linear-attention low-rank gates (`linear_attn.in_proj_a` / `in_proj_b`)
  - MTP speculative-decoding tensors — preserved unmodified in
    `model_extra_tensors.safetensors` for stacks that can use them
- 400 of 607 candidate linear layers quantized; per-layer exclusions are recorded in
  `quantization_config.extra_config` and mirrored in `quantization_config.json`.

## Quality — measured against an FP8 baseline of the same base model

Both sides served by the same vLLM build, identical harness, greedy decoding,
thinking disabled:

| Eval | FP8 (online per-tensor W8A16) | This model (INT4) |
|---|---|---|
| HumanEval pass@1 | 92.7 | **93.9** |
| HumanEval+ pass@1 | 90.9 | **89.6** |
| GSM8K 5-shot (raw-completions harness) | 38.4 | **43.2** |

Coding performance is statistically indistinguishable from the FP8 baseline (n=164,
±2.3 pts). The low GSM8K absolutes on **both** sides are an artifact of the raw few-shot
completions harness (roughly half the responses unparseable for both models); the
A/B delta favors this model. Text-only validation — the (unmodified) vision tower was
not exercised in these evals.

## Speed — 2× Intel Arc B70 32GB, vLLM XPU backend, TP=2

| Metric | FP8 online quant | This model |
|---|---|---|
| Single-stream decode | 32.9 tok/s | **51.9 tok/s** (+58%) |
| Batched, 32 concurrent | 167 tok/s | **206 tok/s** (+23%) |
| Median TTFT (1K-token prompt) | 431 ms | 478 ms |

## Serving

### Docker (recommended on Intel Arc B-series)

The tuned serving stack this model was validated with ships as a Docker image:
[`cyspiegel/vllm-xpu-b70`](https://hub.docker.com/r/cyspiegel/vllm-xpu-b70)
(vLLM XPU fork + custom GDN kernels, MTP speculative decoding, in-image Intel
compute runtime). It downloads this checkpoint on first start and derives the fast
GPTQ-kernel variant automatically:

```bash
docker run -d --name vllm-b70 --privileged --device /dev/dri --ipc host --shm-size 16g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v /path/to/models:/models \
  -e MODEL=CySpiegel/Qwen3.8-27B-Int4-AutoRound \
  cyspiegel/vllm-xpu-b70:latest
```

Serves an OpenAI-compatible API on `:8000` as model name `Qwen3.8-27B-Int4` with MTP
speculative decoding enabled (`PRESET=int4-mtp`; use `PRESET=int4` to disable it).
Measured in-container on 2× Arc B70, TP=2: **57 tok/s** single-stream greedy
(13.1 ms/token), **199 tok/s** at 32 concurrent. Compose file and all knobs:
[`docker/b70/`](https://github.com/CySpiegel/vllm-intel/tree/main/docker/b70) in the
[CySpiegel/vllm-intel](https://github.com/CySpiegel/vllm-intel) fork.

### Plain vLLM

```bash
vllm serve <this-repo> --tensor-parallel-size 2 --language-model-only \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

The checkpoint loads through vLLM's `auto-round` (INC) integration out of the box, and
through Hugging Face `transformers` with the `auto-round` package installed.

**Intel XPU tip:** the fastest serve path in our testing was vLLM's GPTQ kernel
(`int4_gemm_w4a16`). The packed tensors are already GPTQ-layout, so a config-only
rewrite enables it: set `quantization_config` to
`{"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": true, "desc_act": false,
"lm_head": false}` (keep the shards unchanged). On Arc B70 this was +58% decode
throughput over FP8 online quantization; the unmodified `auto-round` path also works
but benchmarked slower at high concurrency on XPU.

## Provenance and license

Base model by the Qwen team, Apache-2.0. This derivative keeps the same license and
carries the original LICENSE file. Quantized with AI assistance (Claude Code) on
2026-08-24; recipe selection, benchmarks, and validation as described above. No
fine-tuning was performed — weights differ from the base model only by INT4
rounding-tuned quantization of the listed layers.
