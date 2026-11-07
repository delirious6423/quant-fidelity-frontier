---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
  - zh
tags:
  - red-ds4
  - glm
  - glm5
  - glm-5.3-flash
  - glm5-next
  - moe
  - nvfp4
  - fp4
  - compressed-tensors
  - vllm
  - vision-language
  - function-calling
  - reasoning
  - mtp
---

# red-ds4

NVFP4 (4-bit) weight quantization of [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — experts-only FP4, served on vLLM. This is the self-hosted `red` checkpoint.

## Model details

| | |
|---|---|
| **Base model** | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| **Architecture** | `Glm5NextForConditionalGeneration` (`glm5_next`) — 45 layers + 1 MTP, hidden 4096, hybrid attention (34 gated-linear KDA + 11 sparse full-attention), MLA, 288 routed experts top-8 + 1 shared, native vision + video |
| **Quantization** | NVFP4 (compressed-tensors `nvfp4-pack-quantized`) on the routed experts; BF16 elsewhere |
| **Format** | safetensors, `weight_packed` (uint8) + `weight_scale` (FP8-E4M3) + `weight_global_scale` (FP32) for experts; BF16 for the rest |
| **Size** | ~191 GiB |
| **Context** | 1,048,576 tokens |
| **Vocabulary** | 154,880 |

## What's quantized

| Component | Precision |
|---|---|
| Routed MoE experts (`mlp.experts.*.{gate,up,down}_proj`, layers 3–44) | NVFP4 (W4, E2M1 group-16 + FP8-E4M3 group scale + FP32 global scale) |
| Attention (MLA + KDA), shared experts, dense MLP (layers 0–2), router, sparse-attention indexer, embeddings, `lm_head`, mHC mixers, norms, vision tower | BF16 |
| mHC / SSM / router F32 aux tensors (`hc_*`, `A_log`, `dt_bias`, `e_score_correction_bias`) | FP32 |

- **Weight-only and data-free.** Expert weights are derived directly from the source checkpoint (symmetric FP4, per-group absmax); activations are quantized dynamically at runtime.
- **MTP head included.** Layer 45 ships as BF16 so the weights match the `num_nextn_predict_layers` the config declares.
- **KV cache is not quantized** (BF16 at runtime).

## Requirements

- A vLLM build with `glm5_next` support. Use `vllm/vllm-openai:glm53-flash-x86_64-cu129` or a build from vllm#53906.
- ~191 GiB of weights plus KV cache. Verified on 8×H100; Blackwell adds native FP4 tensor cores.
- Hopper does not support FP8 KV cache for this model — omit `--kv-cache-dtype fp8` there.

## Usage

```bash
docker run -d --name red-ds4 --gpus all --ipc=host -p 8000:8000 \
  -v /path/to/red-ds4:/model:ro \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e VLLM_KV_CACHE_LAYOUT=HND \
  vllm/vllm-openai:glm53-flash-x86_64-cu129 \
  --model /model --served-model-name red \
  --tensor-parallel-size 8 --max-model-len 8192 --enforce-eager --trust-remote-code
```

Control the thinking budget with `reasoning_effort` (`low` / `high` / `max`) and pass `clear_thinking: true` for chat.

## Evaluation

Measured on these weights against the FP8 source.

| metric | value |
|---|---|
| mean relative error (NVFP4 experts vs FP8) | 9.18% |
| cosine similarity | 0.9958 |
| PPL (wikitext-2) | 2.894 vs 2.788 FP8 (+3.8%) |
| KLD mean | 0.073 |
| Top-1 agreement | 91.7% |

## License

MIT, inherited from the base model. Quantization does not change the underlying license obligations.
