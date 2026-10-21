---
license: other
license_name: glm-5.3-license
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
base_model: zai-org/GLM-5.3
base_model_relation: quantized
quantized_by: bot-lab-21
pipeline_tag: text-generation
library_name: vllm
tags: [glm, glm-5.3, mtp, speculative-decoding, draft-model, compressed-tensors, int8, int4, gptq, pollard, vllm]
---

# GLM-5.3 MTP Draft — layer 78, GPTQ int8 attention / int4 experts (standalone)

The model-author **multi-token-prediction layer of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)** (layer 78: full
MoE decoder block + `eh_proj`/`enorm`/`hnorm` + shared-head norm), packaged as a **standalone speculative draft** for vLLM
`method: mtp`: 13 GB, `compressed-tensors` `pack-quantized`, with the embeddings and `lm_head` it needs to load on its own.
Quantized with the **Pollard-method GPTQ** cook (full-Hessian error feedback, per-expert routed Hessians, in-domain calibration)
at the same allocation as the int8/int4 production draft it replaces — attention **int8** g128, routed experts **int4** g128,
norms/router/indexer bf16.

Companion body: [bot-lab-21/GLM-5.3-Int4-Int8Mix-Pollard-GPTQ](https://huggingface.co/bot-lab-21/GLM-5.3-Int4-Int8Mix-Pollard-GPTQ)
(the same file set ships inside it as `mtp-draft-gptq-int8/`; this repo exists so `speculative-config.model` can be an HF id).

## Why this draft

Same body (int4/int8 GPTQ GLM-5.3), same day, same replay of real traffic, k = 5 adaptive, 8 × DGX Spark:

| layer-78 draft | accepted draft tokens / step |
|---|---|
| round-to-nearest int8 attention (previous production draft) | 1.50 |
| the body's own layer 78 (GPTQ, int4 attention) | 1.59 |
| stock weights, bf16 (unquantized) | 1.64 |
| **this repo (GPTQ, int8 attention)** | **1.84** |

Hessian rounding on the draft layer alone is worth **+22 % acceptance** at identical bits, and a draft quantized the same way as its
target beats an unquantized bf16 copy of the same weights. The probe drifts ~0.3 day to day on a fixed config; compare within a
session. The stock (un-finetuned) head weights are used — six finetuned heads we trained all served worse than stock on this body.

## Usage (vLLM ≥ 0.28, with the companion body)

```bash
vllm serve bot-lab-21/GLM-5.3-Int4-Int8Mix-Pollard-GPTQ --tensor-parallel-size 8 --trust-remote-code \
  --max-model-len 900000 --kv-cache-dtype fp8 \
  --speculative-config '{"model":"bot-lab-21/GLM-5.3-Int4-Int8Mix-Pollard-GPTQ-MTP-Draft","method":"mtp","quantization":"compressed-tensors","num_speculative_tokens":3,"draft_tensor_parallel_size":1}'
```

It should also draft for other int4/int8 GLM-5.3 bodies (the head weights are the author's); acceptance will differ with the body's
quantization noise — measure with a replay of your own traffic.

## Files

`draft-00001…00004.safetensors` (2,348 tensors: `model.layers.78.*`, `model.embed_tokens.weight`, `lm_head.*`, `model.norm.weight`),
`model.safetensors.index.json`, `config.json` (text config, `num_nextn_predict_layers: 1`, `compressed-tensors` groups + ignore rules),
`generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`.

## License / provenance

Derivative of zai-org/GLM-5.3 (GLM-5.3 License, © 2026 Z.AI). Quantization and measurement by bot-lab-21; quantizer core derived
from Pollard Weights' `pollard_gptq` (Apache-2.0). Produced with the help of an AI assistant operating the cluster.
