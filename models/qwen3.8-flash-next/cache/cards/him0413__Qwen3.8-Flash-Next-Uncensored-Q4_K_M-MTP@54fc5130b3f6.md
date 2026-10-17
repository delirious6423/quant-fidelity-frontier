---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - orcarouter/Qwen3.8-Flash-Next-Uncensored
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
language:
  - en
  - zh
tags:
  - qwen
  - qwen4
  - qwen3.8
  - flash-next
  - uncensored
  - abliterated
  - moe
  - gguf
  - llama.cpp
  - speculative
  - mtp
  - vision-language
---

# Qwen3.8-Flash-Next-Uncensored Q4_K_M (Integrated MTP)

An **integrated MTP (Multi-Token Prediction) GGUF** build of the abliterated
`Qwen3.8-Flash-Next-Uncensored` model, quantized to **Q4_K_M** with the MTP
draft head **merged into the same 4-shard split** — no sidecar file needed.

## Model Details

| Property | Value |
|---|---|
| Base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Abliteration | [`orcarouter/Qwen3.8-Flash-Next-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) |
| Quant | Q4_K_M (main trunk) + Q4_K_M (MTP head) |
| Split | 4 shards (3 trunk + 1 MTP), integrated |
| Total size | ~113.7 GiB |
| Context | 262K native |
| Architecture | qwen4exp (Gated DeltaNet + QSA + HyperConnections + PLE) |
| MTP | 1 draft layer, integrated as `blk.48` |

## Usage (llama.cpp with qwen4exp support)

```bash
./build-vulkan/bin/llama-server \
  --model Qwen3.8-Flash-Next-Uncensored-Q4_K_M-MTP-00001-of-00004.gguf \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-adaptive \
  --spec-draft-n-min 0 \
  --spec-draft-n-max 7 \
  --spec-draft-p-min 0.75
```

## What's inside

This repo contains the MTP head tensors fused into the main GGUF:

- `blk.48.nextn.*` — MTP embedding/hidden projections + norms
- `blk.48.nextn.hc_head_*` — draft head output mixer
- Full 48-layer trunk with `nextn_predict_layers = 1`

## ⚠️ Disclaimer

This model is an **abliterated (refusal-removed)** build. It will comply with
harmful, unethical, or illegal requests the original `Qwen3.8-Flash-Next`
would refuse. Released **strictly for legitimate research** — interpretability,
AI-safety / refusal-mechanism study, red-teaming, and robustness evaluation.
**You assume full responsibility** for how you use it and everything it
generates; add your own safety and moderation layers before any deployment.

## License

**Qwen Community License 1.0** (see [LICENSE](LICENSE)), inherited from the
base model [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Abliteration and quantization do not change the underlying license obligations.

Note: per the Qwen Community License, if you operate a **Model as a Service**
or **AI Work Assistant** business commercially, you must obtain a separate
license from Qwen before using this model or its derivatives for commercial
purposes. See LICENSE for full terms.
