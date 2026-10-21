---
language:
- en
- zh
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- rapid-mlx
- qwen3.8
- qwen3.8-27b
- 4-bit
- mtp
- speculative-decoding
- apple-silicon
---

## Run it on your Mac

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
rapid-mlx serve qwen3.8-27b-4bit
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

Use it from OpenCode / Claude Code / Codex: `rapid-mlx agents <name> --setup` · [Agent setup](https://rapidmlx.com/docs/agents)

[Download the Desktop app](https://rapidmlx.com/download) · 32 GB unified memory recommended for the 16.3 GB target and MTP weights.

# Qwen3.8-27B 4-bit MLX + MTP


An official Rapid-MLX, self-contained MLX package of Qwen3.8-27B. It includes
both the 4-bit target weights and the matching native Multi-Token Prediction
(MTP) drafter in `model-mtp.safetensors`.

Users download one repository; the target and acceleration weights stay paired.
The MTP weights were derived from the same upstream Qwen checkpoint as the
target. Upstream model license and limitations continue to apply.

## Rapid-MLX

Use a Rapid-MLX release that supports Qwen3.8 native MTP. Rapid can load the
target and use the colocated `model-mtp.safetensors` as its MTP sidecar. MTP is
speculative decoding: accepted output remains identical to greedy
autoregressive decoding. Runtime should fall back to autoregressive decoding
when MTP is unavailable or is not beneficial on the current Mac.

```bash
rapidmlx serve rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX \
  --speculative-config '{"method":"mtp","model":"rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX"}'
```

## Validation

On an Apple-silicon Studio test system, a paired greedy smoke test produced the
same emitted-token hash with and without MTP. One measured prompt ran at 39.66
tok/s autoregressive and 46.73 tok/s with Rapid MTP (1.18x). Performance varies
by prompt and hardware; this is a compatibility result, not a universal speed
claim.

## Contents

- `model-*.safetensors`: Qwen3.8-27B MLX 4-bit target weights
- `model-mtp.safetensors`: matching MLX 4-bit native MTP drafter
- tokenizer, processor, generation, and model configuration files

The target is based on `mlx-community/Qwen3.8-27B-4bit`; the MTP drafter is
based on `mlx-community/Qwen3.8-27B-MTP-4bit`. Both originate from
`Qwen/Qwen3.8-27B`.
