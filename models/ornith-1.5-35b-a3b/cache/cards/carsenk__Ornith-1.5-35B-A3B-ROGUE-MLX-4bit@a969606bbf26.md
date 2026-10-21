---
language:
- en
library_name: mlx
license: mit
pipeline_tag: text-generation
model_name: Ornith-1.5-35B-A3B-ROGUE-MLX-4bit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
tags:
- mlx
- mlx-lm
- 4-bit
- quantized
- qwen3_5_moe
- mixture-of-experts
- moe
- 35b
- 3b-active
- rogue
- abliterated
- conversational
- not-for-all-audiences
---

# Ornith-1.5-35B-A3B-ROGUE-MLX-4bit

An MLX 4-bit build of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), processed with ROGUE to reduce refusal behavior while retaining measured utility.

ROGUE surgery was applied to the BF16 Hugging Face source before MLX quantization, including the model's fused routed-expert writers.

## About ROGUE

ROGUE is a refusal-direction weight-editing method. It estimates refusal-relevant directions from balanced harmful/harmless activation contrasts, uses held-out scoring and benign-subspace preservation to select intervention depths, and applies a low-rank orthogonal projection to residual-writing matrices. The edit is performed in floating point before quantization.

For Ornith, the residual-writer scope includes attention outputs and the fused routed-expert MLP down projections. The hidden-axis projection is broadcast across all 256 experts in each selected fused tensor. This changes the model weights directly; it is not a prompt, adapter, or inference-time filter. Evaluation remains model- and runtime-specific and does not guarantee universal compliance, unchanged capability, or safe output.

## Model details

- Architecture: Qwen 3.5 MoE, 35.95B total parameters / approximately 3B active parameters per token
- Layers and experts: 40 transformer layers, 256 routed experts
- Runtime: MLX on Apple silicon; tested with `mlx-lm 0.31.3` and `mlx 0.32.0`
- Weight format: affine 4-bit, group size 64; router and shared-expert gates remain 8-bit
- Download size: 19.5 GB across four safetensors shards
- Base license: MIT

## Conversion

- Source: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/10fbf86fed7ecee4a061f8b499a618f46001cac1) (BF16, revision `10fbf86fed7ecee4a061f8b499a618f46001cac1`)
- ROGUE scope: residual writers, attention + MLP
- Strength: 1.4
- Expert handling: fused 3-D `mlp.experts.down_proj` and `mlp.experts.gate_up_proj` writers
- Surgery coverage: 128 writer tensors across 32 selected layers, including 64 fused routed-expert tensors
- Output: MLX affine 4-bit, group size 64; router gates remain 8-bit
- Conversion software: WeightDojo ROGUE pipeline

The four published safetensors shards were verified against the evaluated checkpoint by size and LFS SHA-256 object hash.

## Evaluation

ROGUE Pareto v2 deterministic smoke suite, 40 prompts, MLX backend:

| Metric | Base MLX 4-bit | This model |
|---|---:|---:|
| Refusal-marker rate | 50.0% | 0.0% |
| Retain score | 95.8% | 91.7% |
| Over-refusal rate | 0.0% | 0.0% |
| Mean generation speed | 25.55 tok/s | 25.45 tok/s |

Uncensor gain was `0.50`, retain delta was `-0.0417`, and all configured smoke-gate thresholds passed. The full machine-readable report is included at [`evaluation/rogue-pareto-v2.json`](evaluation/rogue-pareto-v2.json).

This is a deterministic smoke test, not a leaderboard-grade safety or capability evaluation. Refusal classification uses phrase markers, and passing the gate does not guarantee unchanged quality on every task. Measurements were collected on an Apple M1 Ultra.

## Run locally

Install MLX LM:

```bash
uv tool install --upgrade mlx-lm
```

Generate from the Hub:

```bash
mlx_lm.generate \
  --model carsenk/Ornith-1.5-35B-A3B-ROGUE-MLX-4bit \
  --prompt "Explain how mixture-of-experts routing works." \
  --max-tokens 512 \
  --temp 0.6
```

Start an OpenAI-compatible server:

```bash
mlx_lm.server \
  --model carsenk/Ornith-1.5-35B-A3B-ROGUE-MLX-4bit \
  --host 127.0.0.1 \
  --port 8080
```

Then query it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "carsenk/Ornith-1.5-35B-A3B-ROGUE-MLX-4bit",
    "messages": [{"role": "user", "content": "Write a Python merge sort and explain its complexity."}],
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

## Limitations and responsible use

ROGUE intentionally changes refusal behavior. The model can produce inaccurate, unsafe, or objectionable output and is not suitable for every audience. Validate it against your own workloads and apply deployment controls appropriate to your use case.
