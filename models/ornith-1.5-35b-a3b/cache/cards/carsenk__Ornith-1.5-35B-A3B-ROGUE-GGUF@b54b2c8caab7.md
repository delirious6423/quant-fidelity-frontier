---
language:
- en
license: mit
pipeline_tag: text-generation
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
tags:
- gguf
- llama.cpp
- qwen3_5_moe
- mixture-of-experts
- rogue
- abliterated
- conversational
---

# Ornith-1.5-35B-A3B-ROGUE-GGUF

llama.cpp GGUF quantizations of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), processed with ROGUE to reduce refusal behavior while retaining measured utility.

ROGUE was applied to the full-precision source before GGUF conversion and quantization. Ornith's fused three-dimensional routed-expert tensors were included in the transformation.

## About ROGUE

ROGUE is a refusal-direction weight-editing method. It estimates refusal-relevant directions from balanced harmful/harmless activation contrasts, uses held-out scoring and benign-subspace preservation to select intervention depths, and applies a low-rank orthogonal projection to residual-writing matrices. The edit is performed in floating point before conversion into the published runtime format.

For Ornith, the residual-writer scope includes attention outputs and the fused routed-expert MLP down projections. The hidden-axis projection is broadcast across all 256 experts in each selected fused tensor. This changes model weights directly; it is not a prompt, adapter, or inference-time filter. The measured evaluation below describes this release, but does not guarantee universal compliance, unchanged capability, or safe output.

## Files

| File | Intended use | Validation status |
|---|---|---|
| `Ornith-1.5-35B-A3B-ROGUE-Q8_0.gguf` | Highest-fidelity quantized option (34.37 GiB) | Load + coherent-generation smoke passed |
| `Ornith-1.5-35B-A3B-ROGUE-Q4_K_M.gguf` | Recommended balance (19.71 GiB) | Load, coherence, performance, and de-refusal smoke passed |
| `Ornith-1.5-35B-A3B-ROGUE-Q3_K_M.gguf` | Lower-memory option (15.61 GiB) | Load + coherent-generation smoke passed |
| `Ornith-1.5-35B-A3B-ROGUE-IQ2_M.gguf` | Calibrated extreme compression (10.86 GiB) | Load + coherent-generation smoke passed |
| `Ornith-1.5-35B-A3B-ROGUE-IQ1_M.gguf` | Experimental calibrated Q1-class option (7.67 GiB) | Load + coherent-generation smoke passed |

`Q4_K_M` is the recommended starting point. Q3, IQ2, and especially IQ1-class quantization can materially reduce reasoning and instruction-following quality; use them only when the memory savings are necessary.

## Model and conversion details

- Architecture: Qwen 3.5 MoE, approximately 35B total parameters / 3B active parameters
- Layers and experts: 40 transformer layers, 256 routed experts
- Base revision: `10fbf86fed7ecee4a061f8b499a618f46001cac1`
- Base license: MIT
- ROGUE scope: residual writers, attention + MLP
- ROGUE strength: 1.4
- Surgery coverage: 128 writer tensors across 32 selected layers, including 64 fused routed-expert tensors
- GGUF conversion and quantization: llama.cpp build `b10470-34af94cd9`

All published quantizations are derived from the same ROGUE GGUF source. The IQ2 and IQ1-class builds use an importance matrix for calibration.

## Evaluation

### ROGUE behavior evaluation

The ROGUE source artifact was evaluated with the deterministic 40-prompt ROGUE Pareto v2 smoke suite on the MLX runtime:

| Metric | Base MLX 4-bit | ROGUE MLX 4-bit |
|---|---:|---:|
| Refusal-marker rate | 50.0% | 0.0% |
| Retain score | 95.8% | 91.7% |
| Over-refusal rate | 0.0% | 0.0% |
| Mean generation speed | 25.55 tok/s | 25.45 tok/s |

Uncensor gain was `0.50`, retain delta was `-0.0417`, and the configured smoke-gate thresholds passed. These measurements characterize the ROGUE source artifact and are not presented as a GGUF-specific leaderboard evaluation.

### GGUF runtime validation

All five published files loaded and produced coherent benign generations with llama.cpp build `b10470-34af94cd9` on an Apple M1 Ultra. Clean, uncontended measurements were 70.3 tok/s for Q4_K_M, 62.6 tok/s for IQ1_M, and 59.8 tok/s for IQ2_M. Q8_0 and Q3_K_M also passed load/coherence checks; performance figures are reported only where an uncontended measurement was recorded.

Q4_K_M additionally received a held-out adversarial smoke test and produced no refusal marker at 63.7 tok/s. That behavior result is not automatically imputed to every lower-bit quantization. Generated harmful text is not included; the machine-readable report records aggregate classifications and output hashes where available.

These are deterministic smoke tests, not leaderboard-grade capability or safety evaluations. Passing them does not guarantee unchanged quality on every task. Results for the more compressed variants are listed only after each file loads and generates successfully.

Reports:

- [`evaluation/rogue-pareto-v2-mlx.json`](evaluation/rogue-pareto-v2-mlx.json)
- [`evaluation/gguf-runtime-smoke.json`](evaluation/gguf-runtime-smoke.json)
- [`SHA256SUMS`](SHA256SUMS)

## Run locally

Use the recommended Q4 file directly from the Hub:

```bash
llama-cli \
  -hf carsenk/Ornith-1.5-35B-A3B-ROGUE-GGUF:Q4_K_M \
  -ngl all \
  -c 8192 \
  --single-turn \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -p "Explain how mixture-of-experts routing works."
```

Start an OpenAI-compatible server:

```bash
llama-server \
  -hf carsenk/Ornith-1.5-35B-A3B-ROGUE-GGUF:Q4_K_M \
  -ngl all \
  -c 32768 \
  --host 127.0.0.1 \
  --port 8080
```

Query it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Ornith-1.5-35B-A3B-ROGUE-GGUF",
    "messages": [{"role": "user", "content": "Write a Python merge sort and explain its complexity."}],
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

You are responsible for how you deploy and use modified model weights.
