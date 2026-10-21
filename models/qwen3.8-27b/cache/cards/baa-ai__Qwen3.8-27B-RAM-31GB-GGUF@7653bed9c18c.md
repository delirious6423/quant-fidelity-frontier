---
library_name: gguf
tags:
  - gguf
  - quantized
  - mixed-precision
  - imatrix
  - qwen
  - qwen3
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
language: en
---

# Qwen3.8-27B — 31GB (GGUF, imatrix)

Mixed-precision, imatrix-calibrated GGUF of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), prepared by [baa.ai](https://baa.ai). This is the **language model** (text) GGUF; for the vision-preserving build use the [MLX sibling](https://huggingface.co/baa-ai/Qwen3.8-27B-RAM-24GB-MLX).

> ⚠️ **Requires a very recent llama.cpp (build b10360 / Aug 2026 or newer).** Qwen3.8 is a hybrid **Gated Delta Net** (linear-attention) + full-attention architecture (`arch: qwen35`). Support landed in llama.cpp only recently — **stable Ollama and LM Studio do not run this yet.** Use up-to-date `llama.cpp` built from source until downstream runtimes catch up.

## Files

| File | Quant | Size |
|------|-------|------|
| `Qwen3.8-27B-RAM-31GB.gguf` | Mixed (Q4_K–F16) + imatrix | 31.2 GB |

## Metrics

| Metric | Value |
|---|---|
| **Size on disk** | **31.2 GB** |
| Average bits per weight | 9.12 |
| Base type | Q4_K_M (per-tensor overrides via RAM spec) |
| Framework | llama.cpp (GGUF), arch `qwen35` |
| Calibration | Importance matrix (wikitext-2 + 200 MMLU-Pro, seed=99) |
| Source | Qwen/Qwen3.8-27B (BF16, 55.6 GB) |

### Actual tensor-type distribution (866 tensors)

| Type | Count | Role |
|---|---|---|
| F32 | 360 | Norms, biases, Gated Delta Net scalar params |
| Q8_0 | 190 | High-sensitivity RAM-allocated projections |
| Q4_K | 179 | Base type (low-sensitivity + fused tensors) |
| F16 | 114 | Probe-protected sensitive tensors |
| Q6_K | 23 | Medium-sensitivity projections |

> **On RAM allocation coverage:** llama.cpp *fuses* this architecture's attention and Gated Delta Net projections (`attn_qkv`, `ssm_alpha/beta/a/dt`) into tensors that don't map 1:1 onto RAM's per-tensor manifest. RAM's mixed-precision spec therefore applies cleanly to ~74% of weight bytes (the MLP bulk + several SSM tensors); the fused attention/SSM tensors receive the imatrix-calibrated base type. It's a *mostly*-RAM, imatrix GGUF — not a full per-tensor build. The MLX sibling has complete per-tensor allocation.

## Benchmarks

Quality benchmarks for this GGUF are pending. See the [MLX sibling](https://huggingface.co/baa-ai/Qwen3.8-27B-RAM-24GB-MLX) for MMLU (90.0%, matching BF16) and the *Fidelity Is Not Safety* agent-safety screen (PASS / RELIABLE) on the same underlying RAM allocation.

## Usage (llama.cpp — recent build required)

```bash
# Build a current llama.cpp from source (>= b10360) — brew/Ollama/LM Studio may lag.
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j     # add -DGGML_METAL=ON on Apple Silicon

# Download this GGUF
hf download baa-ai/Qwen3.8-27B-RAM-31GB-GGUF --include "*.gguf" --local-dir ./qwen3.8-ram

# Run (Qwen3.8 is a reasoning model — thinking enabled by default)
./build/bin/llama-cli -m ./qwen3.8-ram/Qwen3.8-27B-RAM-31GB.gguf \
    -p "Explain quantum entanglement in one paragraph." -n 512 -ngl 99

# OpenAI-compatible server
./build/bin/llama-server -m ./qwen3.8-ram/Qwen3.8-27B-RAM-31GB.gguf --port 8080 -ngl 99 --ctx-size 8192
```

Note: the Gated Delta Net (SSM) path is not yet fully Metal-offloaded, so generation is partly CPU-bound on Apple Silicon — expect modest tok/s until upstream optimizes it.

## Recommended inference settings

```
temperature: 0.7
top_p: 0.9
top_k: 20
max_tokens: 8192
```

## Quantization method

1. RAM probe allocator measures per-tensor sensitivity across bits 2–8 (random-input, data-free).
2. Path B knapsack re-optimizes allocations in GGUF type space; sensitive tensors held at F16/Q8_0.
3. Importance matrix from 100 chunks of wikitext-2 + 200 MMLU-Pro questions (seed=99, disjoint from eval).
4. `llama-quantize` applies the per-tensor spec with `--imatrix` (base type Q4_K_M for tensors outside the spec).

---
*Quantized by [baa.ai](https://baa.ai)*
