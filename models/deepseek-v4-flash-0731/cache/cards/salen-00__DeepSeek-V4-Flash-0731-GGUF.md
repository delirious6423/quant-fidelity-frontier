---
license: other
license_name: deepseek-license
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-0324/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- gguf
- deepseek
- deepseek-v4
- moe
- quantized
- ablation-guided
model_type: deepseek_v4
quantized_by: salen-00
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731-GGUF (hetero-v2)

Ablation-guided 'heterogeneous' GGUF quantization of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

**95.8 GB** single file. Fits 128 GB memory, leaving ~32 GB for KV cache (~150K tokens) to enable multi-turn sessions.

## Quantization schema

Quantization schema derived from the Deepseek V4 Pro model's quantization recommendations (hence, heterogeneous), refined with ablation signal analysis. I was curious if the models have been trained on and thus have 'internalized' their own architecture such that they could advise an effective distribution.

Three-tier expert quantization, ranked by confabulation-routing ablation signal from a differential heatmap over the model's own expert activation patterns:

| Tensor group | Quant | BPW | Layers / notes |
|---|---|---|---|
| Routed experts (tier 1, hottest) | Q4_K | 4.5 | Layers 32, 40 |
| Routed experts (tier 2) | Q3_K | 3.4 | Layers 13, 26, 36, 42 |
| Routed experts (tier 3, default) | Q2_K | 2.6 | 35 remaining MoE layers |
| Shared experts | Q6_K | 6.6 | Active every token; 1 per layer |
| Attention projections | Q6_K | 6.6 | MLA critical path (compressors, indexer, Q/KV/O) |
| Embeddings | Q8_0 | 8.0 | Input representation fidelity |
| Output head | Q8_0 | 8.0 | Logit accuracy |
| Norms, gates, hyper-connections | F32 | 32.0 | Tiny tensors; kept exact |

Approach is verified by DSL architecture documents regarding MoE routing and attention architecture. 

Layers 32 and 40 showed the highest confabulation-routing signal (routing decisions most likely to produce hallucinated output when degraded), so they get the most bits. Layers 13, 26, 36, 42 are the next tier. Everything else is sparse enough that Q2_K preserves quality.

## How to run

Requires llama.cpp with DeepSeek-V4 support. Mainline llama.cpp gained V4 support in mid-2026; earlier builds may need the [antirez fork](https://github.com/antirez/llama.cpp-deepseek-v4-flash).

```bash
llama-server \
  -m DeepSeek-V4-Flash-0731-hetero-v2.gguf \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --port 8080
```

Minimum context size is 4096 due to MLA compressor bank working set requirements. 32,768 is enough for standard reasoning tasks.

## Architecture

DeepSeek-V4-Flash is a Mixture-of-Experts transformer:

| Parameter | Value |
|---|---|
| Total parameters | ~284B |
| Activated per token | ~13B |
| Layers | 43 |
| Hidden size | 4096 |
| Attention heads | 64 |
| Routed experts | 256 per MoE layer, 6 active per token |
| Shared experts | 1 per layer (always active) |
| Vocabulary | 129,280 tokens |
| Max context | 1,048,576 tokens |
| Attention | MLA (Multi-head Latent Attention) with CSA + HCA hybrid |
| Expert routing | `noaux_tc` with `sqrtsoftplus` scoring + hash routing (3 layers) |
| KV compression | Per-layer ratios alternating 4x / 128x |

V4-specific features not in V3: hash routing tables (`ffn_gate_tid2eid`), MLA compressor banks, DSA indexer, hyper-connections (`hc_*`), attention sinks, grouped output projections.

## Conversion pipeline

Two-phase conversion from the FP4/FP8 source weights:

1. **Phase 1** (`convert_hf_to_gguf.py`): Dequantize FP4 experts and requantize to Q2_K/Q3_K/Q4_K by ablation tier. Base tensors (attention FP8, shared FP8, embeddings BF16) go to Q8_0.
2. **Phase 2** (`llama-quantize`): Requantize attention and shared expert Q8_0 down to Q6_K. Expert tiers and embeddings/output preserved via tensor-type override file.

## Performance notes

Tested on Apple M5 Max, 128 GB unified memory:

- Generation speed comparable to uniform-quant at same expert BPW (~30 t/s)
- Output quality at API parity on qualitative benchmarks (math, translation, code, explanation); see the [v1 report](https://github.com/ok-very/substrate/blob/main/bench-results/v4-flash-hetero-v1-report.md) for methodology. v2 refines the expert tiering but uses the same evaluation approach

## Credits

- **DeepSeek Labs** for the incredible open model 
- **Salvatore Sanfilippo (antirez)** for the original llama.cpp V4 fork and the `--deepseek4-expert-outtypes` conversion flag