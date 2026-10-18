---
license: mit
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: BlivionIaG
tags:
- deepseek
- deepseek-v4
- int4
- w4a16
- fp8
- moe
- compressed-tensors
- vllm
- reap
- pruned
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731-Int4-FP8-REAP-216B

Expert-pruned variant of [BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8](https://huggingface.co/BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8), a mixed-precision quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

**What changed:** routed experts pruned uniformly from **256 → 192 per layer** across all 43 MoE layers. Quantization formats, dense layers, and KV cache are unchanged from the source — see its card for quantization and calibration details.

**How K=192 was selected:** per-expert REAP saliency computed during INT4 calibration,

`0.7 × num_captured × (1 − rel_l2_error) + 0.3 × gate_l2_norm`

keeping the top-192 experts in every layer. The per-layer mask is stored in `reap_saliency.json`; surviving experts are compacted to contiguous indices 0..191.

**Parameters** (INT4 experts unpacked ×8):

| Component | Params |
|---|---:|
| Routed experts (192 × 43 × 25.17 M) | 207.8 B |
| Dense (attention, shared experts, indexer, norms, embeddings) | 7.8 B |
| **Total weights (≈ 216 B)** | **215.6 B** |
| Active per token (6 routed + shared + dense) | ≈ 12 B |

Quantization scales and index tables add ≈ 6.6 B of storage tensors (not counted as model parameters). Before pruning (K=256) the model is ≈ 285 B.

**Usage:**

```bash
vllm serve BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8-REAP-216B \
  --trust-remote-code --kv-cache-dtype fp8
```

**License**: MIT — same as [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

**Cite**:

```
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```
