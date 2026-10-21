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
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731-Int4-FP8

Mixed-precision quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

**Quant format** (`compressed-tensors`):

- Routed experts (`ffn.experts.*`): **INT4 W4A16**, group_size=32, symmetric, `pack-quantized`
- Dense (`attn.wq_*`, `wkv`, `wo_*`, `indexer.wq_b`, `shared_experts`, `main_proj`): **FP8 W8A16**, e4m3 + fp32 block scales, 128×128 blocks, `float-quantized`
- Embeddings, norms, gating, `hc_*`, indexer, `lm_head`: **BF16** passthrough
- KV cache: FP8, tensor-symmetric

**Calibration** (1 500 records):

- [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) `SFT/OpenThoughts3_1.2M_think` — 336
- [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) `train` — 332
- [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) `openhands/qwen35_122b` — 236
- [zai-org/LongCite-45k](https://huggingface.co/datasets/zai-org/LongCite-45k) `train` — 232
- [caskcsg/NExtLong-128K-dataset](https://huggingface.co/datasets/caskcsg/NExtLong-128K-dataset) `train` — 232
- [caskcsg/NExtLong-512K-dataset](https://huggingface.co/datasets/caskcsg/NExtLong-512K-dataset) `train` — 132

**Quality** (per-expert rel-L2 vs FP): mean **1.47 %**, w13 **0.96 %**, w2 **1.68 %**, dead experts 2 750 / 27 136 (10.1 %), worst layer 1 at 3.16 %, worst single expert layer-3/exp-207 w2 at 4.31 %.

**Size**: 165 GB, 46 safetensors shards.

**Usage**:

```bash
vllm serve BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8 \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention-config '{}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
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
