---
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags: [fp8, compressed-tensors, vllm, moe, quantized]
---
# gemma-4-26B-A4B-it — FP8 Dynamic

FP8-dynamic (`compressed-tensors`) quantization of `google/gemma-4-26B-A4B-it` (MoE, vision+audio),
via [llm-compressor](https://github.com/vllm-project/llm-compressor). The MoE expert + attention
Linear layers are FP8; the **vision tower, audio stack, and MoE routers are kept in bf16** (a
quantized vision tower / router is not vLLM-servable). Verified serving + tool-calling on vLLM 0.23.0:

```bash
vllm serve huseinzolkepliscicom/gemma-4-26B-A4B-it-FP8-Dynamic \
  --tensor-parallel-size 1 --tool-call-parser gemma4 --enable-auto-tool-choice \
  --reasoning-parser gemma4 --gpu-memory-utilization 0.9
```
