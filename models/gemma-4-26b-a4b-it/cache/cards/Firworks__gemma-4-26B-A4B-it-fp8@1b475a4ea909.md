---
base_model:
- google/gemma-4-26B-A4B-it
tags:
  - fp8
  - quantized
---
# gemma-4-26B-A4B-it-fp8

**Format:** FP8_DYNAMIC — weights quantized to FP8 statically; activations scaled dynamically at runtime.
**Base model:** `google/gemma-4-26B-A4B-it`
**How it was made:** One-shot datafree quantization with LLM Compressor (FP8_DYNAMIC recipe) on a DGX Spark (GB10 Grace Blackwell). No calibration data required — activations are scaled dynamically at runtime.

> Notes: `lm_head` and multimodal projection layers kept in high precision. Blackwell (GB10/B100/B200) has native FP8 hardware support. Hopper (H100/H200) also supports FP8 natively. Older architectures will fall back to BF16 compute while still benefiting from the reduced model size.

Check the original model card for information about this model.

# Running the model with vLLM in Docker
```sh
sudo docker run --runtime nvidia --gpus all -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model Firworks/gemma-4-26B-A4B-it-fp8 \
  --dtype auto \
  --max-model-len 32768
```
Tested on a DGX Spark (GB10 Grace Blackwell Superchip, 128GB unified memory).

If there are other models you'd like quantized to FP8, let me know.
