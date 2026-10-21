---
license: gemma
library_name: transformers
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
tags:
  - gemma4
  - moe
  - vllm
  - compressed-tensors
  - gptq
  - w8a16
  - q8
---

# gemma-4-26B-A4B-it_q8

This repository contains a vLLM-compatible, expert-only W8A16 quantization of
`google/gemma-4-26B-A4B-it`.

## Quantization

- Quantized modules: routed MoE expert `gate_proj`, `up_proj`, and `down_proj`
- Weight format: signed INT8
- Activations: BF16 (W8A16)
- GPTQ group size: 32
- Symmetric, static weight quantization
- Attention, router, shared MLP, embeddings, LM head, and vision tower remain BF16
- Calibration: 32 coding-oriented samples, sequence length 2,048
- Checkpoint format: `compressed-tensors` / `pack-quantized`
- Weight shards: 7
- Weight size: approximately 28.13 GiB

The `compressed-tensors` ignore metadata includes the missing `v_proj` aliases
for Gemma 4 global-attention layers so that vLLM can consistently fuse Q/K/V.

## Tested configuration

- vLLM: 0.21.0
- GPU: NVIDIA A100 PCIe 40GB
- Context length: 8,192
- GPU memory while loaded: approximately 37.1GB
- Backend: `CompressedTensorsWNA16MarlinMoEMethod`
- OpenAI-compatible `/v1/models` and `/v1/chat/completions`: tested
- Observed generation throughput: approximately 32–42.5 tokens/s for a single short request

## Start with NVIDIA Docker

```bash
docker run -d \
  --name gemma4-vllm \
  --gpus '"device=0"' \
  --ipc=host \
  --ulimit memlock=-1 \
  -p 8000:8000 \
  -v "$PWD:/model:ro" \
  vllm/vllm-openai:v0.21.0 \
  --model /model \
  --served-model-name gemma-4-26B-A4B-it_q8 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90
```

## Tested rootless Podman command

```bash
docker run -d \
  --name gemma4-vllm \
  --userns=keep-id \
  --user "$(id -u):$(id -g)" \
  --device nvidia.com/gpu=1 \
  --ipc=host \
  --ulimit memlock=-1 \
  -p 18080:8000 \
  -v "$PWD:/model:ro,Z" \
  --entrypoint /usr/bin/python3 \
  localhost/vllm/vllm-openai:v0.21.0-cu129-ubuntu2404-media \
  -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name gemma-4-26B-A4B-it_q8 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90
```

Gemma 4 requires `--max-num-batched-tokens` to be greater than its multimodal
item token budget. The tested value is 4,096.

## OpenAI-compatible request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26B-A4B-it_q8",
    "messages": [
      {"role": "user", "content": "Write a Python two_sum function in O(n)."}
    ],
    "max_tokens": 256,
    "temperature": 0.2
  }'
```

## Notes

- A 40GB GPU is recommended for the tested 8K configuration.
- This is an expert-only Q8 checkpoint, not a full-model Q8 conversion.
- Text generation and code generation were tested. Multimodal inference was not
  functionally evaluated.
- The model produced coherent code during testing, but Traditional Chinese
  instruction following was weaker than expected and should be evaluated for
  your application.
- Usage remains subject to the Gemma license and the base model terms.
