---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- openflowlm
- amd-npu
- xdna2
- strix-halo
---

# Qwen3.8-27B for the AMD Strix Halo NPU (OpenFlowLM)

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) packed for the OpenFlowLM (`oflm`) runtime on AMD Strix Halo (XDNA 2 NPU + CPU). Everything the runtime needs is in this repo: weights, MTP draft weights and the NPU kernels.

| | |
|---|---|
| Prompt | 380-760 tok/s (NPU) |
| Decode | 15-23 tok/s (CPU + NPU, MTP drafts, prompt lookup) |
| Memory | about 19 GB |

## Run

```bash
oflm pull qwen3.8:27b
oflm serve qwen3.8:27b        # OpenAI-compatible API on port 52625
```

CPU decode with MTP drafts turns on by itself when `mtp.safetensors` is present. For the fastest decode, send `"frequency_penalty": 0` in requests: the default penalty moves picks off the top token, so fewer drafts hold (about 10 % slower).

## Files

| File | What it is |
|---|---|
| `model.q4nx` | Weights, Q4NX (q4_1 / q8_0 / bf16), from [Atomic-Germ](https://huggingface.co/Atomic-Germ)'s conversion |
| `mtp.safetensors` | The model's MTP head (`mtp.*` tensors, bf16), from the original checkpoint |
| `open_kernels/` | NPU kernel set (int4 GEMMs, causal flash attention, decode GEMV), built from [OpenFlowLM-Next](https://github.com/jalbrethsen-highflame/OpenFlowLM-Next/tree/feat/qwen27b-causal) `open_kernels/` (branch `feat/qwen27b-causal`) |
| `config.json`, `tokenizer*.json`, `chat_template.jinja` | Runtime config, tokenizer, chat template |

## Warning: GPU and NPU together

On Strix Halo boards with AGESA 1.0.0.2, a GPU compute job (ROCm, HIP) that runs at the same time as NPU work can hang the machine ([ROCm/TheRock#8709](https://github.com/ROCm/TheRock/issues/8709); fixed in AGESA 1.0.0.2b). Do not run GPU jobs while this model runs.

## License

Apache-2.0, as the base model. See the [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B).
