---
base_model:
- Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
pipeline_tag: image-text-to-text
tags:
- gguf
- qwen4_exp
- multimodal
- abliteration
---

## Important

**This is an experimental derivative meant for research purposes only in sandboxed environments. It is not intended for use in production applications.**

Reducing a measured refusal direction may weaken safeguards and may also introduce unrelated quality regressions. It is
not a guarantee of unrestricted behavior or improved capability. 

By downloading and using this model you accept all responsibility and hold harmless myself and Alibaba/Tongyi Labs for damage as a consequence of use not in line with its intended purpose.


# Qwen3.8-Flash-Next-Abliterated-s1.5 GGUF

Experimental GGUF builds of an abliterated derivative of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
The base model is pinned to revision
`f5d08274bafd880402bd16f5e3e6c514136ec06c`.

The derivative applies a strength `1.5` refusal-direction projection to 146
architecture-audited tensors. The vision tower, MTP head, LM head, and Gated
Residual controllers were left unchanged.

## Available files

| Quantization | Approximate size | Notes |
|---|---:|---|
| `Q8_0` | 175.29 GiB | Highest-fidelity quantized build |
| `Q6_K` | 156.13 GiB | High-fidelity reduced-size build |
| `Q5_K_M` | 124.89 GiB | Balanced quality and size |
| `Q4_K_M` | 110.96 GiB | Smaller general-purpose build |
| `IQ4_XS` | 91.66 GiB | Smallest provided build |

Each quantized model is provided as one monolithic GGUF. For image input,
download `Qwen3.8-Flash-Next-Abliterated-s1.5-mmproj-BF16.gguf` as well and
place it beside the model file.

## Model information

- Architecture: `qwen4exp`
- Native Context length: 262,144 tokens
- Parameters: 125B A6B + 51B n-gram embedding and 4B MTP parameters
- Experts: 512 total, 10 active per token
- Multimodal vision projector included separately
- Features: Qwen Sparse Attention, Gated DeltaNet, Gated Residual, PLE n-gram embeddings, MTP, tool use, and configurable reasoning
- Chat template: embedded multimodal, tool-use, and reasoning template
- Qwen Community License 1.0

## Running locally

When running llama.cpp directly, selection is explicit:

```
llama-server \
  -m Qwen3.8-Flash-Next-Abliterated-s1.5-Q8_0.gguf \
  --mmproj Qwen3.8-Flash-Next-Abliterated-s1.5-mmproj-BF16.gguf
```

Or use a compatible llama.cpp-based application (such as LM Studio). Select one
quantized GGUF and its BF16 vision projector (if not automatically selected). Start with a conservative context
length for your available memory and increase it as needed.

## Validation

The quantized models were structurally checked and tested with real server
startup, health, and text-generation requests. Subsequent evaluation revealed abliteration was successful without noticeable performance / capability loss.

## License
The upstream Qwen Community License 1.0 remains applicable.