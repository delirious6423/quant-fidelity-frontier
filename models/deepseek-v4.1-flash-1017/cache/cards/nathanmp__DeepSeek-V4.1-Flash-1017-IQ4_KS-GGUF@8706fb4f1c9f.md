---

base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
license: mit
library_name: gguf
pipeline_tag: image-text-to-text
tags:

- gguf
- quantized
- deepseek
- deepseek-v4.1
- iq4_ks
- q8_0
- ik_llama.cpp
- delphi.cpp
- mixture-of-experts
- multimodal
- speculative-decoding

---

# DeepSeek-V4.1-Flash-1017 — IQ4_KS / Q8_0 mixed GGUF

A mixed-precision GGUF quantization of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), with IQ4_KS routed experts and predominantly Q8_0 dense tensors.

Includes separate GGUF files for the optional DSpark speculative-decoding draft and vision projector.

## Compatibility

> Important: This model is NOT compatible with mainline llama.cpp.
>
> This quantization uses `IQ4_KS` tensor types from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), which are not supported by upstream llama.cpp.
>
> Confirmed working runtime: [delphi.cpp](https://github.com/nathanmp/delphi.cpp).
>
> Compatibility with `ik_llama.cpp` itself has not been tested. It is expected to work with a build that supports the required DeepSeek V4.1 Flash architecture and operations, but this remains unverified.

### Recommended runtime: delphi.cpp

[delphi.cpp](https://github.com/nathanmp/delphi.cpp) is a fork of llama.cpp focused on running large mixture-of-experts models efficiently on machines with substantially more system RAM than VRAM.

Notable features include:

- Support for DeepSeek V4.1 Flash, including its specialized attention mechanisms, engrams, and hyper-connections.
- Support for `ik_llama.cpp` quantization types, including `IQ4_KS`.
- NUMA-aware CPU execution for routed experts.
- GPU hot-expert caching and weight streaming.
- DSpark speculative decoding.
- Multimodal inference with a separate vision projector.
- Optimizations for large MoE models spanning CPU and GPU memory.

For installation instructions, tuning information, and additional capabilities, see the [delphi.cpp GitHub repository](https://github.com/nathanmp/delphi.cpp).

## Quantization details

This model uses a heterogeneous quantization scheme.

The primary goal is to preserve higher precision in the dense components while reducing memory requirements for routed experts.

### Tensor precision

| Tensor group                                     | Quantization          |
| ------------------------------------------------ | --------------------- |
| Routed expert gates (`ffn_gate_exps`)            | IQ4_KS                |
| Routed expert up projections (`ffn_up_exps`)     | IQ4_KS                |
| Routed expert down projections (`ffn_down_exps`) | IQ4_KS                |
| Attention tensors                                | Q8_0, mostly          |
| Attention compressor                             | Q8_0                  |
| Shared experts (`ffn_*_shexp`)                   | Q8_0                  |
| Hyper-connection tensors (`hc_*_fn`)             | Q8_0                  |
| Token embeddings (`token_embd`)                  | Q8_0                  |
| Output projection (`output`)                     | Q8_0                  |
| Router (`ffn_gate_inp`)                          | BF16                  |
| Indexer attention key (`indexer.attn_k`)         | BF16                  |
| Indexer projection (`indexer.proj`)              | BF16                  |
| Indexer query (`indexer.attn_q_b`)               | Q8_0                  |
| Engram embeddings (`engram_embd`)                | Q4_K                  |
| Engram WKV (`engram_wkv`)                        | Q8_0                  |
| Engram Q/K (`engram_q`, `engram_k`)              | BF16                  |
| Normalization and other small tensors            | F32                   |

All routed experts across the model's 40 layers use `IQ4_KS`.

### Quantization rationale

Routed expert weights represent a substantial portion of the model's parameters and memory requirements.

Using `IQ4_KS` for these tensors reduces their storage footprint, while retaining `Q8_0` or higher precision for dense operations, attention, shared experts, embeddings, and other sensitive components.

This configuration is particularly relevant to inference systems that keep routed experts in system RAM while executing dense operations on GPUs.

## Files

### Main model

| Filename                                       | Description         |
| ---------------------------------------------- | ------------------- |
| `DeepSeek-V4.1-Flash-1017-00001-of-00002.gguf` | Main model shard 1, backbone parameters. |
| `DeepSeek-V4.1-Flash-1017-00002-of-00002.gguf` | Main model shard 2, engrams tables.      |

Both files are required.

Combined size: approximately 409 GB.

### Optional components

| Filename                                     | Description                             |
| -------------------------------------------- | --------------------------------------- |
| `DeepSeek-V4.1-Flash-1017-DSpark-Q8_0.gguf`  | DSpark speculative-decoding draft model |
| `DeepSeek-V4.1-Flash-1017-mmproj-BF16.gguf`  | Multimodal projector for image input    |
| `deepseek-ai-DeepSeek-V4.1-Flash-1017.jinja` | Chat template                           |

Total size, including the optional GGUF components: approximately 418 GB.

## Usage

### Installation
Clone and build [delphi.cpp](https://github.com/nathanmp/delphi.cpp) according to its documentation. It should also work with [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), but that's untested.

### Basic text inference

From the directory containing the downloaded GGUF files:

```bash
/path/to/delphi.cpp/build/bin/llama-server \
  -m DeepSeek-V4.1-Flash-1017-00001-of-00002.gguf \
  --chat-template-file deepseek-ai-DeepSeek-V4.1-Flash-1017.jinja \
  --spec-type draft-dspark --spec-draft-model DeepSeek-V4.1-Flash-1017-DSpark-Q8_0.gguf \
  --mmproj DeepSeek-V4.1-Flash-1017-mmproj-BF16.gguf \
  -c 65536 \
  --parallel 1
  --temperature 1 --top-p 1 --top-k 0 --min-p 0
```

The runtime automatically loads the remaining GGUF shard when both files are present.

This is a minimal invocation. Because of the model's size, hardware-specific memory placement and performance settings will generally be necessary.

Consult the [delphi.cpp documentation](https://github.com/nathanmp/delphi.cpp/tree/main/docs/delphi/usage) for optimized configurations.

For vision-specific memory management and GPU placement, consult the [delphi.cpp vision documentation](https://github.com/nathanmp/delphi.cpp/blob/main/docs/delphi/usage/vision.md).

See the [delphi.cpp speculative-decoding documentation](https://github.com/nathanmp/delphi.cpp/blob/main/docs/delphi/usage/speculative-decoding.md) for complete usage examples.

To use DeepSeek's recommended sampling parameters, use `--temperature 1 --top-p 1 --top-k 0 --min-p 0`.

## Compatibility and testing status

Runtime compatibility:

- delphi.cpp: Confirmed working by the quantization author.
- ik_llama.cpp: Not tested with this particular model; expected to work with an appropriately updated build.
- Mainline llama.cpp: Not compatible due to unsupported quantization types.
- Other GGUF runtimes: Compatibility has not been established.

No independent quantization-quality benchmarks or perplexity measurements are provided here.

This is an independently prepared community quantization and is not an official DeepSeek release.

## Credits and license

[DeepSeek-AI — DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

The original model is released under the [MIT License](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE).