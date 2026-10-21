---
language:
- multilingual
license: apache-2.0
library_name: mlx
base_model: Qwen/Qwen3.8-27B
tags:
- qwen3.8
- qwen3.5
- mlx
- apple-silicon
- quantized
- oqe
- multimodal
- vision
- text-generation
- chat-template
---

# Qwen3.8-27B oQ6e MTP (MLX)

This repository contains an Apple Silicon / MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

It is a **community conversion**, not an official Qwen release. The original model, model behavior, and upstream documentation belong to Qwen / Alibaba Cloud. This repository packages quantized weights and compatible metadata for local MLX inference.

## What is included

- **Base model:** `Qwen/Qwen3.8-27B`
- **Format:** MLX safetensors
- **Quantization:** oQ6e mixed precision
  - Default: 6-bit affine quantization
  - Group size: 64
  - Selected sensitivity-critical tensors retained at 8-bit affine precision
- **Calibration:** OQE importance-matrix calibration using `oqe_code_multilingual`
  - 128 samples
  - Sequence length: 512
  - 615 linear modules captured
  - 504 importance-matrix entries recorded
- **MTP:** The upstream one-layer MTP metadata is retained (`mtp_num_hidden_layers: 1`).
- **Architecture:** `qwen3_5` multimodal architecture with image and video token support.
- **Files:** 5 model shards / 2,209 indexed tensors
- **Repository payload:** approximately 23.74 GB on disk (23.72 GB of safetensors).

## Chat template

This package includes the **Qwen Sharp Chat Template v22** from [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates).

The template is installed in both locations used by common MLX/OpenAI-compatible runtimes:

- `chat_template.jinja`
- `tokenizer_config.json` → `chat_template`

It preserves Qwen vision placeholders and tool-call formatting while adding direct-answer guidance and configurable thinking controls. The bundled template version marker is:

```text
qwen3.8-froggeric-v22
```

## Tested runtime

This artifact was loaded successfully with **oMLX** on Apple Silicon and served through its OpenAI-compatible chat-completions endpoint.

After placing the repository in your oMLX model directory, select this exact model ID:

```text
Qwen3.8-27B-oQ6e-mtp
```

Example request shape:

```json
{
  "model": "Qwen3.8-27B-oQ6e-mtp",
  "messages": [
    {"role": "user", "content": "Explain the difference between a cache and a database."}
  ]
}
```

## Important notes

- This is a **quantized inference artifact**, not a training checkpoint.
- Quantization can change quality, latency, memory use, and tool/structured-output behavior relative to the upstream BF16 model.
- No formal benchmark suite is claimed for this conversion. Please evaluate it for your own prompts, languages, vision workloads, tool calls, and context lengths before production use.
- The upstream model advertises a 262,144-token maximum context. Practical context length depends on your MLX runtime, available unified memory, concurrent requests, and KV-cache configuration.
- For long agent/tool sessions, validate your runtime's `preserve_thinking` behavior; retaining reasoning history can increase prefill cost.

## Provenance and attribution

- **Upstream model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Upstream license:** Apache License 2.0
- **Chat template:** [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)
- **Template lineage:** Qwen Sharp / froggeric fixed-template family; see the linked template repository for its current documentation and attribution.

Please retain upstream attribution and include the applicable Apache-2.0 license and notices when redistributing this derivative.

## License

The upstream Qwen3.8-27B release is licensed under **Apache-2.0**. This repository is distributed under the same license, subject to the upstream license terms and required attribution/notices.

## Acknowledgements

Thanks to Qwen / Alibaba Cloud for the original model release, the MLX ecosystem for Apple Silicon inference tooling, and the Qwen Sharp template maintainers for the compatible conversation template.
