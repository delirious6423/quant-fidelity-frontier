---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- harness-hub/Qwen3.8-27B-opencode-adapted
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- apple-silicon
- opencode
- 4-bit
---

# Qwen3.8-27B OpenCode MLX Q4

An independently built Apple Silicon checkpoint for local coding with OpenCode. The OpenCode LoRA was trained by harness-hub. This release covers the merge, conversion to MLX, and local validation.

## Quickstart on Apple Silicon

With the Hugging Face CLI and `mlx-lm==0.32.2` installed, download the checkpoint. Allow about 16.1 GB of disk space.

```bash
hf download 0xlean/Qwen3.8-27B-OpenCode-MLX-Q4 \
  --local-dir ./qwen-opencode-mlx-q4
```

Start the local server. The prefill and cache settings below are the ones used in the OpenCode pilot on an M2 Max with 32 GB of unified memory.

```bash
mlx_lm.server --model ./qwen-opencode-mlx-q4 \
  --host 127.0.0.1 --port 18765 --temp 0 \
  --chat-template-args '{"enable_thinking": false}' \
  --kv-bits 4 --quantized-kv-start 0 --prefill-step-size 128 \
  --prompt-cache-bytes 536870912
```

Add this provider to your project's `opencode.json`. If the file already exists, merge the `provider` entry into it.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:18765/v1",
        "apiKey": "dummy"
      },
      "models": {
        "default_model": {
          "name": "Qwen3.8-27B OpenCode MLX Q4",
          "limit": { "context": 8192, "output": 1024 }
        }
      }
    }
  }
}
```

In OpenCode, select `local/default_model`. The LoRA is already merged. For LM Studio, place the entire checkpoint directory at `<models-directory>/local/qwen3.8-27b-opencode-mlx-q4/`.

## Source revisions

| Input | Pinned revision |
| --- | --- |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| [harness-hub/Qwen3.8-27B-opencode-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted) | `b4171c0069685320026d76eac96ccf716ca7e476` |

## Build

The downloaded source files passed Hugging Face checksum verification. The merge script processed one source shard at a time and computed the LoRA update in float64 on CPU before storing merged weights in BF16. It applied the update to 256 text projections. The base checkpoint also contains MTP tensors with overlapping layer names, so the script matches the full language-model prefix. The merged BF16 checkpoint was converted once to MLX affine 4-bit with group size 64. The build used MLX 0.32.4 and mlx-vlm 0.7.11 on an M2 Max with 32 GB unified memory. See [reproduction instructions](REPRODUCE.md) and [checksums](SHA256SUMS).

## Local validation

The 15 GiB checkpoint loaded on an M2 Max with mlx-lm 0.32.2 and returned `2 + 2 = 4` with thinking disabled. MLX reported 15.435 GB peak memory for this short prompt. In a separate five-task synthetic OpenCode pilot, the fused Q4 passed 4/5 tasks and the matched base Q4 passed 5/5. See [evaluation](EVALUATION.md) for per-task outcomes, memory settings, and limits of the comparison.

## Scope and attribution

Validation covers text generation. Native MTP was not exported for this release. Vision has not been evaluated. This is an independent conversion and has not been compared tensor by tensor with other published conversions.

Base model by the Qwen team. LoRA and adapted tokenizer files by harness-hub. This checkpoint is distributed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for changes and attribution.
