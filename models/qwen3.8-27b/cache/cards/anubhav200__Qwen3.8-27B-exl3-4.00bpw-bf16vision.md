---
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
- turboderp/Qwen3.8-27B-exl3
- Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- exl3
- exllamav3
- tensorfold
- qwen3.8
- vision
- dgx-spark
---

# Qwen3.8-27B EXL3 4.00 bpw + full-precision (bf16) vision tower

This is **turboderp's EXL3 4.00 bpw quantization of Qwen3.8-27B**, unchanged, with two additions: the
**official bf16 vision tower** as a separate file, and a **fixed chat template**. It was put together to serve the
model with [TensorFold](https://github.com/ashhart/TensorFold) on one NVIDIA DGX Spark (GB10, 128 GB); the serving
recipe is [27B-2 in Tensorfold-recipies-single-dgx-spark](https://github.com/anubhavgupta/Tensorfold-recipies-single-dgx-spark/tree/main/27B-2).

## Credits

All of the real work is theirs:

- **[turboderp](https://huggingface.co/turboderp)**: the EXL3 format, [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
  and this quantization, [`turboderp/Qwen3.8-27B-exl3`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) branch
  `SC_4.00bpw_H5_V6` (commit `516bf129059031c6da9416768ea6b7a1be00a8fc`): 4.00 bpw, 5-bit output head, 6-bit vision,
  self-calibrated. The model shards, calibration trace and quantization config here are his files, byte for byte.
- **[Qwen team](https://huggingface.co/Qwen)**: the model, [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
  (Apache-2.0), including the vision tower in `vision_bf16.safetensors`. Their model card is kept as `QWEN_README.md`.
- **[froggeric](https://huggingface.co/froggeric)**: the chat template,
  [`froggeric/Qwen-Fixed-Chat-Templates`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) v22.5 (Apache-2.0).

## What differs from `turboderp/Qwen3.8-27B-exl3` @ `SC_4.00bpw_H5_V6`

| File | Change |
|---|---|
| `vision_bf16.safetensors` | **Added.** The 333 `model.visual.*` tensors of `Qwen/Qwen3.8-27B`, bf16, extracted unchanged from its first shard (names, shapes and bytes identical; 879 MiB). |
| `chat_template.jinja` | **Replaced** with froggeric's v22.5 fixed template (the official one is still embedded in `tokenizer_config.json`). It defaults to `reasoning_effort` medium, keeps empty `<think></think>` blocks out of history (prefix caches stay valid across turns), and handles tool calls and mid-conversation system messages more robustly. |
| `README.md` | This card. Upstream's card is `QWEN_README.md`. |
| everything else | Unchanged: `model-0000{1,2}-of-00002.safetensors`, `model.safetensors.index.json`, `config.json`, `quantization_config.json`, tokenizer files, `cal_trace.*`, `qbench_*`, `LICENSE`, ... |

The quantized 6-bit vision tower inside the model shards is still there, untouched, so ExLlamaV3/TabbyAPI load this
repo exactly like upstream (they use that tower and ignore `vision_bf16.safetensors`).

### Why the bf16 vision tower

TensorFold cannot load an EXL3-quantized vision tower, but it can take a separate floating-point one
(`TENSORFOLD_VISION_WEIGHTS`). The earlier workaround was to decode the pack's 6-bit tower to fp16, which keeps its
quantization error. The vision tower is ~0.46B parameters, so serving it in full precision costs ~0.9 GB and removes
that error for image and video input.

## Use with TensorFold

```bash
hf download anubhav200/Qwen3.8-27B-exl3-4.00bpw-bf16vision --local-dir Qwen3.8-27B-exl3-4.00bpw-bf16vision
# then, with the 27B-2 recipe (patched TensorFold 0.6.6; the patches add the EXL3 DFlash2 drafter, image prefix
# caching and faster EXL3 kernels):
MODEL_PATH=$PWD/Qwen3.8-27B-exl3-4.00bpw-bf16vision ./27B-2/start.sh
```

Without the recipe: mount the folder and set `TENSORFOLD_VISION_WEIGHTS=<folder>/vision_bf16.safetensors` with
`tensorfold serve <folder> --vision`. Suggested drafter:
[`igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw`](https://huggingface.co/igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw)
(needs the recipe's drafter-loading patch) or [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2);
both accept the same.

## Measured on one DGX Spark (TensorFold 0.6.6 + the 27B-2 patches)

Quality: the mean log-likelihood of 61,575 thinking-mode response tokens that the 6.00 bpw EXL3 model generated
(this pack's `cal_trace.json`), scored with the same TensorFold forward that serves requests (fp8 KV, bf16 prompt
activations). Lower NLL means closer to the near-full-precision reference (which is itself EXL3, so slightly
favourable to EXL3):

| Checkpoint | Mean NLL (nats/token) | Top-1 agreement |
|---|---|---|
| **this (EXL3 4.00 bpw)** | **0.1705** | **93.50%** |
| `Vontra/Qwen3.8-27B-MLX-4bit` (MLX affine 4-bit, gs 64) | 0.1893 | 92.80% |

Speed (greedy, thinking off; prefill with unique prompts; DFlash2 drafter): single-stream decode 46 tok/s prose,
66-90 tok/s code; prefill 930 tok/s at 10-40K tokens, 840 tok/s at 106K; 8 concurrent ~249K-token conversations
fit in memory (fp8 KV, 262,144-token context). See the recipe README for methods and multi-stream throughput.

## License

Apache-2.0, as `Qwen/Qwen3.8-27B`, turboderp's quantization and froggeric's template. See `LICENSE`.
