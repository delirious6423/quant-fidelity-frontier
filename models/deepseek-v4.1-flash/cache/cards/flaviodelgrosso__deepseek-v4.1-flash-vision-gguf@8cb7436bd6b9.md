---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash
tags:
- gguf
- vision
- deepseek41
---

# DeepSeek V4.1 Flash — vision encoder (GGUF)

The vision tower and aligner of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
at source revision `df42c109f1defefcbfcedbe7d905718a12266e40` (the checkpoint
before 1017) as a separate encoder GGUF.

| File | Bytes | SHA-256 |
|---|---|---|
| `DeepSeek-V4.1-Flash-Vision.gguf` | 970,555,552 | `cc283f032b3e8b8d78aeb5fccaa14e97b859b0c53aae3cd6bffa690ddf0e9e15` |

Tower: 32 blocks, width 1,024, 16 heads of 64, patch 14, 3×3 channel-major
merge to 9,216 and an aligner 9,216 → 5,120 → 5,120.

> [!NOTE]
> This encoder comes from the pre-1017 checkpoint. Its compatibility with the
> 1017 language model
> ([flaviodelgrosso/deepseek-v4.1-flash-1017-gguf](https://huggingface.co/flaviodelgrosso/deepseek-v4.1-flash-1017-gguf))
> has not been established; Inferno does not attach it yet.

## License

MIT, as the upstream DeepSeek checkpoint (see `LICENSE`).
