---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- harness-hub/Qwen3.8-27B-pi-adapted
pipeline_tag: text-generation
tags:
- gguf
- qwen
- tandem
- speculative-decoding
---

# Qwen3.8-27B-pi-TQV12

A 27B GGUF build for running the Pi-adapted Qwen model locally with Tandem. Download one weight file, `Qwen3.8-27B-pi-TQV12.gguf`. The Pi adaptation is included in these weights.

**Inference engine and source code:** [Tandem on GitHub](https://github.com/aseprahman04/tqllm-tandem). The [v0.1.12 release](https://github.com/aseprahman04/tqllm-tandem/releases/tag/v0.1.12) includes the Windows SM120 executable and decode benchmarks.

## Credits

| Contribution | Source |
|---|---|
| Base model and tokenizer | [Qwen / Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Pi harness adaptation | [Harness Hub / Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted). Harness Hub trained the LoRA adapter with reinforcement learning for agent work inside Pi. |
| TQ quantization reference, DFlash2 drafter and draft vocabulary | [YTan2000 / Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ) |
| Distribution of this pi-TQV12 build and Tandem runtime integration | [arn04](https://huggingface.co/arn04) and [the Tandem fork](https://github.com/aseprahman04/tqllm-tandem), based on [turbo-tan/tqllm](https://github.com/turbo-tan/tqllm) |

## Run with Tandem

The following PowerShell commands use the two-GPU configuration tested on two 16 GB RTX 5060 Ti cards. Place the Tandem release executable in the current directory and install its runtime dependencies as described in the release.

```powershell
hf download arn04/Qwen3.8-27B-pi-TQV12 Qwen3.8-27B-pi-TQV12.gguf --local-dir models
hf download YTan2000/Qwen3.8-27B-TQ drafters/Qwen3.8-27B-DFlash2-TQ3.gguf mtp-vocab/atx_65536.txt --local-dir models
$env:TQLLM_TP = "1"
.\tqserve.exe --model models/Qwen3.8-27B-pi-TQV12.gguf -md models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab models/mtp-vocab/atx_65536.txt --context 262144 --kv tq3 --kv-host 0 --prompt-cache on --host 127.0.0.1 --port 11435
```

The server exposes an OpenAI-compatible chat completions API. Code that repeats text from the prompt benefits most from speculative prompt-copy verification. For a comparison with new-code generation, see the [measured workloads and hardware in the Tandem release](https://github.com/aseprahman04/tqllm-tandem/releases/tag/v0.1.12).
