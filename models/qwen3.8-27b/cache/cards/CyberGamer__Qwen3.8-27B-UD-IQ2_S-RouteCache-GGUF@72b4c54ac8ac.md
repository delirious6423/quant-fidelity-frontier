---
tags:
- gguf
- llama.cpp
- ollama
- routecache
library_name: gguf
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
---

# Qwen3.8-27B UD-IQ2_S — RouteCache

Quality-passing UD-IQ2_S GGUF with a RouteCache execution profile embedded directly in GGUF metadata. Quantized tensor payload bytes are preserved from the recorded source GGUF; the whole-file SHA differs because the header metadata is new.

## Ollama

```text
ollama run hf.co/CyberGamer/Qwen3.8-27B-UD-IQ2_S-RouteCache-GGUF
```

The HF→Ollama bridge reads the root-level `params` JSON in this repo. It contains `{"num_ctx":4096}`. `Modelfile` is also included for manual `ollama create`, but direct HF execution does not depend on it.

If you download/clone the entire Hugging Face repository on Windows, you can instead double-click `RUN_OLLAMA.bat`. It uses the local GGUF, installs Ollama through WinGet when available, creates `qwen3.8-27b-routecache` from the included `Modelfile`, and runs it.

## Optimized RouteCache runtime

Use the companion GitHub repository and run `SETUP_AND_RUN.bat` for the hardware-specific RouteCache tensor placement, custom-kernel attempt, warm-up, low-latency streaming, and automatic rolling context compaction. `OLLAMA_SETUP_AND_RUN.bat` is the near-zero-step Ollama path.

## Reproducibility

The [GitHub](https://github.com/MotionflowOffical/routecache-qwen3.8-27b) repository  includes `REPRODUCE_MODEL.bat`, which can rebuild the low-bit topology from official `Qwen/Qwen3.8-27B` weights using the pinned llama.cpp source plus Unsloth's public UD-IQ2_S importance matrix/per-tensor recipe. Source reproduction is optional and requires roughly 120–140 GiB temporary disk.

## Calibration

Private RouteCache calibration on RTX 4060 8 GB: 4096 context, Q8_0 KV, batch/ubatch 256/64, explicit CPU embedding + FFN up/gate blocks 33–46, warmed non-MTP floor around 11.39 tok/s. Quality PPL ratio was about 1.057 against the local Q4 teacher (acceptance threshold 1.08). Hardware-specific speed claim only.

## License

Model/quantization license identifier configured for publication: **apache-2.0**. Verify upstream attribution and any additional terms before publication.
