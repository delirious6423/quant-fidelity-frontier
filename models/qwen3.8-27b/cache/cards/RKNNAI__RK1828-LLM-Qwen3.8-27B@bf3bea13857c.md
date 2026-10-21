---
license: apache-2.0
task: text-generation
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- rknn
---

# RK1828-LLM-Qwen3.8-27B

[English](README.md) | [简体中文](README_CN.md)

## 1. Model Overview

This repository provides the text-only language model component of Qwen3.8-27B for text understanding and generation on RK1828.

Source Model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

- Model ID: `RKNNAI/RK1828-LLM-Qwen3.8-27B`
- Model Display Name: `RK1828-LLM-Qwen3.8-27B`
- RKNN Runtime Version: `v1.1.0`
- Source Model: `Qwen/Qwen3.8-27B`
- Model Type: LLM
- Chip Label: RK1828
- Supported Chips: RK1828

### Available Models

| RKNN Runtime Version | Configuration Directory | Supported Chips | Quantization | NPU Cores | Context Length (tokens) | KVCache | Special Configuration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `v1.1.0` | [Qwen3.8-27B-w4a16-8-51200-normal-kv-int4-4cards](Qwen3.8-27B-w4a16-8-51200-normal-kv-int4-4cards/README.md) | RK1828 | llm_seg0: w4a16; llm_seg1: w4a16; llm_seg2: w4a16; llm_seg3: w4a16 | llm_seg0: 8; llm_seg1: 8; llm_seg2: 8; llm_seg3: 8 | llm_seg0: 51200 (50k); llm_seg1: 51200 (50k); llm_seg2: 51200 (50k); llm_seg3: 51200 (50k) | llm_seg0: int4; llm_seg1: int4; llm_seg2: int4; llm_seg3: int4 | normal-kv-int4-4cards |

## 2. Files

| File or Directory | Description |
| --- | --- |
| `LICENSE` | Source model license |
| `<configuration-directory>/` | Matching model files |
| Subdirectory `README.md` | Configuration documentation (English) |
| Subdirectory `README_CN.md` | Configuration documentation (Chinese) |
| Subdirectory `config.json` | Model configuration and file manifest |
| Subdirectory `SHA256SUMS` | SHA-256 checksums of the files in this directory (excluding this file) |

## 3. Download

Download the full repository from ModelScope:

```bash
modelscope download --model RKNNAI/RK1828-LLM-Qwen3.8-27B --revision v1.1.0 --local_dir ./RK1828-LLM-Qwen3.8-27B
```

Download the full repository from Hugging Face:

```bash
hf download RKNNAI/RK1828-LLM-Qwen3.8-27B --revision v1.1.0 --local-dir ./RK1828-LLM-Qwen3.8-27B
```

Download a specific configuration from ModelScope:

```python
from modelscope import snapshot_download

snapshot_download(
    "RKNNAI/RK1828-LLM-Qwen3.8-27B",
    revision="v1.1.0",
    allow_patterns=["README.md", "README_CN.md", "LICENSE", "Qwen3.8-27B-w4a16-8-51200-normal-kv-int4-4cards/**"],
    local_dir="./RK1828-LLM-Qwen3.8-27B",
)
```

Download a specific configuration from Hugging Face:

```bash
hf download RKNNAI/RK1828-LLM-Qwen3.8-27B --revision v1.1.0 --include "README.md" "README_CN.md" "LICENSE" "Qwen3.8-27B-w4a16-8-51200-normal-kv-int4-4cards/**" --local-dir ./RK1828-LLM-Qwen3.8-27B
```

## 4. SHA-256 Verification

Run in the configuration directory:

```bash
cd ./RK1828-LLM-Qwen3.8-27B/Qwen3.8-27B-w4a16-8-51200-normal-kv-int4-4cards
sha256sum -c SHA256SUMS
```

Deploy only after all entries show `OK`.

## 5. Compatibility and Limitations

- Supported Chips: RK1828.
- Use the matching RKNN Runtime and driver. Check the device-side version with `rknn-smi -v` before running.
- Use the files from the selected configuration directory together.

## 6. Copyright and License

The source model was released by the Qwen team under Apache License 2.0. See [LICENSE](LICENSE) for the full license.
