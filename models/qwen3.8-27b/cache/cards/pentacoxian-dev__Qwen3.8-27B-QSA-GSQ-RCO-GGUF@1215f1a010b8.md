---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - llama.cpp
  - qsa
  - gsq-rco
---

# Qwen3.8-27B QSA, GSQ-RCO IQ3_S

The model files for running Qwen3.8-27B with Qwen Sparse Attention on one 24 GB GPU, at 262K context with vision. These files need the engine at [pentaCoxian/Qwen-QSA-llama.cpp](https://github.com/pentaCoxian/Qwen-QSA-llama.cpp). Stock llama.cpp cannot run the QSA weights.

```sh
hf download pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF --local-dir models
deploy/qsa27b-24gb/serve.sh
```

Run those from a built checkout of the engine. Its README has the build steps.

## Files

| File | Size | What it is |
|---|---:|---|
| `Qwen3.8-27B-QSA-GSQ-RCO-IQ3_S-mtp.gguf` | 12.3 GB | Main model with its MTP head and QSA indexers |
| `Qwen3.8-27B-DFlash2-IQ4_XS.gguf` | 1.0 GB | DFlash2 drafter, for speculative decoding |
| `mmproj-Qwen3.8-27B-QSA-F16.gguf` | 0.9 GB | Vision projector |

`SHA256SUMS` lists their checksums.

## Results

Measured on one RTX 4090 at 262K context, with 255K tokens of input and 4 drafts:

| | |
|---|---:|
| Greedy decode, harmonic mean of 4 tasks | 114.7 tok/s |
| Sampled decode, temperature 1.0, top-p 0.95, top-k 20 | 113.0 tok/s |
| Cold prefill | 2,375 tok/s (108 s) |
| Peak VRAM | 22.8 GiB |
| Fact recall, single fact / two-hop | 24/24, 11/16 |
| Vision, image 255K tokens before the question | 14/14 |

The engine's README has the comparison with stock llama.cpp, per-task results, RTX 5090 results and limits.

## How these files were made

- **Main model:** ISTA-DASLab's `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` with 80 F32 QSA indexer tensors and their metadata added from higa2515's `qsa-indexers.safetensors`. The 866 existing tensors are byte-identical to ISTA-DASLab's file, and the 80 indexers are byte-identical to those in higa2515's published Q8_0 GGUF.
- **Drafter:** z-lab's `Qwen3.8-27B-DFlash2-BF16.gguf`, quantized to IQ4_XS with `llama-quantize` and no imatrix.
- **Vision projector:** higa2515's `mmproj-Qwen3.8-27B-QSA-F16.gguf`, unchanged.

## Credits

All the source models are Apache-2.0, and so are these files.

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), the base model
- [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF), the quantized weights
- [higa2515/Qwen3.8-27B-QSA-GGUF](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF), the QSA conversion, indexers and vision projector
- [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), the drafter
