---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
- unsloth
- qwen
- qwen3_5
- GGUF
- llama.cpp
- MTP
datasets:
- eaddario/imatrix-calibration
metrics:
- perplexity 
---
# Qwen3.8-27B-48Gb
## Overview
This repository provides a highly optimized, custom-quantized GGUF model of Qwen3.8-27B, specifically engineered for local deployment on Dual RTX 3090 setups. 
The primary research objective of this quantization is to achieve an extreme context length (full 262K tokens in F16 KV Cache) while maximizing inference speed through adapted BPW and Multi-Token Prediction (MTP / Self-Speculative Decoding) and retaining most of the original model's capacities. To achieve this, the base network was quantized to `Q5_0` and `Q8_0` using a custom iMatrix, while the critical `NextN` layers and embeddings were strictly preserved in `Q8_0`.
The base model used for this requantization is [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).

## Research & Methodology
### Selective precision Quantization for high-speed inference
Most standard quantization pipelines compress the entire model, which severely degrades the quality, the inference speeds and the NextN layer responsible for Multi-Token Prediction (which is, sometimes, completely suppress it).

To maintain capacities while optimizing for a Dual RTX 3090 (NVLink) setup, I have implemented a Selective Precision Mapping strategy. By carefully partitioning the model into Q8_0 (High Precision) and Q5_0 (Balanced Efficiency) tensors, it preserves the critical activation flows, specifically the Multi-Token Prediction (NextN) layer, without sacrificing the throughput necessary for processing massive contexts (full 262K tokens).

#### Strategic Layer Mapping
The following quantization scheme was applied :

``` bash
--tensor-type 'token_embd\.weight=q8_0'
--tensor-type 'output\.weight=q8_0'
--tensor-type 'blk\.64\..*=q8_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.attn_qkv\.weight=q8_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.attn_q\.weight=q8_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.attn_k\.weight=q8_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.attn_v\.weight=q8_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.attn_output\.weight=q8_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.attn_gate\.weight=q5_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.ssm_alpha\.weight=q5_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.ssm_beta\.weight=q5_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.ssm_out\.weight=q8_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_up\.weight=q5_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_down\.weight=q5_0'
--tensor-type 'blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_gate\.weight=q5_0'
```

This scheme keeps the most important layers at high precision (Q8_0) and lowers the GGUF size on disk, enabling full offload on dual RTX 3090 setups. I chose to go for Q5_0 and Q8_0 as it both retains many capacities while lowering the needs for complex mathematical kernel calculations. 

### iMatrix Calibration
The model was calibrated using a custom, shuffled iMatrix to ensure high fidelity across coding, instruction-following, and bilingual tasks (English/French). 
The dataset was built by merging and shuffling the following subsets from [`eaddario/imatrix-calibration`](https://huggingface.co/datasets/eaddario/imatrix-calibration):
- `code_small`
- `tools_small`
- `text_en_small`
- `text_fr_small`

### Accuracy Analysis
I compared the perplexity on Wiki-Text-raw to evaluate the precision loss after the mixed-quantization scheme:

| Model Version | Precision | Wiki-text-raw (PPL) | Delta vs BF16 |
| :--- | :--- | :--- | :--- |
| **Q8_0-Plus** | **Q8_0** + **F16** | **5.8285** | **+0.0204** |
| Base (Source) | BF16 (Original) | 5.8081 | - |

**Key Findings:**
* **Near-Lossless:** The perplexity degradation is minimal at +0.0204, indicating that this mixed precision layout preserves the original model's reasoning capabilities.

## Recommended Usage

To replicate the optimal performance (200K context, F16 Cache, Multi-GPU) using `llama.cpp`, use the following `llama-server` command. Note the specific use of `--split-mode tensor` and `--tensor-split 1,1` for optimal PCIe bandwidth management across dual RTX 3090s. This command appeared to be the best one I could come across using an NVLink.

``` bash
/path/to/llama.cpp/build/bin/llama-server \
    -m /path/to/Qwen3.8-27B-Q8_0-plus\
    --mmproj /path/to/mmproj-F16.gguf \
    --split-mode tensor \
    --tensor-split 1,1 \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 262144 \
    --parallel 1 \
    --gpu-layers 999 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --flash-attn on \
    -b 2048 -ub 2048 \
    --spec-type draft-mtp \
    -sps 0.70 \
    --image-min-tokens 1024 \
    --alias Qwen3.6-27b \
    --jinja
```

If I may add, I also developed a proxy to enable users to select thinking or non-thinking behaviors and applied the recommended sampling parameters AND the `"Preserve Thinking"` option. You may find it on my [GitHub](https://github.com/AlexanderKyng/Qwen3.6-reasoning-toggle-proxy).

## Hardware Requirements
- **Target VRAM:** 48 GB (Tested on 2x NVIDIA RTX 3090 24GB).
- **RAM:** Minimum 32GB system RAM (Prompt caching and system overhead).
- **Context limit:** The command above loads ~13GB of KV cache across the two GPUs. If you experience OOM (Out of Memory) errors, consider reducing `--ctx-size` or using 8-bit cache (`--cache-type-k q8_0 --cache-type-v q8_0`).

## Acknowledgments

This project was made possible thanks to the outstanding tools and contributions from the open-source AI community. Special thanks to:

*   **The Qwen Team**: For researching and releasing the exceptional Qwen3.8 architecture.
*   **llama.cpp**: Using the new Tensor split mode, it finally achieves extremelly high performances on dual-GPU setups.
*   **eaddario**: For the extremely diverse `imatrix-calibration` dataset, which was crucial in building the custom, multilingual, and code-heavy iMatrix.
