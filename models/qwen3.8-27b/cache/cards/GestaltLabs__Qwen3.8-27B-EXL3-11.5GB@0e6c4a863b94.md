---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: exllamav3
tags:
  - exl3
  - exllamav3
  - quantized
  - conversational
  - vision
  - speculative-decoding
  - mtp
  - dflash
---

# Qwen3.8-27B — 11.5 GB native EXL3, with MTP and vision retained

**This release would not have been possible without the generosity of [selimaktas](https://huggingface.co/selimaktas), who donated the compute time for this work.**

A complete, standalone **native EXL3 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**, published by GestaltLabs. This is **Qwen, not the unrelated Ornstein fine-tune**. It includes the target's native multi-token prediction (MTP) and vision components, calibrated on an expanded full-corpus dataset.

**EXL3 is the work of [Turboderp and the ExLlamaV3 contributors](https://github.com/turboderp-org/exllamav3), not an invention of GestaltLabs.** EXL3 builds on [QTIP from Cornell RelaxML](https://github.com/Cornell-RelaxML/qtip). Credit for the base model belongs to Qwen; credit for DFlash belongs to [z-lab and its contributors](https://github.com/z-lab/dflash). GestaltLabs' contribution here is full-corpus calibration, checkpoint recovery/selection, draft adaptation, measurement, and publication.

| At a glance | This release |
|---|---|
| Base | Qwen3.8-27B, dense vision-language model |
| Weight format / runtime | Native EXL3 / **ExLlamaV3 1.4.6** |
| Decoder allocation | Mixed **2.866379 bits per weight**; 4-bit output head; native F8_E4M3 token embeddings |
| Calibration dataset | **598 independent conversations** (4,898,816 tokens, 8K context: `598 × 8192`) |
| Package footprint | Exactly **11,497,448,942 bytes** (~11.50 GB decimal / 10.708 GiB), within 11.5 GB budget |
| Recommended text profile | Built-in native MTP (stock ExLlamaV3 1.4.6) |
| Speculative drafting options | **Native 4-bit MTP head** (built into this package); DFlash1 companion supported |
| License | [Apache-2.0](LICENSE); dependencies retain their respective licenses |

**Use the native runtime below.** These safetensors contain EXL3 codes, not ordinary dense Transformers weights, GGUF, or EXL2. Do not load them with a generic Transformers `from_pretrained`, vLLM, SGLang, or llama.cpp command. No LoRA, inference adapter, dense recovery graph, or training framework is needed.

## What is included

- Three native weight shards and their index: the complete target model, not a delta.
- Native **4-bit MTP weights inside the target** (39 tensors, 15 modules). MTP does not require downloading an external draft repository.
- Native vision tensors, model configuration, tokenizer, chat template, [image processor configuration](preprocessor_config.json), and [video processor configuration](video_preprocessor_config.json). There is no separate GGUF `mmproj` file.
- [inference.py](inference.py): plain, MTP, and companion DFlash text inference; [vision.py](vision.py): plain native single-image inference.
- [benchmark_native_spec.py](benchmark_native_spec.py), [requirements.txt](requirements.txt), and portable [inference_settings.json](inference_settings.json).
- [Evidence](evidence/), [release manifest](release_manifest.json), license, and [CITATION.cff](CITATION.cff).

The integrity verification confirms preservation of all **3,080 tensor names**, including all **1,026 vision and MTP tensors** (987 `model.visual.*` and 39 `mtp.*`) and tokenizer/processor assets.

**Precision metadata:** this is a mixed-bit decoder, not a uniformly quantized model. Its exact average is recorded as `decoder_bpw` in the native configuration; the Hub's integer-only uniform `bits` field is intentionally not used. Full per-tensor formats remain in [quantization_config.json](quantization_config.json).

## Download and install

Use a CUDA-capable NVIDIA system supported by [ExLlamaV3](https://github.com/turboderp-org/exllamav3#how-to). The measured environment was Linux, Python **3.12.14**, **PyTorch 2.13.0+cu130**, and **ExLlamaV3 1.4.6**, on one **NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB**. Other devices and environments may differ; the disk footprint is not a promise that this runs on an 11.5 GB or 12 GB GPU.

Create an environment and download the complete target snapshot:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install huggingface_hub
hf download GestaltLabs/Qwen3.8-27B-EXL3-11.5GB --local-dir ./qwen-exl3

# Install CUDA PyTorch separately, before ExLlamaV3.
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r ./qwen-exl3/requirements.txt
```

The PyPI ExLlamaV3 distribution requires a compatible CUDA toolkit and C++ build prerequisites; it is not a universal prebuilt CUDA extension. If using a wheel instead, choose a **1.4.6** wheel matching your Python, PyTorch, CUDA, and platform from [upstream releases](https://github.com/turboderp-org/exllamav3/releases). Follow [PyTorch's installation guidance](https://pytorch.org/get-started/locally/) for a driver-compatible CUDA build. The tested runtime reference is revision [`499890c75d20d8e7c9d061f37189ae611a5c9f0b`](https://github.com/turboderp-org/exllamav3/tree/499890c75d20d8e7c9d061f37189ae611a5c9f0b).

Activate the environment before launching the helpers so Ninja is on `PATH`. For a source/JIT build, set `CC=gcc-11` and `CXX=g++-11` if your system defaults point to different compiler versions.

## Speculative Decoding: Native MTP & Companion Drafters

### 1. Built-in Native MTP (Recommended — Zero Extra VRAM)

The target model includes Qwen3.8's official **multi-token prediction (MTP) head** quantized to 4-bit EXL3 directly inside the weight shards (`mtp.fc`, `mtp.pre_fc_norm_*`, `mtp.layers.0.*`, `mtp.norm`, 39 tensors total). Drafting requires **no extra download and no extra draft weights in VRAM**:

```bash
python ./qwen-exl3/inference.py --model ./qwen-exl3 --mode mtp \
  --prompt "Explain a safe design for an LRU cache." --max-new-tokens 512
```

In hardware smoke verification on Blackwell RTX PRO 6000, native text+MTP achieved **100% draft acceptance** (4/4 drafts accepted), and multimodal vision+MTP achieved **7/12 draft acceptance**. MTP uses the target's prediction head with adaptive draft length and confidence **0.4**.

### 2. External Companion Drafters (DFlash1)

For users running external speculative drafters on stock ExLlamaV3 1.4.6:
- **DFlash1**: [GestaltLabs/Qwen3.8-27B-DFlash-EXL3-4bit](https://huggingface.co/GestaltLabs/Qwen3.8-27B-DFlash-EXL3-4bit) — 1.73B companion draft model adapted specifically to this target.

```bash
python ./qwen-exl3/inference.py --model ./qwen-exl3 --mode dflash \
  --draft ./qwen-dflash --prompt "Explain a safe design for an LRU cache." \
  --max-new-tokens 512
```

*Note on DFlash2:* Community DFlash2 block-diffusion models (such as `z-lab/Qwen3.8-27B-DFlash2` or `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw`) require an experimental fork of ExLlamaV3 with dedicated DFlash2 kernel and selector additions; stock ExLlamaV3 1.4.6 natively runs built-in MTP and DFlash1, but not DFlash2.

### 3. Plain Decoding (No Speculation)

```bash
python ./qwen-exl3/inference.py --model ./qwen-exl3 --mode plain \
  --prompt "What is 17 * 19? Reply only with the number." --max-new-tokens 512
```

## Multimodal Inference (Vision + Text)

Use the bundled native image helper for multimodal inputs:

```bash
python ./qwen-exl3/vision.py --model ./qwen-exl3 --image ./example.jpg \
  --prompt "Describe what is visible in this image." --max-new-tokens 256
```

All 27 vision transformer blocks (987 tensors) and image/video preprocessing configurations are retained natively in EXL3 format.

## Full-Corpus Calibration & Evaluation Results

This release was quantized using an expanded full-corpus calibration set of **598 independent multi-turn conversations** (4,898,816 tokens in `598 × 8192` format, 9.34× larger than the 64-conversation pilot).

### 1. Full-Vocabulary Distribution Fidelity (KL Divergence to Official FP8)

Metric: Full-vocabulary natural-log $D_{\text{KL}}(P_{\text{FP8}} \parallel P_{\text{EXL3}})$ measured across all 248,320 vocabulary columns at 8 frozen prompt-prefix positions per task. **Lower is better.**

| Model | Development Mean KL | Acceptance Mean KL | Combined All Mean KL |
|---|---:|---:|---:|
| **This Release (Expanded 8K, 598 rows)** | **0.16220** | **0.25899** | **0.21060** |
| Prior Release (Pilot Calibration) | 0.18597 | 0.26780 | 0.22689 |
| Pilot 8K Candidate (64 rows) | 0.23696 | 0.33439 | 0.28568 |

**Domain Breakdown on Acceptance Split:**
- **General (20 tasks)**: Mean KL **0.16523** (vs. 0.23461 prior, **−29.6% closer to FP8**)
- **Math (20 tasks)**: Mean KL **0.14196** (vs. 0.30028 prior, **−52.7% closer to FP8**)
- **Code (16 tasks)**: Mean KL **0.17880** (vs. 0.19573 prior, **−8.6% closer to FP8**)
- **Structured (8 tasks)**: Mean KL 0.94636

Across General, Math, and Code domains, the expanded 598-conversation calibration brought next-token distribution fidelity substantially closer to official FP8 than prior releases.

### 2. Objective Task Accuracy

Evaluation protocol: 128 tasks (64 development, 64 acceptance), greedy argmax decoding, non-thinking, `max_new_tokens=512`, isolated Docker sandbox with zero infrastructure errors.

| Model | Development Split | Acceptance Split | Combined All (Descriptive) |
|---|---:|---:|---:|
| Official FP8 (Audited Reference) | 49/64 (76.56%) | 50/64 (78.13%) | 99/128 (77.34%) |
| Prior Release (Pilot Calibration) | 50/64 (78.13%) | 48/64 (75.00%) | 98/128 (76.56%) |
| **This Release (Expanded 8K)** | 43/64 (67.19%) | 44/64 (68.75%) | 87/128 (67.97%) |

**Domain Performance on Acceptance Split:**
- **Math**: **19/20 (95.0%)** vs. 18/20 prior (+1 net win)
- **General**: 7/20 (35.0%) vs. 7/20 prior (tied, both cap-limited)
- **Structured**: 7/8 (87.5%) vs. 7/8 prior (tied)
- **Code**: 11/16 (68.75%) vs. 16/16 prior

While distribution fidelity to FP8 reached its highest measured level across all major domains, task-level coding accuracy showed trade-offs on 512-token capped generation. Users prioritizing strict coding accuracy over general distribution fidelity should evaluate both releases against their target prompts.

## Historical Baseline Speed & Memory Reference

The following table records the historical speculative-decoding throughput baseline measured on an **NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB**, with a 16,384-token FP16 KV cache under the prior release configuration:

| Profile | Decode tokens/s | End-to-end tokens/s | Mean TTFT | Peak allocated VRAM |
|---|---:|---:|---:|---:|
| Plain (INT8 activation GEMV disabled) | 66.6 | 65.8 | 185 ms | 11.28 GB |
| Plain (default INT8 activation GEMV) | 80.1 | 79.0 | 182 ms | 11.28 GB |
| **Built-in MTP (INT8 activation GEMV disabled)** | **159.2** | **153.4** | 264 ms | 12.80 GB |
| Trained 4-bit DFlash companion | 149.4 | 145.7 | 192 ms | 13.68 GB |

*Note:* These numbers reflect single-request greedy benchmarks under prior release test workloads. The new full-corpus model was functionally verified for MTP and vision execution via native hardware smoke tests, but was not independently re-benchmarked across the full 16K throughput suite.

## Troubleshooting and Limitations

- **Loader rejects weights:** Ensure ExLlamaV3 1.4.6 is installed. These are native EXL3 safetensors, not dense Hugging Face weights or GGUF.
- **Speculative decoding setup:** For MTP, run with `--mode mtp` directly on this model directory. For DFlash companions, provide the companion directory via `--draft`.
- **Memory expectations:** The model package on disk is 11.497 GB. Runtime VRAM depends on KV cache size and context length (~11.3 GB for plain 16K, ~12.8 GB for MTP 16K).
- **Thinking / Chat template:** Generation scripts default to non-thinking greedy mode. Adjust sampling and templates per upstream Qwen recommendations for conversational use.

## Credits, Citation, and Support

Please credit the upstream research and tooling:
1. **Qwen Team (Alibaba)** — [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
2. **Turboderp & ExLlamaV3 Contributors** — [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [EXL3 Format](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md).
3. **Cornell RelaxML / QTIP Contributors** — [*QTIP: Quantization with Trellises and Incoherence Processing*](https://arxiv.org/abs/2406.11235).
4. **z-lab / DFlash Contributors** — [*DFlash: Block Diffusion for Flash Speculative Decoding*](https://arxiv.org/abs/2602.06036).

If you would like to support GestaltLabs' checkpoint preparation, experiments, and open publication work: **[Ko-fi — djlougen](https://ko-fi.com/djlougen)**.
