---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
tags:
- gemma4
- unsloth
- gemma
- google
- gguf
- split
---

# Gemma 4 26B A4B — GGUF Split

This repository contains **split GGUF shards** of [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF), optimized for use with **llama.cpp** (CPU & GPU inference). Files are split into ≤5 GB shards for easier downloading and HuggingFace LFS compatibility.

Inspired by [Nielk38/Qwen3.6-35B-A3B-GGUF-SPLIT](https://huggingface.co/Nielk38/Qwen3.6-35B-A3B-GGUF-SPLIT).

## Available Quantizations

| Quantization | Total Size | Shards | Prompt Speed (CPU) | Generation Speed (CPU) | Notes |
|---|---|---|---|---|---|
| `UD-Q4_K_M` | ~16.9 GB | 4 | 52.9 t/s | 12.3 t/s | Best quality 4-bit |

> CPU benchmark: AMD Ryzen 7 9800X3D (8c/16t, AVX-512 Zen4 backend), 16 threads, CPU-only (no GPU offload).

## Multimodal Projection Files

| File | Use |
|---|---|
| `mmproj-BF16.gguf` | Vision encoder (BF16, recommended) |
| `mmproj-F16.gguf` | Vision encoder (F16) |

## How to Download & Use

### Download shards
```bash
huggingface-cli download Nielk38/gemma-4-26B-A4B-it-GGUF-SPLIT \
  --include "*.gguf" \
  --local-dir ./gemma4-gguf
```

### Run with llama.cpp (first shard auto-loads all)
```bash
llama-cli -m ./gemma4-gguf/gemma-4-26B-A4B-it-UD-Q4_K_M-00001-of-00004.gguf \
  --mmproj ./gemma4-gguf/mmproj-BF16.gguf \
  -ngl 0 -t 16 -p "What is 2+2?" -n 128
```

### Merge shards (optional)
```bash
llama-gguf-split --merge \
  ./gemma4-gguf/gemma-4-26B-A4B-it-UD-Q4_K_M-00001-of-00004.gguf \
  ./gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
```

---

## Benchmark Comparison: Gemma 4 26B A4B vs Qwen3.6 35B A3B

Both are MoE models with ~4B active parameters at inference.

| Benchmark | Gemma 4 26B A4B | Qwen3.6 35B A3B |
|---|---|---|
| **MMLU Pro** | 82.6% | 85.2% |
| **AIME 2026 (no tools)** | 88.3% | 92.7% |
| **LiveCodeBench v6** | 77.1% | 80.4% |
| **Codeforces ELO** | 1718 | — |
| **GPQA Diamond** | 82.3% | 86.0% |
| **BigBench Extra Hard** | 64.8% | — |
| **MMMU Pro (Vision)** | 73.8% | 75.3% |
| **HLE (no tools)** | 8.7% | 21.4% |
| **SWE-bench Verified** | — | 73.4% |
| **MMLU-Redux** | — | 93.3% |
| **Total Parameters** | 25.2B | 35B |
| **Active Parameters** | 3.8B | 3B |
| **Context Length** | 256K | 262K |
| **Vision** | Yes | Yes |
| **License** | Apache 2.0 | Apache 2.0 |

> Qwen3.6 leads on reasoning/coding benchmarks; Gemma 4 26B A4B has a larger active parameter count (3.8B vs 3B) and multimodal (image) support out of the box in llama.cpp via mmproj.

---

# Original unsloth README

---

license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
tags:
- gemma4
- unsloth
- gemma
- google

---

# Read our How to [Run Gemma 4 Guide!](https://docs.unsloth.ai/models/gemma-4)
<div>
  <p style="margin: 0 0 0px 0; margin-top: 0px;">
    <em>See <a href="https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf">Unsloth Dynamic 2.0 GGUFs</a> for our quantization benchmarks.</em>
  </p>
  <div style="display: flex; gap: 5px; align-items: center; margin-bottom: 0px;">
    <a href="https://github.com/unslothai/unsloth/">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="133">
    </a>
    <a href="https://discord.gg/unsloth">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/Discord%20button.png" width="173">
    </a>
    <a href="https://unsloth.ai/docs/models/gemma-4">
      <img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png" width="143">
    </a>
  </div>

  <ul style="margin: 0;">
    <li><b>Apr 11 Update:</b> Re-download for Google's latest chat template and llama.cpp fixes.</li>
    <li>Gemma 4 can now be run and fine-tuned in <a href="https://unsloth.ai/docs/new/studio">Unsloth Studio</a>. <a href="https://unsloth.ai/docs/models/gemma-4">Read our guide</a>.</li>
    <li>See all versions of Gemma 4 (GGUF, 16-bit etc.) <a href="https://huggingface.co/collections/unsloth/gemma-4">in our collection</a>.</li>
  </ul>
</div>

---

<div align="center">
  <img src=https://ai.google.dev/gemma/images/gemma4_banner.png>
</div>

<p align="center">
    <a href="https://huggingface.co/collections/google/gemma-4" target="_blank">Hugging Face</a> |
    <a href="https://github.com/google-gemma" target="_blank">GitHub</a> |
    <a href="https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/" target="_blank">Launch Blog</a> |
    <a href="https://ai.google.dev/gemma/docs/core" target="_blank">Documentation</a>
    <br>
    <b>License</b>: <a href="https://ai.google.dev/gemma/docs/gemma_4_license" target="_blank">Apache 2.0</a> | <b>Authors</b>: <a href="https://deepmind.google/models/gemma/" target="_blank">Google DeepMind</a>
</p>

Gemma is a family of open models built by Google DeepMind. Gemma 4 models are multimodal, handling text and image input (with audio supported on small models) and generating text output. This release includes open-weights models in both pre-trained and instruction-tuned variants. Gemma 4 features a context window of up to 256K tokens and maintains multilingual support in over 140 languages.

Featuring both Dense and Mixture-of-Experts (MoE) architectures, Gemma 4 is well-suited for tasks like text generation, coding, and reasoning. The models are available in four distinct sizes: **E2B**, **E4B**, **26B A4B**, and **31B**. Their diverse sizes make them deployable in environments ranging from high-end phones to laptops and servers, democratizing access to state-of-the-art AI.

Gemma 4 introduces key **capability and architectural advancements**:

* **Reasoning** – All models in the family are designed as highly capable reasoners, with configurable thinking modes.
* **Extended Multimodalities** – Processes Text, Image with variable aspect ratio and resolution support (all models), Video, and Audio (featured natively on the E2B and E4B models).
* **Diverse & Efficient Architectures** – Offers Dense and Mixture-of-Experts (MoE) variants of different sizes for scalable deployment.
* **Optimized for On-Device** – Smaller models are specifically designed for efficient local execution on laptops and mobile devices.
* **Increased Context Window** – The small models feature a 128K context window, while the medium models support 256K.
* **Enhanced Coding & Agentic Capabilities** – Achieves notable improvements in coding benchmarks alongside native function-calling support, powering highly capable autonomous agents.
* **Native System Prompt Support** – Gemma 4 introduces native support for the `system` role, enabling more structured and controllable conversations.

## Models Overview

| Property | 26B A4B MoE |
| :---- | :---- |
| **Total Parameters** | 25.2B |
| **Active Parameters** | 3.8B |
| **Layers** | 30 |
| **Sliding Window** | 1024 tokens |
| **Context Length** | 256K tokens |
| **Vocabulary Size** | 262K |
| **Expert Count** | 8 active / 128 total and 1 shared |
| **Supported Modalities** | Text, Image |
| **Vision Encoder Parameters** | ~550M |

## Benchmark Results

|  | Gemma 4 31B | Gemma 4 26B A4B | Gemma 4 E4B | Gemma 4 E2B | Gemma 3 27B (no think) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| MMLU Pro | 85.2% | 82.6% | 69.4% | 60.0% | 67.6% |
| AIME 2026 no tools | 89.2% | 88.3% | 42.5% | 37.5% | 20.8% |
| LiveCodeBench v6 | 80.0% | 77.1% | 52.0% | 44.0% | 29.1% |
| Codeforces ELO | 2150 | 1718 | 940 | 633 | 110 |
| GPQA Diamond | 84.3% | 82.3% | 58.6% | 43.4% | 42.4% |
| Tau2 (average over 3) | 76.9% | 68.2% | 42.2% | 24.5% | 16.2% |
| HLE no tools | 19.5% | 8.7% | — | — | — |
| HLE with search | 26.5% | 17.2% | — | — | — |
| BigBench Extra Hard | 74.4% | 64.8% | 33.1% | 21.9% | 19.3% |
| MMMLU | 88.4% | 86.3% | 76.6% | 67.4% | 70.7% |
| **Vision** | | | | | |
| MMMU Pro | 76.9% | 73.8% | 52.6% | 44.2% | 49.7% |
| MATH-Vision | 85.6% | 82.4% | 59.5% | 52.4% | 46.0% |
| **Long Context** | | | | | |
| MRCR v2 8 needle 128k | 66.4% | 44.1% | 25.4% | 19.1% | 13.5% |

## Best Practices

* `temperature=1.0`, `top_p=0.95`, `top_k=64`
* Enable thinking: include `<|think|>` token in system prompt
* Place images **before** text in multimodal prompts
* Vision token budgets: 70, 140, 280, 560, 1120 — use lower for speed, higher for OCR/documents

## Getting Started

```python
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-26B-A4B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
processor.parse_response(response)
```

## License

Apache 2.0 — see [license link](https://ai.google.dev/gemma/docs/gemma_4_license).
