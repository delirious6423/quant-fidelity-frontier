---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
library_name: transformers
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
- glq
- quantization
- e8-lattice
---

# gemma-4-26B-A4B-it — GLQ 4bpw

> [!IMPORTANT]
> **Superseded by [`xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw`](https://huggingface.co/xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw).**
> the trellis (TCQ) codebook has been the recommended format since v0.7 — same bit-rate, faster single-stream decode, and better quality at low bpw.
>
> ```bash
> vllm serve xv0y5ncu/gemma-4-26B-A4B-it-GLQ-trellis-3inst-4bpw --quantization glq
> ```
>
> These weights remain available and unchanged, so existing references keep working.

**GLQ (4.0 bits/weight)** quantization of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) using an **E8-lattice codebook + randomized Hadamard transform (RHT) + LDLQ**. Weights stay compressed in memory and are dequantized on the fly by fused CUDA kernels.

> **Where GLQ shines:** the size-vs-accuracy win is strongest at **2–4 bits/weight**. For even more headroom, combine it with the **E8 KV cache** (≈4× smaller KV cache → longer context in the same VRAM).

- **Base model:** [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Method:** E8 shell codebook + RHT + LDLQ — 4.0 bpw uniform
- **Avg weight SQNR:** 18.89 dB over 7885 quantized layers
- **Calibration:** 128 samples × 2048 tokens (WikiText-2)

## Quantization fidelity — paired GLQ vs bf16 (thinking mode)

GLQ-4bpw and the bf16 base were run **head-to-head on the same questions**, in thinking mode (this model's intended setting):

| Benchmark | bf16 base | GLQ 4bpw |
| :-- | :-- | :-- |
| MMLU-Pro (n=60, paired) | 91.7% (55/60) | **93.3% (56/60)** |
| AIME-2024 (n=30, paired) | 93.3% (28/30) | **93.3% (28/30)** |

At **~15 GB vs ~50 GB** (~⅓ the weight size), 4-bit GLQ is **close to** bf16 on both reasoning evals.

**AIME thinking-budget sensitivity** — 4-bit reasons ~20% longer (mean 12.5k vs 10.5k tokens), so a tight cap penalizes its *chains*, not its *answers*:

| AIME-2024 budget | bf16 | GLQ 4bpw |
| :-- | :-- | :-- |
| 32k | 93.3% (28/30) | 86.7% (26/30) |
| 64k | 93.3% (28/30) | **93.3% (28/30)** |

bf16 is budget-flat (it finished every problem under 32k); GLQ recovers fully at 64k → **GLQ = bf16 = 28/30**. On this set, the 4-bit cost is slightly lower token-efficiency — *eval thinking models with generous budget*.

**Harder, newer AIME (2025 + 2026, n=60):**

| AIME 2025+2026 (n=60) | bf16 | GLQ 4bpw |
| :-- | :-- | :-- |
| Accuracy | 93.3% (56/60) | 85.0% (51/60) |
| Truncated @ 64k | 7 | 14 |

Here GLQ trails by ~5 problems, and it **truncated 2× as often** (it reasons longer: mean 24k vs 18k tokens, and hits the 64k cap more). Whether that reflects **token-inefficiency** (recoverable with more budget) or **degraded reasoning on the hardest problems** (which more budget would *not* fix) is unresolved.

<sub>Paired, seed-fixed subsets (identical questions per precision). gemma-4 chat template, `enable_thinking=True`, sampling `temperature=1.0, top_p=0.95, top_k=64`, vLLM; thinking budget 16384 (MMLU-Pro) / 65536 (AIME — a 32k cap undersizes GLQ's longer chains and drops it to 86.7%). These are **small-n fidelity** comparisons (95% CI ≈ ±8% MMLU-Pro, ±15% AIME), single-sample pass@1 — *not* leaderboard scores, and not directly comparable to the full-benchmark figures in the original card below. (The default lm-eval `mmlu_pro` task — no chat template, thinking off — scores ~31% for *both* precisions, an answer-extraction artifact; always evaluate this model in thinking mode.)</sub>

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'glq[vllm]' "transformers>=5.13.1,<5.15"
```
**Pin transformers below 5.15** — 5.15.0 moved gemma-4 to a per-layer config, so
`config.head_dim` raises and vLLM dies before loading a single weight. Verified on
5.14.1 with vLLM 0.27.1. Not specific to GLQ: stock bf16 gemma-4 fails the same way.

## Use with vLLM (recommended)

```python
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw",
        quantization="glq",
        dtype="bfloat16",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},  # text-only serving
    )
    out = llm.generate(["The capital city of New Zealand is"],
                       SamplingParams(max_tokens=64, temperature=0))
    print(out[0].outputs[0].text)

# The __main__ guard is REQUIRED when running this as a script. vLLM switches to the
# "spawn" multiprocessing start method once CUDA is initialised, so its worker processes
# re-import this file; without the guard the script spawns itself recursively and dies
# with "An attempt has been made to start a new process before the current process has
# finished its bootstrapping phase" -- before the model ever loads.
if __name__ == "__main__":
    main()
```

`glq` registers with vLLM automatically via its plugin entry point — no extra import needed.

## Use with Transformers

```python
import glq.hf_integration  # registers the GLQ quantization method
from transformers import AutoModelForImageTextToText, AutoTokenizer
import torch

model = AutoModelForImageTextToText.from_pretrained(
    "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw", device_map="cuda", dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw")

msgs = [{"role": "user", "content": "What is the capital city of New Zealand?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
print(tok.decode(model.generate(ids, max_new_tokens=64)[0][ids.shape[1]:], skip_special_tokens=True))
```

> This is a multimodal base model. GLQ quantizes the **text decoder** (the vision/audio towers are kept in their native format). vLLM is the recommended runtime.

## Use with coding agents (pi-code, opencode)

Serve an OpenAI-compatible endpoint, then point your agent at it:

```bash
vllm serve xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw --port 8000
```

**pi-code** — `~/.pi/agent/models.json`:
```json
{
  "providers": {
    "glq": {
      "baseUrl": "http://localhost:8000/v1",
      "api": "openai-completions",
      "apiKey": "glq",
      "models": [
        {
          "id": "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw"
        }
      ]
    }
  }
}
```

**opencode** — `~/.config/opencode/opencode.json`:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "glq": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "GLQ (local vLLM)",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "glq"
      },
      "models": {
        "xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw": {
          "name": "gemma-4-26B-A4B-it-GLQ-4bpw"
        }
      }
    }
  }
}
```

## Smaller KV cache (E8) — long context in the same VRAM

GLQ also ships an **E8-quantized KV cache** (~4× smaller than fp16). Enable it at serve time:

```bash
GLQ_KV_QUANT=e8_relaxed:2 \
GLQ_KV_E8_SIDECAR=1 GLQ_KV_E8_SIDECAR_READ=1 GLQ_KV_E8_COMPRESSED_ALLOC=1 \
GLQ_KV_E8_FUSED_GATHER=1 GLQ_KV_E8_FUSED_WRITE=1 \
vllm serve xv0y5ncu/gemma-4-26B-A4B-it-GLQ-4bpw
```

`e8_relaxed:2` ≈ 4-bit KV, `:1` ≈ 2-bit, `:3` ≈ 6-bit. This is the part of GLQ that keeps paying off **above** 4-bit weights.

## Decode throughput

Batched MoE decode routes to a **grouped-GEMM** kernel — tokens are sorted by expert, then each expert runs one batched tensor-core GEMM — the default in `glq ≥ 0.6.0`. Single-stream (batch 1) decode stays on the bit-exact block-diagonal matvec.

| GPU | batch | decode tok/s | vs. prior block-diag path |
| :-- | :-- | :-- | :-- |
| NVIDIA L4 24 GB (sm89) | 1 | 16.5 | — (b1 path unchanged) |
| NVIDIA L4 24 GB (sm89) | 8 | 63.6 | **1.83×** |
| RTX PRO 6000 96 GB (sm120) | 32 | 607.8 | **6.9×** |

<sub>Measured with vLLM under FULL cudagraph on this 4bpw model, greedy decode. "vs. prior block-diag" is the speedup of the grouped-GEMM default over GLQ's previous per-(token, expert) block-diagonal path on the same GPU/config — the grouped win grows with batch size and GPU. Absolute tok/s depend on GPU, batch, and context length. To force the previous path: `GLQ_MOE_GROUPED=0`.</sub>

> **GLQ vs bf16 speed — be realistic.** The table above is GLQ-vs-GLQ (the new kernel vs GLQ's old one). Versus the **bf16 base** on a *large* GPU where it fits (e.g. 96 GB), native bf16 decode is **faster** than 4-bit GLQ — the dequant cost isn't repaid when there's no memory pressure. GLQ's advantage there is the **~3× smaller footprint** (more KV cache / longer context in the same VRAM). GLQ's *speed* advantage shows up on **memory-constrained GPUs** (e.g. 24/32 GB) where the bf16 model doesn't fit or is bandwidth-bound.

## How GLQ works

GLQ compresses weights in groups of 8. A **randomized Hadamard transform** (a fixed random sign-flip + Hadamard rotation on each side) makes the weights and the calibration Hessian *incoherent* — spreading outliers so an 8-dimensional lattice quantizes them well. **LDLQ** then rounds each group to the nearest entry of an **E8 shell codebook** (a 65,536-point subset of the E8 lattice), feeding each group's rounding error forward across the remaining input dimensions via the Hessian's LDL factorization. At 3–4 bpw a second residual codebook quantizes the leftover error (two-stage residual VQ).

E8 is the 8-dimensional root lattice; the "Golay/Leech" in the project name refers to its higher-dimensional lattice cousins. Code, kernels and quantizer: **[GLQ on GitHub](https://github.com/cnygaard/glq)**.

---

<details>
<summary>Original model card (google/gemma-4-26B-A4B-it)</summary>

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

## **Models Overview**

Gemma 4 models are designed to deliver frontier-level performance at each size, targeting deployment scenarios from mobile and edge devices (E2B, E4B) to consumer GPUs and workstations (26B A4B, 31B). They are well-suited for reasoning, agentic workflows, coding, and multimodal understanding.

The models employ a hybrid attention mechanism that interleaves local sliding window attention with full global attention, ensuring the final layer is always global. This hybrid design delivers the processing speed and low memory footprint of a lightweight model without sacrificing the deep awareness required for complex, long-context tasks. To optimize memory for long contexts, global layers feature unified Keys and Values, and apply Proportional RoPE (p-RoPE). 

### Dense Models

| Property | E2B | E4B | 31B Dense |
| :---- | :---- | :---- | :---- |
| **Total Parameters** | 2.3B effective (5.1B with embeddings) | 4.5B effective (8B with embeddings) | 30.7B |
| **Layers** | 35 | 42 | 60 |
| **Sliding Window** | 512 tokens | 512 tokens | 1024 tokens |
| **Context Length** | 128K tokens | 128K tokens | 256K tokens  |
| **Vocabulary Size** | 262K | 262K | 262K |
| **Supported Modalities** | Text, Image, Audio | Text, Image, Audio | Text, Image |
| **Vision Encoder Parameters** | *~150M* | *~150M* | *~550M* |
| **Audio Encoder Parameters** | *~300M* | *~300M* | No Audio |

The "E" in E2B and E4B stands for "effective" parameters. The smaller models incorporate Per-Layer Embeddings (PLE) to maximize parameter efficiency in on-device deployments. Rather than adding more layers or parameters to the model, PLE gives each decoder layer its own small embedding for every token. These embedding tables are large but are only used for quick lookups, which is why the effective parameter count is much smaller than the total.

### Mixture-of-Experts (MoE) Model

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
| **Vision Encoder Parameters** | *~550M* |

The "A" in 26B A4B stands for "active parameters" in contrast to the total number of parameters the model contains. By only activating a 4B subset of parameters during inference, the Mixture-of-Experts model runs much faster than its 26B total might suggest. This makes it an excellent choice for fast inference compared to the dense 31B model since it runs almost as fast as a 4B-parameter model.

## **Benchmark Results** 

These models were evaluated against a large collection of different datasets and metrics to cover different aspects of text generation. Evaluation results marked in the table are for instruction-tuned models.

|  | Gemma 4  31B | Gemma 4  26B A4B | Gemma 4  E4B | Gemma 4  E2B | Gemma 3  27B (no think) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| MMLU Pro | 85.2% | 82.6% | 69.4% | 60.0% | 67.6% |
| AIME 2026 no tools | 89.2% | 88.3% | 42.5% | 37.5% | 20.8% |
| LiveCodeBench v6 | 80.0% | 77.1% | 52.0% | 44.0% | 29.1% |
| Codeforces ELO | 2150 | 1718 | 940 | 633 | 110 |
| GPQA Diamond | 84.3% | 82.3% | 58.6% | 43.4% | 42.4% |
| Tau2 (average over 3) | 76.9% | 68.2% | 42.2% | 24.5% | 16.2% |
| HLE no tools | 19.5% | 8.7% | - | - | - |
| HLE with search | 26.5% | 17.2% | - | - | - |
| BigBench Extra Hard | 74.4% | 64.8% | 33.1% | 21.9% | 19.3% |
| MMMLU | 88.4% | 86.3% | 76.6% | 67.4% | 70.7% |
| **Vision** |  |  |  |  |  |
| MMMU Pro | 76.9% | 73.8% | 52.6% | 44.2% | 49.7% |
| OmniDocBench 1.5 (average edit distance, lower is better) | 0.131 | 0.149 | 0.181 | 0.290 | 0.365 |
| MATH-Vision | 85.6% | 82.4% | 59.5% | 52.4% | 46.0% |
| MedXPertQA MM | 61.3% | 58.1% | 28.7% | 23.5% | - |
| **Audio** |  |  |  |  |  |
| CoVoST | - | - | 35.54 | 33.47 | - |
| FLEURS (lower is better) | - | - | 0.08 | 0.09 | - |
| **Long Context** |  |  |  |  |  |
| MRCR v2 8 needle 128k (average) | 66.4% | 44.1%  | 25.4% | 19.1% | 13.5% |

## **Core Capabilities**

Gemma 4 models handle a broad range of tasks across text, vision, and audio. Key capabilities include:

* **Thinking** – Built-in reasoning mode that lets the model think step-by-step before answering.
* **Long Context** – Context windows of up to 128K tokens (E2B/E4B) and 256K tokens (26B A4B/31B).
* **Image Understanding** – Object detection, Document/PDF parsing, screen and UI understanding, chart comprehension, OCR (including multilingual), handwriting recognition, and pointing. Images can be processed at variable aspect ratios and resolutions.
* **Video Understanding** – Analyze video by processing sequences of frames.
* **Interleaved Multimodal Input** – Freely mix text and images in any order within a single prompt.
* **Function Calling** – Native support for structured tool use, enabling agentic workflows.
* **Coding** – Code generation, completion, and correction.
* **Multilingual** – Out-of-the-box support for 35+ languages, pre-trained on 140+ languages.
* **Audio** (E2B and E4B only) – Automatic speech recognition (ASR) and speech-to-translated-text translation across multiple languages.

## Getting Started

You can use all Gemma 4 models with the latest version of Transformers. To get started, install the necessary dependencies in your environment:

`pip install -U transformers torch accelerate`

Once you have everything installed, you can proceed to load the model with the code below:

```python
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-26B-A4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)
```

Once the model is loaded, you can start generating output:

```python
# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

# Process input
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True, 
    enable_thinking=False
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)
```

To enable reasoning, set `enable_thinking=True` and the `parse_response` function will take care of parsing the thinking output.

Below, you will also find snippets for processing audio (E2B and E4B only), images, and video alongside text:

<details>
<summary>Code for processing Audio</summary>

Instead of using `AutoModelForCausalLM`, you can use `AutoModelForMultimodalLM` to process audio. To use it, make sure to install the following packages:


`pip install -U transformers torch librosa accelerate`

You can then load the model with the code below:

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-E2B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)
```

Once the model is loaded, you can start generating output by directly referencing the audio URL in the prompt:


```python
# Prompt - add audio after text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
            {"type": "audio", "audio": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/journal1.wav"},
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)
```

</details>

<details>
<summary>Code for processing Images</summary>

Instead of using `AutoModelForCausalLM`, you can use `AutoModelForMultimodalLM` to process images. To use it, make sure to install the following packages:


`pip install -U transformers torch torchvision accelerate`

You can then load the model with the code below:

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-26B-A4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)
```

Once the model is loaded, you can start generating output by directly referencing the image URL in the prompt:


```python
# Prompt - add image before text
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)
```

</details>


<details>
<summary>Code for processing Videos</summary>

Instead of using `AutoModelForCausalLM`, you can use `AutoModelForMultimodalLM` to process videos. To use it, make sure to install the following packages:

`pip install -U transformers torch torchvision torchcodec librosa accelerate`

You can then load the model with the code below:

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "google/gemma-4-26B-A4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)
```

Once the model is loaded, you can start generating output by directly referencing the video URL in the prompt:


```python
# Prompt - add video before text
messages = [
    {
        'role': 'user',
        'content': [
            {"type": "video", "video": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4"},
            {'type': 'text', 'text': 'Describe this video.'}
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)
```

</details>


## **Best Practices**

For the best performance, use these configurations and best practices:

### 1. Sampling Parameters

Use the following standardized sampling configuration across all use cases:

* `temperature=1.0`  
* `top_p=0.95`  
* `top_k=64`

### 2. Thinking Mode Configuration

Compared to Gemma 3, the models use standard `system`, `assistant`, and `user` roles. To properly manage the thinking process, use the following control tokens:

* **Trigger Thinking:** Thinking is enabled by including the `<|think|>` token at the start of the system prompt. To disable thinking, remove the token.   
* **Standard Generation:** When thinking is enabled, the model will output its internal reasoning followed by the final answer using this structure:  
  `<|channel>thought\n`**[Internal reasoning]**`<channel|>`  
* **Disabled Thinking Behavior:** For all models except for the E2B and E4B variants, if thinking is disabled, the model will still generate the tags but with an empty thought block:  
  `<|channel>thought\n<channel|>`**[Final answer]**

> [!Note]
> Note that many libraries like Transformers and llama.cpp handle the complexities of the chat template for you.

### 3. Multi-Turn Conversations

* **No Thinking Content in History**: In multi-turn conversations, the historical model output should only include the final response. Thoughts from previous model turns must *not be added* before the next user turn begins.

### 4. Modality order

For optimal performance with multimodal inputs, place:

* Image content **before** the text in your prompt.
* Audio content **after** the text in your prompt.

### 5. Variable Image Resolution

Aside from variable aspect ratios, Gemma 4 supports variable image resolution through a configurable visual token budget, which controls how many tokens are used to represent an image. A higher token budget preserves more visual detail at the cost of additional compute, while a lower budget enables faster inference for tasks that don't require fine-grained understanding.

* The supported token budgets are: **70**, **140**, **280**, **560**, and **1120**.  
  * Use *lower budgets* for classification, captioning, or video understanding, where faster inference and processing many frames outweigh fine-grained detail.   
  * Use *higher budgets* for tasks like OCR, document parsing, or reading small text.

### 6. Audio

Use the following prompt structures for audio processing:

* **Audio Speech Recognition (ASR)**

```text
Transcribe the following speech segment in {LANGUAGE} into {LANGUAGE} text.

Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
```

* **Automatic Speech Translation (AST)**

```text
Transcribe the following speech segment in {SOURCE_LANGUAGE}, then translate it into {TARGET_LANGUAGE}.
When formatting the answer, first output the transcription in {SOURCE_LANGUAGE}, then one newline, then output the string '{TARGET_LANGUAGE}: ', then the translation in {TARGET_LANGUAGE}.
```

### 7. Audio and Video Length

All models support image inputs and can process videos as frames whereas the E2B and E4B models also support audio inputs. Audio supports a maximum length of 30 seconds. Video supports a maximum of 60 seconds assuming the images are processed at one frame per second.

## **Model Data**

Data used for model training and how the data was processed.

### **Training Dataset**

Our pre-training dataset is a large-scale, diverse collection of data encompassing a wide range of domains and modalities, which includes web documents, code, images, audio, with a cutoff date of January 2025. Here are the key components:

* **Web Documents**: A diverse collection of web text ensures the model is exposed to a broad range of linguistic styles, topics, and vocabulary. The training dataset includes content in over 140 languages.  
* **Code**: Exposing the model to code helps it to learn the syntax and patterns of programming languages, which improves its ability to generate code and understand code-related questions.  
* **Mathematics**: Training on mathematical text helps the model learn logical reasoning, symbolic representation, and to address mathematical queries.  
* **Images**: A wide range of images enables the model to perform image analysis and visual data extraction tasks.

The combination of these diverse data sources is crucial for training a powerful multimodal model that can handle a wide variety of different tasks and data formats.

### **Data Preprocessing**

Here are the key data cleaning and filtering methods applied to the training data:

* **CSAM Filtering**: Rigorous CSAM (Child Sexual Abuse Material) filtering was applied at multiple stages in the data preparation process to ensure the exclusion of harmful and illegal content.  
* **Sensitive Data Filtering**: As part of making Gemma pre-trained models safe and reliable, automated techniques were used to filter out certain personal information and other sensitive data from training sets.  
* **Additional methods**: Filtering based on content quality and safety in line with [our policies](https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf).

## **Ethics and Safety**

As open models become central to enterprise infrastructure, provenance and security are paramount. Developed by Google DeepMind, Gemma 4 undergoes the same rigorous safety evaluations as our proprietary Gemini models. 

### **Evaluation Approach**

Gemma 4 models were developed in partnership with internal safety and responsible AI teams. A range of automated as well as human evaluations were conducted to help improve model safety. These evaluations align with [Google’s AI principles](https://ai.google/principles/), as well as safety policies, which aim to prevent our generative AI models from generating harmful content, including:

* Content related to child sexual abuse material and exploitation   
* Dangerous content (e.g., promoting suicide, or instructing in activities that could cause real-world harm)   
* Sexually explicit content  
* Hate speech (e.g., dehumanizing members of protected groups)   
* Harassment (e.g., encouraging violence against people)

### **Evaluation Results**

For all areas of safety testing, we saw major improvements in all categories of content safety relative to previous Gemma models. Overall, Gemma 4 models significantly outperform Gemma 3 and 3n models in improving safety, while keeping unjustified refusals low. All testing was conducted without safety filters to evaluate the model capabilities and behaviors. For both text-to-text and image-to-text, and across all model sizes, the model produced minimal policy violations, and showed significant improvements over previous Gemma models' performance. 

## **Usage and Limitations**

These models have certain limitations that users should be aware of.

### **Intended Usage**

Multimodal models (capable of processing vision, language, and/or audio) have a wide range of applications across various industries and domains. The following list of potential uses is not comprehensive. The purpose of this list is to provide contextual information about the possible use-cases that the model creators considered as part of model training and development.

* **Content Creation and Communication**  
  * **Text Generation**: These models can be used to generate creative text formats such as poems, scripts, code, marketing copy, and email drafts.  
  * **Chatbots and Conversational AI**: Power conversational interfaces for customer service, virtual assistants, or interactive applications.  
  * **Text Summarization**: Generate concise summaries of a text corpus, research papers, or reports.  
  * **Image Data Extraction**: These models can be used to extract, interpret, and summarize visual data for text communications.  
  * **Audio Processing and Interaction**: The smaller models (E2B and E4B) can analyze and interpret audio inputs, enabling voice-driven interactions and transcriptions.  
* **Research and Education**  
  * **Natural Language Processing (NLP) and VLM Research**: These models can serve as a foundation for researchers to experiment with VLM and NLP techniques, develop algorithms, and contribute to the advancement of the field.  
  * **Language Learning Tools**: Support interactive language learning experiences, aiding in grammar correction or providing writing practice.  
  * **Knowledge Exploration**: Assist researchers in exploring large bodies of text by generating summaries or answering questions about specific topics.

### **Limitations**

* **Training Data**  
  * The quality and diversity of the training data significantly influence the model's capabilities. Biases or gaps in the training data can lead to limitations in the model's responses.  
  * The scope of the training dataset determines the subject areas the model can handle effectively.  
* **Context and Task Complexity**  
  * Models perform well on tasks that can be framed with clear prompts and instructions. Open-ended or highly complex tasks might be challenging.  
  * A model's performance can be influenced by the amount of context provided (longer context generally leads to better outputs, up to a certain point).  
* **Language Ambiguity and Nuance**  
  * Natural language is inherently complex. Models might struggle to grasp subtle nuances, sarcasm, or figurative language.  
* **Factual Accuracy**  
  * Models generate responses based on information they learned from their training datasets, but they are not knowledge bases. They may generate incorrect or outdated factual statements.  
* **Common Sense**  
  * Models rely on statistical patterns in language. They might lack the ability to apply common sense reasoning in certain situations.

### **Ethical Considerations and Risks**

The development of vision-language models (VLMs) raises several ethical concerns. In creating an open model, we have carefully considered the following:

* **Bias and Fairness**  
  * VLMs trained on large-scale, real-world text and image data can reflect socio-cultural biases embedded in the training material. Gemma 4 models underwent careful scrutiny, input data pre-processing, and post-training evaluations as reported in this card to help mitigate the risk of these biases.  
* **Misinformation and Misuse**  
  * VLMs can be misused to generate text that is false, misleading, or harmful.  
  * Guidelines are provided for responsible use with the model, see the [Responsible Generative AI Toolkit](https://ai.google.dev/responsible).  
* **Transparency and Accountability**  
  * This model card summarizes details on the models' architecture, capabilities, limitations, and evaluation processes.  
  * A responsibly developed open model offers the opportunity to share innovation by making VLM technology accessible to developers and researchers across the AI ecosystem.

**Risks identified and mitigations**:

* **Generation of harmful content**: Mechanisms and guidelines for content safety are essential. Developers are encouraged to exercise caution and implement appropriate content safety safeguards based on their specific product policies and application use cases.  
* **Misuse for malicious purposes**: Technical limitations and developer and end-user education can help mitigate against malicious applications of VLMs. Educational resources and reporting mechanisms for users to flag misuse are provided.   
* **Privacy violations**: Models were trained on data filtered for removal of certain personal information and other sensitive data. Developers are encouraged to adhere to privacy regulations with privacy-preserving techniques.  
* **Perpetuation of biases**: It's encouraged to perform continuous monitoring (using evaluation metrics, human review) and the exploration of de-biasing techniques during model training, fine-tuning, and other use cases.

### **Benefits**

At the time of release, this family of models provides high-performance open vision-language model implementations designed from the ground up for responsible AI development compared to similarly sized models.

</details>

---

*Derivative work of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it), quantized with [GLQ](https://github.com/cnygaard/glq). It inherits the base model's license (`apache-2.0`) — please respect the base model's terms.*

⭐ [GLQ on GitHub](https://github.com/cnygaard/glq)
