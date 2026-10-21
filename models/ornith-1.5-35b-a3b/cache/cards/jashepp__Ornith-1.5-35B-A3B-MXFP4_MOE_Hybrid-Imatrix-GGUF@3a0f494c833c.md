---
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B
language:
- en
library_name: transformers
base_model_relation: quantized
pipeline_tag: text-generation
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
tags:
- qwen
- qwen3
- qwen3.5
- moe
- distillation
- chain-of-thought
- agentic
- tool-use
- chained-distill
- imatrix
- GGUF
- mxfp4
- quantized
- conversational
---

# 💎 Ornith-1.5-35B-A3B - Custom Mixed Precision GGUFs with Imatrix

> **Ornith-1.5** extends the self-scaffolding framework introduced in Ornith-1.0 into a more complete self-improvement loop:\
> The model **proposes new tasks, generates task-specific scaffolds, and produces solution rollouts** for reinforcement learning, continuously creating new learning experiences from which it can improve.

[![Base model](https://img.shields.io/badge/🤗_Base-ornith--ai%2FOrnith--1.5--35B-blue)](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
[![Ornith AI Blog](https://img.shields.io/badge/📄%20Article-DeepReinforce%20Ornith%201.5%20Blog%20Post%20-FD8E5B)](https://ornith.ai/ornith_1_5.html)
[![License](https://img.shields.io/badge/License-MIT-blue)](https://choosealicense.com/licenses/mit/)

This repository contains custom, highly optimized, **multi-tier mixed precision** GGUF weights for [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

**Ornith-1.5 35B** is the direct successor of [**Ornith-1.0 35B**](https://huggingface.co/ornith-ai/Ornith-1.0-35B), which achieves state-of-the-art performance among open-source models of comparable size across a broad range of agentic coding benchmarks.\
It brings improved instruction following & improved thinking/reasoning, among other benefits.

> [!TIP]
> **Highly Recommended:** Always keep **reasoning/thinking enabled**.\
> **Ornith** thoroughly **plans** and **reasons** through code edits **before** execution, ensuring an **efficient and clean output**.\
> Unlike baseline Qwen models, which frequently execute blindly and backtrack after generating broken code.

<img style="width: 100%; max-width: 900px;" src="https://ornith.ai/ornith_1_5/ornith_35b_eval_1787116402.webp" alt="Ornith 1.5 35B A3B Benchmark Results" title="Ornith 1.5 35B A3B Benchmark Results">

To learn more about Ornith 1.5, [read their blog post](https://ornith.ai/ornith_1_5.html).\
To learn more about how to use Ornith 1.5 35B A3B, [view the base model](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).\
A smaller variant is also available: [Ornith-1.5-9B](https://huggingface.co/jashepp/Ornith-1.5-9B-MXFP4_Hybrid-Imatrix-GGUF)

These quants were generated using manual layer targeting to maximize quality while shrinking the massive VRAM footprint of the Mixture of Experts layers.

## 📄 GGUF Files

In order of quality:

| Filename | Size | Quants |
| :--- | :--- | :--- |
| **[Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0_F16-Imatrix.gguf](https://huggingface.co/jashepp/Ornith-1.5-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF/tree/main?show_file_info=Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0_F16-Imatrix.gguf)** | **20.7 GB** | **MXFP4_MOE + Q8_0 + F16** |
| **[Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf](https://huggingface.co/jashepp/Ornith-1.5-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF/tree/main?show_file_info=Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf)** | **19.8 GB** | **MXFP4_MOE + Q8_0** |
| **[Ornith-1.5-35B-A3B-MXFP4_MOE-Only-Imatrix.gguf](https://huggingface.co/jashepp/Ornith-1.5-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF/tree/main?show_file_info=Ornith-1.5-35B-A3B-MXFP4_MOE-Only-Imatrix.gguf)** | **18.5 GB** | **MXFP4_MOE *Only*** |

**Updated 2026-08-22:**
 - Re-uploaded models without MTP layer

## 📊 Importance Matrix (Imatrix)

The imatrix is a combination of:
  - [bartowski/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF)'s imatrix file
  - 1.6m tokens of my own markdown data from prompts, results, audits, skills & documentation generated via [Ornith-1.0 35B A3B](https://huggingface.co/jashepp/Ornith-1.0-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF) + this model

> [!NOTE]
> The `MXFP4` quantized layers **include imatrix data**, using [this commit](https://github.com/ggml-org/llama.cpp/commit/d9b625edb6d2cba294d51007ed751e3c17879668) on-top of llama.cpp.

---
## 🔍 Precision Matrix & Flavor Variations

Standard global quantization presets (like stock `MXFP4_MOE`) compress the backbone layers uniformly, which degrades the delicate reasoning capabilities of advanced agent models.\
This repository provides multiple distinct manual configuration layouts to balance precision and memory constraints:

### 1. The Tri-Quant Hybrid Flavor (`MXFP4` + `Q8_0` + `F16`)
`Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0_F16-Imatrix.gguf` - Designed for maximum quality preservation, this layout implements a strict **3-Tier Precision Matrix**:

* **Tier 1 (Core & Mamba Gating - F16 Precision):**
  - `token_embd.weight`, `output.weight` - Protects the critical input/output vocabulary mappings. Dramatically prevents text degradation.
  - `ssm_alpha`, `ssm_beta` - Protects the integrity of the Mamba state-space calculations across long-range context tokens.
* **Tier 2 (Backbone & Shared - Q8_0 Precision):** `ssm_out`, `*._shexp` - Keeps the attention mechanics, and all trailing shared experts at high quality, to protect the logical research loops.
* **Tier 3 (Routed Experts - MXFP4 Precision):** `ffn_down_exps`, `ffn_gate_exps`, `ffn_up_exps` - Shrink the massive background expert parameters directly to `MXFP4`.

### 2. The Dual-Quant Hybrid Flavor (`MXFP4` + `Q8_0`)
`Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf` - Designed for a slightly leaner memory profile, this layout utilizes **2-Tier Precision**:
* **Tier 1 (Backbone - Q8_0 Precision):** All attention blocks, Mamba structures, vocabulary embeddings, and internal routers use the universal `Q8_0` format.
* **Tier 2 (Experts - MXFP4 Precision):** The heavy sparse expert blocks are target-quantized directly to `MXFP4`.

### 3. Bonus Single-Quant (`MXFP4`)
`Ornith-1.5-35B-A3B-MXFP4_MOE-Only-Imatrix.gguf` - Using only `MXFP4`, this shrinks the model down to `18.5 GB`. The quality is not the best, but it can still do decent work.

- **Single Tier (All Layers - MXFP4 Precision):** All layers are target-quantized directly to `MXFP4`, for speed and a low VRAM footprint.

---
## 📝 Exact Conversion Details

These files were converted via `llama-quantize` utilizing the following manual recipe parameters:

### Convert SafeTensors to GGUF:
```sh
# Requires python3.12, with `pip install --upgrade transformers`
python convert_hf_to_gguf.py "Ornith-1.5-35B-A3B/" --outtype f16 --outfile "Ornith-1.5-35B-A3B_F16.gguf" --no-mtp
```

### Generate Tri-Quant MXFP4_MOE + Q8_0 + F16:
```sh
llama-quantize \
  --tensor-type ".*_shexp\.weight=Q8_0" \
  --tensor-type "token_embd\.weight=F16" \
  --tensor-type "^output\.weight=F16" \
  --tensor-type "blk\..*\.(ssm_alpha|ssm_beta)\.weight=F16" \
  --tensor-type "blk\..*\.(ffn_down_exps|ffn_gate_exps|ffn_up_exps)\.weight=MXFP4" \
  --imatrix "imatrix.gguf" \
  "Ornith-1.5-35B-A3B_F16.gguf" \
  "Ornith-1.5-35B-A3B-A3B-MXFP4_MOE_Q8_0_F16-Imatrix.gguf" \
  Q8_0
```

### Generate Dual-Quant MXFP4_MOE + Q8_0:
```sh
llama-quantize \
  --tensor-type ".*_shexp\.weight=Q8_0" \
  --tensor-type "blk\..*\.(ffn_down_exps|ffn_gate_exps|ffn_up_exps)\.weight=MXFP4" \
  --imatrix "imatrix.gguf" \
  "Ornith-1.5-35B-A3B_F16.gguf" \
  "Ornith-1.5-35B-A3B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf" \
  Q8_0
```

### Generate Single-Quant MXFP4_MOE:
```sh
llama-quantize \
  --tensor-type ".*_shexp\.weight=MXFP4" \
  --tensor-type "token_embd\.weight=MXFP4" \
  --tensor-type "^output\.weight=MXFP4" \
  --tensor-type "blk\..*\.(ssm_alpha|ssm_beta|ssm_out|attn_gate|attn_qkv|ffn_down|ffn_gate|ffn_up|attn_k|attn_q|attn_v|attn_output)\.weight=MXFP4" \
  --tensor-type "blk\..*\.(ffn_down_exps|ffn_gate_exps|ffn_up_exps)\.weight=MXFP4" \
  --imatrix "imatrix.gguf" \
  "Ornith-1.5-35B-A3B_F16.gguf" \
  "Ornith-1.5-35B-A3B-MXFP4_MOE-Only-Imatrix.gguf" \
  MXFP4_MOE
```

---
## 📝 Local Deployment & `llama-server` Configuration (`config.ini`)

To maintain long solid thinking/reasoning, prevent repetitive loops, have fewer hallucinations, and get higher-quality output, I recommend the following sampling settings & server parameters.

```ini
# --- Samplers (Dynamic & Expressive) ---
# Establishes the foundational pooling and filtering layers to balance creativity with logical precision.
temperature = 0.60
top-k = 35
top-p = 0.93
min-p = 0.10
top-n-sigma = 0.60

# --- Penalties (Prevent Syntax & Reasoner Corruption) ---
# Excluded from the pipeline to protect recurring folder paths and directory prefixes from corruption.
#repeat-penalty = 1.05
#presence-penalty = 1.1

# --- DRY Sampler (Protects Indentation & Structural Boilerplate) ---
# Intelligently limits phrase duplication and structural looping without punishing syntax punctuation or code dividers.
dry-multiplier = 0.8
dry-base = 1.75
dry-allowed-length = 3
dry-penalty-last-n = 1024
dry-sequence-breaker = [ "\n", "```\n", ":", "\t", "\"", "|", "-", "}", "]", "/", "\\" ]

# --- Enforced Execution Graph ---
# Clears the vast vocabulary tail early for speed and lets DRY safely block path duplication loops in a wide pool, 
# while temp prepares multi-token schemas so a late-stage Top-N-Sigma can slice out single-character spelling typos.
samplers = top_k;top_p;min_p;dry;temp;top_n_sigma
```

> [!TIP]
> **Accuracy Tips:**
> - Small amounts of details are mis-remembered during long context windows (>100k).
> - Use `cache-type-k = f16` as anything lower suffers from mis-remembered details (at any context size).
> - If you *must* use `q8_0` for kv cache, or the `MXFP4 + Q8_0` or the `MXFP4 Only` variants, try tweaking `top-n-sigma` to increase accuracy.

> [!NOTE]
> Compared to [Ornith-1.0-35B](https://huggingface.co/jashepp/Ornith-1.0-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF), I have the values tuned for higher quality output.

Highly Recommended: Always keep reasoning/thinking enabled, for better quality results. The higher the budget, the more the model verifies/validates tasks.

```ini
# --- Reasoning ---
chat-template-kwargs = { "enable_thinking":true }
reasoning = on
reasoning-format = auto
reasoning-budget = 32768
```

This works well with `256k` context window.
```ini
fit-ctx = 262144
```

> [!TIP]
> **For Maximum Quality at 100k+ Context:** \
> Use the **MXFP4 + Q8_0 + F16** split-quantized version.
> - **Preserved at F16**: `token_embd.weight`, `output.weight`, `*.ssm_alpha.weight`, and `*.ssm_beta.weight`.
> - **Why this matters**: Keeping these critical layers at full precision **prevents** the model from dropping most of the **fine details** during extreme "needle-in-a-haystack" retrieval tasks (large context windows).
> - **What to avoid**: If `output.weight` or the embedding layers are quantized to `Q8_0` or lower, logit precision rounds off, causing the model to **lose accuracy** and **forget specific details** in long-context scenarios, more often.

**Updated 2026-09-05:**
 - Improved sampling settings **again**. Less over-confidence & higher accuracy (for coding & agentic tasks).

**Updated 2026-08-29:**
 - Added accuracy tips section & updated notes

<details>
  <summary>Earlier Changes</summary>

**Updated 2026-08-26:**
 - Improved sampling settings **again**. It is now almost flawlessly calling correct tool calls, correct path names, variable names, etc. Moved `top_n_sigma` to the end.

**Updated 2026-08-24:**
 - Improved sampling settings for even **higher quality** output, such as moving & tweaking `top_n_sigma`, and removing `penalties` (using `dry` instead). It can still make minor typos, but it's *much better* at Powershell commands and tool calls now.

**Updated 2026-08-23:**
 - Improved sampling settings for higher quality output & added `top_n_sigma`+`penalties`

**Updated 2026-08-22:**
 - Improved sampling settings

</details>

---
### ℹ️ Misc Details

I'm doing this as a side hobby, with my AMD 5900X, 64GB DDR4, **RTX 3060 12GB** & **RTX 5060 Ti 16GB**.

In addition to the above configuration, I also use:

```ini
slots = 1
parallel = 1
no-warmup = true

flash-attn = on
mlock = false
no-mmap = true
context-shift = false

batch-size = 2048
ubatch-size = 256

fit = on
fit-target = 1024
cache-ram = 4096
main-gpu = 0
split-mode = layer
n-gpu-layers = 999
n-cpu-moe = 0
tensor-split = 16,13
override-tensor = (token_embd)=CUDA0,(vision|vpm|nextn)=CPU

fit-ctx = 262144
cache-type-k = q8_0
cache-type-v = q8_0

jinja = true
chat-template = jinja
chat-template-file = chat_template.jinja
```

For further quality and better ssm behaviour, this configuration can help:

```ini
context-shift = false
cache-type-k = f16
cache-type-v = f16
```

---
## 🤝 Support the Journey

As a passionate developer, I'm always programming, automating, or experimenting with new ideas.\
I love building open-source tools, trying out new web tech, and creating things that don't yet exist, including local AI & quantizing models. 

I love sharing these creations to give back to the community.\
If my projects have saved you time or helped you out, consider supporting my work below!

👉 **[Support me on Ko-fi](https://ko-fi.com/jashepp)**

---
## ✨ Acknowledgments
- [ornith-ai](https://huggingface.co/ornith-ai) for the exceptional `Ornith-1.5` base model.
- [bartowski/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF) for the imatrix gguf that I then combined with my own imatrix data.

## 📜 License

Released under **MIT**.

## 🔗 Citation

```bibtex
@misc{ornith_1_5,
    title = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
    url = {https://ornith.ai/ornith_1_5.html},
    author = {{Ornith Team}},
    year = {2026}
}
```