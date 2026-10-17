---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags:
  - qwen
  - qwen3.8
  - qwen35
  - gguf
  - nvfp4
  - mtp
  - speculative-decoding
  - vision
  - llama.cpp
  - ollama
language:
  - en
  - zh
---

# Qwen3.8-27B — NVFP4 + MTP + VL (GGUF, 1M YaRN)

Community **llama.cpp / Ollama** pack of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

This is **not** the official Transformers / vLLM / SGLang checkpoint. The official repo is BF16 safetensors (~55 GB) for Hugging Face Transformers and dedicated servers. This repo is a **quantized GGUF split**: NVFP4 language + trained MTP, plus an F16 CLIP projector, with GGUF metadata patched so runners that honor `n_ctx_train` will accept **1,048,576** tokens (official static YaRN).

| | Official [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | This repo |
|--|--|--|
| Format | Safetensors (Transformers) | GGUF (`qwen35`) |
| Language weights | BF16 | **NVFP4** (embd/output Q5_K) |
| Vision | In the same checkpoint | Separate **F16 mmproj GGUF** (already converted; not fused into the 27B file) |
| MTP | Trained in the model | Kept in the language GGUF (`draft-mtp`) |
| Native context | 262,144 | 262,144 (unchanged weights) |
| 1M context | YaRN in `config.json` / engine flags | **YaRN written into GGUF KV** |
| Typical serve | vLLM, SGLang, TokenSpeed, Transformers | llama.cpp, Ollama |

License is the same **Apache License 2.0** as [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE). The [`LICENSE`](LICENSE) file in this repo is a verbatim copy of that official text (including the Alibaba Cloud 2026 appendix notice).

---

## What you get

| File | Role | Size |
|------|------|------|
| `Qwen3.8-27B-NVFP4-Q5K-mtp-yarn1m.gguf` | Language + MTP only | ~14.6 GiB |
| `mmproj/mmproj-Qwen3.8-27B-F16.gguf` | CLIP vision projector (complete GGUF, already converted) | ~888 MiB |
| `Modelfile` | Ollama dual-`FROM` example | tiny |

**The projector is not inside the 27B GGUF.** That file is `qwen35` language + MTP. CLIP lives in its own `mmproj` GGUF (`general.type = mmproj`, F16). You do **not** convert the official tower, merge tensors, or use `ADAPTER` (that is LoRA). Point the runner at the file we already shipped: `--mmproj` or a second `FROM`.

`mmproj/` is only a **folder** so Hugging Face indexes the 27B language file (Ollama app filter, ~27B / `qwen35`) instead of the 0.5B CLIP file. Same projector weights; not a second copy you have to “embed.”

**NVFP4 applies only to language (and MTP) matmuls.** The vision tower stays F16. 1M context is **not** extra trained weights; it is rope/context metadata so the runner will allocate a 1M window.

---

## Why this pack exists

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is a native VLM (images and video) with thinking on by default, trained MTP, **262,144** native context, and an official path to **1,000,000** tokens via **static YaRN** (`factor: 4.0`, `original_max_position_embeddings: 262144`). Qwen’s hosted API is expected to default to 1M; the open Transformers card still ships 262K in `config.json` and documents YaRN as an override.

Goal for local serve (DGX Spark / Ollama 0.32 / llama.cpp):

1. Keep **NVFP4** so ~27B fits in ~16 GB instead of ~55 GB BF16.
2. Keep **MTP** so speculative decode works (`--spec-type draft-mtp`).
3. Keep **vision** (official CLIP projector).
4. Make **1M** actually loadable in **Ollama**, which caps `num_ctx` at GGUF `n_ctx_train`.

---

## Strategy (what we did and did not do)

### What we did not do

- We did **not** retrain Qwen3.8.
- We did **not** NVFP4-quantize from BF16 on this tree. Current llama.cpp `llama-quantize` **has no NVFP4 target** in `QUANT_OPTIONS` (NVFP4 exists as a runtime ftype). Do not run `llama-quantize … NVFP4` and expect a good file.
- We did **not** bake 1M into the tensors. KV at 1M is still ~61–64 GiB FP16 plus the ~16 GB weights.
- We did **not** put vision into the NVFP4 blob. That would be the wrong precision and the wrong packaging.

### Pipeline

```
Qwen/Qwen3.8-27B (official BF16 VLM + MTP, 262K native / 1M YaRN recipe)
        │
        ├─ language+MTP NVFP4 GGUF
        │     Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF
        │     (text-only file; VL tower not inside)
        │
        ├─ CLIP mmproj F16
        │     convert official HF with llama.cpp / Ollama
        │     (language_model_only: false, vision_config present)
        │
        └─ copy language GGUF → rewrite KV only
              qwen35.context_length = 1048576
              rope.scaling.type = yarn
              rope.scaling.factor = 4.0
              rope.scaling.original_context_length = 262144
              freq_base 1e7, mrope sections [11, 11, 10] already present
```

Ollama is assembled with **two `FROM` lines** (language blob + projector blob). `ADAPTER` is LoRA and is the wrong instruction. `RENDERER`/`PARSER` `qwen3-vl` is for the older `qwen3-vl` arch, not `qwen35`.

### Why the YaRN rewrite

On Ollama 0.32, `PARAMETER num_ctx 1048576` against the stock 262K NVFP4 GGUF is rejected:

`requested context size too large for model num_ctx=1048576 n_ctx_train=262144`

Unified memory was not the limiter. After the KV patch, `ollama show` reports **context length 1048576**. The shipped Modelfile still defaults to **`num_ctx 262144`** so a casual chat does not allocate a 1M KV cache. Pass `options.num_ctx: 1048576` only when you want that window.

Qwen’s own card: static YaRN can **hurt short prompts**. Prefer 8K–262K for chat; use 1M for long jobs.

---

## Use

Needs a llama.cpp / Ollama build that understands **`qwen35`**, **NVFP4**, **mmproj**, and **draft-mtp**.

### llama.cpp

```bash
huggingface-cli download mixbits/Qwen3.8-27B-NVFP4-MTP-VL-GGUF --local-dir ./qwen38-nvfp4

llama-server \
  -m ./qwen38-nvfp4/Qwen3.8-27B-NVFP4-Q5K-mtp-yarn1m.gguf \
  --mmproj ./qwen38-nvfp4/mmproj/mmproj-Qwen3.8-27B-F16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -c 8192
```

Use `-c 262144` or `-c 1048576` only if you have the RAM.

### Ollama

```bash
# after download, from the folder that contains both GGUFs + Modelfile
ollama create qwen38-27b-nvfp4-mtp-vl -f Modelfile
```

```dockerfile
FROM ./Qwen3.8-27B-NVFP4-Q5K-mtp-yarn1m.gguf
FROM ./mmproj/mmproj-Qwen3.8-27B-F16.gguf

PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER min_p 0.0
PARAMETER repeat_penalty 1.0
PARAMETER presence_penalty 0.0
PARAMETER num_ctx 262144
PARAMETER draft_num_predict 4
```

`draft_num_predict 4` → `--spec-type draft-mtp` on Ollama 0.32.x.

Thinking is **on by default** (same as the official model). A vision request with thinking enabled can return **HTTP 200 and empty `content`**. Use instruct / `"think": false` for images.

Open WebUI: leaving `num_ctx` as **Default** uses the Modelfile (262K). Toggling the control **pre-fills 2048** and will silently shrink the window. Set `num_ctx` to `1048576` on this model only if you intend a ~64 GiB KV alloc. Do not raise a global Ollama `OLLAMA_CONTEXT_LENGTH` to 1M.

---

## Sampling (from the official card)

| Mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|------|-------------|-------|-------|-------|------------------|--------------------|
| Thinking (default) | 1.0 | 0.95 | 20 | 0 | 0 | 1.0 |
| Instruct (no think) | 0.7 | 0.80 | 20 | 0 | 1.5 | 1.0 |

Official `reasoning_effort` (`xhigh` / `medium` / `low`) and `preserve_thinking` are Transformers / Qwen Cloud features. GGUF+Ollama exposes thinking on/off more coarsely.

---

## Validation (DGX Spark, GB10, Ollama 0.32.11)

| Check | Result |
|--------|--------|
| Offload | 66/66 GPU, `--mmproj` + `--spec-type draft-mtp` |
| Text | short ping OK |
| Vision | red\|blue PNG described correctly with `think: false` |
| MTP (text-only NVFP4, 128 tok, before VL attach) | ~26 t/s decode, ~470 t/s prompt, ~64% draft accept |
| 1M metadata | `ollama show` context **1048576** |
| 1M KV fill | **not** published as a bench in this card |

Official [text and VL benchmark tables](https://huggingface.co/Qwen/Qwen3.8-27B) are for the **BF16** model, not this quant.

---

## Official model (condensed)

From [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B):

- Causal LM + vision encoder; 27B; hidden 5120; 64 layers; vocab 248,320 (padded)
- Hybrid layout: gated DeltaNet + gated attention (RoPE dim 64, `partial_rotary_factor` 0.25)
- MTP trained with multiple steps
- Native VLM (images and video); thinking on by default
- Context: **262,144** native, extensible to **1,000,000** with YaRN

YaRN fields used here match the official recipe:

```json
{
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": 4.0,
  "original_max_position_embeddings": 262144
}
```

Video `longest_edge` tuning from the official card is a **Transformers / vLLM** concern; this GGUF pack is validated for **images**, not hour-scale video.

---

## Attribution

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — architecture, training, license, YaRN recipe, sampling
- [Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF) — NVFP4 + MTP language GGUF this pack copies and metadata-patches
- This repo ([mixbits](https://huggingface.co/mixbits)) — YaRN/1M GGUF KV, official CLIP mmproj attach, Ollama dual-`FROM` layout, Spark validation

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```
