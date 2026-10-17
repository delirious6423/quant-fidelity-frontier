---
license: apache-2.0
library_name: mlx
tags:
  - mlx
  - krill
  - qwen3_5
  - qwen3.8
  - vision-language
  - 4-bit
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B — MLX int4 (complete VLM, Krill-native)

A **mirror** of the MLX 4-bit quantization of
**[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**, packaged for
[Krill](https://github.com/srvsngh99/Krill)'s native Apple-silicon runtime.

> **Provenance, plainly.** The original model and weights are by
> **[Qwen (Alibaba)](https://huggingface.co/Qwen)**. The 4-bit MLX conversion is
> **[mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)** —
> this repo mirrors those weights **byte-for-byte**, it does not re-quantize
> them. It exists so Krill's `qwen3.8-27b` alias resolves to a pinned, tested
> blob. If you are not using Krill, prefer pulling from mlx-community directly.

## Why this build

- 👁️ **Complete vision-language model — the vision tower is included** and kept
  at bf16 while the decoder is quantized, so image and video understanding work,
  not just text.
- ⚡ **Native Krill runtime.** Runs as **native Swift + MLX** on Apple Silicon.
  Qwen3.8 is a `qwen3_5`-class hybrid, and Krill ships a from-scratch native
  runtime for that architecture — the GatedDeltaNet (SSM) linear-attention
  layers, the gated full-attention layers, the vision tower and the 3D
  interleaved mRoPE are all Swift. No Python bridge, no mlx_vlm passthrough.
- 🧠 **Thinking mode with tunable depth**, including the `reasoning_effort`
  control (`xhigh` / `medium` / `low`).
- 📏 **262,144-token context** natively.

## Fit before you pull

**~16.1 GiB resident.** That fits a 24 GB Apple-silicon box, but with little
headroom — expect memory pressure under concurrent load, and prefer closing
other large apps. On a 16 GB machine it will swap; don't.

## Run in Krill (recommended)

```bash
# install Krill
brew tap srvsngh99/krill && brew install krill
# or:
curl -fsSL https://raw.githubusercontent.com/srvsngh99/Krill/main/install.sh | sh

# run Qwen3.8-27B (pulls this repo)
krill run qwen3.8-27b "Explain why a hybrid linear-attention decoder saves memory."

# with an image
krill run qwen3.8-27b --image diagram.png "What does this diagram show?"

# dial the reasoning depth down for short answers
KRILL_REASONING_EFFORT=low krill run qwen3.8-27b "One-line summary of RAII."
```

## Run with mlx_vlm

```bash
pip install -U mlx-vlm
python -m mlx_vlm generate --model srv-sngh/Qwen3.8-27B-mlx-4bit \
  --prompt "Describe this image." --image path/to/image.jpg --max-tokens 200
```

## About Qwen3.8-27B

A dense **27B vision-language model** built on the Qwen3.5 architecture: 64
layers arranged as **16 × (3 × Gated DeltaNet → 1 × Gated Attention)**, hidden
size 5120, FFN 17408, 24 query / 4 KV heads at head-dim 256, and a
Qwen3-VL-style vision tower (depth 27, patch 16, spatial merge 2). Thinking mode
is on by default; context is 262,144 tokens natively and extensible to ~1M with
YaRN.

Selected results from the [official model card](https://huggingface.co/Qwen/Qwen3.8-27B)
(full model, not this quantization):

| Benchmark | Qwen3.8-27B |
|---|---|
| SWE-bench Pro | 61.7 |
| QwenSWEBench | 79.0 |
| Terminal Bench 2.1 (Terminus) | 73.0 |
| CoWorkBench (long-horizon office work) | 70.7 |
| LiveCodeBench v6 | 90.3 |
| GPQA Diamond | 89.2 |
| IFBench | 79.5 |

## Quantization

| field | value |
|---|---|
| mode | affine |
| bits | 4 |
| group size | 64 |
| vision tower | preserved (bf16, not quantized) |
| MTP head | stripped |
| on disk | ~16.1 GiB |

## Sampling

Per Qwen's recommendations:

- **Thinking mode:** `temperature=1.0`, `top_p=0.95`, `top_k=20`
- **Instruct (non-thinking) mode:** `temperature=0.7`, `top_p=0.80`, `top_k=20`,
  `presence_penalty=1.5`

## License

Apache-2.0, inherited from the upstream model. Credit for the model belongs to
**[Qwen](https://huggingface.co/Qwen)**; credit for the MLX conversion belongs to
**[mlx-community](https://huggingface.co/mlx-community)**.

---

## About this build

Quantized and tested by [Sourav Singh](https://souravailabs.ai) for [Krill](https://github.com/srvsngh99/Krill), a Mac-native LLM runtime and coding agent in one Swift + MLX binary. More Apple Silicon builds: [huggingface.co/srv-sngh](https://huggingface.co/srv-sngh). Notes on building with local models and frontier AI: [souravailabs.ai](https://souravailabs.ai) · [GitHub](https://github.com/srvsngh99) · [LinkedIn](https://www.linkedin.com/in/srv-sngh) · [YouTube](https://www.youtube.com/@SouravAILabs)

