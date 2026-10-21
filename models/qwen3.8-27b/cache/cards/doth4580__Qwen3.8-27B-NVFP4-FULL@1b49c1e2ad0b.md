---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
tags:
- nvfp4
- compressed-tensors
- gb10
- veloGB10
- qwen3.5
---

# Qwen3.8 27B — NVFP4 for veloGB10 (NVIDIA DGX Spark / GB10)

> **Original model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Alibaba's Qwen team,
> [Apache-2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE). This artifact contains
> **modified weights** (NVFP4 quantization); the original model is © its authors.

NVFP4-quantized variant of **Qwen3.8 27B** (27B). **This quantized artifact exists first and foremost to run with the
[veloGB10](https://github.com/sf-stav/veloGB10) inference engine** (`gb10_inference`) on NVIDIA
DGX Spark / GB10 systems — it is produced, validated, and tuned for veloGB10's kernels.
The weights use the standard HuggingFace `compressed-tensors` layout (`nvfp4-pack-quantized`),
so this artifact can be used **for any purpose, with any framework that reads the format**. It is
fit and *proven* to work with veloGB10 — greedy-lossless speculative decoding, two-node TP=2 serving,
and the engine's correctness suite all pass on this artifact.

> **Note on modality:** the underlying model is Qwen's hybrid Qwen3.5-family architecture and the
> checkpoint ships a vision tower (`Qwen3_5ForConditionalGeneration`). This artifact is currently
> presented as a **text/language** model — the vision (image/video) part is **still being worked on**
> and is not yet validated end-to-end. The language half (the dense GatedDeltaNet + GQA stack) is the
> tested, production path.

## Specifications

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Architecture | dense hybrid Qwen3.5 (48 GatedDeltaNet + 16 full-GQA layers), hidden 5120, 24 Q / 4 KV heads, head_dim 256 |
| Attention | hybrid GatedDeltaNet (linear attention, fixed-size recurrent state) + periodic full GQA (every 4th layer) |
| MTP head | native multi-token-prediction head (used for speculative decoding) |
| Context | 262,144 tokens |
| Vocabulary | 248,320 |
| Full-attention interval | 4 |
| Quantization | `nvfp4-full` (`recipe: all`) — all GEMMs NVFP4; the fastest recipe, the right 27B default at every context length |
| Format | `nvfp4-pack-quantized` (compressed-tensors compatible) |
| On-disk size | ~16 GB |

## Running with veloGB10

Full instructions: **[veloGB10 README](https://github.com/sf-stav/veloGB10)** (build or download a release binary, then).
The binary loads its PTX kernel artifacts from the current directory — run it from wherever the
binary + `*.ptx` live.

This model is launched with the bundled Qwen3.8-27B **DFlash2** drafter for speculative decoding
(`--spec-source dflash2-auto` + `--draft-dir`). It uses the native MTP head alongside the drafter as
the auto policy decides. Both TP=4 and TP=2 layouts are supported.

### Four nodes — TP=4

```bash
./gb10_inference --server \
  --model-dir /path/to/3.8-27b-nvfp4-full-all \
  --tp 4 \
  --nodes <peer-ip1>:29500, <peer-ip2>:29500, <peer-ip3>:29500 \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on \
  --default-presence-penalty 1.5 \
  --mtp=auto \
  --spec-source dflash2-auto \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2
```

(Three peer IPs listed because the head is the fourth node.)

### Two nodes — TP=2

```bash
./gb10_inference --server \
  --model-dir /path/to/3.8-27b-nvfp4-full-all \
  --tp 2 \
  --nodes <peer-ip>:29500 \
  --port 9000 \
  --max-seq-len 262144 \
  --max-batch 1 \
  --max-tokens 65536 \
  --prefix-cache on \
  --default-presence-penalty 1.5 \
  --mtp=auto \
  --spec-source dflash2-auto \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2
```

> **You must also download the DFlash2 drafter** (the speculative-decoding draft model used via
> `--spec-source dflash2-auto`). It is published at
> **[doth4580/Qwen3.8-27B-DFlash2](https://huggingface.co/doth4580/Qwen3.8-27B-DFlash2)** — grab it
> with e.g. `hf download doth4580/Qwen3.8-27B-DFlash2 --local-dir /path/to/Qwen3.8-27B-DFlash2`
> (or via the web UI), then point `--draft-dir` at that local directory. If you only care about the
> model on its own you can omit `--spec-source` / `--draft-dir` and run without the drafter.

## Measured on veloGB10 (greedy, NVFP4)

Single-stream decode, greedy, NVFP4 with DFlash 2. Figures are representative; real numbers vary
with content type. **Highest performance is on code generation.** "Average" is a representative
blend across content types (a code-heavy run averages much higher).

| Mode | Average | Bottoms | Peaks | Max sustained (code) |
|---|---:|---:|---:|---:|
| Single node | **> 40 tok/s** | ~11 tok/s | ~100 tok/s | **~70 tok/s** |
| TP=2 | **~56 tok/s** | ~18 tok/s | ~105 tok/s | **~85 tok/s** |
| TP=4 | **~85 tok/s** | ~32 tok/s | ~150 tok/s | **~125 tok/s** |

> Averages span a wide range by content type: mixed-content runs average ~25 / ~38 / ~51 tok/s
> (single / TP=2 / TP=4) vs code-heavy sustained runs ~70 / ~85 / ~125 tok/s. Peaks are the top
> reached on code-generation content.

Life throughput traces and the benchmark methodology are in the
[veloGB10 README](https://github.com/sf-stav/veloGB10).

## License & credits

- **Original model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — © Alibaba Qwen team,
  [Apache License 2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE). A copy of the license
  is included in this repository (`LICENSE`).
- **This artifact:** quantized from the original with veloGB10's offline quantizer
  (`nvfp4-full`). Weights are modified relative to the original; tokenizer and chat template are
  unchanged from the original model.
- **Engine:** [veloGB10](https://github.com/sf-stav/veloGB10) (Apache-2.0).
