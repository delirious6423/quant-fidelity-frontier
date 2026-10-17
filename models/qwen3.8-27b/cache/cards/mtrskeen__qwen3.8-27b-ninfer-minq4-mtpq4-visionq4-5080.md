---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - multimodal
  - conversational
  - cuda
  - blackwell
  - rtx-5080
  - 16gb
  - min-q4
  - mtp
---

# Qwen3.8-27B NInfer — RTX 5080 16 GB fork artifact (min-Q4 + MTP-Q4 + Vision-Q4)

This repository contains **one** NInfer `.ninfer` artifact for
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), converted and
requantized for a single **RTX 5080 16 GB** (`sm_120`) by the fork
[mtrskeen/ninfer-5080](https://github.com/mtrskeen/ninfer-5080), which extends
[Neroued/ninfer](https://github.com/Neroued/ninfer) and the A5000 16 GB fork
[aljazceru/ninfer](https://github.com/aljazceru/ninfer).

The artifact is intended **only for NInfer**. It is not a Transformers
checkpoint, safetensors distribution, GGUF, or MLX file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_minq4_mtpq4_visionq4.ninfer` |
| Size | 15,514,935,296 bytes (14.45 GiB) |
| SHA-256 | `c7eb6fdde74bfd70e168a9ccce113b08fb34ad7ce3349ea989590c4830591fb8` |
| Container version | 2 (`NINFER\0\x02`) |
| NInfer target key | `qwen3_8_27b` |
| Weights ID | `groupwise-int` |
| Stored objects | 1,115 tensors + 7 resources (per server load report) |
| Max context tested | 124,928 tokens (text); 32,768 (vision) |

Contents versus the registered full-precision-preserving profile:

- **Text body**: minimal all-Q4 layout (MLP, GDN/attention inputs via fused
  Q4/Q4 kernels, outputs via Q4 `linear_add`, Q4 output head); token embedding
  is stored Q6. Quality gate: simulated PPL +0.04 versus the IQ3_XXS anchor.
- **MTP draft head**: matrices requantized to `Q4G64_F16S`.
- **Vision tower**: large matrices requantized to `Q4G64_F16S`; Vision buffers
  are allocated lazily.
- All tensors are 256-byte aligned and the payload is 4096-byte aligned.

## Verification

```bash
sha256sum qwen3_8_27b_minq4_mtpq4_visionq4.ninfer
# c7eb6fdde74bfd70e168a9ccce113b08fb34ad7ce3349ea989590c4830591fb8
```

## Usage

See [mtrskeen/ninfer-5080](https://github.com/mtrskeen/ninfer-5080) for the
build and serving instructions. Recommended 16 GB profile:

```bash
ninfer-serve qwen3_8_27b_minq4_mtpq4_visionq4.ninfer \
  --model-id qwen3.8-27b-16gb \
  --host 0.0.0.0 --port 18100 \
  --max-context 124928 --kv-capacity 124928 \
  --kv-dtype i4 --spec mtp --draft-tokens 3 \
  --prefill-chunk 256 --no-cuda-graph
```

Vision is a separate mode: add `--vision` and cap context at 32,768 (the Vision
runtime has a hard 32,768 merged-token envelope, and `--vision` at 124,928 OOMs
at startup on 16 GB).

## Measured performance (RTX 5080 16 GB, this artifact)

Baseline profile with `--prefill-chunk 256` (validated, not the old 64):

| prompt tokens | prefill tok/s | TTFT | peak VRAM |
|---:|---:|---:|---:|
| 32,714 | 1,500 | 22 s | 15,858 MiB |
| 97,873 | 1,116 | 88 s | 15,858 MiB |
| 122,861 | 1,017 | 121 s | 15,858 MiB |

MTP3 decode is strongly content-dependent: ~144 tok/s on structured output
(97 % acceptance, 3.90 tokens/round), ~117–133 tok/s on code/translation,
~71–86 tok/s on story/long reasoning (31–43 % acceptance). MTP0 is a flat
~53 tok/s. `gsm8k_cot` (8-shot, lm-evaluation-harness, 50 samples): **0.96**. RULER NIAH retrieval at 8K/32K/64K: 3/3 each.

Full methodology, per-fixture results, raw CSV/JSON and the context/OOM
boundaries are in the fork's `BENCHMARKS.md`.

## Limitations

- RTX 5080 / `sm_120a` build. Other `sm_120` 16 GB cards are **expected
  compatible, not hardware-tested**.
- Vision context is capped at 32,768 merged tokens by the runtime.
- Text + MTP3 safe ceiling on 16 GB is 124,928; MTP0 reaches 131,072.
- Greedy MTP3 output is bit-identical to MTP0 on short answers but can diverge
  on long generations at near-tie tokens (both remain coherent).
- `--prefill-chunk 256` is a performance profile: it can change exact token
  choices versus chunk 64 at near-ties (decode speed is unchanged), and it
  lowers the absolute capacity boundary to 125,952 while the safe ceiling stays
  124,928.
- `--lm-head-draft` does not fit at the 124,928 ceiling and needs a smaller
  context.

## Attribution and license

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
  (Apache-2.0).
- Runtime: [Neroued/ninfer](https://github.com/Neroued/ninfer).
- 16 GB min-Q4 conversion and sm_86 port: `aljazceru/ninfer`.
- Blackwell MTP-Q4 / Vision-Q4 requantization, 16 GB serving profile,
  validation: `mtrskeen/ninfer-5080`.

Licensed under Apache-2.0; see `LICENSE`.
