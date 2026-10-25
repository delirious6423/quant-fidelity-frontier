---
license: apache-2.0
base_model:
  - google/gemma-4-26B-A4B-it
base_model_relation: quantized
quantized_by: xero0000
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - gemma4
  - moe
  - imatrix
  - mixed-precision
  - iq4_xs
  - all-vram
  - llama.cpp
language:
  - en
---

# Gemma-4-26B-A4B-it — vram14 (IQ4_XS + imatrix)

**~13.3 GB all-VRAM mixed GGUF** of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
(26B MoE, ~4B active), tuned so the full weight set fits dual mid-range GPUs and still
beats the Q8 source on a coding/agent holdout.

> TL;DR: 4.52 bpw custom mix (experts IQ4_XS, attention Q5_K, dense FFN Q6_K) + imatrix.
> On a 3060 Ti 8 GB + 3080 10 GB desktop: **~3060 t/s prefill / ~98 t/s decode** at 128K
> all-VRAM — about **13.5× prefill and 5.3× decode** vs the same model’s Q8 production
> recipe (CPU-MoE offload). Perplexity is quality-neutral vs Q8 on the target corpus.

---

## File

| File | Size | BPW | Notes |
|---|---:|---:|---|
| `gemma-4-26B-A4B-it-vram14-iq4xs-imat.gguf` | **13.30 GiB** | **4.52** | production daily driver |

Source was the official instruct `Q8_0` GGUF (~25 GiB), requantized with an importance
matrix (`--allow-requantize`).

## Architecture notes

Gemma-4 26B-A4B GGUF layout (relevant for the recipe):

- **30 layers**, **128 experts / 8 active**
- Fused expert mat `ffn_gate_up_exps` + `ffn_down_exps` dominate size (~90% of Q8 weights)
- **Tied embeddings** (no separate `output.weight`)
- `ffn_down_exps` has **704 columns** → not 256-divisible; stock IQ4_XS fails there, so
  that tensor uses **IQ4_NL** instead

This arch currently needs a **Gemma-4–capable llama.cpp build** (TurboQuant / gemma4 fork
or equivalent). Plain older mainline binaries that predate `gemma4` will not load it.

## Recipe

Built with `llama-quantize` (TurboQuant gemma4 tree), imatrix-guided:

| Tensor class | Type | Rationale |
|---|---|---|
| `ffn_gate_up_exps` | **IQ4_XS** | size lever (experts dominate) |
| `ffn_down_exps` | **IQ4_NL** | 704-col fallback (IQ4_XS not valid) |
| `attn_(q\|k\|v\|output).weight` | **Q5_K** | every-token attention path |
| dense `ffn_(up\|gate\|down).weight` | **Q6_K** | shared / non-expert FFN |
| `ffn_gate_inp` (router) | **Q8_0** | routing is tiny and quality-critical |
| token embeddings | **Q6_K** | tied embd / output path |

Base ftype `q4_K` so custom `--tensor-type` rules apply. Approximate command:

```bash
llama-quantize --allow-requantize --imatrix gemma4-it.imatrix.gguf \
  --token-embedding-type q6_K \
  --tensor-type 'ffn_gate_up_exps=iq4_xs' \
  --tensor-type 'ffn_down_exps=iq4_nl' \
  --tensor-type 'attn_(q|k|v|output)\.weight=q5_K' \
  --tensor-type 'ffn_(up|gate|down)\.weight=q6_K' \
  --tensor-type 'ffn_gate_inp=q8_0' \
  gemma-4-26B-A4B-it-Q8_0.gguf \
  gemma-4-26B-A4B-it-vram14-iq4xs-imat.gguf \
  q4_K
```

**Importance matrix:** computed on a local coding/agent calibration corpus
(~90 chunks, same domain family as the holdout). Imatrix here acts as light
domain adaptation as well as rounding guidance.

## Quality (perplexity)

Holdout corpus, `n_ctx=512`, 24 chunks (same methodology as the local vram13 series).
**Gemma PPL scale is not comparable to Qwen-family numbers** (different tokenizer /
corpus fit) — only within-family deltas matter.

| Model | PPL | vs Q8 same-placement |
|---|---:|---|
| Q8_0 (auto-fit placement) | 26.588 | baseline |
| Q8_0 (nc23 CPU-MoE offload) | 27.061 | placement alone ~+1.8% |
| **vram14-iq4xs-imat (this file)** | **25.954** | **−2.4%** |
| Q4_K_M-imat (stock, discarded) | 24.834 | −6.6% (larger / slower / not all-VRAM) |

Verdict on this holdout: **4-bit is quality-neutral to slightly better than Q8**, because
the imatrix + domain match more than offsets quantization noise. A stock Q4_K_M scored a
bit lower PPL but was **larger (15.6 GiB)**, needed CPU-MoE offload on this rig, and lost
badly on runtime — it was deleted after the bench.

## Runtime

Rig: **RTX 3060 Ti 8 GB + RTX 3080 10 GB** (18 GB total), Ryzen 5950X, DDR4.
Engine: Gemma-4–capable llama.cpp (TurboQuant fork), `q4_0` KV, flash-attn on.
Perf prompt ≈ 6K-token prefill + 160 decode.

| Config | Prefill | Decode | Checks |
|---|---:|---:|---|
| **vram14 @128K all-VRAM** (ts 44,56, ub 512) | **3061.5 t/s** | **97.5 t/s** | exact ✓ |
| **vram14 @192K all-VRAM** | **3038.6 t/s** | **98.1 t/s** | exact ✓ reasoning ✓ browser ✓ |
| Q8 @256K nc23 no-op-offload (production ref) | 226.2 t/s | 18.5 t/s | exact ✓ |
| vram14 @256K or ubatch 1024 @192K | — | — | OOM (GDN chunked compute buffer) |

**vram14 vs Q8 production: ~13.5× prefill, ~5.3× decode.**

### VRAM ceilings on 18 GB dual-GPU

| ctx | fit | slack (approx) | recommendation |
|---|---|---|---|
| **128K** | all-VRAM | ~1.4 GiB | **daily driver** |
| **192K** | all-VRAM | ~750 MiB | light desktop only |
| **256K** | OOM | — | needs offload / more VRAM |

At razor-edge VRAM the dual-GPU **pipeline-parallel compute buffer reserve**
(~1.3–1.9 GiB on CUDA0) can log a transient allocation failure; the server then
retries without pipeline parallelism and recovers at the speeds above. That log
line is expected and harmless on this class of rig when fully loaded.

## How to run

**Thinking-capable instruct model** — leave thinking enabled for agent/tool loops
(official IT is trained to plan). Recommended sampling (Gemma defaults):
`temp 1.0`, `top_k 64`, `top_p 0.95`. Clients that force temp ≈ 0.2 make tool
loops pathologically deterministic.

### Production (128K, all-VRAM)

```bash
./llama-server \
  -m gemma-4-26B-A4B-it-vram14-iq4xs-imat.gguf \
  --jinja --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on \
  --ctx-size 131072 --parallel 1 --n-gpu-layers 99 \
  --split-mode layer --tensor-split 44,56 \
  --batch-size 2048 --ubatch-size 512 \
  --temp 1.0 --top-k 64 --top-p 0.95 \
  --no-mmap --threads 8 --no-warmup \
  --port 8000
```

### Max context on 18 GB dual mid-range (192K, tight)

Same as above with `--ctx-size 196608`. Expect ~750 MiB free after settle — fine for
interactive use, risky for long multi-tool sessions with a busy desktop compositor.

### Single GPU

- **≥16 GB** with headroom: drop `--tensor-split` / use a single-card split.
- **≤12 GB**: you will need expert offload (`--n-cpu-moe` / equivalent) and will lose
  most of the all-VRAM speedup; prefer a smaller quant or more VRAM.

Adjust `--tensor-split` for your card sizes (44,56 targets 8+10 GB).

## Intended use & limitations

- **Target:** local chat / coding / agent workloads on ~16–18 GB total VRAM where you
  want Gemma-4 IT quality without Q8’s CPU-MoE tax.
- Multimodal (image) support depends on the **runtime and GGUF export**, not just
  weights — this release is validated as a **text** server quant.
- 4-bit experts are the quality floor vs full Q8; on *this* holdout the gap was
  closed by imatrix domain match, but other domains may differ.
- Inherits capabilities, refusal behavior, and biases of
  **google/gemma-4-26B-A4B-it**. No fine-tune — pure quantization.

## Provenance

| | |
|---|---|
| Base | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (Google DeepMind) |
| Source GGUF | official instruct Q8_0 (Unsloth-labeled metadata in header) |
| Quant + imatrix + benches | **xero0000**, 2026-07-18 |
| Tooling | TurboQuant / gemma4 `llama-quantize` + `llama-imatrix` |

**License:** Apache-2.0 (same family as the base —
[Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license)).
Quantization does not change the model license.
