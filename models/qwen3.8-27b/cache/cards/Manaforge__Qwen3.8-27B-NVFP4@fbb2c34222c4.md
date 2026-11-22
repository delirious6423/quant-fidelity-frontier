---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: evoke
pipeline_tag: text-generation
inference: false
tags:
  - manaforge
  - evoke
  - qwen3.8
  - quantized
  - nvfp4
  - fp4
---

# Qwen3.8-27B-NVFP4

An NVFP4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Manaforge, built for the [evoke](https://github.com/manaforged/evoke) inference engine.

Against the BF16 model, it has a mean KL divergence of 0.0169 and picks the same next token at 95.50% of positions, averaged over five corpora. It is 24.9 GB on disk.

Released 2026-11-21 (`evoke 0.1.0`).

## Model overview

| | |
| --- | --- |
| Base model | Qwen/Qwen3.8-27B (Apache-2.0) |
| Format | NVFP4 (E2M1 weights, groups of 16, E4M3 block scales), with sensitive layers in FP8 and BF16 (see [Precision map](#precision-map)) |
| Size on disk | 24.9 GB |
| Multi-token prediction | The MTP head is kept in BF16 |
| Input and output | Text and inline images (vision encoder kept in BF16) |
| Developer | Manaforge Technologies |
| Version | 0.1.0 |

## Hardware and engine

| GPU | Path | Accuracy |
| --- | --- | --- |
| NVIDIA `sm_120` (RTX 5090, RTX PRO 6000) | Native NVFP4: 4-bit weights and activations | Measured, this card |
| Older NVIDIA GPUs (`sm_80`–`sm_90`) | Emulated W4A4 FP4 MMA; needs ~25 GB+ VRAM (two 24 GB cards with `--tensor-parallel-size 2`) | Not yet measured |

The numbers on this card come from evoke on one `sm_120` GPU (`--tensor-parallel-size 1`).

The checkpoint is tested with evoke only. We have not checked whether vLLM, SGLang, or other engines load it.

## Usage

Install evoke (`pip install evoke-llm` or build from [github.com/manaforged/evoke](https://github.com/manaforged/evoke)), then start an OpenAI- and Anthropic-compatible server on one GPU:

```sh
evoke serve Manaforge/Qwen3.8-27B-NVFP4 --host 127.0.0.1 --port 8000
```

To split the model across two GPUs, add `--tensor-parallel-size 2`.

Generate without a server:

```sh
evoke generate Manaforge/Qwen3.8-27B-NVFP4 --prompt "The capital of France is" --max-tokens 64
```

**Two-term activations (opt-in).** By default, evoke reads one activation plane per linear. To enable two-term activations (the code plane plus its remainder plane) on the modules listed in `hf_quant_config.json`, pass `--two-term-rows every` (or `--two-term-rows generated` for decode rows only). Using two-term FP8 activations on the linear-attention and MLP layers lowers mean KL to 0.0152 and raises top-1 agreement to 96.04%, at a cost in prefill time.

## Quantization method

Weights were quantized with GPTQ. Each NVFP4 block scale is chosen inside the GPTQ solve, then the scales are refit twice with the codes fixed. Layers that lost the most accuracy in 4 bits were moved to FP8, one group at a time, by measured KL. The recipe is `weights-gptq-replay-refit-codefirst-pfmap-chatcal` (full manifest in `conversion-manifest.json`).

### Precision map

| Precision | Modules |
| --- | --- |
| NVFP4 | All other quantized linears: the gated attention projections, and MLP layers 0-47 |
| FP8 | Gated DeltaNet projections `in_proj_qkv`, `in_proj_z`, `out_proj` (all 48 linear-attention layers); MLP `gate_proj`, `up_proj`, `down_proj` in layers 48-63 |
| BF16 | `lm_head`, `embed_tokens`, Gated DeltaNet `conv1d`, `in_proj_a`, `in_proj_b`, the MTP head, the vision encoder |

### Calibration

1,536 samples (sequence length 512, 774,462 tokens) from a code-first mix of source code (`bigcode/the-stack-smol-xs`, 480 samples), distilled chat/code/math (`a-m-team/AM-Qwen3-Distilled`, 896 samples), and multilingual web text (`allenai/c4` en/de/fr/es/ja/zh/ar, 160 samples). Exact revisions and splits are recorded in `conversion-manifest.json`.

## Fidelity against BF16

KL is the per-token KL divergence of this checkpoint from the BF16 reference over the full vocabulary (measured with two-term activations enabled). Top-1 agreement is the share of positions where both models rank the same token first. Each corpus has 16,384 positions.

| Corpus | Mean KL | Median KL | p99 KL | Top-1 agreement | Perplexity (this / BF16) |
| --- | --- | --- | --- | --- | --- |
| Code (codeparrot-clean valid) | 0.0084 | 0.0016 | 0.0924 | 97.19% ± 0.13% | 3.569 / 3.559 |
| Multilingual (fineweb-2 test: deu, fra, jpn, arb) | 0.0126 | 0.0054 | 0.1142 | 94.46% ± 0.18% | 10.243 / 10.149 |
| Math (hendrycks_math test) | 0.0074 | 0.0022 | 0.0914 | 96.34% ± 0.15% | 3.303 / 3.289 |
| Chat (ultrachat_200k test_sft) | 0.0424 | 0.0046 | 0.4803 | 94.66% ± 0.18% | 13.415 / 13.381 |
| Wikitext (wikitext-2-raw-v1 test) | 0.0137 | 0.0052 | 0.1409 | 94.84% ± 0.17% | 7.273 / 7.216 |
| **Mean of 5** | **0.0169** | **0.0038** | **0.1838** | **95.50%** | |

Top-1 agreement is a per-token measure. It is not comparable to task-level accuracy recovery percentages.

## Comparison with other NVFP4 checkpoints

Same method for every row: one engine (evoke), one BF16 reference, the same five corpora and tokens. Each public checkpoint was served as published, except the row marked "two-term activations", which is the unchanged checkpoint with evoke's two-term activations turned on.

| Checkpoint | Mean KL | Mean top-1 agreement |
| --- | --- | --- |
| This checkpoint, maximum-accuracy setting (opt-in) | 0.0152 | 96.04% |
| **This checkpoint** | **0.0169** | **95.50%** |
| unsloth, two-term activations | 0.0234 | 94.13% |
| unsloth, as published | 0.0341 | 92.80% |
| RadixArk, BF16 LM head | 0.0459 | 91.96% |
| NVIDIA | 0.0480 | 90.61% |
| QUASAR | 0.0559 | 90.96% |
| RadixArk | 0.0584 | 89.53% |

Mean KL is 2.0x lower than the best public NVFP4 checkpoint as published (0.0341), and 28% lower than that checkpoint served with the same two-term activations (0.0234). The table is not size-matched; public checkpoints may score differently on the engines they target.

## Speed

See [BENCHMARKS.md](https://github.com/manaforged/evoke/blob/main/BENCHMARKS.md) in the evoke repository for RTX 5090 TP1 and TP2 throughput, latency, and determinism benchmarks.

## Limits

- Accuracy is measured on evoke only. On an engine without two-term activations, expect a higher KL.
- Accuracy on emulated W4A4 paths for pre-Blackwell GPUs (`sm_80`–`sm_90`) is not yet measured.
- KL and top-1 agreement measure fidelity to BF16, not downstream task quality.
- Chat text accounts for roughly half of the mean KL.
- The base model's limitations and intended uses apply. See the [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B).

## License

Apache-2.0, the license of the base model. See [LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE).

## Citation

```bibtex
@misc{manaforge2026qwen38nvfp4,
  title  = {Qwen3.8-27B-NVFP4},
  author = {{Manaforge Technologies}},
  year   = {2026},
  url    = {https://huggingface.co/Manaforge/Qwen3.8-27B-NVFP4}
}
```

Base model:

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```

## Contact

Manaforge Technologies: thomas@manaforge.tech, [github.com/manaforged](https://github.com/manaforged). Report problems in the [Community tab](https://huggingface.co/Manaforge/Qwen3.8-27B-NVFP4/discussions).
