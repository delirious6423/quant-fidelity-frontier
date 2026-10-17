---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- reasoning
- vision-language
- personal-model
- control
- baseline-comparator
- mlx
- apple-silicon
- 8-bit
---

# Qwen3.8-27B-Opus-Reasoning-Control-MLX-8bit

This is the immutable trained-control comparator for the release family. It contains the reasoning QLoRA merge before Abliterix, allowing direct comparison and fallback without presenting the winner as universally dominant.

**Comparator status:** trained reasoning control, not the untouched upstream Qwen base and not an Abliterix model.

## Why this release

The target was a practical personal reasoning/VLM model with much lower reflexive refusal while retaining measured capability and an immutable fallback. The benchmark table is the decision record: it shows where the selected winner improved, where the control stayed stronger, and why both are published. The family therefore describes the winner as a **practical personal-model selection with measured deviations**, not universal dominance.

## What this variant is

- **Role:** Baseline comparator
- **Format:** Affine 8-bit MLX (group size 64) with an optional native MTP drafter under `mtp/`.
- **Base:** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Pipeline:** image + text to text
- **License:** Apache-2.0

## Frozen local results

> **These are self-run, frozen local project benchmarks, not official Qwen benchmarks.** The same local harness compared `control-bf16` with `abliterix-pass1-bf16`.

| Frozen local metric | Control | Abliterix winner |
|---|---:|---:|
| Harmful hard refusal | 43.2% | **0.0%** |
| Harmful soft deflection | 14.6% | **0.2%** |
| Harmful substantive response | 47.0% | **99.4%** |
| Capability macro | 17.6859% | **21.0086%** |
| Full code | **16/421** | 10/421 |
| HumanEval | **7.9268%** | 4.2683% |
| Long-form pass | 54.1667% | **62.5000%** |
| MMMU30 | 9/30 | **11/30** |
| Held-out loss ratio | 1.000000 | 1.024478 |
| Benign KL | 0.000000 | 0.093614 |

The winner improved the local capability macro by 3.3227 points, long-form pass rate by 8.3333 points, and MMMU30 by 2 correct answers while reducing harmful hard refusal by 43.2 points. The control remained better on full code/HumanEval.

Full machine-readable values are in [`benchmark-results.json`](./benchmark-results.json).

## Strict deviations and code pathology

- Benign KL was `0.093614`, above the strict `0.05` limit.
- Incoherence was `4.3077%`, above the `2.7692%` strict comparison point.
- HumanEval was `4.2683%` versus control `7.9268%`; full code was `10/421` versus `16/421`.
- Long-form maximum repeated-4gram fraction was `5.8632%`, above the `5%` limit.
- Prompt leakage was detected: 3 exact winner prompt echoes versus 2 for control; both failed the leakage hard gate.
- 376/421 winner code generations hit the 512-token cap. Among generations that reached execution, the winner passed 10/46 (21.74%) versus control 16/103 (15.53%), pointing to severe termination/extraction pathology rather than a clean latent-code estimate.

## Training data and run

| Aggregate source label | Rows |
|---|---:|
| `opus-10000x` | 9,633 |
| `opus-3000x` | 2,326 |
| `reasoning-700x` | 633 |
| `high-reasoning-250x` | 250 |
| **Raw total** | **12,842** |

After 208 deduplications and 20 invalid-row removals, 12,614 rows remained: 12,349 train, 127 validation, and 138 test. The processed-manifest SHA-256 is `6e0a36ad20732c5f98ff592c4565a4c86876fead4e9f94bc6ceedfad1339a94d`. Only aggregate counts and hashes are published; raw/private rows are not.

Training ran for 1,544 optimizer steps with 108,789,760 trainable LoRA parameters. Final validation loss was `0.23739749`; token accuracy was `91.7594%`.

## Architecture and lineage

The model uses `Qwen3_5ForConditionalGeneration` as recorded by the released config: a 64-layer, hidden-size-5120 text stack with a 3:1 linear/full-attention schedule and a configured 262,144-token maximum position range, paired with a 27-layer, hidden-size-1152 vision encoder. It is an image/text conditional-generation model, not a text-only checkpoint.

### Method

1. Started from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0.
2. Prepared 12,842 raw reasoning rows: 12,614 accepted after removing 208 duplicates and 20 invalid rows, then split into 12,349 train / 127 validation / 138 test rows.
3. Trained a reasoning QLoRA on the 12,349-row train split for 1,544 optimizer steps with 108,789,760 trainable LoRA parameters. Final validation loss was 0.23739749 and token accuracy was 91.7594%.
4. Merged the adapter to BF16 to form immutable `control-bf16`.
5. Produced the selected winner with Abliterix 1.12.2 pass 1 (seed 42): orthogonal/projected, winsorized single-direction residual steering over the output/down-projection writer components. Q/K/V projections were excluded.
6. Preserved BF16 releases and converted both variants to affine MLX 8-bit, group size 64. Native one-layer MTP drafters were split and validated separately.

The final selection label is `selected_practical_winner_with_measured_deviations`. Private training data, raw harmful/benign prompt sets, operational receipts, local paths, and Drive metadata are intentionally not published.

## Tensor and conversion integrity

The merged control retained all 1,199 tensor keys, including 15 native MTP tensors and 333 vision tensors. It received no Abliterix residual-writer edits.

MLX native proof ran through `mlx-vlm` on arm64 macOS/Metal. Ordinary and MTP-assisted generation both produced the exact answer `323` for winner and control. The winner MLX build used `mlx` 0.32.0, `mlx-lm` 0.31.3, `mlx-vlm` 0.6.13, and affine 8-bit group-64 quantization.

## Hash and size summary

- `MLX control 8-bit model`: 15 files, 29,521,446,704 bytes, sealed aggregate SHA-256 `832ee3922e9efd4c4a0ac7fd46c181146d715458c022f300330745fb82b66391`
- `Native MTP drafter`: 4 files, 869,393,759 bytes, sealed aggregate SHA-256 `09f4819f5e6660e7c3e262ed3a5903ca56b35cf79fdba2568f364f88453cbef2`

The deliberately published model payload for this repository is 17 files / 30,390,836,197 bytes. Per-file source SHA-256 values are in [`SHA256SUMS`](./SHA256SUMS) and [`manifests/artifact-manifest.json`](./manifests/artifact-manifest.json). Sealed aggregate hashes above cover the complete local source components, including private entries excluded from publication; use the per-file public manifest for the Hub payload.

## Use with MLX

Install a current `mlx-vlm` on Apple Silicon:

```bash
pip install -U mlx-vlm huggingface_hub
python -m mlx_vlm generate \
  --model timteh673/Qwen3.8-27B-Opus-Reasoning-Control-MLX-8bit \
  --prompt "Calculate 17 multiplied by 19. Show concise reasoning and end with the numeric answer." \
  --max-tokens 192 --temperature 0 --seed 42 \
  --thinking-mode enabled --thinking-budget 128 --skip-special-tokens --no-verbose
```

For the optional native MTP drafter, download the repository once so the `mtp/` subdirectory is addressable:

```bash
hf download timteh673/Qwen3.8-27B-Opus-Reasoning-Control-MLX-8bit --local-dir ./qwen38-model
python -m mlx_vlm generate \
  --model ./qwen38-model \
  --draft-model ./qwen38-model/mtp --draft-kind mtp --draft-block-size 3 \
  --prompt "Calculate 17 multiplied by 19. Show concise reasoning and end with the numeric answer." \
  --max-tokens 192 --temperature 0 --seed 42 \
  --thinking-mode enabled --thinking-budget 128 --skip-special-tokens --no-verbose
```

Image input uses the same root model plus `--image /path/to/image.jpg`. The root is deliberately laid out for direct `mlx-vlm` loading; `mtp/` is optional.

## Known limitations

- The control retained much more refusal behavior (43.2% hard refusal on the frozen harmful suite) and scored below the winner on capability macro, long-form, and MMMU30.
- It is a project comparator, not an official Qwen baseline result.
- MLX 8-bit quantization trades precision for Apple-Silicon practicality; numeric behavior can differ from BF16.
- MTP acceptance was validated, but speed is workload-dependent.
- The local benchmark suite and private training/evaluation data are not included. Results may not transfer to other prompts, languages, runtimes, or sampling settings.
- Generated content can be incorrect, insecure, or incomplete; review it before consequential use.

## Release family

- [Recommended MLX 8-bit winner](https://huggingface.co/timteh673/Qwen3.8-27B-Opus-Abliterix-Reasoning-MLX-8bit)
- [BF16 winner](https://huggingface.co/timteh673/Qwen3.8-27B-Opus-Abliterix-Reasoning-BF16)
- [MLX 8-bit control](https://huggingface.co/timteh673/Qwen3.8-27B-Opus-Reasoning-Control-MLX-8bit)
- [BF16 control](https://huggingface.co/timteh673/Qwen3.8-27B-Opus-Reasoning-Control-BF16)

See [`RELEASE_FAMILY.md`](./RELEASE_FAMILY.md) for the role and integrity summary of every variant.

## Responsible use

Evaluate this model for your own setting, isolate untrusted code/tool output, and comply with applicable law and the policies of systems you connect it to. The uncensoring/abliteration work changes refusal behavior; it does not make outputs accurate or safe by default.

## License and attribution

Released under Apache License 2.0. This is a derivative of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), Copyright 2026 Alibaba Cloud. Qwen/Alibaba Cloud attribution and the full license text are retained in [`LICENSE`](./LICENSE). Modifications include the personal reasoning QLoRA merge, selected Abliterix residual edits for winner variants, and MLX quantization where applicable.

## Support

If this release is useful, follow [`timteh673`](https://huggingface.co/timteh673), star the repository, and share reproducible benchmark or runtime findings in the Hub community tab.
