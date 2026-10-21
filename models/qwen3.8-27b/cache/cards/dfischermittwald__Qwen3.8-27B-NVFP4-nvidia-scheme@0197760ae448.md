---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- nvfp4
- fp4
- quantized
- compressed-tensors
- llm-compressor
- vllm
- speculative-decoding
- dflash2
- blackwell
- sm120
---

# Qwen3.8-27B-NVFP4-nvidia-scheme

> **This is a control experiment, not the recommended checkpoint.** Use
> [`Qwen3.8-27B-NVFP4-DFlash2`](https://huggingface.co/dfischermittwald/Qwen3.8-27B-NVFP4-DFlash2)
> instead: 24 GB against this one's 29 GB, 109.7 against 91.9 decode tok/s, same
> DFlash2 compatibility, same measured fidelity. This checkpoint is published because it
> is the run that *proves* where the bottleneck was — see "why this exists".

> ### This checkpoint rests on one upstream assumption
>
> The **only** difference from [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
> is that `lm_head` is left unquantized, and the only reason that matters is that DFlash2, as
> written in [vllm-project/vllm#52816](https://github.com/vllm-project/vllm/pull/52816),
> *refuses* a quantized target LM head:
>
> ```
> ValueError: DFlash2 requires an unquantized target LM head for candidate TopK
> ```
>
> That requirement is an implementation choice in an unmerged PR, not a property of the
> method. [#52883](https://github.com/vllm-project/vllm/pull/52883) already relaxed the guard
> once (from `UnquantizedEmbeddingMethod` to also accept `UnquantizedLinearMethod`). If review
> relaxes it further into a real fallback, unsloth's 22 GB checkpoint becomes DFlash2-capable
> and this 24 GB one has no advantage left. **Everything measured below stays true; the reason
> to prefer this checkpoint does not.**

> ### ⚠️ Work in progress, not production-blessed
>
> This checkpoint only does what it is for on a vLLM build that carries **two unmerged pull
> requests**: [#52816](https://github.com/vllm-project/vllm/pull/52816) (DFlash2) and
> [#52883](https://github.com/vllm-project/vllm/pull/52883) (its LM-head guard fix). Neither
> is in any release. Concretely:
>
> - **No stock vLLM can use the DFlash2 path.** Without those patches this serves as an
>   ordinary mixed-precision checkpoint and nothing more.
> - **The PR is moving.** It went from 1 to 9 commits in two days under active review by vLLM
>   maintainers. The measurements here were taken against head `ed34bf91`; the PR has since
>   moved on. Numbers are internally consistent, but a rebuild today gets a different drafter.
> - **The reason this checkpoint exists could be reviewed away.** DFlash2 currently *refuses*
>   a quantized target LM head. That is an API wart, and if reviewers replace the hard error
>   with a fallback, `unsloth/Qwen3.8-27B-NVFP4` becomes DFlash2-capable at 22 GB and this
>   24 GB checkpoint loses its advantage entirely.
> - **`--async-scheduling` cannot be used** with `method: dflash`; vLLM raises. That is a real
>   structural disadvantage against MTP in production, not a misconfiguration.
> - Measured on **one** node (RTX PRO 6000 Blackwell, SM 12.0, TP=1). No multi-node, no TP>1,
>   no long-horizon soak beyond the numbers stated below.
>
> Correctness and fidelity were gated before any speed number was quoted (see Validation), so
> the risk here is about *availability and longevity*, not about the weights being wrong.

NVFP4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced on a single NVIDIA RTX PRO 6000 Blackwell Server Edition (SM 12.0).

The scheme is copied from NVIDIA's own NVFP4 recipe. The one property that makes this
checkpoint different from the NVFP4 quantizations already on the Hub is in the name:
**`lm_head` is not quantized.**

## Why this exists

[vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816) adds DFlash2, a
block-diffusion drafter for speculative decoding. Its candidate selector runs a TopK
over the *target* model's LM head, and it refuses a quantized one:

```
ValueError: DFlash2 requires an unquantized target LM head for candidate TopK
```

[`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
quantizes it — its `config_groups.group_0.targets` contains `re:.*lm_head` at
`num_bits: 8`. So that checkpoint cannot host DFlash2 at all, no flag changes this,
and the fastest weight format available for this model was the one combination that
could not be measured.

That mattered because DFlash2 measurably wins on this model. Measured on one RTX PRO
6000 Blackwell, single stream, `min_tokens 400`, decode throughput timed separately
from TTFT, best of two runs, `temperature 1.0 / top_p 0.95 / top_k 20`:

| Target weights | Drafter | ctx 2048 | ctx 8192 | ctx 32768 | acceptance length @2048 |
|---|---|---:|---:|---:|---:|
| BF16 | none | 26.2 | 26.0 | — | — |
| BF16 | MTP, n=3 | 42.5 | 40.5 | — | 2.09 |
| BF16 | **DFlash2, n=7** | **68.9** | **51.2** | — | **3.17** |
| FP8 | none | 44.3 | 44.1 | 41.9 | — |
| FP8 | MTP, n=3 | 65.4 | 57.5 | 39.2 | 2.16 |
| FP8 | **DFlash2, n=7** | **84.9** | **81.2** | **49.8** | **2.61** |

Decode tokens/s. DFlash2 beats the model's built-in MTP head at every context length
on both weight formats — 2.63× over autoregressive on BF16, 1.92× on FP8, against
1.62× and 1.48× for MTP. At 32k context MTP drops *below* the autoregressive baseline
(39.2 against 41.9) while DFlash2 still gains.

## Measured result, and the honest conclusion

It works, and it answers the question — but not in this checkpoint's favour.

| Configuration | size | ctx 2048 | ctx 8192 | ctx 32768 | acceptance |
|---|---:|---:|---:|---:|---:|
| this checkpoint, no drafter | 29 GB | 46.1 | 46.0 | 44.8 | — |
| this checkpoint, MTP n=3 | 29 GB | 65.7 | 70.2 | — | 2.06 |
| **this checkpoint, DFlash2 n=7** | 29 GB | **91.9** | **88.3** | **87.8** | 2.69 |
| `unsloth/Qwen3.8-27B-NVFP4`, MTP n=3 | 22 GB | 83.4 | 82.9 | 85.3 | 2.13 |
| **`Qwen3.8-27B-NVFP4-DFlash2`** (the recipe to use), DFlash2 n=7 | 24 GB | **109.7** | **104.8** | **110.8** | 2.77 |

DFlash2 gives **1.99×** over autoregressive on this checkpoint and **1.40×** over its
built-in MTP head on the very same weights (91.9 against 65.7) — in line with the 1.92×
and 1.30× measured on FP8. It is also remarkably context-stable: 91.9 → 87.8 from 2k
to 32k, where the FP8 checkpoint fell 84.9 → 49.8 (FP8 KV cache and a 1.26M-token pool).

And it still loses to unsloth's checkpoint driven by the *weaker* drafter, because of size.

**NVIDIA's scheme does not transfer well to this model.** It declines to quantize
attention, and in Qwen3.8-27B attention (`self_attn` plus `linear_attn` across 64 layers)
is a large share of the parameters — so "NVFP4" here weighs 29 GB, exactly as much as plain
FP8. Gemma-4-31B, where NVIDIA cut this recipe, has a smaller attention share and the
recipe pays off there. unsloth quantizes attention to FP8 and lands at 22 GB.

Size explains most of that gap: 29 GB against the 24 GB of the mixed-precision recipe is
1.21 on bytes, and 46.1 against 51 autoregressive tokens/s is 1.11. Not calibration, not
the kernel, not the recipe's correctness. All figures same-node, same-protocol.

**So if you want DFlash2 on a fast NVFP4 Qwen3.8, do not copy NVIDIA's scheme; copy
unsloth's (FP8 attention + NVFP4 MLP) and exclude `lm_head`.** That was done, it is
`Qwen3.8-27B-NVFP4-DFlash2`, and it measures 109.7 tok/s. This checkpoint is the
controlled experiment that isolates the drafter from the quantization; it is not the
fast configuration.

## Quantization recipe

Read directly off [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)'s
`config.json` and transposed to this model:

| | NVIDIA (Gemma-4-31B) | this checkpoint (Qwen3.8-27B) |
|---|---|---|
| Format | NVFP4, W4A4 | same |
| `group_size` | 16 | 16 |
| Scales | static (`dynamic: false`) | same |
| `targets` | `["Linear"]` | `["Linear"]` |
| KV cache | FP8 (`kv_cache_scheme`) | same |
| Actually quantized | MLP linears only | MLP linears only |
| Kept at original precision | `lm_head`, vision tower, **every** `self_attn*` | `lm_head`, `model.visual.*`, **every** `self_attn.*` **and** `linear_attn.*`, plus `mtp.*` |

The two extra `ignore` entries are a transposition, not an extension:

- **`linear_attn.*`** — Qwen3.8-27B is a hybrid model. Of its 64 language-model layers
  only about 16 use `self_attn`; the rest use `linear_attn` (Gated DeltaNet). That *is*
  this model's attention. NVIDIA excludes every attention module in Gemma, so quantizing
  `linear_attn` here would be a change of scheme, not a port of it.
- **`mtp.*`** — the model's built-in multi-token-prediction draft head, which Gemma has
  no counterpart for. Left alone so the checkpoint can still serve MTP speculative
  decoding, which is the arm DFlash2 has to be compared against. `unsloth/Qwen3.8-27B-NVFP4`
  excludes it too.

Calibration: 512 samples of `HuggingFaceH4/ultrachat_200k` at 2048 tokens with the
model's chat template applied (NVFP4's activation scales are static, so calibration is
required — unlike a data-free dynamic FP8 recipe).

### What this is not

This is **not** an attempt at a better Qwen3.8 quantization than the ones already
published. There is no per-layer bit search, no mixed 8/4-bit config groups, no
KLD-driven tuning of the ignore list. It is NVIDIA's published scheme applied to a
different model, changing exactly one thing, for exactly one reason.

### Two deviations from NVIDIA, stated plainly

NVIDIA produced their checkpoint with `modelopt` — their `config.json` says
`producer: {name: modelopt, version: 0.37.0}`. This one uses `llm-compressor`. The
numeric format is the same either way (NVFP4 is per-group-of-16 E4M3 scales plus a
per-tensor FP32 global scale), but the emitted `quant_method` differs —
`compressed-tensors` here against `modelopt` there — and that selects a different
loader path in vLLM.

`llm-compressor` was chosen because compressed-tensors NVFP4 is the path already
proven for **this exact architecture** (`Qwen3_5ForConditionalGeneration`, via the
unsloth checkpoint, serving correctly in production), whereas modelopt-NVFP4 had only
ever been exercised on Gemma here. Reproducing NVIDIA's loader path on an untested
architecture would have added a second unknown to an experiment that already had one.

Second, the activation scales differ slightly in kind. NVIDIA's config says
`input_activations.dynamic: false`; llm-compressor emits `dynamic: local`, meaning the
per-group-of-16 scales are computed at runtime under a static per-tensor global scale.
Same NVFP4 format and the same group size; the local scale is derived rather than frozen.

## MTP head — a silent trap worth knowing about

Qwen3.8-27B ships a built-in multi-token-prediction head as 15 `mtp.*` tensors inside its
regular shards. **transformers does not model it** — MTP is a vLLM-side module — so
`from_pretrained` never loads those tensors and `save_pretrained` never writes them. Any
llm-compressor round-trip silently drops the MTP head.

What that looks like: vLLM starts happily with `--speculative-config '{"method":"mtp",...}'`,
logs no missing-weight warning whatsoever, and serves at an acceptance length of exactly
**1.00** — every draft token rejected — at 32.3 tok/s, which is *below* the autoregressive
baseline, because it pays for drafting and verification and accepts nothing.

This checkpoint has them restored (copied verbatim from the base model; they are BF16 and
excluded from quantization anyway), and the recipe now fails the job if they are absent.
If you quantize this model yourself, check for `mtp.*` in your output index.

## Correctness caveat — read this before quoting throughput

[vLLM issue #48898](https://github.com/vllm-project/vllm/issues/48898): NVFP4's
FlashInfer/CUTLASS kernels can produce garbage or NaN output on SM 12.0, which is the
architecture this was built and tested on. The counter-evidence is that unsloth's
NVFP4 Qwen3.8 serves correct answers on this hardware today. That makes the format
viable for this model; it does not make any newly produced checkpoint correct by
inheritance.

This checkpoint was gated before any throughput number was quoted. The engine log
confirms it selects exactly the suspect kernel — `Using FlashInferCutlassNvFp4LinearKernel
for NVFP4 GEMM` — so the check was aimed, not ceremonial: six deterministic probes (German
factual, arithmetic, fluent-German generation, code, plus image shape-counting and OCR
against a generated ground-truth image), three runs each at `temperature 1.0`, run
side-by-side against the known-good unsloth NVFP4 checkpoint. **36/36 passed on both.**

Validate your own outputs before trusting a speed number. A checkpoint that is fast and
quietly degraded is worse than no checkpoint.

## Serving

Requires a vLLM build containing DFlash2 if you want the drafter; PR #52816 is not in
any release as of 2026-08-19. Without it, this checkpoint still serves normally as a
plain NVFP4 model, and the unquantized head costs about 1.3 GB of extra VRAM.

```bash
vllm serve <path-to-this-checkpoint> \
  --served-model-name Qwen3.8-27B-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```

Two things that will bite you:

- **`--async-scheduling` is incompatible with `method: dflash`.** vLLM raises
  `Currently, async scheduling is only supported with EAGLE/MTP/Draft Model/NGram
  GPU/DSpark kind of speculative decoding`. This is a real structural disadvantage of
  DFlash2 against MTP in production, not a configuration mistake.
- **Greedy sampling is unsupported by the DFlash2 selector.** Benchmark and evaluate at
  `temperature 1.0 / top_p 0.95 / top_k 20` (Qwen3.8's own recommended sampling).
  `temperature 0` is an invalid test here, not a stricter one.

`method` is the string `dflash`, not `dflash2`. The DFlash2 path is selected by the
*draft* model's architecture (`DFlash2DraftModel`), in
`vllm/v1/worker/gpu/spec_decode/__init__.py::init_speculator`.

## Reproducing

Recipe, calibration script and the Kubernetes Job that produced this are in
mittwald's internal `deploy-quantizations` repository under
`recipes/qwen3-8-27b-nvfp4/`. The script re-verifies after saving that `lm_head`
carries no quantization scales and fails the job if it does — otherwise the entire
point of the checkpoint would be lost silently.

## License

Apache-2.0, inherited from
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Quantization adds no
additional restrictions.
