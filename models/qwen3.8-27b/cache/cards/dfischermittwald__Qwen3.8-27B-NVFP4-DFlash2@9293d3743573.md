---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- nvfp4
- fp4
- mixed-precision
- quantized
- compressed-tensors
- llm-compressor
- vllm
- speculative-decoding
- dflash2
- blackwell
- sm120
---

# Qwen3.8-27B-NVFP4-DFlash2

> **This is the TARGET model, not the drafter.** `z-lab/Qwen3.8-27B-DFlash2` is the
> 2 B draft model with an almost identical name. You need **both**, at the same time:
> this checkpoint goes in `--model`, theirs goes inside `--speculative-config`. Loading
> this one as a drafter, or theirs as a target, will not work.

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

Mixed-precision quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
built for one purpose: to be the fastest Qwen3.8-27B target that DFlash2 speculative
decoding can actually run on. 24 GB.

## Why this exists

[vLLM PR #52816](https://github.com/vllm-project/vllm/pull/52816) adds DFlash2, a
block-diffusion drafter. Its candidate selector runs a TopK over the *target* model's LM
head and refuses a quantized one:

```
ValueError: DFlash2 requires an unquantized target LM head for candidate TopK
```

Every fast NVFP4 Qwen3.8 checkpoint on the Hub quantizes `lm_head` —
[`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) has
`re:.*lm_head` in its FP8 group. So the fastest weights and the best drafter could not be
combined at all. This checkpoint is that combination.

## Measured

One RTX PRO 6000 Blackwell Server Edition (SM 12.0), TP=1, single stream, `min_tokens 400`,
decode throughput timed separately from TTFT, best of two, `temperature 1.0 / top_p 0.95 /
top_k 20`. Decode tokens/s:

| Configuration | size | ctx 2048 | ctx 8192 | ctx 32768 |
|---|---:|---:|---:|---:|
| **this checkpoint + DFlash2 n=7** | 24 GB | **109.7** | **104.8** | **110.8** |
| this checkpoint, no drafter | 24 GB | 51 | 50 | 50 |
| `unsloth/Qwen3.8-27B-NVFP4` + MTP n=3 | 22 GB | 83.4 | 82.9 | 85.3 |
| `Qwen/Qwen3.8-27B-FP8` + DFlash2 n=7 | 29 GB | 84.9 | 81.2 | 49.8 |
| `Qwen/Qwen3.8-27B` (BF16) + DFlash2 n=7 | 52 GB | 68.9 | 51.2 | — |
| NVIDIA-scheme NVFP4 + DFlash2 n=7 | 29 GB | 91.9 | 88.3 | 87.8 |

Every row is same-node, same-protocol, measured within the same window.

**2.15× over autoregressive, and 1.32× over the fastest previously available configuration
for this model** (`unsloth/Qwen3.8-27B-NVFP4` driven by its built-in MTP head), at every
context length. The flatness matters as much as the peak: 110.8 tok/s at 32k
context, against 49.8 for the FP8 checkpoint with the same drafter, and against MTP on FP8
which at 32k drops to 39.2 — *below* its own 41.9 autoregressive baseline. DFlash2 keeps
paying off where MTP stops.

One number deserves a warning, because it is easy to quote and easy to get wrong. An earlier
published measurement of `unsloth/Qwen3.8-27B-NVFP4` + MTP n=3 reads **102.3 / 100.4 / 97.0**
and was taken with **greedy** sampling on a different cluster. Greedy raises MTP acceptance
(2.2–2.6 there against 2.13 measured here under sampling), so it is not comparable to
anything in the table above — DFlash2 cannot do greedy at all, so a greedy row can only ever
exist for the MTP arm. Comparing 109.7 against 102.3 understates the difference by more than
half; the honest figure is 109.7 against 83.4.

## Quantization recipe

The scheme is `unsloth/Qwen3.8-27B-NVFP4`'s, read off their `config.json`, with **exactly
one change**: `re:.*lm_head` comes out of the FP8 group and `lm_head` goes into `ignore`.

| | value |
|---|---|
| `format` | `mixed-precision` (`quant_method: compressed-tensors`) |
| group_0 — FP8 W8A8 | weights 8-bit float, per-channel, static; activations 8-bit float, per-token, dynamic |
| group_0 targets | `self_attn.(q\|k\|v\|o)_proj`, `linear_attn.(in_proj_qkv\|in_proj_z\|out_proj)`, `layers.(56..63).mlp.(gate\|up\|down)_proj` |
| group_1 — NVFP4 W4A4 | weights 4-bit float, `group_size` 16, `tensor_group`, `actorder: static`, `scale_dtype` float8_e4m3fn; activations 4-bit float, group 16, `dynamic: local` |
| group_1 targets | all remaining `mlp.(gate\|up\|down)_proj` |
| KV cache | FP8, static, per-tensor |
| Kept BF16 | **`lm_head`**, vision tower (`model.visual.*`), `linear_attn.(norm\|in_proj_a\|in_proj_b)`, the MTP head (`mtp.*`) |

Calibration: 512 samples of `HuggingFaceH4/ultrachat_200k` at 2048 tokens, chat template
applied. Tool: `llm-compressor`.

Two details in unsloth's scheme are what make it 5 GB smaller than a naive port of
NVIDIA's NVFP4 recipe, and neither is obvious:

1. **`linear_attn` is split.** The three large projections go to FP8; the two small ones
   and the norm stay BF16. Qwen3.8-27B is hybrid — about 48 of its 64 layers use
   `linear_attn` (Gated DeltaNet) rather than `self_attn` — so this is where the bytes are.
2. **The last eight layers' MLPs (56–63) are FP8, not NVFP4.** Higher precision where it
   apparently matters more, at a small size cost.

For contrast: NVIDIA's own NVFP4 recipe (from `nvidia/Gemma-4-31B-IT-NVFP4`) declines to
quantize attention at all. Ported faithfully to this model it produces a **29 GB**
checkpoint — the same size as plain FP8 — and 91.9 tok/s with DFlash2 against this one's
109.7. Gemma has a smaller attention share, so the recipe pays off there and not here.

Size tracks throughput across the whole set, though not as tightly as a single pair of
numbers can be made to look: 29 GB against 24 GB is 1.21 on bytes and 51 against 46.1 is
1.11 on autoregressive tokens/s, so size explains most of the gap and not all of it.

The unquantized `lm_head` costs about 1.3 GB (248320 × 5120, BF16 instead of FP8). That is
the whole price of DFlash2 compatibility.

## Validation

**Correctness.** vLLM issue [#48898](https://github.com/vllm-project/vllm/issues/48898)
says NVFP4 kernels can emit garbage or NaN on SM120 without raising, and the engine log
confirms this checkpoint selects the named suspect
(`Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`). Six deterministic probes —
German factual, arithmetic, fluent German generation, code, plus image shape-counting and
OCR against a generated ground-truth image — three runs each, side by side with the
known-good unsloth checkpoint: **36/36 passed on both**.

**Distribution fidelity.** Truncated KL divergence against the BF16 original, teacher-forced
over five fixed texts, top-20 logprobs per position — see `KLD.md` in the recipe directory
for the numbers and the method's limits. Note it is a *truncated* KL: vLLM's API exposes at
most 20 logprobs, never full logits over the 248320-token vocabulary, so it is a proxy
suited to ranking quantizations against each other rather than an absolute figure.

## Serving

Needs a vLLM build containing DFlash2. PR #52816 is not in any release as of 2026-08-19;
it and its bugfix #52883 are Python and Triton only, so they patch cleanly onto a matching
nightly.

```bash
vllm serve <path-to-this-checkpoint> \
  --served-model-name Qwen3.8-27B-NVFP4-DFlash2 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```

Four things that will bite you:

- **`method` is `dflash`, not `dflash2`.** The DFlash2 path is selected by the *draft*
  model's architecture (`DFlash2DraftModel`) in
  `vllm/v1/worker/gpu/spec_decode/__init__.py::init_speculator`.
- **`num_speculative_tokens` must be 7**, i.e. the draft's `block_size` of 8 minus one.
- **`--async-scheduling` is incompatible with `method: dflash`** and raises. vLLM allows it
  only for EAGLE/MTP/draft_model/NGram/DSpark. This is a genuine structural disadvantage of
  DFlash2 against MTP in production, not a misconfiguration — and note that all throughput
  numbers above have it off in *every* arm, so the comparison is fair.
- **Greedy sampling is unsupported** by the DFlash2 selector. Evaluate at `temperature 1.0
  / top_p 0.95 / top_k 20`. `temperature 0` is an invalid test here, not a stricter one.

Without a DFlash2-capable engine this still serves as an ordinary mixed-precision
checkpoint, and its built-in MTP head works too (it is preserved, see below).

## The MTP head, and a trap worth knowing about

Qwen3.8-27B ships a built-in multi-token-prediction head as 15 `mtp.*` tensors inside its
regular shards. **transformers does not model it** — MTP is a vLLM-side module — so
`from_pretrained` never loads them and `save_pretrained` never writes them. Any
llm-compressor round-trip drops the MTP head, and there is no warning anywhere: vLLM starts
happily with `method=mtp`, serves at an acceptance length of exactly **1.00** — every draft
token rejected — at 32.3 tok/s, which is *below* the autoregressive baseline, because it
pays for drafting and verification and accepts nothing.

Restoring the tensors is only half the fix. `llm-compressor` expands `ignore` patterns
against the transformers model, so an `re:^mtp.*` entry matches zero modules and vanishes
from the emitted config; vLLM then builds the head quantized and crashes with
`no module or parameter named 'fc.weight' in Qwen3_5MultiTokenPredictor`. unsloth's
checkpoint carries the **unexpanded** `re:^mtp.*` in its ignore list for exactly this reason.

This checkpoint has both halves: 15/15 `mtp.*` tensors present and `re:^mtp.*` in `ignore`.
Verified serving with `method: mtp` at acceptance 2.06. If you quantize this model yourself,
check for both.

## Reproducing

`recipes/qwen3-8-27b-nvfp4-dflash2/` in mittwald's internal `deploy-quantizations`
repository, with copy-pasteable re-run instructions in that repo's README. The script gates
its own output and fails the job rather than emitting a checkpoint that DFlash2 would
refuse or that is missing its MTP head.

## License

Apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Quantization adds no additional restrictions.
