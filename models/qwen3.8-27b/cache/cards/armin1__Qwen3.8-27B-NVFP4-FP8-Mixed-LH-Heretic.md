---
base_model:
  - Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - heretic
  - abliterated
  - uncensored
  - nvfp4
  - fp8
  - modelopt
  - mixed-precision
  - sglang
  - dflash
  - dgx-spark
  - blackwell
---

# Qwen3.8-27B NVFP4/FP8 mixed, Local-Hessian, Heretic (abliterated, SGLang + DFlash ready)

> [!WARNING]
> **Responsible use: this is an uncensored model.** Its refusal behaviour has been partly removed (abliterated), so
> it can produce harmful, offensive, biased or false content without warning or refusal.
>
> - It is shared **for research and educational purposes**, such as studying alignment, refusal behaviour and
>   quantization.
> - It is **not a safe default** for products, public-facing services or anyone who expects safety filtering. If
>   you deploy it, add your own safeguards.
> - **You are solely responsible** for how you use this model and its outputs, and for complying with the laws and
>   regulations that apply to you and with the license terms.
> - It is provided **as is, without warranty of any kind** (see the Apache 2.0 license). The author accepts no
>   liability for any use or misuse.

An **abliterated** version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized with exactly
the same recipe as our non-abliterated [NVFP4-FP8-Mixed-LH](https://huggingface.co/armin1/Qwen3.8-27B-NVFP4-FP8-Mixed-LH) checkpoint. It's in NVIDIA ModelOpt format, built for single-GPU Blackwell machines (developed and measured on a DGX
Spark / GB10).

- **Uncensoring cost almost nothing.** Mean KL from the BF16 original is 0.0188, against 0.0178 for the
  non-abliterated checkpoint.
- **Closer to the original than NVIDIA's and RadixArk's NVFP4 checkpoints,** which scored 0.0244 and 0.0245. It is
  closer on 52 of 58 test documents against each.
- **Equal on long reasoning.** Full GSM8K with thinking on (1,319 questions): 95.83%, a statistical tie with the
  non-abliterated checkpoint (96.06%), RadixArk (96.21%) and Qwen's official FP8 (95.98%). It thinks as long as the non-abliterated checkpoint, question by
  question.
- **Same speed as the non-abliterated checkpoint.** With the DFlash2 drafter (16 draft tokens) on a DGX Spark: 102 tok/s counting, 50 code, 27 prose, single
  stream.
- **Partly uncensored.** 28/100 refusals on Heretic's test prompts, down from 98/100. See
  [Refusals](#refusals) for what that number does and doesn't measure.

## How it was made

1. **Abliteration with [Heretic](https://github.com/p-e-w/heretic)** (v2.0.0.dev0, commit `3521f86`) on the BF16
   original, with no 4-bit shortcut during the search.
   - **Method:** norm-preserving biprojected abliteration (`row_normalization = "full"`,
     `orthogonalize_direction = true`) with per-layer refusal directions, applied as a rank-16 LoRA and merged.
   - **Edited:** all three kinds of residual writers. That's 16 full-attention `o_proj`, 48 Gated-DeltaNet
     `out_proj` and 64 MLP `down_proj`. Tools that skip the Gated-DeltaNet `out_proj` leave 48 paths for refusals to
     come back.
   - **Data:** `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`, train[:400] for the directions and
     test[:100] for scoring. Seed 1234.
   - **Search:** 200 Optuna trials, about 11 hours on a DGX Spark. The full Heretic config is in
     `heretic-config.toml`.
2. **Selection:** trial 134, the Pareto-front trial with the fewest refusals.

   | Trial | Refusals | Heretic KL |
   |---|---|---|
   | Original | 98/100 | 0 |
   | **134 (this model)** | **28/100** | **0.0638** |
   | 138 | 34/100 | 0.0449 |
   | 155 | 42/100 | 0.0416 |
   | 91 | 66/100 | 0.0298 |

3. **Quantization with the non-abliterated checkpoint's exact recipe.** That's NVIDIA ModelOpt `87f7d14`, `hf_ptq.py` with `recipe.yaml`,
   `local_hessian` calibration on Nemotron post-training v3 (512 × 2,048). The layer assignment is identical to
   the non-abliterated checkpoint's: 233 FP8 and 168 NVFP4 linear modules, 22.6 GB on disk. The tokenizer, chat template and preprocessor configs
   are byte-identical to the original's.

Heretic's KL (0.064) and our KL (0.0188) are different rulers. Heretic's measures only the first token of answers
to 100 harmless prompts, the most refusal-sensitive spot. Ours is the average over 110,155 positions of real
documents. Only ours is comparable with the other checkpoints below.

## What's inside

| Part | Precision |
|---|---|
| MLP layers 0–55 | NVFP4 W4A4 (group 16, FP8 E4M3 block scales, static weight scales) |
| MLP layers 56–63 | FP8 W8A8 |
| Full-attention projections (16 layers) and Gated-DeltaNet projections (48 layers) | FP8 W8A8 |
| Gated-DeltaNet `in_proj_a` / `in_proj_b` | BF16 |
| `lm_head` | FP8 W8A8 |
| KV cache | FP8 (scale 1.0) |
| Vision tower, embeddings, norms | BF16 |
| MTP head | **not included** (see Limitations) |

The Hub's parameter badge counts packed NVFP4 bytes (two 4-bit weights per `U8`), so it shows fewer parameters than
the model's real 27B.

## Evaluation

All numbers were measured on one DGX Spark with the same SGLang image, flags and drafter for every checkpoint.

**Faithfulness to BF16** (teacher-forced): 58 documents of code, prose, long context (16K / 32K) and chat, 110,155
positions. KL is computed over the top-20 tokens, so it is a lower bound on the full KL.

| Checkpoint | Mean KL | p99 KL | Top-1 agreement | ΔNLL |
|---|---|---|---|---|
| [NVFP4-FP8-Mixed-LH](https://huggingface.co/armin1/Qwen3.8-27B-NVFP4-FP8-Mixed-LH) (not abliterated) | 0.0178 | 0.206 | 95.98% | +0.0111 |
| **This model** | **0.0188** | **0.217** | **95.81%** | **+0.0117** |
| nvidia/Qwen3.8-27B-NVFP4 | 0.0244 | 0.272 | 94.89% | +0.0187 |
| RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead | 0.0245 | 0.302 | 95.36% | +0.0165 |

The extra change from abliteration is small and spread out. KL is slightly higher than the non-abliterated checkpoint's on 47 of 58 documents,
and the largest single increase is +0.013.

**Long reasoning:** GSM8K test (1,319 questions), thinking on, temperature 1.0, top_p 0.95, top_k 20, 16K cap.

| Checkpoint | Accuracy | McNemar χ² vs this model | Hit 16K cap | Median tokens |
|---|---|---|---|---|
| RadixArk BF16-LMHead | 96.21% | 0.43 | 3 | 292 |
| NVFP4-FP8-Mixed-LH (not abliterated) | 96.06% | 0.11 | 1 | 289 |
| Qwen FP8 | 95.98% | 0.03 | 2 | 291 |
| **This model** | **95.83%** | — | 4 | 296 |

- Every pair is a statistical tie (3.84 would be a real difference).
- Thinking length matches the non-abliterated checkpoint question by question (median ratio 1.004).

**Small task checks** (equal to the non-abliterated checkpoint within noise): GSM8K 49/50, code 25/25 (executed), JSON 14/15, instruction
following 8/10.

**Speed:** DGX Spark, SGLang, DFlash2 drafter with 16 draft tokens, single stream, thinking off.

| Checkpoint | Counting | Code | Prose |
|---|---|---|---|
| **This model** | **102.4** | **49.9** | **26.8** |
| NVFP4-FP8-Mixed-LH (not abliterated) | 98.1 | 53.6 | 27.2 |

Code and prose speeds move about ±10% with the text, because they depend on drafter acceptance. The prefix cache
works: a 62K-token prompt goes from cold to warm 69× faster.

## Refusals

- **What 28/100 means:** Heretic's keyword scorer counted refusal phrases in the answers to 100 prompts from the
  `mlabonne/harmful_behaviors` test split, with thinking skipped and answers capped at 100 tokens.
- **Why it isn't a precise measure:** a keyword count can score an answer that complies but adds a warning as a
  refusal. It wasn't re-checked with a judge model, so treat the model as **partly** uncensored.
- **How it compares:** other Qwen3.8-27B abliterations report, from Heretic or a Heretic-based tool, anywhere from
  53/100 refusals at KL 0.008 to 0/100 by keyword (36/100 by a judge) at KL 0.099. Each used its own prompt set,
  so compare those refusal counts only roughly.

## How to run (SGLang + DFlash)

This needs a Blackwell GPU with FP4 support, and an SGLang build with DFlash speculative decoding. We used the
`lmsysorg/sglang` nightly image (digest `sha256:febfb971…`, tag `nightly-dev-20260814-c4271c3f`) with the DFlash2
overlay from [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38). The engine reports version
`0.0.0.dev0+qwen38.27b.g561c8f3`.

```sh
python3 -m sglang.launch_server \
  --model-path <this-repo> --trust-remote-code --tp-size 1 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal \
  --speculative-draft-model-revision bd7a934213c47a9e7ef69eef36bb3325f47fd1f1 \
  --speculative-num-draft-tokens 16 --speculative-draft-model-quantization modelopt_fp4 \
  --attention-backend flashinfer --chunked-prefill-size 8192 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --mem-fraction-static 0.70
```

Without the drafter, drop the five `--speculative-*` flags. Recommended sampling is Qwen's own: thinking mode
temperature 1.0, top_p 0.95, top_k 20.

## Limitations

- **Partly uncensored:** 28/100 keyword refusals (see [Refusals](#refusals)).
- **No MTP head.** Heretic's export drops it. `config.json` still declares one MTP layer (the configuration we
  tested), but its weights aren't in the checkpoint, so don't enable MTP / NEXTN speculative decoding. DFlash, as
  above, works.
- Evaluated on text only. The vision tower is kept at BF16 but was not tested.
- Long context was not re-tested for this variant (the non-abliterated checkpoint, with the same recipe, was
  correct up to 256K).
- vLLM loading was not tested.
- Speed was measured single-stream; concurrent throughput was not benchmarked.
- These evaluations come from our own harness and test sets. Compare the numbers with each other, not with scores
  published elsewhere.
- Refusal behaviour is reduced, not removed, and the model can produce harmful or false content. You are
  responsible for how you use it and for complying with the law.

## Credits

- [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B.
- [Heretic](https://github.com/p-e-w/heretic) (p-e-w) for the abliteration tool, and grimjim for norm-preserving
  biprojected abliteration.
- [mlabonne](https://huggingface.co/mlabonne) for the harmful/harmless prompt datasets.
- [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer) for the quantization tool and the Local-Hessian
  recipe.
- [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) for the layer layout idea.
- [maurienne-ai](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal) for the DFlash2 drafter.
- z-lab for DFlash, and the SGLang team.

License: Apache 2.0, the same as the base model.
