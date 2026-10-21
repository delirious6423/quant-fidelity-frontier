---
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - nvfp4
  - fp8
  - modelopt
  - mixed-precision
  - sglang
  - dflash
  - dgx-spark
  - blackwell
---

# Qwen3.8-27B NVFP4/FP8 mixed, Local-Hessian (SGLang + DFlash ready)

A mixed-precision quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in NVIDIA
ModelOpt format, built for single-GPU Blackwell machines (developed and measured on a DGX Spark / GB10).

- **27% closer to the original than NVIDIA's and RadixArk's NVFP4 checkpoints.** Mean KL 0.0178 vs 0.0244
  (`nvidia/Qwen3.8-27B-NVFP4`) and 0.0245 (`RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`). It is closer on 54 and 56
  of 58 test documents.
- **Equal on long reasoning.** Full GSM8K with thinking on (1,319 questions): 96.06%. Qwen's official FP8
  scores 95.98% on the same harness, a statistical tie.
- **Fast.** With the DFlash2 drafter (16 draft tokens) on a DGX Spark: 98 tok/s on a counting prompt, 54 on code, 27 on prose,
  single stream. Qwen's official FP8 runs 25–30% slower on the same setup (72 / 37 / 21 tok/s).
- **Loads in SGLang** with speculative decoding (DFlash) and prefix caching.

It combines two proven ideas:
- Unsloth's choice of which layers to keep at 8 bits (the last 8 MLPs, the attention projections and the
  lm_head).
- NVIDIA's Local-Hessian calibration.

Both are packaged in ModelOpt `MIXED_PRECISION` format, which SGLang's ModelOpt loader reads directly.

## What's inside

| Part | Precision |
|---|---|
| MLP layers 0–55 | NVFP4 W4A4 (group 16, FP8 E4M3 block scales, static weight scales) |
| MLP layers 56–63 | FP8 W8A8 |
| Full-attention projections (16 layers) and Gated-DeltaNet projections (48 layers) | FP8 W8A8 |
| Gated-DeltaNet `in_proj_a` / `in_proj_b` | BF16 |
| `lm_head` | FP8 W8A8 |
| KV cache | FP8 (scale 1.0) |
| Vision tower, MTP head, embeddings, norms | BF16 (unchanged) |

It is 23.4 GB on disk (the BF16 original is 55.6 GB), with 233 FP8 and 168 NVFP4 linear modules. The exact
assignment is in `hf_quant_config.json`. The Hub's parameter badge counts packed NVFP4 bytes (two 4-bit weights per
`U8`), so it shows fewer parameters than the model's real 27B.

## How it was made

- **Tool:** NVIDIA ModelOpt, commit `87f7d14` (0.47.0rc1.dev101), `hf_ptq.py` with the recipe in `recipe.yaml`.
- **Algorithm:** `local_hessian`, layerwise, with activations taken from the previous quantized layer and an FP8
  scale sweep.
- **Calibration data:** Nemotron post-training v3, 512 samples × 2,048 tokens.
- **No fine-tuning or other training.**

The recipe was chosen by changing one variable at a time against a fixed set of evaluations. Each step, measured:

| Step | Mean KL (lower = closer to BF16) |
|---|---|
| RadixArk recipe reproduced bit for bit (max calibration, BF16 head) | 0.0242 |
| NVIDIA Local-Hessian recipe + BF16 lm_head | 0.0209 |
| + last 8 MLPs in FP8 | 0.0171 |
| + FP8 lm_head (**this model**, back to the fast speed class) | 0.0178 |

Things that did **not** help:
- Chat-templated calibration data: 0.0210 vs 0.0209.
- Calibrated FP8 KV-cache scales: 0.0209 vs 0.0209.
- Attention in NVFP4 was harmful: +46% KL.

## Evaluation

All numbers were measured on one DGX Spark with the same SGLang image and flags for every checkpoint.

**Faithfulness to BF16** (teacher-forced): 58 documents of code, prose, long context (16K / 32K) and chat, 110,155
positions. KL is computed over the top-20 tokens, so it is a lower bound on the full KL.

| Checkpoint | Mean KL | p99 KL | Top-1 agreement | ΔNLL |
|---|---|---|---|---|
| Qwen/Qwen3.8-27B-FP8 | 0.0034 | 0.037 | 98.22% | +0.0023 |
| **This model** | **0.0178** | **0.206** | **95.98%** | **+0.0111** |
| nvidia/Qwen3.8-27B-NVFP4 | 0.0244 | 0.272 | 94.89% | +0.0187 |
| RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead | 0.0245 | 0.302 | 95.36% | +0.0165 |
| RadixArk/Qwen3.8-27B-NVFP4 | 0.0301 | 0.330 | 94.41% | +0.0220 |

**Long reasoning:** GSM8K test (1,319 questions), thinking on, temperature 1.0, top_p 0.95, top_k 20, 16K cap.

| Checkpoint | Accuracy | Hit 16K cap | Median tokens |
|---|---|---|---|
| RadixArk BF16-LMHead | 96.21% | 3 | 292 |
| **This model** | **96.06%** | **1** | **289** |
| Qwen FP8 | 95.98% | 2 | 291 |

- Every pair is a statistical tie (McNemar χ² ≤ 0.11; 3.84 would be a real difference).
- Thinking length matches the FP8 model question by question (median ratio 0.996). We saw no sign of the
  "overthinking" sometimes reported for aggressive 4-bit quants.
- 28 questions are missed by all three checkpoints: the hard or mislabelled core of GSM8K.

**Small task checks** (all equal to BF16 within noise): GSM8K 49/50, code 25/25 (executed), JSON 14/15,
instruction following 9/10.

**Speed:** DGX Spark, SGLang, DFlash2 drafter with 16 draft tokens, single stream, thinking off.

| Checkpoint | Counting | Code | Prose |
|---|---|---|---|
| **This model** | **98.1** | **53.6** | **27.2** |
| RadixArk BF16-LMHead | 93.6 | 46.8 | 26.0 |
| nvidia NVFP4 | 104.0 | 52.4 | 31.3 |
| Qwen FP8 | 72.1 | 37.4 | 20.5 |

Code and prose speeds move about ±10% with the text, because they depend on drafter acceptance. Counting shows
the per-step cost most cleanly. The prefix cache works: a 62K-token prompt goes from cold to warm about 71×
faster.

**Long context:** DGX Spark, DFlash2 drafter with 16 draft tokens, single stream, thinking off. Every prompt hides a value the answer must
contain, at 60% depth.

| Prompt | Time to first token (cold) | Prefill tok/s | Decode tok/s (code) | Correct |
|---|---|---|---|---|
| 1K–8K | 0.5–4 s | ~2,000–2,300 | 51–61 | ✓ (code, prose, JSON) |
| 16K | 8.6 s | 1,911 | 65 | ✓ |
| 64K | 46 s | 1,411 | 54 | ✓ |
| 128K | 126 s | 1,038 | 42 | ✓ |
| 256K (258K tokens) | 373 s | 691 | 38 | ✓ |

**Follow-up turns** (the same conversation plus one more message) reuse 92–100% of the prompt from the prefix
cache, so the first token arrives in 0.2–0.35 s at every size, including 256K. With the same engine and flags, the
RadixArk BF16-LMHead checkpoint prefills at the same speed from 16K to 256K and decodes code 8–13% slower.

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

Without the drafter, drop the five `--speculative-*` flags. The drafter revision is pinned to the one we measured.
Recommended sampling is Qwen's own: thinking mode temperature 1.0, top_p 0.95, top_k 20.

## Limitations

- Evaluated on text only. The vision tower is kept at BF16 but was not tested.
- The MTP head is kept but was not tested (we use DFlash).
- vLLM loading was not tested.
- Speed was measured single-stream; concurrent throughput was not benchmarked.
- These evaluations come from our own harness and test sets. Compare the numbers with each other, not with scores
  published elsewhere.
- GSM8K at this level cannot rank the checkpoints (they all tie). It shows that long reasoning is intact.

## Credits

- [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B.
- [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer) for the tool and the Local-Hessian Qwen3.8 recipe
  this builds on.
- [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) for the layout idea (FP8 last layers, attention and
  head).
- [RadixArk](https://huggingface.co/RadixArk) for the reference checkpoints we reproduced and compared against.
- [maurienne-ai](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal) for the DFlash2 drafter.
- z-lab for DFlash, and the SGLang team.

License: Apache 2.0, the same as the base model.
