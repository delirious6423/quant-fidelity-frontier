---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- mxfp8
- fp8
- modelopt
- vllm
- quantized
---

# Qwen3.8-27B-MXFP8

MXFP8 (OCP microscaling FP8, block size 32) quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with
[NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 0.45.0.

Checkpoint size is 32.0 GB across 8 shards.

## What is quantized

Only the language-model linear layers are converted to MXFP8. 155 modules are
left in BF16:

| Kept in BF16 | Why |
| --- | --- |
| `model.visual*` (entire vision tower) | Vision encoder untouched |
| `linear_attn.conv1d`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b` (all 48 linear-attention layers) | Hybrid-attention layers not MX-quantizable in this recipe |
| `model.language_model.embed_tokens`, `lm_head` | Standard exclusions |
| `mtp.*` | Built-in MTP draft head (849 MB) |

KV cache is **not** quantized (`kv_cache_quant_algo: null`).

The built-in MTP head is preserved, so `{"method":"mtp"}` speculative decoding
still works.

## Hardware support

MXFP8 has native block-scaled tensor-core support only on Blackwell (SM100+).
On older hardware, engine support varies:

- **vLLM** — runs on SM80+. The Marlin kernel dequantizes weights and computes
  in BF16, so pre-Blackwell you get the memory footprint of FP8 with BF16
  compute, not FP8 math.
- **SGLang** — refuses to load below SM100
  (`get_min_capability()` returns `100` for MXFP8); the checkpoint will not
  start on Ada or Hopper without forcing a block-FP8 conversion path.

This checkpoint was verified serving on 2x RTX 6000 Ada (SM 8.9) under vLLM via
the Marlin path.

## Serving with vLLM

```bash
vllm serve chriswritescode/Qwen3.8-27B-MXFP8 \
  --quantization modelopt_mxfp8 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --attention-backend flashinfer \
  --trust-remote-code
```

## Evaluation: KL divergence from BF16

`KL(BF16 reference ‖ candidate)`, full vocabulary, no top-k, over 136 held-out
contexts of 2,048 tokens (278,392 scored positions). Hidden states are captured
at the final norm and both operands are replayed through **one shared BF16 LM
head**, so no candidate's own head quantization is counted. Confidence intervals
are a bootstrap over the 41 source clusters, 10,000 samples.

| candidate | mean KLD | bootstrap 95 % CI | median | top-1 |
| --- | --: | --- | --: | --: |
| **this checkpoint** | **0.008471** | [0.006421, 0.010920] | 0.001795 | **96.80 %** |
| `Qwen/Qwen3.8-27B-FP8` | 0.013365 | [0.009955, 0.017457] | 0.002434 | 96.11 % |

Paired on identical contexts, this checkpoint is **−0.004894** lower,
95 % CI [−0.006547, −0.003522], winning **136 / 136 contexts** — 36.6 % less
divergence from BF16 than the official FP8 release.

By stratum:

| stratum | this checkpoint | official FP8 |
| --- | --: | --: |
| literary | 0.01526 | 0.02473 |
| code | 0.00869 | 0.01375 |
| encyclopedic | 0.00607 | 0.00920 |
| multilingual | 0.00290 | 0.00411 |
| scientific | 0.00293 | 0.00411 |

### Controls

Both candidates were captured on the same hardware (RTX 6000 Ada, TP1, stock
vLLM, `enforce_eager`, prefix caching off) against the same BF16 reference.

- **Replay reproducibility** — re-scoring the suite author's own published FP8
  captures through this pipeline gives 0.013201 against their published
  0.013126 (+0.6 %). Inputs are SHA-256-identical, so that residual is
  GPU-arithmetic difference, not method difference.
- **Capture-environment floor** — official FP8 captured here versus the same
  model captured by the suite author differs by **+0.000164**, 95 % CI
  [−0.000096, +0.000434]. The interval includes zero, so this environment
  introduces no systematic bias. The measured advantage above is 30x that floor.

### What this does and does not say

This measures distributional fidelity to BF16, not task accuracy. **No public
task benchmark has been run** on this checkpoint. Absolute KLD magnitudes are
suite-specific; the paired comparison against official FP8 on identical contexts
is the transferable result.

Corpus, BF16 reference captures, shared LM head and harness come from
[`malaiwah/qwen38-27b-fidelity-suite-v3`](https://huggingface.co/datasets/malaiwah/qwen38-27b-fidelity-suite-v3)
and [`malaiwah/qwen38-27b-exl3`](https://github.com/malaiwah/qwen38-27b-exl3)
(`tools/fidelity.py`), used unmodified.

## License

Apache 2.0, inherited from the base model. See `LICENSE`.
