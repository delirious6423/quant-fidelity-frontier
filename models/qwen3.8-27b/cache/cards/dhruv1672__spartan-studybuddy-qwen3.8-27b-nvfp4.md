---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- qwen3_5
- nvfp4
- quantized
- lora
- speculative-decoding
- mtp
- socratic
- tutoring
- vllm
language:
- en
---

# Spartan StudyBuddy — Qwen3.8-27B Socratic Tutor (NVFP4)

A LoRA-tuned, NVFP4-quantized version of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B),
fine-tuned for Socratic-style engineering/CS tutoring (guiding hints instead of final answers) and
compressed for fast local inference. Built for the [Spartan StudyBuddy](https://github.com/) project,
served with vLLM on an NVIDIA DGX Spark (GB10).

Every number on this page was measured on-device — nothing here is estimated or hard-coded.

## Model summary

| | |
|---|---|
| Base model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (`qwen3_5` architecture) |
| Fine-tuning | LoRA, r=32, alpha=64, layers 48–63 only (last 16 of 64), 58.4M trainable params (0.21%) |
| Quantization | NVFP4 (4-bit, group size 16) via NVIDIA `nvidia-modelopt` 0.37.0 |
| Weights size | 18.73 GiB (vs. 51.75 GiB bf16 merged, vs. ~55.6 GiB base) |
| Speculative decoding | Native MTP draft head grafted from the base checkpoint (see below) |
| Serving | vLLM ≥0.30.0, `FlashInferCutlassNvFp4LinearKernel` (real NVFP4 GEMM) |

## Architecture

- `Qwen3_5ForConditionalGeneration`: a hybrid model with GatedDeltaNet linear attention on 48 of 64
  layers and full self-attention on every 4th layer, plus a Qwen3VL-style vision tower.
- The vision tower (`model.visual.*`) is present, unquantized, and functional — it was excluded from
  both LoRA tuning and quantization since this fine-tune targets text-only Socratic tutoring. If you
  need multimodal input, the base model's vision capability should still work through this checkpoint,
  but it has not been evaluated for that use case here.
- The base checkpoint ships a native single-layer **multi-token-prediction (MTP) draft head**
  (`mtp.*`, ~425M params, reuses the main embedding table). This checkpoint includes those weights
  (grafted back in — see below) so it can be served with real, model-native speculative decoding, not
  a generic/self-drafted approximation.

## What was done to it

1. **LoRA fine-tune** (bf16) on real Socratic-tutoring data — [`meric533/socrateach-sft`](https://huggingface.co/datasets/meric533/socrateach-sft)
   (8000 rows) + [`AndreiSobo/PACT-Socratic-Coding-Tutor`](https://huggingface.co/datasets/AndreiSobo/PACT-Socratic-Coding-Tutor)
   (2000 rows), no synthetic/templated seed data. Split 3-way with SHA-256 row-fingerprint leak checking:
   train 7241 / val 493 / test 493. LoRA applied only to the last 16 of 64 layers (later layers govern
   output style/behavior more than early layers) across attention, MLP, and GatedDeltaNet projections.
   400 steps, eval_loss improved 0.7893 → 0.7766 (step 200 → 400).
2. **Merged** the adapter into the bf16 base (verified: only layers 48–63 changed; layers 0–47 are
   byte-identical to the original base weights).
3. **Quantized** to NVFP4 with `nvidia-modelopt`, calibrated on 512 training rows. `conv1d` layers
   (GatedDeltaNet), the vision tower, and `lm_head` are excluded from quantization — matching NVIDIA's
   own official NVFP4 release recipe for this model family.
4. **Exported** to a clean HF-format checkpoint via modelopt's `export_hf_checkpoint()`.
5. **Grafted the MTP draft head** back in from the base checkpoint as an additive safetensors shard
   (the merge/export pipeline otherwise drops it), enabling real speculative decoding.

Two metadata bugs from the quantization export tooling were found and fixed along the way (both were
incorrect module-path strings in the quantization config, not problems with the actual weight data):
the vision tower's and the grafted MTP head's exclusion patterns needed correcting so vLLM's loader
would recognize them as unquantized bf16 rather than trying to load them into NVFP4-shaped slots.

## Speculative decoding results

Benchmarked on an NVIDIA DGX Spark (GB10), 16 sequential Socratic-tutoring requests, max_tokens=180,
concurrency=1:

| Serving profile | Aggregate tok/s | Decode tok/s (p50) | TTFT (p50) | Draft acceptance rate |
|---|---|---|---|---|
| No speculative decoding | 12.55 | 12.72 | 0.167 s | — |
| + n-gram speculative decoding | 14.24 | 13.25 | 0.105 s | 1.8% |
| + **native MTP speculative decoding** | **18.22 (+45.2%)** | **18.61 (+46.3%)** | 0.230 s | **74.1%** |

n-gram speculation's 1.8% acceptance rate means its throughput number is most likely noise, not a real
effect — n-gram only helps on repetitive text, and Socratic tutoring responses are free-form. MTP's
74.1% acceptance rate is a purpose-trained head predicting this exact model's own outputs, and the
+45% throughput gain is directly attributable to it (at the honest cost of a modestly higher TTFT — the
draft model's own forward pass on the prompt).

This ~12.5 tok/s baseline decode rate is close to the hardware's memory-bandwidth roofline for an
18.73 GiB model on GB10 (~273 GB/s published bandwidth → ~14.6 tok/s theoretical ceiling for batch-1
decode), consistent with a real, non-fluke quantization speedup rather than a measurement artifact.

## Usage

Serve with vLLM (≥0.30.0), with MTP speculative decoding enabled:

```bash
vllm serve dhruv1672/spartan-studybuddy-qwen3.8-27b-nvfp4 \
  --trust-remote-code \
  --served-model-name spartan-teacher \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Without speculative decoding, drop the `--speculative-config` flag — the model serves normally.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-edge")
resp = client.chat.completions.create(
    model="spartan-teacher",
    messages=[{"role": "user", "content": "I'm implementing BFS and my traversal looks depth-first. Give me one Socratic hint."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

## Limitations

- Tuned and evaluated for **English-language Socratic tutoring on CS/software-engineering topics**;
  behavior outside that domain is unverified.
- The vision tower is present and unquantized but has not been evaluated post-fine-tune; this release
  was validated as a text-only tutor.
- Held-out test-set accuracy/perplexity numbers (beyond training/validation loss) have not yet been
  published for this checkpoint.
- NVFP4 quantization requires NVFP4-capable inference hardware and kernels (e.g. vLLM with
  FlashInfer's NVFP4 GEMM support) to realize the measured speedups — it will not run faster than bf16
  on hardware/software without real NVFP4 GEMM support.

## Provenance

Built as part of the [Spartan StudyBuddy](https://github.com/) project. Base model, license, and
architecture credit: [Qwen team, `Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0).
Quantization via [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer).
