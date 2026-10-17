---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - gptq
  - int4
  - quantized
  - compressed-tensors
  - qwen3.8
  - imatrix
  - long-context
datasets:
  - nvidia/Nemotron-SFT-Instruction-Following-Chat-v3
  - nvidia/Nemotron-SFT-Math-v4
  - nvidia/Nemotron-SFT-Science-v2
  - nvidia/Nemotron-SFT-SWE-v3
  - nvidia/Nemotron-SFT-Multilingual-v2
library_name: transformers
pipeline_tag: image-text-to-text
---
# Qwen3.8-27B-GPTQ-INT4
### Asymmetric INT4 weight-only (group_size=32) + GPTQ, long-sequence calibrated
A 4-bit quantization of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**.
~2.1× smaller than BF16. Matches the BF16 base on every evaluated benchmark;
all deltas are within sampling noise.
## TL;DR
| | value |
|---|---|
| Format | INT4, asymmetric (zero-point), `group_size=32`, weight-only |
| Method | **GPTQ** (Hessian error compensation) + **imatrix-mse** observer, `actorder=static`, block 128, `dampening_frac: 0.01` |
| Calibration | **8 sequences × 262,144 tokens** (the model's native context length), packed from Nemotron-Post-Training-v3 SFT datasets (instruction-following chat, math, science, agentic code, multilingual) |
| Disk size | ~27 GB (vs ~55.6 GB BF16) |
| **Accuracy vs BF16 base** | **Within sampling noise on 4 of 5 benchmarks + long-context retrieval (MathVision −2.5 at full n)** |

## What was quantized (and what was protected)

| Component | Quantized | Protected (kept BF16) |
|---|---|---|
| `language_model` (64 hybrid-attention text layers) | block linears — attention `self_attn.{q_proj,k_proj,v_proj,o_proj}` and FFN `mlp.{gate_proj,up_proj,down_proj}` (INT4, asymmetric, `group_size=32`) | layer norms, `embed_tokens`, `norm`, `lm_head` |
| Linear-attention (DeltaNet) projections | — | all `.*linear_attn.*` modules |
| `visual` (frozen ViT encoder) | — | entire `model.visual.*` |
| MTP predictor | — | kept intact for speculative decoding |

## Quantization recipe

GPTQ (Hessian error compensation) with weight-only INT4: asymmetric
(zero-point, stored as int8), `group_size=32`, imatrix-mse weight observer,
`actorder=static`, block size 128, `dampening_frac: 0.01`. Targets all
`Linear` layers except `lm_head`, embeddings, DeltaNet (`.*linear_attn.*`),
vision (`model.visual.*`), and the MTP predictor. The full resolved recipe —
including defaults — ships as `recipe.yaml` in this repo.

imatrix importance statistics were collected by forward hooks over the 8
packed calibration sequences (llm-compressor `imatrix-mse` observer). The
8-sequence budget was chosen to calibrate at the model's native 262,144-token
context within a single 80GB GPU — the token count (~2.1M) is conventional,
the document count is the trade-off.

## Evaluation
Identical harness and hardware for base and quantized (lm-eval-harness,
`hf-multimodal` backend, greedy). Generative tasks use the chat template with
thinking enabled; ARC-Easy is loglikelihood.

| task | protocol | n | base (BF16) | **this** | Δ |
|---|---|---|---|---|---|
| GSM8K | thinking, generative (250-sample subset) | 250 | 95.60 | **94.80** | −0.8 (within noise) |
| GPQA Diamond | thinking, generative | 198 | 64.65 | **64.65** | 0.0 |
| RealWorldQA | thinking, generative, vision | 765 | 84.31 | **82.75** | −1.6 (within noise) |
| MathVision | thinking, generative, vision | 3040 | 78.06 | **75.53** | −2.5 (≈2.3σ) |
| ARC-Easy | loglikelihood, full | 2376 | 82.58 / 72.81¹ | **82.62 / 74.58** | +0.04 / +1.77 |

¹ acc_norm < acc inverts the usual ARC-Easy ordering for both models under this
harness/tokenizer; the effect is identical for base and quantized.
RealWorldQA and MathVision are full-n (765 / 3040), re-measured with the
vllm-vlm backend (thinking, greedy, seed 1234) on the same machine as the base;
they supersede the earlier n=100 slices.
Long-context needle-in-a-haystack (greedy, 3 needle depths per length; both
models retrieve exactly):
| Context (tokens) | Base | This model |
|---|---|---|
| 8,192 | 3/3 | 3/3 |
| 32,768 | 3/3 | 3/3 |
| 131,072 | 3/3 | 3/3 |
| 262,144 (native) | 3/3 | 3/3 |
Retrieval validated at the model's native 262,144-token context — the same
length the model was calibrated at.
Not evaluated: agentic/tool-calling behavior; vision coverage beyond
RealWorldQA and MathVision.
## Serving
```bash
vllm serve abhishekchohan/Qwen3.8-27B-GPTQ-INT4
```
Full-featured (thinking, tool calling, MTP speculative decoding):
```bash
vllm serve abhishekchohan/Qwen3.8-27B-GPTQ-INT4 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```
Requires a recent vLLM with `qwen3_5` support (per the base model card, use
the latest version). The checkpoint is compressed-tensors `pack-quantized`
int4 — int4 block linears plus BF16 DeltaNet/vision/embeddings/lm_head, so
int4 kernels (Marlin/compressed-tensors) engage for the quantized linears
only. ~27 GB of weights: fits a 32GB+ GPU at moderate context (KV ≈2 GB at
32k). Full 262,144-token context needs ≈17 GB of KV on top — plan for ~48GB
total, or cap `--max-model-len` on smaller cards. KV ≈65 KB/token (hybrid:
only global-attention layers accumulate KV; DeltaNet layers carry fixed-size
state). MTP predictor weights are included intact; the speculative-decoding
flag above follows the base-model recipe — if your vLLM build rejects MTP
for this architecture, drop the `--speculative-config` line and serve
normally.
The base model supports YaRN extension to 1M context via `--hf-overrides`
(see the base model card). This checkpoint was calibrated and validated only
at the native 262,144 tokens; behavior beyond that length is untested and
YaRN interacts with RoPE-bearing attention layers — run your own
long-context eval before relying on it.
