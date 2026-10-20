---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
tags:
- onnx
- onnxruntime
- gemma4
- mixture-of-experts
- inference-metadata
- text-generation
pipeline_tag: text-generation
library_name: onnxruntime
---

# Gemma-4 26B-A4B-it — ONNX MoE text decoder (fp16, CUDA) with canonical inference metadata

A **real-weight** ONNX export of the text decoder of
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) —
the **Mixture-of-Experts** member of the Gemma-4 family — built with
[mobius](https://github.com/onnxruntime/mobius) and shipped with a canonical,
hashless `inference_metadata.yaml`.

> **This is the MoE example.** It is deliberately distinct from the **dense**
> pair [`justinchuby/onnx-genai-example-gemma4-e2b`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b)
> (target) and [`…-assistant`](https://huggingface.co/justinchuby/onnx-genai-example-gemma4-e2b-assistant)
> (speculative drafter), which are **dense** (`enable_moe_block=false`). MoE is
> **not** invented anywhere — it is real only in this 26B-A4B checkpoint.

- **Source**: `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (Apache-2.0)
- **dtype / EP**: `float16` / `CUDAExecutionProvider` (validated on NVIDIA H200)
- **mobius**: `272bce2b` · onnxruntime-gpu `1.29.0` · onnx_ir `1.0.0` · transformers `5.14.1`
- **Weights**: real (52 GB fp16) — **no tiny/random substitutes**

## Truthful graph contracts

The exported decoder faithfully preserves Gemma-4 26B-A4B's exotic geometry:

| Contract | Value |
|---|---|
| Layers | 30 |
| **Mixture of Experts** | **128 experts, top-8** per token, `moe_intermediate_size=704`, `gelu_pytorch_tanh` |
| MoE topology | **hybrid parallel** dense MLP (`intermediate_size=2112`) **+** MoE block each layer → `post_ff_norm(dense + moe) + residual` |
| Router | scale-free RMSNorm → learned `scale × hidden_size^-0.5` → linear → softmax(fp32) → top-8 → renormalize → `per_expert_scale` |
| Hybrid attention | sliding-window(1024) + `full_attention` at layers **5, 11, 17, 23, 29** (5 full / 25 sliding) |
| **Heterogeneous head geometry** | sliding/local: **8** KV heads × **256** dim (GQA 2×); global/full: **2** KV heads × **512** dim (GQA 8×); 16 query heads |
| Shared-KV | `num_kv_shared_layers=0` → **all 30 layers own a KV cache** (contrast E2B: 20 borrow) |
| Per-layer input | `hidden_size_per_layer_input=0` (none; contrast E2B: 256) |
| Logit softcap | `final_logit_softcapping=30.0`, `tie_word_embeddings=true` |

`inference_metadata.yaml` (canonical onnx-genai **`v1`**, hashless) encodes the
model's MoE capability (`model.mixture_of_experts`: 128 routed experts, 8 per
token, expert intermediate 704, softmax top-k router) and the hybrid cache as
two state-service groups — `full_attention` (the 5 global owner layers, head_dim
512) and `sliding_attention` (the 25 local owner layers, head_dim 256,
`evictable_prefix: true`) — over the **real** graph ports
(`past_key_values.N.key` / `present.N.key`). It is **validated with the
authoritative onnx-genai PR #1716 `validate_metadata` Rust validator**
(cross-references + real ONNX graph ports, not just JSON-Schema).

`inference_metadata.mobius.yaml` is the raw mobius emitter output (schema 1.0)
for reference.

## Numerical parity (L4) and generation (L5) — NVIDIA H200, onnxruntime-gpu 1.29

Compared against pinned HuggingFace `transformers` (fp16) for the prompt
*"Explain gravity in one sentence."*:

| Level | Metric | Result |
|---|---|---|
| **L4** last-position logits | top-1 argmax / cosine | **1.0 / 0.99999** |
| **L4** all 19 positions | top-1 argmax / cosine | 0.947 / **0.9996** |
| **L5** greedy, 24 new tokens | exact token match vs HF | **24 / 24 ✓** |

L5 ONNX output is **token-identical** to HF greedy:

> *"Gravity is the fundamental force of attraction that pulls objects with mass toward one another, governing everything from the fall of an apple"*

Stage parity (real layer-0 weights): attention output cosine **1.0**; router
top-8 selection agreement **1.0**; the isolated MoE block (unfused ONNX,
fp16 weights) matches an fp32 HuggingFace reference to cosine **0.998**.

Performance (H200, prompt 19 tok): prefill **84 ms**, decode **12.8 tok/s**,
peak VRAM **55.3 GB**. See `performance.json`.

## MoE dispatch: why this export uses the vectorized *unfused* path

mobius normally emits the fused `com.microsoft::MoE` op for CUDA. However, the
**released-onnxruntime fused MoE CUDA kernel fp16-accumulates the expert GEMMs**,
which loses accuracy at this checkpoint's hidden size (2816): per-MoE-layer
cosine drops to ~0.97 and all-position top-1 parity collapses to 0.684. This is
a **runtime kernel limitation**, not an export error — verified because the
*exact same expert weights*, evaluated with standard `MatMul`/`Einsum` (which
accumulate in **fp32** on CUDA even for fp16 inputs), match HuggingFace to
cosine 0.998.

This package therefore ships the **numerically-faithful vectorized unfused MoE**
decomposition (`supports_fused_moe=False`): batched `Einsum` gate/up + down GEMMs
masked by a `ScatterElements`'d top-K routing vector — O(1) ONNX nodes per layer
(~2k total) rather than a per-expert unroll (~183k nodes, impractical to load).
Once a future ORT ships an fp32-accumulate fused MoE kernel, the smaller fused
graph becomes usable with identical numerics.

**Runtime tip:** use plain-string providers
`["CUDAExecutionProvider","CPUExecutionProvider"]`. onnxruntime-gpu 1.29 silently
ignores the `("CUDAExecutionProvider", {"device_id": N})` tuple form (new
plugin-EP API) and falls back to CPU.

## ORT / native parity fixture

The canonical `inference_metadata.yaml` workflow is a single deterministic step —
`invoke decoder` → `emit logits` — so token selection is external greedy argmax
(the L5 parity above is exact-token). The package ships **no RNG token sampler**
(no `policies/`; a counter-based PRNG sampler casts to `uint64`, which the native
CPU EP cannot execute), so it runs identically under ORT and native backends.

## Files

- `model.onnx` + `model.onnx.data` — fp16 MoE text decoder (real weights, ~52 GB)
- `inference_metadata.yaml` — canonical onnx-genai v1 (hashless, Rust-validated)
- `inference_metadata.mobius.yaml` — raw mobius emitter (schema 1.0)
- `policies/` — generation policy ONNX helpers
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
- `source_provenance.json`, `runtime_versions.json`, `performance.json`
- `evidence/` — L4/L5 harness + JSON results + MoE root-cause scripts

## Run directly with onnxruntime

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])  # plain strings
# feed input_ids, attention_mask, and empty past_key_values.{0..29}.{key,value}
# (KV geometry per layer: full layers 2×512, sliding layers 8×256 — see metadata)
```

Licensed Apache-2.0 (see `SOURCE_LICENSE.md`); derived from a pinned Apache-2.0
checkpoint.

<!-- inference-metadata-annotation:start -->
## Annotated inference metadata

Review [`inference_metadata.annotated.yaml`](./inference_metadata.annotated.yaml) for inline explanations of this package's workflow, tensor/state/cache contracts, and fail-closed omissions. [`inference_metadata.yaml`](./inference_metadata.yaml) remains the canonical machine-authored contract; automated validation confirms both files parse to the same metadata object.
<!-- inference-metadata-annotation:end -->
