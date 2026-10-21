---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
tags:
- quantized
- compressed-tensors
- gptq
- w4a16
- vllm
- ampere
quantized_by: aleada
license: other
license_name: see-base-model-license
license_link: https://huggingface.co/Qwen/Qwen3.8-27B
---

<img src="assert-logo-light.png" alt="ASSERT" height="26">

# Qwen3.8-27B — W4A16 (compressed-tensors)

Standard W4A16 quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced with
[llm-compressor](https://github.com/vllm-project/llm-compressor) (the
official vLLM-team quantization toolkit) inside a reproducible Docker
container. The artifact saves in **`compressed-tensors`** format and
is **drop-in loadable by vLLM** — no upstream patches, no client-side
shims; vLLM auto-detects the quantization config from the embedded
`config.json` at load time.

This release is part of an ongoing series of vLLM-friendly quantized
packs maintained by **atlas**, a self-evolving agent project run by
[**Alex Adamopoulos**](https://huggingface.co/aleada) at
[**assert.gr**](https://assert.gr).

## Reproducibility

| Parameter | Value |
|---|---|
| Source model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Quantization tool | llm-compressor 0.12.0 (Neural Magic / vLLM team) |
| Quantization recipe | GPTQModifier |
| `scheme` | `W4A16` |
| `targets` | `Linear` |
| `ignore` | `re:.*lm_head`, `re:.*visual.*` |
| `graft` (kept in source dtype) | `re:^mtp\.` |
| `sequential_targets` | `Qwen3_5DecoderLayer` |
| `dampening_frac` | 0.01 |
| Calibration dataset | `ultrachat-200k` |
| Calibration samples | 256 |
| `max_seq_length` | 2048 |
| Quantized size | 18.11 GiB |
| Quantization time | — |

Every targeted module was quantized by GPTQ proper: no Hessian inversion failed, so none silently degraded to round-to-nearest.

## Preserved auxiliary weights

Tensors matching `re:^mtp\.` are carried over from the source
checkpoint **verbatim, in their original dtype**, and re-attached to the
artifact after compression.

This matters because `from_pretrained` does not materialise auxiliary
heads that the AutoModel class has no slot for. They are therefore
invisible to the quantizer — an `ignore` entry cannot save a module that
was never loaded — and they drop out of the saved weights entirely. For
a multi-token-prediction head the symptom is not a crash but **0% draft
acceptance**: speculative decoding stays enabled and silently does
nothing.

## Speculative decoding — measured, not assumed

Quantization can silently break a draft head. The model still loads,
serves, and answers correctly while every draft is rejected: the
speed-up the head exists to provide is gone, and nothing in the logs
says so. This pack was served with its drafter and measured.

| Metric | Value |
|---|---|
| Mean accepted length | 2.45 |
| Acceptance, position 1 | 80 % |
| Acceptance, position 2 | 64 % |
| Overall draft acceptance | 72 % |
| Decode, drafter on | 70.3 tok/s |
| Decode, drafter off | 48.4 tok/s |
| Speed-up | 1.45x |

Conditions: vLLM nightly, 2x RTX 3090 (TP=2), num_speculative_tokens=2, temperature 0, max_model_len 8192. Acceptance from the cumulative Prometheus counters over 12 sequential chat requests; decode from 3 runs of 400 tokens after a warm-up request, same prompt in both configurations.

Acceptance depends heavily on workload and sampling temperature, so
these numbers describe the run above rather than a universal property.
Reproduce them against your own server's Prometheus endpoint:

```promql
# overall acceptance
vllm:spec_decode_num_accepted_tokens_total
  / vllm:spec_decode_num_draft_tokens_total

# per-position acceptance
vllm:spec_decode_num_accepted_tokens_per_pos
  / vllm:spec_decode_num_drafts

# mean accepted length (conventionally counts the bonus token)
1 + (vllm:spec_decode_num_accepted_tokens_total
     / vllm:spec_decode_num_drafts)
```

## Beside our other packs, measured the same way

Same recipe, same two RTX 3090s, same protocol — only packs whose
measurement conditions match this one's exactly appear here.

| | **this pack** | [Qwen3.6-27B-W4A16](https://huggingface.co/aleada/Qwen3.6-27B-W4A16) |
|---|---|---|
| Draft acceptance | 72.3 % | 85.2 % |
| Mean accepted length | 2.45 | 2.70 |
| Decode, drafter on | 70.3 tok/s | 70.2 tok/s |
| Decode, drafter off | 48.4 tok/s | 39.8 tok/s |
| Speed-up | 1.45x | 1.76x |

Read the speed-up against the row above it rather than on its own. A
larger multiplier does not mean a faster pack; it means the drafter is
recovering more, which happens when the model decodes slower without
one. Two packs can differ by thirteen points of acceptance and land on
the same tokens per second.

## License

Inherits the license of the base model. By using this artifact you
agree to the original license at the source link above. Atlas /
assert.gr adds no additional restrictions on the quantized weights.

## Usage with vLLM

```bash
docker run --runtime=nvidia --gpus all \
    -p 8000:8000 \
    -e HF_TOKEN=hf_XXX \
    vllm/vllm-openai:latest \
    aleada/Qwen3.8-27B-W4A16 \
    --limit-mm-per-prompt 'image=1' \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching
```

The model is the first **positional** argument — vLLM's `--model` flag
is deprecated and slated for removal.

vLLM auto-detects `compressed-tensors` from the model's config — no
`--quantization` flag required (it is accepted as a redundant hint).
vLLM also picks the model's full native context window from
`config.json`. If you hit KV-cache OOM on a smaller GPU, pin a shorter
window with `--max-model-len 16384` (or smaller) — leave it off to get
the maximum the model was trained for.
Once vLLM is running, hit it with any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="aleada/Qwen3.8-27B-W4A16",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

### Turning the draft head on

The acceptance rate above is **not** what the snippet above measures —
speculative decoding is off unless you ask for it:

```bash
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

If the model has hybrid linear-attention (Gated DeltaNet) layers and
you also pass `--enable-prefix-caching`, add `--mamba-cache-mode=align`.
Prefix caching otherwise selects mamba cache mode `all`, and the MTP
class raises `NotImplementedError` on that combination — the server
will not start.

Confirm the head actually loaded before trusting any speed-up:

```bash
docker logs <container> 2>&1 | grep -c "not found in params_dict"
```

That must print **0**. A non-zero count means the runtime rejected the
draft weights and left the head randomly initialised — the model still
answers correctly, and every draft is rejected.

## Hardware target

Requires CUDA compute-capability ≥ 8.0 (Ampere or newer). Verified on
**NVIDIA RTX 3090** (compute 8.6) where the W4A16 path runs the
language tower at INT4 weights / BF16 activations through vLLM's
compressed-tensors kernels. Vision encoder + multimodal projector
remain BF16 by design — quantizing them gives negligible memory
benefit relative to accuracy cost (matches the upstream
llm-compressor multimodal-vision recommendation).

Weight-only INT4 is the point on this class of card: FP8 and NVFP4
checkpoints are native on Hopper and Blackwell but emulated or
unusable on Ampere, where the INT4 Marlin kernels are what actually
run fast.

## Check this pack yourself

Quantization can drop or disable part of a model without failing: the
pack loads, serves, and answers correctly while something its card
says it kept is absent, or present and ignored by the runtime. Nothing
errors, and the card still promises it.

Three tools read any published repo's metadata — safetensors headers
and `config.json`, no weights — entirely in your browser, so each reads
exactly what you could read yourself. Point them at this pack. Point
them at someone else's.

**[Pack integrity check](https://huggingface.co/spaces/aleada/pack-integrity-check)**
— whether the exclusion entries name real modules, whether anything
from the source model failed to reach the pack, and whether anything is
left at source precision without being declared.

**[Precision map](https://huggingface.co/spaces/aleada/pack-precision-map)**
— how much of a "4-bit" pack is actually 4-bit, and what stayed whole.
Never all of it: embeddings, the output head and the norms are usually
kept, so the honest figure is a fifth to two fifths of the bytes. It
reads the dtypes rather than the tensor names, because five of the nine
quantization toolchains store the packed payload under the plain name
`weight` — a name-driven reader calls those packs full precision.

**[Reasoning-parser advisor](https://huggingface.co/spaces/aleada/reasoning-parser-advisor)**
— whether a model needs vLLM's `--reasoning-parser` and which one, read
from its chat template. Getting this wrong is invisible: the wrong
parser claims the entire output and `content` comes back empty, with no
error anywhere.

## About the maintainer

Alex Adamopoulos is the founder of [assert.gr](https://assert.gr) and
the engineer behind the **atlas** self-evolving AI agent platform.
Atlas runs a planner→executor→supervisor loop over a skill registry,
backed by Postgres, Redis, Qdrant, and a multi-LLM vLLM deployment.
Quantization releases like this one keep the open-source model
ecosystem usable on consumer-grade hardware for self-hosted agent
research.

Connect:

* HuggingFace: [@aleada](https://huggingface.co/aleada)
* Company: [assert.gr](https://assert.gr)
