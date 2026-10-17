---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
  - nvfp4
  - quantized
  - modelopt
  - vllm
  - mtp
  - blackwell
  - dgx-spark
  - qwen3.8
language:
  - en
---

# Qwen3.8-27B-NVFP4

An **NVFP4 (4-bit floating point)** quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with **NVIDIA
TensorRT Model Optimizer 0.45.0** so the model fits and runs on a single **128 GB DGX
Spark (GB10)** through vLLM's Blackwell FP4 path.

**No training or fine-tuning was involved.** This is a numeric format conversion of Qwen's
finished checkpoint: 51.75 GiB of bf16 weights rewritten to 4-bit with per-block scales.
It is a **lossy** conversion, and **no quality evaluation of the 4-bit result against the
bf16 source was run here** — see [Limitations](#limitations). Treat the upstream
capability claims below as Qwen's, measured on their bf16 weights, not as verified
properties of this quantization.

---

## What this quantization actually is

NVFP4 stores each weight as an `E2M1` 4-bit float (1 sign, 2 exponent, 1 mantissa bit),
with an 8-bit scale shared across each block of 16 weights.

**It is a mixed-precision checkpoint, not a uniformly 4-bit one, and that is deliberate.**
ModelOpt 0.45.0 leaves these in higher precision:

| Component | Precision | Why |
|---|---|---|
| Gated DeltaNet (linear attention) | bf16 | Recurrent state is low-tolerance; 4 bits degrades it silently |
| MTP head | bf16 | vLLM force-unquantizes `mtp.fc` for NVFP4 anyway |
| Vision tower | bf16 | Auto-detected as a VLM; the tower is left alone |
| Embeddings / LM head | bf16 | 248,320-row tables; quantizing them is a known quality loss |
| FFN + full-attention projections | **NVFP4** | The bulk of the parameters, and the tolerant part |

---

## Upstream model description

*The following is Qwen's own description of the base model, reproduced for reference. These
are upstream's claims about the bf16 model.*

> Following the widespread community adoption of the Qwen3.5 and Qwen3.6 series, we are
> pleased to introduce Qwen3.8, the most capable generation in the Qwen open-model family to
> date.
>
> Built on the architectural foundation of Qwen3.5, Qwen3.8 delivers substantial gains
> across coding, professional work, research, and long-horizon agentic tasks. Qwen3.8-27B
> brings these advances to a compact, deployment-friendly dense model: a native
> vision-language model that understands images and videos, with flexible thinking control,
> designed to carry complex, multi-step tasks through to completion with greater reliability.

### Qwen3.8 highlights

- **Core Capabilities** — Comprehensive improvements across coding, professional work,
  research, and long-horizon agentic tasks.
- **Agent Execution** — Stronger autonomous planning and better handling of environment
  feedback, leading to more reliable end-to-end task completion.
- **Downstream Compatibility** — Broader support for popular harnesses and development
  tools, making it easier to integrate into your existing stack.
- **Flexible Thinking Control** — Thinking mode is on by default and can be disabled per
  request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from
  historical messages is retained via `preserve_thinking`.
- **Vision-Language Understanding** — Native support for image and video understanding,
  from STEM diagrams and documents to hour-scale videos.

### Model overview

| | |
|---|---|
| Type | Causal Language Model with Vision Encoder |
| Training stage | Pre-training & Post-training |
| Number of parameters | 27B |
| Hidden dimension | 5120 |
| Token embedding | 248,320 (padded) |
| Number of layers | 64 |
| Hidden layout | 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)) |
| Gated DeltaNet | 48 linear attention heads for V, 16 for QK; head dim 128 |
| Gated Attention | 24 heads for Q, 4 for KV; head dim 256; RoPE dim 64 |
| Feed forward network | Intermediate dimension 17,408 |
| LM output | 248,320 (padded) |
| MTP (Multi-Token Prediction) | Trained with multiple steps |
| Context length | 262,144 natively, extensible up to 1,000,000 tokens |

---

## Best practices

*Upstream's recommendations. They apply to this quantization too — nothing in the format
conversion changes sampling behaviour.*

### Sampling parameters

| Mode | Settings |
|---|---|
| **Thinking** | `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0` |
| **Instruct** (non-thinking) | `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` |

For supported frameworks, `presence_penalty` can be adjusted between 0 and 2 to reduce
endless repetition. Higher values may occasionally cause language mixing and a slight
decrease in performance.

### Adequate output length

To optimize performance on agentic tasks, allocate sufficient output length. For frameworks
supporting separate token limits for internal reasoning and final outputs, within the 1M
context length:

- **Reasoning content** — maximum output length 262,144 tokens.
- **Final response** — maximum output length 131,072 tokens.

These settings provide capacity for complex reasoning while ensuring space for high-quality
final deliverables.

### Processing ultra-long texts

Qwen3.8-27B natively supports 262,144 tokens. For long-horizon tasks where the total length
(input plus output) exceeds this limit, use RoPE scaling techniques such as **YaRN**.

### Long video understanding

To optimize inference efficiency for plain text and images, the `size` parameter in the
released `video_preprocessor_config.json` is conservatively configured. To enable
higher frame-rate sampling for hour-scale videos, set `longest_edge` to **469,762,048**
(corresponding to 224k video tokens):

```json
{"longest_edge": 469762048, "shortest_edge": ...}
```

> **Note for this repo:** the `video_preprocessor_config.json` shipped here is upstream's
> conservative default, byte-identical to Qwen's. Raise `longest_edge` yourself if you want
> hour-scale video.

---

## Serving

Requires a Blackwell-class GPU (`sm_120`/`sm_121`) for the NVFP4 path, and vLLM **0.25.1**
or newer.

```bash
vllm serve PassingByPixels/Qwen3.8-27B-NVFP4 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 32768
```

**Two gotchas on a DGX Spark:**

1. `--gpu-memory-utilization` is a fraction of the **entire 121.7 GiB unified pool**, not of
   free memory. The default `0.7` reserves 85 GiB — far more than this model needs. `0.35`
   fits the weights plus a real KV cache.
2. Thinking mode is **on by default**. If you do not want it, disable it per request, or the
   model spends your `max_tokens` budget in the reasoning channel and returns empty
   `content` once the reasoning parser splits it out.

---

## Conversion recipe

Full write-up, including the nine traps that break this conversion, plus a copy-paste
quickstart if you only want to run it:
**[github.com/PassingByPixels/qwen38-27b-nvfp4-dgx-spark](https://github.com/PassingByPixels/qwen38-27b-nvfp4-dgx-spark)**

Short version:

```bash
python hf_ptq.py \
  --pyt_ckpt_path=/path/to/Qwen3.8-27B \
  --export_path=/path/to/out \
  --qformat=nvfp4 --calib_size=512 --batch_size=0 \
  --inference_tensor_parallel=1 --dataset cnn_dailymail
```

Called **directly**, not through `huggingface_example.sh` — that wrapper never forwards a
dataset flag, so it always falls back to its **gated** default and fails on auth.

Calibration: 512 samples of `cnn_dailymail` at sequence length 2048. Calibration only
measures activation ranges to set scales; there are no gradients and no training.

---

## Limitations

- **No quality evaluation was run.** Nothing here compares this 4-bit checkpoint against the
  bf16 source on any benchmark. NVFP4 is lossy. If quality matters for your use, measure it
  on your own workload.
- **Upstream's capability claims are not verified here.** The Qwen3.8 highlights above
  describe the bf16 model as Qwen released it.
- **Vision path is structurally correct but not quality-tested.** The vision tower is
  preserved in bf16 and the processor configs are byte-identical to upstream's, but no
  image or video benchmark was run against this quantization.
- **Requires Blackwell.** NVFP4 needs `sm_120`/`sm_121`. It will not run on Ampere or Ada.
- **Provenance of the three vision processor configs.** `processor_config.json` was emitted
  by ModelOpt during the export. `preprocessor_config.json` and
  `video_preprocessor_config.json` are not written by ModelOpt and were filled from a
  same-architecture sibling ([`nvidia/Qwen3.6-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
  lineage). Both were checked byte-for-byte against
  [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)'s own copies and are
  **identical**, and the two models share identical vision geometry (27 layers, hidden 1152,
  patch 16, merge 2, 2304 position embeddings). Verified, not assumed.

---

## Measurements

Measured on a **DGX Spark GB10** (121.7 GiB unified memory, sm_121a, arm64), vLLM
**0.25.1**, this checkpoint served at **32,768 context**, `max_num_seqs=16`,
`--gpu-memory-utilization 0.55`. Workload: 512-token prompts, 256 max tokens, 16 requests
per point, **one run per point, prose only**.

| | bf16 source | this NVFP4 checkpoint |
|---|---|---|
| On disk | 51.75 GiB | **19.20 GiB** |
| Shards | 18 | 3 |
| Tensors | 1,199 | 2,399 (weights + block scales) |

### Throughput

| concurrency | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| **no speculation** — aggregate tok/s | 18.3 | 46.8 | 81.4 | **138.5** |
| **MTP k=2** — aggregate tok/s | **21.7** | **50.1** | **104.4** | 123.7 |
| TPOT p50, no speculation | 81 ms | 83 ms | 87 ms | 102 ms |
| TPOT p50, MTP k=2 | **59 ms** | **61 ms** | **65 ms** | 109 ms |

Zero errors at every point in both runs.

### MTP speculative decoding works on this checkpoint

`mtp.fc.weight` is **present** — it survives the ModelOpt export, so MTP needs no repair.
Serve it with:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Measured acceptance on prose: mean acceptance length **2.0–2.3**, per-position acceptance
**0.64 / 0.40** rising to **0.74 / 0.53**, average draft acceptance **51–63%**.

MTP is a clear win up to 8 concurrent streams (**+36% single-stream**, ~27% lower per-token
latency) and **costs 11% at 16 streams**, where vLLM clamps `max_num_scheduled_tokens` to
2048 to make room for draft-token KV slots. Whether raising `max_num_batched_tokens`
recovers that case is **untested**.

### What these numbers are not

- **One run per point**, no repeat rounds, no standard deviations.
- **A single prose prompt shape.** Speculative-decoding acceptance is highly
  workload-dependent; code and file-editing workloads typically accept far more, and were
  not measured.
- **MTP TTFT at concurrency 4 (4874 ms) is an unexplained outlier**, out of line with its
  neighbours at 8 (1635 ms) and 16 (2645 ms). Reported rather than dropped; not
  characterised.
- **Not a quality measurement.** See [Limitations](#limitations).

---

## License

Apache 2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
