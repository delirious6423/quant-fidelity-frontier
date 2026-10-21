---
base_model: Qwen/Qwen3.8-27B
library_name: mlx-vlm
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- qwen3.8
- quantized
- image-text-to-text
- thinking
- reasoning
- tool-calling
- function-calling
- conversational
- apple-silicon
---

# Qwen3.8-27B mixed-bit 3.80-BPW MLX-VLM

This is an Apple-silicon MLX-VLM conversion of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The language tower uses an explicit mixed-bit affine MLX policy. The vision
tower is the unquantized BF16 MLX-VLM conversion. The source checkpoint's MTP
weights are intentionally **not included** in this release; normal
autoregressive text, image, and video generation do not require an MTP
companion.

[![Bit allocation of model](https://og.tensorlens.dev/latest/og-v1/maglun/Qwen3.8-27B-MLX-Mixed-3.80bpw.png)](https://tensorlens.dev/maglun/Qwen3.8-27B-MLX-Mixed-3.80bpw)
_Fig 1. Bit allocation of model as shown on TensorLens (see link for details)_

## Size and composition

- Full tensor payload: 12.76 GiB.
- Text tensor payload: 11.90 GiB, 3.8016 aggregate BPW over 26,895,998,464 text parameters.
- BF16 vision tensor payload: 0.86 GiB.
- Serialized tensors: 1,655 text + 333 vision + 0 MTP.

The aggregate includes packed codes, affine metadata, and the requested BF16
tensor families. Human layers 1-7 and 55-64 receive edge protection (model
indices 0-6 and 54-63).

| Text tensor family | Effective BPW |
|---|---:|
| Embeddings and untied LM head | 3.25 |
| Attention Q / K | 3.5; 5.5 in protected edge layers |
| Attention V / O | 4.5; 5.5 in protected edge layers |
| MLP gate | 3.25 |
| MLP up | 3.5 |
| MLP down | 3.5; 5.5 in protected edge layers |
| Linear-mixer input (`in_proj_qkv`, `in_proj_z`) | 4.5 |
| Linear-mixer output | 4.5 |
| Linear-mixer state (`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`) | 16.0 |
| Linear-mixer convolution and all norms | 16.0 |

The 3.25 rate is affine 3-bit with group size 128. The 3.5, 4.5, and 5.5 rates
are affine 3-, 4-, and 5-bit with group size 64; the fractional overhead is the
BF16 scale and bias for each group.

## Quick paired quality smoke

These are small, deterministic launch-day checks, not statistically powered
leaderboard results. All three models used identical prompts/task IDs, greedy
decoding, and Qwen's explicit `reasoning_effort=low` thinking mode for
HumanEval+ and GSM8K, with an 8,192-token safety cap so reasoning is allowed to
finish naturally. The middle column is the published 4.9510-BPW release; the
last column is this 3.8016-BPW candidate.

| Check | BF16 text | 4.95-BPW release | 3.80-BPW release |
|---|---:|---:|---:|
| Held-out chat-framed PPL ↓ (n=6) | 5.6728 | 5.7428 | 6.2212 |
| HumanEval+ pass@1 (n=8) | 75.0% | 75.0% | 75.0% |
| GSM8K accuracy (n=12) | 91.7% | 100.0% | 91.7% |

Raw per-item outcomes, exact task identities, corpus hashes, timings, and the
paired summary are retained in the project release dossier rather than written
inside the frozen model directory. The PPL panel is a numerical retention
sanity check; HumanEval+ and GSM8K are more relevant behavioral signals for this
coding/reasoning model, but the sample sizes remain too small for close claims.

## Where this build sits among the 4-bit-class quants

Added 2026-10-15. Every row below was measured on **one host** — `orion`, Apple M5 Pro, 64 GB —
under `mlx 0.32.3` / `mlx-vlm 0.7.6` / `mlx-lm 0.32.0`, with identical prompts, task IDs, seeds,
greedy decoding, `reasoning_effort=low`, and an 8,192-token cap. PPL is teacher-forced on a fixed
held-out corpus of 4 documents / 6,057 tokens.

> The launch-day table above used a **6-document** corpus; this one uses **4 documents**. The PPL
> columns of the two tables are therefore **not interchangeable** — compare within a table, not
> across them. The HumanEval+ and GSM8K item sets are the same in both.

| build | GiB | GB | bpw | PPL ↓ | HumanEval+ (n=8) | GSM8K (n=12) | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 text source (cross-host) | 51.75 | 55.56 | 16 | 5.8107 | 75.0% | 91.7% | — |
| sibling release — mixed affine 4.95 bpw | 16.36 | 17.57 | 4.95 | 5.9056 | 75.0% | 100.0% | 15.96 |
| MXFP4, text only | 13.31 | 14.29 | 4.251 | 6.0271 | 75.0% | 100.0% | 18.19 |
| MXFP4, with BF16 vision + MTP drafter | 14.17 | 15.21 | 4.251 | 6.0271 | 75.0% | 100.0% | 18.17 |
| **this release** — mixed affine 3.80 bpw | 12.76 | 13.70 | 3.80 | 6.4916 | 75.0% | 91.7% | 19.93 |

GiB is 1024-based and GB is 1000-based; the same artifact differs by about 7 % between them, so the
unit matters when size-matching against another release.

**The launch-day behavioural scores reproduced on different hardware.** This release returned
**75.0 %** HumanEval+ and **91.7 %** GSM8K here, exactly as published in August on the retired Apple
M4 Pro, as did the 4.95-BPW sibling. PPL is not cross-checkable between the two tables because the
corpora differ, but the sibling's PPL reproduced to 0.047 % on its own matched corpus. Both
artifacts were verified against the seal files committed at release time before evaluation, so
these are the published bytes, not a rebuild.

**What this means for this build.** At 3.80 bpw this is the **smallest and fastest** of the set:
19.93 tok/s decode, 0.59 GB below the 4.25 bpw MXFP4 builds and 3.87 GB below the 4.95-BPW release.
The cost is real and visible on both quality axes — 0.46 PPL worse than MXFP4 (7.7 % relative) and
the one GSM8K item that the 4.25 and 4.95 bpw builds get right. Pick this when the memory or
throughput budget requires it, and the 4.95-BPW sibling when quality is the priority.

**What these probes can and cannot show.** HumanEval+ reads 75.0 % for *every* row including BF16,
so at n=8 it does not discriminate between these builds at all; GSM8K separates only the 3.80-BPW
build and BF16, where a single item is 8.3 %. PPL is the only quality axis here with resolution.
These are screening-tier counts: treat the table as a measurement, not a ranking, and do not read
"no measurable difference" as "no difference".

`decode tok/s` is derived from the existing evaluation logs — generated tokens over both generative
axes divided by their wall time — not from a dedicated benchmark. Prompt and generation lengths
differ per row and prefill is folded in, so it is comparable within this table only. It is monotone
in bit width (19.93 / 18.17 / 15.96 at 3.80 / 4.25 / 4.95 bpw), which is what a bandwidth-bound
decode should do.

The BF16 row was measured on a different machine (the retired M4 Pro) and is therefore not
host-matched to the rest of the table; it is included as the quality reference, not as a
size or speed comparison.

## Validation

The assembled bundle passed `mlx_vlm.load(..., strict=True)`. A text prompt
returned `The capital of Sweden is Stockholm.` and a real image-path smoke returned
`Two tabby cats are lying on a pink blanket, each positioned next to a remote control.`. Peak unified memory for strict load plus
both short generations was 13.81 GiB on
an Apple M4 Pro. Long-context and video behavior were not tested in this quick
release pass.

## Runtime

Validated with MLX-VLM commit
[`738e44063f145f7df24acc375e33c379053982d5`](https://github.com/Blaizzy/mlx-vlm/commit/738e44063f145f7df24acc375e33c379053982d5)
(`mlx-vlm==0.6.14`, `mlx==0.32.0`).

```bash
python -m pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@738e44063f145f7df24acc375e33c379053982d5"
```

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_id = "maglun/Qwen3.8-27B-MLX-Mixed-3.80bpw"
model, processor = load(model_id, strict=True)
config = load_config(model_id)
images = ["image.jpg"]
prompt = apply_chat_template(
    processor, config, "Describe this image.", num_images=len(images)
)
result = generate(model, processor, prompt, image=images, max_tokens=256)
print(result.text)
```

## Provenance and limitations

- Text quantization was produced by the project's Swift streaming quantizer and independently re-derived before atomic publication.
- The BF16 vision tower was sanitized by the pinned MLX-VLM Qwen3.5-family loader before rebundling; raw Hugging Face vision tensors were not copied blindly.
- Every text shard hash is retained byte-for-byte from the sealed text quant artifact.
- MTP tensors are omitted and no MTP/speculative-drafting behavior is claimed.
- This is an independently produced quantization. Refer to the upstream model card for intended use, safety information, and full architecture details.

## License

Apache License 2.0, inherited from the upstream Qwen3.8-27B release. See
`LICENSE` and the [upstream model card](https://huggingface.co/Qwen/Qwen3.8-27B).
