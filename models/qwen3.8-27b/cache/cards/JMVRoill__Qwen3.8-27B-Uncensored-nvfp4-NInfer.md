---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - orcarouter/Qwen3.8-27B-Uncensored
  - orcarouter/Qwen3.8-27B-Uncensored-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - fp8
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
  - abliterated
  - uncensored
  - not-for-all-audiences
---

# Qwen3.8-27B Uncensored NVFP4 for NInfer

This repository contains an **abliterated** (refusal-removed) build of Qwen3.8-27B in the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format. It combines the BF16
weights of [orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored)
with the packed NVFP4/FP8 Text weights of
[orcarouter/Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4).
The artifact is intended only for NInfer; it is not a Transformers checkpoint, Safetensors
distribution, or GGUF file.

This is the registered `nvfp4` weight profile of the `qwen3_8_27b` target. Text layers 0–55 use
NVFP4 MLP weights, while the token embedding, attention input/output projections, GDN Q/K/V/Z and
output projections, full output head, and Text layers 56–63 MLP weights use row-scaled FP8. BF16
control weights and the registered MTP and Vision allocations are retained.

> [!WARNING]
> **Built from weights whose safety alignment has been substantially removed.** On a sibling FP8
> build, the upstream author reports harmful-prompt refusal collapsing from 64–99% to 0–6%.
> See the disclaimer below.

> [!WARNING]
> **This artifact cannot be reproduced from stock NInfer.** Building it required a converter
> patch that is not upstream: `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` ships `lm_head`
> unquantized, and the registered recipe requires it as row-scaled FP8, so normal conversion
> fails at preflight. **Running it needs no patch**: the modification is in the Python
> converter, not the engine. See [Converter modification](#converter-modification).

> [!IMPORTANT]
> **This artifact reports the same identity as the official build.** Its `model_id` is
> `qwen3.8-27b` and its `weights_id` is `nvfp4`, and it is the same size as
> [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).
> Nothing at load time distinguishes them. Keep the `_uncensored` filename, and pass
> `--model-id` to separate them when serving.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4_uncensored.ninfer` |
| Size | 23,719,496,192 bytes (22.09 GiB) |
| SHA-256 | `43025bb64f2cb558d9ede6269f8a3c2ed8ebfa4619344fe1ba8e955ba4979218` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `nvfp4` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,190 (1,184 tensors and 6 resources) |
| NVFP4 tensors | 112 |
| Row-scaled FP8 tensors | 146 |

Objects by name prefix: `text` 773 · `vision` 333 · `dflash2` 66 · `mtp` 12 · `frontend` 6.
Tensor formats: BF16 579 · FP32 208 · FP8_E4M3FN_ROW_BF16S 146 · NVFP4 112 · Q4G64_F16S 55 ·
Q5G64_F16S 54 · W8G32_F16S 28 · Q6G64_F16S 1 · I32 1.

The file contains the registered Text, Vision, MTP, DFlash2, optimized proposal-head, tokenizer,
chat-template, generation, and media-processor objects required by NInfer.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '43025bb64f2cb558d9ede6269f8a3c2ed8ebfa4619344fe1ba8e955ba4979218' \
  'qwen3_8_27b_nvfp4_uncensored.ninfer' | sha256sum --check
```

## Requirements

- [NInfer](https://github.com/Neroued/ninfer) built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

Observed to load, generate and serve on NInfer revision
`a16b6442856620b7e4856acb25215acbf7e3c750`. The minimum compatible revision has not been
determined; earlier revisions are untested for this artifact.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#quick-start) for source-build dependencies.

## Download and run a CLI example

```bash
hf download JMVRoill/Qwen3.8-27B-Uncensored-nvfp4-NInfer \
  qwen3_8_27b_nvfp4_uncensored.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_8_27b_nvfp4_uncensored.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For images, videos, and structured chat history, see the
[CLI guide](https://github.com/Neroued/ninfer/blob/master/docs/cli.md).

## Start a local server

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4_uncensored.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --model-id qwen3.8-27b-uncensored \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. The shared 240,000-token Device KV pool admits
two active requests when their combined completion reservations fit; either request may use the
full pool while running alone. Two extra Device checkpoint slots, eight pinned Host State slots,
and 8 GiB of pinned Host KV retain reusable continuations under resource pressure.

`--model-id` is what distinguishes this artifact from the official one over `/v1/models`; the
artifact identity in the engine log cannot tell them apart.

See the [HTTP serving guide](https://github.com/Neroued/ninfer/blob/master/docs/serving.md) for
the API surface and the
[resource scheduling reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/resource-scheduling-and-context-cache.md)
for cache and admission semantics.

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- DFlash2 with draft windows from one to fifteen using the included DFlash2 companion weights
  (`--spec dflash2 --draft-tokens 7`, optionally `--lm-head-draft`);
- BF16, INT8, FP8, NVFP4, and K8V4 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages serving.

Directly exercised on this artifact: the CLI and server invocations above, `--kv-dtype fp8` and
`--kv-dtype nvfp4`, `--spec mtp --draft-tokens 3`, `--spec dflash2 --draft-tokens 6`,
`--lm-head-draft`, and `--max-concurrency 2`. The remaining entries are capabilities of the
registered `qwen3_8_27b` target whose objects are present in the file but which were not
individually exercised here.

## Performance

**No benchmark has been run to NInfer's published methodology, and no performance figures are
claimed.**

The artifact loads, generates and serves. An ad-hoc K sweep exists on the maintainer's machine
but is **not comparable** to any published NInfer result: it omitted `--no-prefix-reuse`, left
thinking enabled under a short token cap, used invented prompts rather than the committed fixture
corpus, used no fixed seeds, computed decode rate as `completion_tokens / decode_seconds` where
NInfer uses `(completion_tokens - 1) / decode_seconds`, and ran at `--device-state-slots 2` which
is a deployment value rather than NInfer's benchmark profile. Those rates are biased upward and
are deliberately omitted here.

Numbers will be added only when produced by the reproduction command in
[`docs/performance/methodology.md`](https://github.com/Neroued/ninfer/blob/master/docs/performance/methodology.md).

### What was actually observed

| Check | Result |
|---|---|
| CLI, `--kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft` | exit 0, coherent output, MTP acceptance 63.4% |
| CLI, `--kv-dtype nvfp4 --spec dflash2 --draft-tokens 6` | exit 0, coherent output, 0 fallback steps |
| `ninfer-serve`, the profile above | `engine ready \| qwen3.8-27b/nvfp4 \| total 16.3s` |
| `ninfer-serve`, `dflash2` K=6 / NVFP4 / 262,144 | `/v1/models` returned `qwen3.8-27b-uncensored`, 28,616 MiB resident |

These are **single runs without fixed seeds or a fixture corpus**. The acceptance figure is one
sample and is not a benchmark result.

## Evaluation

**No capability evaluation has been run on this artifact.** No EvalScope suite, no perplexity
measurement, and no quality comparison against the official build.

Two smoke tests have been run over HTTP with `enable_thinking:false`: one coherence check (merge
two sorted lists; correct output) and one refusal probe (profane limerick; complied, no
refusal). Those establish that it loads, is coherent, and did not refuse on that prompt. **Two
prompts cannot support a quality or safety claim.**

The figures below are **orcarouter's, measured on their separate
[FP8 build](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8)**, not on the NVFP4
checkpoint this artifact derives from, and not on this artifact. They argue the numbers transfer
because abliteration is "a weight edit shared across all releases" of the model:

| orcarouter, measured on the FP8 build | Value |
|---|---|
| Harmful-prompt refusal, thinking off | 64–99% (base) → 0–6% |
| Harmful-prompt refusal, thinking on | ≤ 1.7% |
| Benign over-refusal (XSTest-safe) | 5.6% → 0.4% |
| MMLU | 84.3 → 84.7 |
| MMLU-Pro | 77.6 → 76.8 |
| GSM8K | 90.0 → 88.7 |
| CMMLU | 81.4 → 80.8 |

orcarouter states that capability retention and perplexity **for the NVFP4 + FP8 checkpoint have
not yet been measured**, and that a mixed 4-bit/8-bit build is expected to trade some accuracy
against the FP8 build. The additional effect of conversion to `.ninfer` is likewise unmeasured.

## ⚠️ Disclaimer — read before use

This model has had its **safety alignment substantially removed** via *abliteration*
(orthogonalizing the refusal direction out of the residual stream). As a direct consequence:

- **It will comply with harmful, unethical, offensive, or illegal requests** that the original
  `Qwen3.8-27B` would refuse. It has no meaningful built-in guardrails.
- It is released **strictly for legitimate research** — interpretability, AI-safety and
  refusal-mechanism study, red-teaming, robustness evaluation, and controlled experiments.
- **You assume full responsibility and liability** for how you use it and for everything it
  generates. Do not deploy it to end users or in production without adding your own safety,
  moderation, and abuse-prevention layers.
- Use must comply with the **[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)**
  inherited from the base model, and all laws and regulations that apply to you.
- The authors and uploaders **accept no liability** for any misuse or harm arising from this
  model. Its outputs do **not** reflect the views of the uploaders or of Qwen / Alibaba.

This disclaimer is carried forward from the
[upstream model card](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4); the
abliterated weights it describes are the weights in this artifact, and conversion to `.ninfer`
does not restore any refusal behaviour.

### Method

The upstream author describes the abliteration as following
[Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717),
using a single refusal direction (k = 1), and states that the vision tower is untouched and the
MTP head is abliterated consistently with the main model.

Refusal behaviour of *this converted artifact* has not been systematically measured. The only
direct evidence here is a single refusal probe (profane limerick) which it complied with. One
prompt is not a safety evaluation.

## Converter modification

The registered `qwen3_8_27b_nvfp4-v2` recipe sources `text/output_head` from the quantized
checkpoint as row-scaled FP8, but `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` leaves `lm_head`
unquantized: it ships `lm_head.weight` with no `lm_head.weight_scale`, so preflight fails.

Two changes were made:

1. **`text/output_head` is sourced from the BF16 checkpoint** and encoded locally with
   `MAXABS_BF16S_RECIP_E4M3FN_RNE_V1`, the same canonical row-scaled FP8 encoder the recipe
   already applies to `text/token_embedding`. Both are FP8 `(248320, 5120)` in the inventory.
2. **`config_groups.targets` comparison accepts explicit module lists** as well as the registered
   `re:` pattern forms. The upstream checkpoint spells its targets as 232 and 168 fully expanded
   module names, verified to select exactly the same module set as the patterns, with none missing and none
   extra.

## Provenance

| Field | Value |
|---|---|
| Base repository | `orcarouter/Qwen3.8-27B-Uncensored` |
| Base download source | `huggingface.co/orcarouter/Qwen3.8-27B-Uncensored` |
| Base revision | `404ea47aaa5d8a8b00049c9e9750089aca011ab2` |
| Quantized source repository | `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` |
| Quantized source revision | `69d21348b2d6c11439fb69368f40414c2256e44e` |
| DFlash2 source repository | `z-lab/Qwen3.8-27B-DFlash2` |
| DFlash2 source revision | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Upstream of both orcarouter repos | `Qwen/Qwen3.8-27B` |
| Conversion recipe | `qwen3_8_27b_nvfp4-v2` (modified, see above) |
| Embedding and output-head encoder | `MAXABS_BF16S_RECIP_E4M3FN_RNE_V1` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Converter base revision | `a16b6442856620b7e4856acb25215acbf7e3c750` + local patch |
| Minimum runtime revision | not determined; observed on `a16b6442856620b7e4856acb25215acbf7e3c750` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |
| Frontend resources | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The six frontend resources (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`) are the
official Qwen files, SHA-256-verified against the converter's pinned digests. The converter
requires them byte-exact, and NInfer re-verifies `chat_template.jinja` against a runtime
whitelist at load.

> [!NOTE]
> The `.ninfer.conversion.json` report emitted alongside this artifact records
> `Qwen/Qwen3.8-27B` and `unsloth/Qwen3.8-27B-NVFP4` as its sources. Those strings are hard-coded
> constants in the recipe module and are **not** the sources used for this build. The table above
> is authoritative.

## License

This NInfer artifact is distributed under the Apache License 2.0. The base repository
[orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored),
the [quantized source](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4), the
[DFlash2 companion](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), the upstream
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[NInfer](https://github.com/Neroued/ninfer) are all licensed under Apache-2.0. Users remain
responsible for complying with the license and applicable laws.
