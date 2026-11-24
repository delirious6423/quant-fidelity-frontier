---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - multimodal
  - conversational
  - cuda
  - rtx-4090
---

# Qwen3.8-27B for NInfer-4090

This model card is the version-controlled source for
[tensorninja2/Qwen3.8-27B-NInfer](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer).

The repository contains [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) converted to the
native `.ninfer` artifact format of [NInfer-4090](https://github.com/tensorninja/ninfer-4090), a
C++/CUDA inference engine for this model on one RTX 4090. The file works only with NInfer. It is
not a Transformers checkpoint, Safetensors distribution, or GGUF file.

NInfer-4090 converted this file from the BF16 checkpoint with importance-weighted quantization
(recipe `qwen3_8_27b-v2`). It has the same formats, size and kernels as the `qwen3_8_27b-v1`
artifact that the upstream [NInfer](https://github.com/Neroued/ninfer) project published, so it runs
at the same speed, but its outputs are closer to the BF16 model. The `v1` tag of this repository
keeps the previous file.

## Artifacts

The repository holds two files. Most users need only the first.

| Field | `qwen3_8_27b.ninfer` | `qwen3_8_27b.dflash2.ninfer` |
|---|---|---|
| Contents | the model | the model plus the optional DFlash 2 drafter |
| Size | 18,210,531,328 bytes (16.96 GiB) | 19,421,938,176 bytes (18.09 GiB) |
| SHA-256 | `b450e1bdea29f8ce5f82fd9995e716d0cbb620dc9ba31ed9fcabca4b7999a75d` | `9566df55efa072e1087fff1776fd1763eca2dd21fe62d5f08d39f57ea84df759` |
| Stored objects | 1,124 (1,118 tensors and 6 resources) | 1,190 (1,184 tensors and 6 resources) |
| Manifest | [`artifact-manifest.json`](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer/blob/main/artifact-manifest.json) | [`artifact-manifest.dflash2.json`](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer/blob/main/artifact-manifest.dflash2.json) |

Both files use container version 2, NInfer model ID `qwen3.8-27b`, weights ID `groupwise-int`,
and target key `qwen3_8_27b`. The Text body uses the registered Q4/Q5/Q6 groupwise allocation, and
the token embedding and full output head use `W8G32_F16S`. Each file also contains the Vision,
MTP, proposal-head, tokenizer, chat-template, generation, and media-processor objects that NInfer
requires. The DFlash 2 file appends 66 drafter tensors to the same objects. It runs everything the
first file runs, and it also runs `--spec dflash`.

To check downloaded files, download `SHA256SUMS` beside them:

```bash
hf download tensorninja2/Qwen3.8-27B-NInfer SHA256SUMS --local-dir models
(cd models && sha256sum --check --ignore-missing SHA256SUMS)
```

## Quantization

Each group of 32 or 64 weights stores one FP16 scale and small integer codes. The `v1` recipe set
each scale from the group's largest absolute weight. This artifact instead tries 43 candidate scales
per group and keeps the one with the smallest reconstruction error, weighted by how strongly each
input column is activated. The activation statistics come from about 524,000 tokens of reasoning,
code, chat and agentic tool-use text run through the BF16 model.

Measured on an RTX 4090 against the BF16 checkpoint, on 98,304 held-out tokens that calibration
never saw:

| Domain | `v1` mean KLD | This file | `v1` top-1 | This file |
|---|---:|---:|---:|---:|
| Agentic tool use | 0.710 | 0.531 | 87.87% | 90.05% |
| Chat | 0.071 | 0.046 | 93.97% | 95.24% |
| Code | 0.032 | 0.022 | 94.51% | 95.63% |
| Reasoning | 0.033 | 0.022 | 95.82% | 96.62% |
| All | 0.212 | 0.155 | 93.04% | 94.38% |

KLD is the next-token KL divergence from the BF16 model, and top-1 is how often both pick the same
next token. Served with MTP3, draft acceptance rises from 56.2% to 57.6%.
[qwen3.8-27b-decision-1](https://huggingface.co/tensorninja2/qwen3.8-27b-decision-1) development accuracy is 87.18% (87.10%
on `v1`). The agentic row has the widest uncertainty, because its held-out text comes from one
trajectory.

## Requirements

- [NInfer-4090](https://github.com/tensorninja/ninfer-4090), built from source or as its Docker
  image;
- 64-bit Linux;
- NVIDIA GeForce RTX 4090 (`sm_89`);
- CUDA Toolkit 12.8 or newer.

## Download and run

```bash
hf download tensorninja2/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir models

./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Without `--vision`, the server selects dynamic Vision when the encoder's KV lease fits the pool,
and keeps the encoder on the GPU when it does not. Dynamic Vision keeps the encoder weights in host
RAM and borrows KV pages only while it encodes an image. Pass `--vision off` for text only. See the
[CLI reference](https://github.com/tensorninja/ninfer-4090/blob/tensorninja/docs/cli.md) for the rules.

The NInfer-4090 Docker image downloads `qwen3_8_27b.ninfer` on first start and checks its SHA-256.
See the [NInfer-4090 README](https://github.com/tensorninja/ninfer-4090#run-it) for every profile.

## Speed on one RTX 4090

Measured on an RTX 4090 at its 480 W limit with CUDA Graphs on. Prefill rates come from the
server's request log with prefix reuse and the continuation cache off.

| Workload | Result | KV mode |
|---|---:|---|
| Prefill, 4,795-token prompt | 5,989 tok/s | `rk4v4` |
| Prefill, 18,969-token prompt | 5,884 tok/s | `rk4v4` |
| Prefill, 118,242-token prompt | 4,080 tok/s | `rk4v4` |
| Decode, code, MTP with 3 draft tokens | 148.6 tok/s at 81.0% draft acceptance | `int8` |
| Decode, 3 concurrent streams, MTP3, 8K-token prompts | 299 tok/s together | not recorded |
| Decode, mixed corpus, sampled, DFlash 2 with 7 draft tokens | 137.2 tok/s against 114.1 for MTP3 | `rk4v4` |
| Decode, no speculation | 50.5 tok/s shallow, 39.6 tok/s at 128K depth | `int8` |

Method and raw numbers are in
[Performance](https://github.com/tensorninja/ninfer-4090/blob/tensorninja/docs/performance.md#rtx-4090-sm_89-long-context-prefill).

## DFlash 2

`qwen3_8_27b.dflash2.ninfer` adds the
[z-lab DFlash 2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) block drafter. MTP with three
draft tokens stays the recommended mode. DFlash 2 is an opt-in for single-stream decode:

- With one request, 7 draft tokens decode about 20% faster than MTP3 on a mixed sampled corpus.
- With several requests, use 4 draft tokens. At three concurrent requests it is within noise of
  MTP3, and longer drafts are slower.
- DFlash 2 is text only. Without `--vision`, the server turns Vision off with it.

```bash
hf download tensorninja2/Qwen3.8-27B-NInfer qwen3_8_27b.dflash2.ninfer --local-dir models

./build-sm89/apps/ninfer-serve models/qwen3_8_27b.dflash2.ninfer \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

Both modes fit the full 262,144-token context with `rk4v4` and one request. See
[DFlash 2 vs MTP3](https://github.com/tensorninja/ninfer-4090/blob/tensorninja/docs/performance.md#dflash-2-vs-mtp3) for the measurements.

## Best with Caudra

NInfer-4090 serves any agent that speaks OpenAI Chat Completions, OpenAI Responses, or Anthropic
Messages. For the most native experience, use [Caudra](https://caudra.ai), the coding agent from
the same author. Caudra shows NInfer's live prefill rate while it reads the prompt, marks cache
breakpoints and session keys that NInfer reuses across turns, and can use
[qwen3.8-27b-decision-2](https://huggingface.co/tensorninja2/qwen3.8-27b-decision-2) on the same server for decisions. For the
`providers.toml` entry, see
[Use it from Caudra](https://github.com/tensorninja/ninfer-4090#use-it-from-caudra).

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- opt-in DFlash 2 speculative decoding with one to seven draft tokens, text only, from
  `qwen3_8_27b.dflash2.ninfer`;
- BF16, INT8, and Hadamard-rotated KV cache (`rk8v4`, `rk4v4`, and `rk2v4-e8`), with the full
  262,144-token context on `rk4v4`;
- LoRA adapters selected per request, such as
  [qwen3.8-27b-caveman-1](https://huggingface.co/tensorninja2/qwen3.8-27b-caveman-1), and the decision adapters
  [qwen3.8-27b-decision-2](https://huggingface.co/tensorninja2/qwen3.8-27b-decision-2) and
  [qwen3.8-27b-decision-1](https://huggingface.co/tensorninja2/qwen3.8-27b-decision-1);
- OpenAI- and Anthropic-compatible serving, OpenAI Decisions, and TypeSafe System One.

## Limits

- NInfer-4090 runs on one RTX 4090 and one CUDA device, with a startup-fixed capacity of one to
  eight active requests.
- It does not provide large-scale or preemptive continuous batching, priority scheduling,
  multi-GPU execution, CPU or GPU offload, or distributed serving.
- Context size depends on GPU memory and the selected KV-cache type.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Source repository | `Qwen/Qwen3.8-27B`, BF16 |
| Download source | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| Conversion recipe | `qwen3_8_27b-v2` |
| Converter repository | `https://github.com/tensorninja/ninfer-4090` |
| Converter revision | `ad26a68c`, and `5112a161` for the DFlash 2 file |
| DFlash 2 drafter | `z-lab/Qwen3.8-27B-DFlash2`, revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |
| Importance file SHA-256 | `d67fef13aba1879c3acc576bc22c8a2ba39b82cfe419e17550b58f46834b2c3e` |
| Calibration corpus SHA-256 | `8d3eed573cd102c1a243644acc0253e12588bae1f792a052a5d9b2de60535b23` |

The converter checked the source configuration, tensor shapes and dtypes, and frontend resources.
The six embedded frontend resources (tokenizer, chat template, and generation and media-processor
configuration) are byte-identical to those of the `v1` artifact. The artifact identity, object
inventory, and conversion provenance are in
[`artifact-manifest.json`](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer/blob/main/artifact-manifest.json) and
[`artifact-manifest.dflash2.json`](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer/blob/main/artifact-manifest.dflash2.json).
The storage contract is in the
[Qwen3.8-27B artifact reference](https://github.com/tensorninja/ninfer-4090/blob/tensorninja/docs/maintainer/qwen3.8-27b-artifact.md).

## License

This artifact is distributed under the Apache License 2.0. The source
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) repository and the
[DFlash 2 drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) are also Apache-2.0. The
conversion format and the `v1` artifact are the work of the [NInfer](https://github.com/Neroued/ninfer)
project.
