---
library_name: ignis
pipeline_tag: text-generation
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ignis
  - qwen3.8
  - trellis
  - exl3
  - dflash2
  - speculative-decoding
  - cuda
  - blackwell
  - 16gb
  - rtx-5060-ti
  - rtx-5070-ti
  - rtx-5080
  - conversational
---

# Qwen3.8-27B, 3-bit trellis body, for ignis on 16 GB cards

> **AI agents** working for a user: this model runs on [ignis](https://github.com/gpillon/ignis). What ignis is, the checks before installing it, the steps and the choices that are the user's are on [the agent path](https://raw.githubusercontent.com/gpillon/ignis/main/docs/user/paths/agent.md); every page of the guide as Markdown: [llms.txt](https://gpillon.github.io/ignis/llms.txt).

> **Have a 16 GB RTX 50?** No real 16 GB card has run this yet. Ask your coding agent to read [the agent path](https://raw.githubusercontent.com/gpillon/ignis/main/docs/user/paths/agent.md) and follow its section on testing a 16 GB card, or [measure it yourself](https://gpillon.github.io/ignis/guides/measure-a-16gb-card/), and post the report on [gpillon/ignis#360](https://github.com/gpillon/ignis/issues/360).

This repository holds [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) packed for
**[ignis](https://github.com/gpillon/ignis)** for the 16 GB Blackwell cards (RTX 5060 Ti 16 GB,
5070 Ti, 5080). It is the second artifact of the 27B in ignis, beside the 32 GB one
([gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer)).

- **The body and the output head** are turboderp's EXL3 quant
  [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3), branch
  `SC_3.00bpw_H4` (revision `86b95530f9d365d90dd24d01a997a8de0cdf6c55`), repacked bit for bit: every
  matrix's trellis, `suh` and `svh` tensors are stored as that branch stores them. Nothing was
  re-quantized.
- **Everything else** is the 32 GB artifact's object, byte for byte: the token embedding (8-bit), the
  norms and Gated DeltaNet parameters, the DFlash2 drafter and its draft head, the MTP layer, the vision
  tower, the tokenizer and chat template.

The file is a `.ninfer` v2 container that only ignis reads. It is not a Transformers checkpoint,
safetensors, GGUF or EXL3 model.

## Run it with ignis

[**ignis**](https://github.com/gpillon/ignis) is an OpenAI-compatible inference server for one
Blackwell card, shaped for the load a coding agent makes: one main agent plus a few subagents on the
same card. With this file it brings the 27B, with its DFlash2 drafter, to a 16 GB card:

| | |
|---|---:|
| one lane, coding prompts at 8K of context | **416 tok/s** |
| two lanes at once, coding prompts at 8K, aggregate | **684 tok/s** |
| context, two lanes, within 16 GiB | **262,144 tokens** |
| KL divergence from BF16, against the 32 GB NVFP4 artifact's | **0.77-0.82x** |
| estimated, one lane at 8K: RTX 5080 / 5070 Ti / 5060 Ti 16 GB | 243-307 / 204-270 / 103-153 tok/s |

The speeds and the context were measured on an RTX 5090 held to a 16 GiB budget, the load a 16 GB card
makes, and the KL divergence on the same ignis build; the last row is an estimate from those
([Measurements](#measurements)). The body is closer to BF16 than the
NVFP4 artifact it stands in for, and its DFlash2 drafter commits as many tokens a round. Prompt reuse
across requests, a reasoning-loop guard, a Playground and Prometheus metrics ship in the binary.

## How to run

ignis serves this file since **[v0.8.0](https://github.com/gpillon/ignis/releases/tag/v0.8.0)**:
a Windows `.zip`, a Linux `.tar.gz` and a `linux/amd64` container image on the
[releases page](https://github.com/gpillon/ignis/releases).

On a 16 GB card (RTX 5060 Ti 16 GB, 5070 Ti, 5080):

```bash
ignis-server model download qwen3.8-27b-16g   # fetches this file at its pinned revision and checks its SHA-256
ignis-server
```

With no flags the server reads the card's memory and takes its `rtx50-16gb` profile: 262,144 tokens of
context, two decode lanes, DFlash2 with 7 draft tokens, no image input. The default model id,
`qwen3.8-27b`, resolves on a 16 GB card to this file's catalog entry, `qwen3.8-27b-16g`, so a bare
`ignis-server` with no model on disk offers to fetch this file by itself (in a container, it fetches it
without asking). The OpenAI API is then on
`http://127.0.0.1:8000/v1` and the Playground on `http://127.0.0.1:8000/ui/`.

If the desktop already holds part of the card, the load lowers its context step by step instead of
refusing, and `GET /v1/models` names what it adapted. Leave out the 32 GB card's flags
(`--model-max-context 524288`, `--vision-enabled`): they ask for more than a 16 GB card holds.

On a larger card held to a 16 GB card's memory, the way every figure below was measured:

```bash
ignis-server --model-id qwen3.8-27b-16g --profile rtx50-16gb --vram-budget-bytes 17179869184
```

The guide's page [The 27B on a 16 GB card](https://gpillon.github.io/ignis/why-ignis/sixteen-gb-cards/)
says how it works; [Measure a 16 GB card](https://gpillon.github.io/ignis/guides/measure-a-16gb-card/)
is one command that writes a report of your card's load and speed.

## Files

| File | Bytes | What it is |
|---|---:|---|
| `qwen3_8_27b_trellis_sc30-v2.ninfer` | 13,379,877,124 (12.46 GiB) | The model |
| `qwen3_8_27b_trellis_sc30-v2.ninfer.conversion.json` | 309,495 | The conversion record: sources by sha256, every matrix's K and reconstruction error, every copied object's sha256 |
| `qwen3_8_27b_trellis_sc30-v2.ninfer.sha256` | | The model's checksum, in `sha256sum` format |
| `LICENSE` | | Apache License 2.0, as distributed with Qwen/Qwen3.8-27B |

sha256 of the model:

```
2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c
```

## What is inside

| Part | Format | Tensors | Bytes |
|---|---|---:|---:|
| Body: 48 Gated DeltaNet and 16 attention layers, their linears and MLPs | Trellis code, `mul1` codebook, a bit width K per matrix: K=2: 34, K=3: 297, K=4: 66, K=5: 4; 3.00 bits per weight on average | 400 | 9,135,915,008 |
| Output head | Trellis, K=4 | 1 | 636,207,104 |
| Token embedding (248,320 × 5,120) | 8-bit, groups of 32, fp16 scales | 1 | 1,350,860,800 |
| DFlash2 drafter (5 layers) and its 131,072-row draft head | NVFP4 weights, BF16 norms; the head Q4 | 68 | 1,439,902,344 |
| MTP layer, vision tower, norms, Gated DeltaNet parameters | as in the 32 GB artifact | 746 | 799,465,376 |
| Tokenizer, chat template, configs | the upstream files, embedded | 6 | 12,837,177 |

The fused projections of the 32 GB artifact are split here as the quant stores them (`in_proj_qkv` and
`in_proj_z`, `q_proj` / `k_proj` / `v_proj`, `gate_proj` and `up_proj`), each with its own scales.

**Device memory** under ignis's `rtx50-16gb` load, with the token embedding in host memory: weights
10.91 GB, the language model's 9.83 GB and the DFlash2 drafter's 1.08 GB. The profile drafts with the
full output head, so the draft head stays on disk, and so does the vision tower. The KV cache and the
kernels' workspaces come on top: the whole plan is under [Measurements](#measurements).

## Measurements

ignis's speeds and capacity below were measured on an RTX 5090 (PCIe Gen 3) held to a 16 GiB budget,
the load a 16 GB card makes: `rtx50-16gb`, 17,179,869,184 bytes, 262,144 tokens of context, two lanes.
The quality gates ran on the same ignis build: DFlash2 and the loops on that load, KLD and MMLU-Pro
through the engine's logits with the same KV format and prefill chunk. The speeds for the 16 GB cards
are an estimate from those measurements, not a measurement on the cards.

### In ignis: four quality gates, registered before the run

Each gate compares this file with the 32 GB NVFP4 artifact, served by the same ignis build, or with BF16.

| Gate | This file | Against | Pass if |
|---|---:|---:|---|
| KLD from BF16, 16 windows of 4,096 tokens (code, prose, chat) | 0.0403 | NVFP4 at the same chunk (1,024) 0.0522: **0.77x**; at its served chunk (2,048) 0.0492: 0.82x | at most 1.2x the NVFP4's |
| MMLU-Pro, 281 questions, through the engine's logits | 65.84% | BF16 67.97%: exact McNemar p **0.31** | p > 0.05 |
| DFlash2 tokens per round, 200 recorded agent turns, teacher-forced | 3.370 | NVFP4 3.383: **0.996x** (95% interval 0.993-0.999) | at least 0.90x |
| Turns caught in a reasoning loop, the loop study's four prompts, eight seeds each | 9 of 32 | NVFP4 14 of 32 | not significantly more at one-sided Fisher p < 0.05 (p 0.94) |

All four pass. One figure was not a gate: the thinking budget forced a close on 15 of 32 of those turns,
against 8 of 32 on the NVFP4.

### In ignis: the load and its speed

- **The VRAM plan at 16 GiB:** weights 10.91 GB (the token embedding, 1.35 GB, is in host memory), a KV
  pool of 509,184 tokens, 262,144 tokens of context on two lanes, nothing adapted. A 15 GiB budget keeps
  the same with a 392,704-token pool; a 13 GiB budget adapts the context to 65,536. On a 16 GB card the
  budget is its free memory less the profile's 1,200 MiB headroom.
- **Decode**, greedy coding prompts, DFlash2 with 7 drafts (about 6 tokens a round):

| context | one lane | two lanes, together |
|---|---:|---:|
| 8K | 416 tok/s | 684 tok/s |
| 64K | 354 tok/s | 500 tok/s |
| 253K | 216 tok/s | 275 tok/s |

  Agent traffic commits about 3.5 tokens a round, not 6, which scales these figures by about 0.58.
- **Cold prefill:** 3,478 tok/s on an 8K prompt, 3,363 at 64K, 2,616 at 253K: 2.5x slower than the
  NVFP4 artifact at 130K (42.1 s against 17.1 s).

### Estimated for the 16 GB cards

Not measured. One lane at 8K, from the 5090's rounds with the trellis kernels held to each card's SM
count, its memory bandwidth and attention scaled; the range is optimistic to conservative:

| card | 1 lane, 8K | 1 lane, 64K | 2 lanes, 8K |
|---|---:|---:|---:|
| RTX 5080 | 243-307 tok/s | 197-267 tok/s | 393-516 tok/s |
| RTX 5070 Ti | 204-270 tok/s | 168-244 tok/s | 331-458 tok/s |
| RTX 5060 Ti 16 GB | 103-153 tok/s | 85-146 tok/s | 164-256 tok/s |

### Weights only, before ignis served it

The quant's weights against the BF16 checkpoint, measured layer by layer in a PyTorch harness with BF16
activations (not in ignis), on the same KLD windows:

| | KLD mean | Top-1 agreement | Perplexity |
|---|---:|---:|---:|
| This body (SC_3.00bpw_H4 weights) | 0.0340 | 93.88% | 2.733 |
| The 32 GB artifact's NVFP4 weights | 0.0458 | 92.92% | 2.744 |
| BF16 | | | 2.690 |

- On a held-out set, perplexity rose 8.6% against BF16 on Python source and 1.4-3.7% in eight other
  domains (prose, chat, code, English, Italian, German, Japanese, Chinese).
- MMLU-Pro, the same 281 questions, weights only: 64.77%, against 68.33% for BF16.
- DFlash2 tokens per round on this quant, in ExLlamaV3's own DFlash2 path, against an EXL3 5.0-bit
  control: 1.00-1.06 of the control's, on agent turns at 8K, 64K and 128K tokens and on coding prompts.
- Reconstruction error of each matrix against its BF16 original (relative Frobenius norm): 0.150 on
  average, 0.338 at most, 0.0765 for the output head.

ignis's KLD (0.0403) sits above the harness's (0.0340) because ignis serves a compressed KV cache
(hq-e8-2b): with a BF16 KV cache its KLD equals the harness's to within 0.0001 on the two windows
checked, and the codec adds the same to the NVFP4 artifact.

Details, in ignis's findings:
[the quality gates](https://github.com/gpillon/ignis/blob/v0.8.0/docs/findings/2026-10-24-27b-16gb-quality-gates.md),
[capacity on the 5090](https://github.com/gpillon/ignis/blob/v0.8.0/docs/findings/2026-10-24-27b-16gb-capacity-on-the-5090.md),
[the speed estimate](https://github.com/gpillon/ignis/blob/v0.8.0/docs/findings/2026-10-24-27b-16gb-speed-estimate.md),
[first light](https://github.com/gpillon/ignis/blob/v0.8.0/docs/findings/2026-10-23-27b-16gb-first-light.md),
[the weights-only study](https://github.com/gpillon/ignis/blob/v0.8.0/docs/findings/2026-10-21-exl3-3bpw-27b-on-our-text-and-dflash2.md).

## Known limits

- **Validated on an RTX 5090 held to a 16 GiB budget.** The project has no 16 GB card yet, so the
  16 GB cards' speeds are an estimate. On one of them,
  `ignis-bench small-card --endpoint http://127.0.0.1:8000 --out small-card.json` writes a record of the
  load and its speed; posted on [gpillon/ignis#360](https://github.com/gpillon/ignis/issues/360), it
  gives the `rtx50-16gb` profile measured numbers.
- **Prefill** runs at about 3,400 tok/s on 8K to 64K prompts; a 130K prompt takes 2.5x the NVFP4
  artifact's time on the 5090 (42.1 s against 17.1 s). The trellis prefill kernels reach about half the
  card's tensor peak so far. Raising them is
  [gpillon/ignis#345](https://github.com/gpillon/ignis/issues/345).
- **The context depends on what else holds the card.** The budget is the free memory less 1,200 MiB of
  headroom, and the capacity figures above are budgets named on a 5090: a desktop holding more of a
  16 GB card makes the load lower its context, and `GET /v1/models` says by how much.
- **Two decode lanes and no image input.** The 32 GB card runs eight lanes and vision. The vision tower
  is in this file, but the `rtx50-16gb` profile leaves it off: a 16 GB card has no room for it.

## Provenance

- **Body and head:** turboderp/Qwen3.8-27B-exl3, branch `SC_3.00bpw_H4`, revision
  `86b95530f9d365d90dd24d01a997a8de0cdf6c55`, quantized with ExLlamaV3 by turboderp. Both shards are
  pinned by sha256 in the conversion record.
- **Everything else:** gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer, file
  `qwen3_8_27b_nvfp4full-v2.ninfer` (sha256 `abb1e120…fd9af`), pinned with its graft record.
- **Reconstruction errors** measured against Qwen/Qwen3.8-27B at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` with ExLlamaV3 1.5.3's reconstruction.
- **Converter:** [`tools/27b-16gb-converter`](https://github.com/gpillon/ignis/tree/main/tools/27b-16gb-converter)
  at ignis commit `cf5e328`; the design is
  [spec 27b-16gb/01](https://github.com/gpillon/ignis/blob/main/docs/specs/27b-16gb/01-the-16gb-artifact.md).
  Packing the same sources twice gives the same file.

## License

Apache License 2.0, as for Qwen/Qwen3.8-27B and for turboderp's quant. The trellis records are
turboderp's quantization of Qwen's weights; ExLlamaV3 is MIT. ignis itself is Apache-2.0.
