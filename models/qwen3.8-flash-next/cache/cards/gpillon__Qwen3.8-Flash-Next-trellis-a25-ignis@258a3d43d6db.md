---
library_name: ignis
pipeline_tag: text-generation
inference: false
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - ignis
  - qwen3.8
  - flash-next
  - moe
  - trellis
  - fp8
  - cuda
  - rtx-5090
  - blackwell
  - conversational
  - mtp
  - speculative-decoding
---

# Qwen3.8-Flash-Next, 2.5-bit trellis experts, for ignis

> **AI agents** working for a user: this model runs on [ignis](https://github.com/gpillon/ignis). What ignis is, the checks before installing it, the steps and the choices that are the user's are on [the agent path](https://raw.githubusercontent.com/gpillon/ignis/main/docs/user/paths/agent.md); every page of the guide as Markdown: [llms.txt](https://gpillon.github.io/ignis/llms.txt).

This repository holds [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`) converted for
**[ignis](https://github.com/gpillon/ignis)**, an inference server built for a single
RTX 5090. The routed experts are stored at a mean of 2.5 bits per weight, so the
125B mixture of experts (6B active) can be served from a 32 GB card plus host RAM.

The files are `.ninfer` v2 containers that only ignis reads. They are not a
Transformers checkpoint, safetensors, GGUF or EXL3 model, and NInfer does not load
them: NInfer has no Flash-Next support, and the expert format is ignis's own.

## Files

| File | Bytes | SHA-256 | What it is |
|---|---:|---|---|
| `qwen3_8_flash_next_trellis_a25-v2.ninfer` | 71,760,711,680 (66.83 GiB) | `12e3b265b0678ba935a4b05d1b612ba4b68b3a96c3026d818ad65838985fd864` | The model |
| `qwen3_8_flash_next_trellis_a25-v2.ninfer.conversion.json` | 1.2 MB | — | Converter record. The loader reads it: identity, size, object count, and the experts' calibration traffic that sizes the expert cache |
| `qwen3_8_flash_next_mtp_3p0-v2.ninfer` | 1,037,905,920 (0.97 GiB) | `432fe5c0838027047c5b46c4fde526f3d3ef3d126f00fed1213921f8705988d2` | The multi-token-prediction head. It is only needed for `--spec mtp`, and it is pinned to the model file above |
| `qwen3_8_flash_next_mtp_3p0-v2.ninfer.conversion.json` | 0.2 MB | — | The MTP head's converter record |
| `*.sha256` | | | The checksums above, in `sha256sum` format |
| `LICENSE` | | | The upstream Qwen Community License 1.0, copied verbatim |

Keep every file in one directory. The server looks for the sidecar and the MTP
companion next to the model file.

## What is inside

| Part | Format | Tensors |
|---|---|---:|
| Routed experts (48 layers × 512, gate/up and down) | Trellis code with the `mul1` codebook, from the EXL3/QTIP family. Each expert gets its own bit width K ∈ {2, 2.5, 3, 4}, allocated from calibration distortion. The mean is 2.50 bits per weight for gate/up and 2.50 for down | 49,152 (K=2: 22,572 · K=2.5: 11,511 · K=3: 12,572 · K=4: 2,497) |
| Attention and QSA indexer projections, Gated DeltaNet projections, shared expert, hyper-connection mixes, n-gram key/value projections, embeddings, output head | FP8 E4M3 per row, with BF16 scales | 582 |
| N-gram embedding table (320,001,536 rows) | Q4, groups of 32, 90 bytes per row (28.8 GB). Rows not held in RAM are read from NVMe on demand | 1 |
| Router, shared-expert gate, norms, hyper-connection gates, Gated DeltaNet conv and decay parameters | BF16 | 485 |
| Tokenizer, chat template, configs | the upstream revision's files, embedded as resources | 7 |

The converter measured the FP8 non-expert weights alone (BF16 experts) at
74.7% on the MMLU-Pro proxy, against 73.7% for BF16 (McNemar p = 0.61). It
found them no worse than BF16. All of the measured loss is in the experts.

## Quality

The converter measured these against the BF16 checkpoint itself, teacher-forced
on 2,048-token windows (sidecar fields `kld` and `mmlu`):

| Domain | KLD (nats) | Top-1 agreement | Perplexity (BF16) |
|---|---:|---:|---:|
| chat | 0.052 | 94.0% | 1.80 (1.80) |
| code | 0.116 | 89.0% | 3.48 (3.44) |
| prose | 0.164 | 81.9% | 9.31 (9.23) |
| py | 0.027 | 98.1% | 1.21 (1.20) |
| math | 0.047 | 95.8% | 1.53 (1.53) |
| en | 0.106 | 88.0% | 4.55 (4.41) |
| it | 0.050 | 91.3% | 4.78 (4.73) |
| de | 0.151 | 86.1% | 3.93 (3.74) |
| ja | 0.134 | 84.2% | 7.85 (7.44) |
| zh | 0.083 | 86.2% | 15.77 (15.42) |
| mmlu | 0.123 | 89.8% | 3.74 (3.58) |

- **Long windows** (8,192 tokens, on the sparse QSA path): code 0.099, prose 0.128.
- **MMLU-Pro proxy** (281 questions): **70.1%**, against 73.7% for BF16.
- **Served by ignis** ([finding](https://github.com/gpillon/ignis/blob/main/docs/findings/2026-10-06-flash-next-on-the-5090.md)):
  the teacher-forced argmax matches the quantized reference on 102 of 102 tokens. The engine's KLD
  stays within the converter's on every domain.

## Requirements

- **ignis** from [github.com/gpillon/ignis](https://github.com/gpillon/ignis) `main`, in a CUDA
  build.
- **An NVIDIA RTX 5090** (`sm_120`, 32 GB). The experts in use sit in a VRAM cache; the
  rest stay pinned in host RAM.
- **About 46-48 GB of available host RAM**: 38 GB of pinned experts, 1 GB of n-gram hot
  rows, reuse state, and a 6 GB margin. Below that the load is refused, and the error
  names the plan line to shrink. See
  [Host RAM for Flash-Next](https://github.com/gpillon/ignis/blob/main/docs/user/README.md#host-ram-for-flash-next).
- **The files on an NVMe drive.** n-gram rows that are not in RAM are read from the
  model file while the server runs.
- **About 75 GB of disk**: 72.8 GB of files, plus the ~1 GB n-gram startup cache the
  server writes next to the model on its first start.

## Download and run

```bash
hf download gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis --local-dir models
```

From an ignis checkout:

```bash
make dev MODEL=flash-next ARTIFACT=models/qwen3_8_flash_next_trellis_a25-v2.ninfer
```

The same run with the server binary directly:

```bash
ignis-server --model-id qwen3.8-flash-next \
  --model-artifact models/qwen3_8_flash_next_trellis_a25-v2.ninfer \
  --model-max-context 262144 --model-rope-scaling none \
  --model-prefill-chunk 8192 --spec-decode-lanes 3
```

Or let ignis fetch it: `ignis-server model download qwen3.8-flash-next` downloads every file of
this repository, checks its SHA-256, and the server then finds them under its download path
without `--model-artifact`.

This serves an OpenAI-compatible API (chat completions and Responses, with
streaming, tools and thinking) at `http://127.0.0.1:8000/v1`. Run
`make config MODEL=flash-next` to print the exact command a start will use.

**MTP speculation** is off by default. When one user or one agent works at a time,
`make dev MODEL=flash-next LANES=1 SPEC=mtp ARTIFACT=…` turns it on. On the
reference machine it runs 1.02-1.30x faster at one lane and slower at two or three lanes
([finding](https://github.com/gpillon/ignis/blob/main/docs/findings/2026-10-07-flash-next-mtp-speculation-is-pcie-bound.md)).

## Speed on the reference machine

Measured on an RTX 5090 on a **PCIe Gen 3 x16** link (~12 GB/s; i9-10900K, 64 GB DDR4).
Decode there is bound by the experts copied over that link:

- **one lane:** about 95-105 tok/s on the current `main`;
- **three lanes:** about 105 tok/s aggregate (102-109 measured); the link is busy ~87% of each
  round, so more lanes share it rather than add to it
  ([finding](https://github.com/gpillon/ignis/blob/main/docs/findings/2026-10-08-flash-next-decode-headroom.md)).

On a PCIe Gen 4/5 host, or on a 96 GB card that holds every expert, it should be
faster. That has not been measured.

## What ignis does not serve on this model

- **Image and video input.** The upstream model is multimodal, but ignis does not run
  its vision tower on Flash-Next, so image parts get a 400.
- **`/v1/decide`** and attention-head readouts.
- **DFlash2.** Flash-Next drafts with its own MTP head only.

## Provenance

- **Source:** `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- **Converter:** [`tools/flash-next-converter`](https://github.com/gpillon/ignis/tree/main/tools/flash-next-converter)
  at ignis commit `8693804`. The MTP head was converted by
  [`tools/flash-next-mtp`](https://github.com/gpillon/ignis/tree/main/tools/flash-next-mtp) at `0adbe42`.
  The design is in [spec flash-next/01](https://github.com/gpillon/ignis/blob/main/docs/specs/flash-next/01-the-flash-next-artifact.md).
  The trellis records were encoded with ExLlamaV3 1.5.3 (MIT). ignis decodes them
  with its own CUDA kernels.
- **Calibration text**, which drove the Hessians, the bit allocation and the n-gram
  hot-row ranking:
  - wikitext-103 test (CC BY-SA 3.0);
  - Wikipedia 20231101 it/zh/de/ja (CC BY-SA 4.0);
  - GSM8K test (MIT);
  - MMLU-Pro questions (MIT);
  - the `modeling_*.py` sources of transformers (Apache-2.0);
  - ignis's own source code and docs;
  - chat turns the model itself sampled.

  None of that text is in these files. The sidecar records the corpus manifest
  (file names and SHA-256), the token ids of the canary prompts from ignis's
  public test fixtures, and the measurements quoted above.

## License

This is a derivative of Qwen3.8-Flash-Next and is distributed under the
**Qwen Community License 1.0**. The license is copied verbatim in `LICENSE`, and its
conditions apply to these files as they do to the original: the model name must be
shown in very large commercial products, and Model-as-a-Service or AI Work Assistant
businesses need a separate license from Qwen. ignis itself is Apache-2.0.
