---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
- qwen
- abliterated
- uncensored
- refusal-removal
- vision
- video
---

# Qwen3.8-Flash-Next-Abliterated (baked)

The refusal-abliterated Qwen3.8-Flash-Next with the edit **merged into the weights**, so it runs on
engines that cannot load a LoRA adapter. The base is
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) served from the
[Qwen3.8-Flash-Next-GGUF](https://huggingface.co/apetersson/Qwen3.8-Flash-Next-GGUF) package; the
edit is the same one shipped as
[Qwen3.8-Flash-Next-Abliterated-Adapter](https://huggingface.co/apetersson/Qwen3.8-Flash-Next-Abliterated-Adapter).

Nothing else changes: the BF16 n-gram table, the vision projector and the MTP head are the stock
files already published in the base package. Only the trunk carries new weights.

## Performance beyond refusal removal

**Instruction following improved by two passes (+25 percentage points) on the IFEval slice.**
Math produced one additional verified correct answer; code and knowledge kept the same pass counts.

| Check | Stock Q5 | Baked Q5 | Observed change |
|---|---:|---:|---|
| Instruction following (IFEval) | 5/8 | **7/8** | **+2 passes** |
| Math (MATH-500) | 6/8 | **7/8** | **+1 verified correct answer** |
| Code (HumanEval+/MBPP+) | 10/12 | 10/12 | Same pass count |
| Knowledge (MMLU-Pro slice) | 6/8 | 6/8 | Same pass count |

Denominators include all questions. Stock math had two unscorable cases, baked math had one, and
stock knowledge had one. The historical summary instead uses scorable denominators for those cases.
These small non-thinking samples ran on different llama.cpp revisions and do not establish a
general capability improvement. The Q5 adapter matched stock's instruction and math counts;
the gains above belong to this baked release. Sources: [per-question scores](evaluation/capability-scores.json)
and [stock/adapter comparison](evaluation/validation.json).

## What is in this repository

| File | Size | Role |
|---|---:|---|
| `Qwen3.8-Flash-Next-Abliterated-Q5_K_M-00001-of-00002.gguf` | 78072.8 MiB | Baked trunk (shard 1 of 2) |
| `Qwen3.8-Flash-Next-Abliterated-Q5_K_M-00002-of-00002.gguf` | 95.37 GiB | Stock BF16 n-gram table (shard 2 of 2), byte-identical to the base package's copy |
| `link-ngrams.sh` | | Replaces shard 2 with a hard link/clone of a table you already have (saves 95 GiB) |
| `evaluation/validation.json` | | Load, behaviour, capability and bake-fidelity evidence with pinned cohort hashes |
| `evaluation/capability-scores.json` | | Per-row capability scores (frozen scorers) |
| `evaluation/tensor-integrity.json` | | Per-tensor proof that the 1078 untouched tensors are byte-identical |
| `bake-audit.json` | | Method and per-type numbers for how much of the edit survives re-quantization |
| `manifest.json`, `SHA256SUMS`, `LICENSE`, `LLAMA-CPP-LICENSE` | | Pins, hashes, licenses |

Both GGUF shards are included. Shard 2 is the unchanged BF16 n-gram table from the base package;
llama.cpp finds it automatically beside shard 1. Existing owners can
[reuse their table](#reuse-an-existing-table-optional) to avoid storing another copy.

## Quick start

### 1. Download

Download this model, then the base package's optional vision projector and MTP head:

```bash
hf download apetersson/Qwen3.8-Flash-Next-Abliterated --local-dir Qwen3.8-Flash-Next-Abliterated
hf download apetersson/Qwen3.8-Flash-Next-GGUF --local-dir Qwen3.8-Flash-Next-GGUF \
  --include "*.md" "*LICENSE" "*.json" "shared/mmproj-*" "shared/mtp-*"
```

Run both download and launch commands from the same parent directory. Skip the second download
for text-only inference without MTP.

### 2. Run with stock llama.cpp

Use a build with `qwen4exp` support (upstream PR #27742). No adapter or runtime patch is needed.

**128 GB Mac / 96 GB RTX PRO 6000: 256K context, vision + MTP:**

```bash
/path/to/llama-server \
  --model ./Qwen3.8-Flash-Next-Abliterated/Qwen3.8-Flash-Next-Abliterated-Q5_K_M-00001-of-00002.gguf \
  --mmproj ./Qwen3.8-Flash-Next-GGUF/shared/mmproj-Qwen3.8-Flash-Next-f16.gguf \
  --model-draft ./Qwen3.8-Flash-Next-GGUF/shared/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 -ngld 99 \
  --fit off --lazy-mode on -ngl 99 -fa on --jinja --parallel 1 \
  --ctx-size 262144 --batch-size 512 --ubatch-size 128 --no-context-shift \
  --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 \
  --host 127.0.0.1 --port 18080
```

Both support files come from the base package's `shared/`. Drop the four `--model-draft`/`--spec-*`
arguments for text without speculative decoding, and `--mmproj` too for text-only.

### Reuse an existing table (optional)

If you already have the base package's BF16 table, run this inside the downloaded model directory:

```bash
rm Qwen3.8-Flash-Next-Abliterated-Q5_K_M-00002-of-00002.gguf
./link-ngrams.sh /path/to/Qwen3.8-Flash-Next-GGUF/shared/ngrams-Qwen3.8-Flash-Next-BF16.gguf
```

The helper verifies the source checksum, then tries a hard link or copy-on-write clone before
falling back to a full copy, and checks the result. Linking or cloning avoids another 95.37 GiB
of storage; the copy fallback does not.

## Which machine runs it

The following stock-base memory measurements include the vision projector and MTP head.
The baked trunk uses the same tensor sizes and quantization layout.

| Machine | Context | Notes |
|---|---|---|
| 128 GB Mac (Ultra) | 262,144 | The reference configuration. The base package measured **95.5 GiB peak wired memory** for the 262,144-token run with vision + MTP, no swap growth. |
| 96 GB Mac | 131,072–262,144 | Tighter. Use Q8 KV caches (`--cache-type-k q8_0 --cache-type-v q8_0`) or a smaller context; for the 96 GB path the base package prefers its IQ4 trunk **with the adapter**, which keeps 262K comfortable. |
| RTX PRO 6000 Blackwell (96 GB) | 262,144 | Measured **90.51 GiB peak including driver reservation, 5.09 GiB free** (base package, same trunk layout). The BF16 table stays CPU-side. |
| RTX PRO 6000 Blackwell (96 GB) | 1,048,576 | Not with Q5. Use the IQ4 trunk with the adapter and Q8 KV + YaRN (measured 87.85 GiB stock). |

The trunk must fit in unified memory/VRAM. The 95.37 GiB table is read lazily from disk
(`--lazy-mode on`), so keep it on fast storage with room for the OS page cache.

## Engine support

| Engine | Status | Notes |
|---|---|---|
| **llama.cpp (stock, master ≥ 2026-08-27)** | **Yes** | Verified here: the two-shard pair loads with the stock binary, no patch, no adapter. |
| llama.cpp + shared-ngrams patch | Yes | Also loads the pair; `--lazy-mode on` keeps the table on CPU. |
| **DS4 (`ivanfioravanti/ds4`, branch `qwen3.8-flash-next`)** | **No** | Requires a dedicated pack. See [Building a DS4 pack](#building-a-ds4-pack). |
| Ollama | Untested | Its runtime carries `qwen4exp`, but this pair was not verified here. |
| LM Studio | Untested | Works only if its llama.cpp build includes `qwen4exp`; no LoRA is involved, which is why this repository exists. |
| Jan / text-generation-webui / KoboldCpp | Untested | Any build with `qwen4exp` should load the pair. |

## Verification

**Loading.** Stock llama.cpp `df03399` loaded the pair and completed the arithmetic smoke at
24.9 generation tokens/s on an M1 Ultra, with the table streamed from SSD. The evaluation harness
checked the live process against its runtime and policy pins.

**Behaviour.** The baked trunk was run through the sealed 6-cell sentinel panel (non-thinking +
thinking variants) on that stock runtime: **6/6 substantive answers, zero refusals, zero evasions,
zero fake compliance**. The unedited trunks refused 6/6 on the same cells. An independent reviewer
labelled the transcripts with the frozen rubric; the repetition-loop cell that the adapter carrier
failed once did **not** reproduce here.

**Bake fidelity.** The merge dequantizes each target, adds `B·A`, and requantizes into its original
storage type. Seven sampled tensors cover all five target families and four storage types.
Their effective retained projection is the realized/intended delta norm ratio times their cosine:

| Storage type | Sampled tensors | Effective projection retained |
|---|---:|---|
| Q5_K (attention/PLE residual writers) | 3 | **0.97–1.00** |
| Q8_0 (shared-expert down) | 2 | **0.99–1.01** |
| Q5_1 (shared-expert down) | 1 | **0.86** |
| IQ4_NL (routed-expert down) | 1 | **0.34** |

The routed-expert sample retained much less of the intended projection. These samples do not
establish fidelity for every edited tensor. [Full audit and method](bake-audit.json).

**Untouched-tensor integrity.** All 1078 non-target tensors are **byte-identical** to the published
Q5 trunk (verified tensor by tensor over the whole file); only the 145 targets differ, and the split
metadata (`split.no=0`, `split.count=2`, `split.tensors.count=1224`) matches the stock table shard.

## Building a DS4 pack

DS4's Qwen3.8-Flash-Next branch is a closed pack format. To produce an abliterated DS4 pack you need
the edit in the **BF16 checkpoint**, not in a GGUF:

1. Apply the same 145-tensor rank-one edit to the official BF16 checkpoint (the recipe, direction
   bundle and strengths are recorded in `manifest.json`).
2. Run the fork's converter, e.g.
   `uv run --script gguf-tools/qwen4_pack.py --src /path/to/Qwen3.8-Flash-Next-metadata --out /path/to/qwen3.8-flash-next-q4 --remote-repo Qwen/Qwen3.8-Flash-Next --source-revision <pinned> --tokenizer-template /path/to/qwen-tokenizer-template.gguf`.
   It writes the base artifact, the external `Qwen3.8-Flash-Next-PLE-Q4_1.gguf` sidecar and optional
   vision/MTP sidecars, and its state file resumes per source shard.

Budget: the converter needs the final pack plus at most the largest source shard, and the mutated
BF16 checkpoint itself is about 335 GB.

## Provenance and licensing

- Base weights, BF16 n-gram table, vision projector and MTP head: see the
  [base package](https://huggingface.co/apetersson/Qwen3.8-Flash-Next-GGUF) for full credits
  (Qwen, Unsloth's IQ4 trunk, bartowski's iMatrix, llama.cpp contributors).
- The edit is a refusal-direction projection over 145 trunk writer tensors; the adapter repository
  carries the same edit as a runtime LoRA and documents the verification protocol in detail.
- Weights are derivative of the base model and stay under the
  [Qwen Community License 1.0](LICENSE) and its acceptable-use terms.

## Support

If this is useful to you, sign up through my [Runpod affiliate link](https://runpod.io?ref=bjuhotnb).
You get signup credit and I may earn a commission.
