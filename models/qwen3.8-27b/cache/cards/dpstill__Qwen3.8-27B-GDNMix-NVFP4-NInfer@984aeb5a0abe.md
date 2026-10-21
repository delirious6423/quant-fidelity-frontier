---
license: apache-2.0
library_name: ninfer
pipeline_tag: text-generation
base_model:
  - neroued/Qwen3.8-27B-nvfp4-NInfer
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - qwen3.8
  - qwen3.5
  - ninfer
  - nvfp4
  - fp8
  - mixed-precision
  - quantized
  - gated-deltanet
  - gdn
  - cuda
  - blackwell
  - speculative-decoding
  - mtp
  - rtx-5090-laptop
  - windows
---

# Qwen3.8-27B GDNMix NVFP4 for NInfer

**Sensitivity-guided mixed FP8/NVFP4 Gated DeltaNet (GDN) projections for Qwen3.8-27B in native NInfer format.**

Designed for single-GPU inference on NVIDIA Blackwell GPUs with 24 GB VRAM. Tested on RTX 5090 Laptop with a configured maximum context of 139,264 tokens, NVFP4 KV cache, and MTP speculative decoding.

This repository distributes a modified version of the official [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) artifact. **Only the fused GDN Q/K/V/Z projection in 37 of the 48 GDN layers was replaced** using pre-quantized NVFP4 tensors from [`minima-ai/mnma_qwen3.8_27b_nvfp4`](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4). The remaining 11 GDN Q/K/V/Z projections, and the rest of the official NInfer weight profile, were retained.

The 37/11 precision split was informed by an independent, single-layer sensitivity sweep over **all 48 GDN layers**. The measurements are included as [`gdn-sensitivity-256.json`](./gdn-sensitivity-256.json).

> **Format:** The `.ninfer` file is a native NInfer model artifact. It is **not** a Transformers/Safetensors checkpoint or GGUF model and cannot be loaded directly by Transformers, vLLM, or llama.cpp.

## Files

| File | Description |
| --- | --- |
| [`qwen3_8_27b_gdnmix_nvfp4.ninfer`](./qwen3_8_27b_gdnmix_nvfp4.ninfer) | Complete mixed-precision NInfer model artifact (approximately 22.4 GB as displayed by Hugging Face). |
| [`gdn-sensitivity-256.json`](./gdn-sensitivity-256.json) | Single-GDN replacement sensitivity results, baseline, and per-layer diagnostics. |

## Precision profile

| Component | GDNMix precision | Origin |
| --- | --- | --- |
| MLP, text layers **0–55** | NVFP4 | Unchanged official NInfer base |
| MLP, text layers **56–63** | Row-scaled FP8 | Unchanged official NInfer base |
| Full-attention input/output projections | Row-scaled FP8 | Unchanged official NInfer base |
| GDN Q/K/V/Z, **11 selected layers** | Row-scaled FP8 | Unchanged official NInfer base |
| GDN Q/K/V/Z, **37 selected layers** | NVFP4 | Imported from Minima |
| GDN output projections | Row-scaled FP8 | Unchanged official NInfer base |
| GDN A/B projection and other control tensors | Base artifact formats (including BF16/FP32) | Unchanged official NInfer base |
| Embedding and output head | Row-scaled FP8 | Unchanged official NInfer base |

The format labels above describe the **text weight profile**. The artifact may also contain additional model/frontend resources inherited from the selected NInfer base revision. Optional modality and draft paths require a compatible runtime and separate validation.

**FP8 GDN Q/K/V/Z layers (11):**

```text
0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13
```

**Minima NVFP4 GDN Q/K/V/Z layers (37):**

```text
14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30,
32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46,
48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62
```

Layer indices are **zero-based** within the model's 64 text layers. The 16 full-attention layers are not included in the 48-layer GDN list.

### Why Minima?

The official NInfer NVFP4 model retains GDN Q/K/V/Z projections in FP8. Minima provides already-quantized **NVFP4 W4A4** GDN `in_proj_qkv` and `in_proj_z` weights, including the packed weights and scaling metadata. Minima's PTQ process also harmonizes scales for fused projection execution. This made its model a suitable **source of encoded NVFP4 GDN data**, without re-quantizing an already quantized FP8 projection.

The import targets NInfer's fused `query_key_value_z` parent (combining the Minima `in_proj_qkv` and `in_proj_z` representations). **This repository is not a full Minima conversion:** the other GDN components and all non-GDN-QKVZ weights remain from the official NInfer base.

For Minima's quantization method and its own evaluations, see [its model card](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4) and the [accompanying paper](https://arxiv.org/abs/2609.04098). Minima's published evaluation numbers **must not** be interpreted as results for this mixed artifact.

## Sensitivity-guided selection

Before assembling GDNMix, the effect of replacing each GDN Q/K/V/Z projection was measured independently:

1. **Reference (`teacher`):** the official NInfer artifact, retaining FP8 GDN Q/K/V/Z.
2. **Candidate (`student`):** replace **exactly one** GDN Q/K/V/Z fused parent with its Minima NVFP4 counterpart, leaving all other parents unchanged.
3. Evaluate teacher/student output distributions at **256 positions** in a **4,096-token context**, with a **BF16 KV cache** and `NINFER_CUDA_SYNC=blocking`.
4. Measure `KL(student || teacher)` as the primary distribution-level sensitivity metric, together with top-1 flips, target log-probability/rank changes, and perplexity-related diagnostics.
5. Repeat for each of the **48 GDN layers**, then choose the mixed precision allocation for the assembled model.

Two reference runs in the recorded experiment were **byte-identical**, supporting baseline reproducibility under that test setup.

The early GDN layers showed greater sensitivity to the FP8-to-NVFP4 substitution than many deeper layers. Accordingly, the first 11 GDN Q/K/V/Z parents were retained in FP8, while the remaining 37 were taken from Minima in NVFP4. All 48 independent measurements are provided in the included JSON file.

**Interpretation caveat:** These are *single-layer* intervention measurements. They do **not** by themselves prove that the combined 37-layer substitution has no quality loss, nor do they establish superior quality to any other model. The final combined artifact requires a matched end-to-end evaluation for such claims.

## Running the model

**Requirements:** A compatible build of [NInfer](https://github.com/Neroued/ninfer), a recent CUDA runtime/driver, and an NVIDIA GPU with support for the artifact's NVFP4 execution path. This release was used on **Windows with an RTX 5090 Laptop GPU (24 GB VRAM)** and a locally built `ninfer-serve.exe`. Compatibility, context capacity, and optimal settings depend on the NInfer revision and device.

### Download

```powershell
hf download dpstill/Qwen3.8-27B-GDNMix-NVFP4-NInfer qwen3_8_27b_gdnmix_nvfp4.ninfer --local-dir D:\Ai\models\gdnmix
```

### Windows PowerShell / NInfer server example

The following is an **observed single-user configuration**, not a universal performance optimum. Update the executable and model paths for your installation.

```powershell
$env:NINFER_CUDA_SYNC = 'blocking'

& 'D:\dev\ninfer.laptop\build-windows\apps\ninfer-serve.exe' `
  'D:\Ai\models\gdnmix\qwen3_8_27b_gdnmix_nvfp4.ninfer' `
  --model-id qwen3.8-27b-ninfer `
  --device 0 `
  --host 127.0.0.1 --port 8080 `
  --max-context 139264 --kv-capacity 139264 --kv-dtype nvfp4 `
  --prefill-chunk 4096 --max-concurrency 1 `
  --spec mtp --draft-tokens 8 --lm-head-draft `
  --preserve-thinking
```

This is a **139,264-token configured runtime context limit**, not a claim that every installation can reserve that context or that long-context task accuracy has been benchmarked at that length. It uses an NVFP4 KV cache, which is separate from the weight precision profile.

Check the server's advertised model ID:

```powershell
curl.exe http://127.0.0.1:8080/v1/models
```

For OpenCode or another OpenAI-compatible client, use `http://127.0.0.1:8080/v1` as the base URL and the **exact model ID** returned by `/v1/models`. The `--model-id` value is an API alias and does not need to match the filename.

## Status and limitations

- **Verified by the author:** final artifact creation and structural checks; successful startup and text-agent use through a local Windows NInfer server and OpenCode with NVFP4 KV and static MTP.
- **Reported test hardware:** RTX 5090 Laptop GPU, 24 GB VRAM. The sample settings use `max_concurrency=1`.
- **Not yet provided as GDNMix-specific benchmark results:** matched BF16-vs-GDNMix or official-NInfer-vs-GDNMix accuracy, GPQA/MMLU/AIME, full long-context retrieval accuracy, or standardized tokens-per-second comparisons.
- **Not an official Neroued or Minima release.** Optional Vision/DFlash2 paths and behavior with other NInfer revisions should be tested separately before advertising support for them.

## Provenance and acknowledgements

This work modifies and redistributes weights from the following projects, with attribution:

- **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** — original model family and checkpoint.
- **[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)** — direct **base `.ninfer` artifact** used for GDNMix. Its official card describes the text weight allocation and its additional sources.
- **[minima-ai/mnma_qwen3.8_27b_nvfp4](https://huggingface.co/minima-ai/mnma_qwen3.8_27b_nvfp4)** — donor of the NVFP4 GDN Q/K/V/Z weight representations selectively imported at 37 GDN layer positions.
- **[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)** — source credited by the official NInfer base artifact for its packed text weight profile.
- **[NInfer](https://github.com/Neroued/ninfer)** — runtime and artifact ecosystem.

The source repositories listed above identify their model distributions as **Apache-2.0**. This derivative artifact is released under **Apache License 2.0**, subject to the original upstream notices and terms. The accompanying experiment file describes independently measured GDN interventions; it is not part of Minima's own published benchmark suite.

## Reproduction notes

To independently reproduce the *precision-selection logic*, begin with the exact official NInfer base artifact and Minima checkpoint revisions used for the conversion, then perform the one-parent-at-a-time sensitivity sweep described above. The selection results alone are not a complete build script; full bit-for-bit reproduction also requires the conversion implementation, source revision IDs, and artifact integrity hashes. These details can be added as supplementary provenance metadata when available.
