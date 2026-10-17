---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: llama.cpp
tags:
  - gguf
  - ternary
  - mtp
  - quantized
  - research
---
# Ternary Bonsai 2 27B, RTX 4070 research artifact

This repository is a community-derived publication of a CPU-prepared MTP draft-quantization artifact for the [Prism ML Ternary Bonsai 2 27B GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

## Artifact

`Ternary-Bonsai-2-27B-PQ2_0-MTP-Q4_K_M.gguf` is the combined Prism PQ2 target plus a Q4_K_M-only MTP head preparation. The target tensors and target metadata were preserved from the combined source; 8 MTP matrices were quantized and 7 MTP norm tensors were preserved unchanged. This is a CPU-prepared packaging/quantization artifact, not a new trained model.

The controlled lab results for this exact artifact were 81.65 tok/s on the short fixture, 68.30 tok/s with a filled 46K context, and 51.37 tok/s with a filled 60K context; the small-math gate was 20/20. These are historical RTX 4070 measurements from the accompanying research campaign, not a universal performance promise.

The source artifact is pinned by SHA256 in `provenance.json`; tensor mappings are in `artifact-tensor-manifest.json`. See the [final measured report](https://github.com/trevor050/escha-rtx4070-lab/blob/main/bonsai/reports/BONSAI_FINAL_REPORT_2026-09-19.md) and the [RTX 4070 inference lab](https://github.com/trevor050/escha-rtx4070-lab) for the implementation and experiment history.

Download the reviewed files with `hf download trevor000/Ternary-Bonsai-2-27B-RTX4070 --local-dir Ternary-Bonsai-2-27B-RTX4070`. The exact artifact SHA256 and source paths remain in `provenance.json`.

## Runtime

Use a Bonsai-compatible Prism/llama.cpp runtime that understands the packed rotation and MTP layout, following the [Prism model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b). The lab runtime pin is Prism PR205, commit `518ad108f0b72bac4f397a486695a5786e33a58d`. Follow the [launcher and profile instructions](https://github.com/trevor050/escha-rtx4070-lab/blob/main/bonsai/work/bonsai-chat/README.md); the Windows entry point is `bonsai/work/bonsai-chat/start-bonsai-chat.ps1 -Profile balanced`. Stock llama.cpp is not asserted compatible with this artifact.

The reported short result is a three-run median with 48K allocated FP16 context and a fixed 2,048-token output. The 46K result uses filled history and FP16 KV; the 60K result uses Q4 keys and FP16 values. These profiles include the lab's custom small-batch BF16 dispatch overlay. The 20/20 arithmetic result is a small public regression test, not a general intelligence score; factual hallucinations remain a documented checkpoint limitation.

## Attribution and license

This artifact is derived from Prism ML’s Bonsai 2 27B, which is built from Qwen3.8-27B. It is not an official Prism ML or Qwen release. Preserve [Prism’s notice](NOTICE.txt), [Prism’s Apache-2.0 license](LICENSE), and the included Qwen license. Please cite Prism ML and credit the upstream Qwen authors.

The MTP preparation follows the independent [ProCreations donor experiment](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP), not an official Prism ML or Qwen release. The donor artifact's historical source revision was not captured in this lab snapshot, so the public provenance pins the donor identity and this derived artifact's SHA rather than inventing a donor commit. Historical donor numbers were measured on an RTX PRO 6000 Blackwell and are not claimed here.
