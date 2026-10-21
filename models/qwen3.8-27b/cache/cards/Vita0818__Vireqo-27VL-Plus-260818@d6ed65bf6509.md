---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: image-text-to-text
language:
  - zh
  - en
base_model:
  - Qwen/Qwen3.8-27B
  - prism-ml/Ternary-Bonsai-27B-gguf
tags:
  - gguf
  - vision-language
  - multimodal
  - qwen3-vl
  - mmproj
  - ternary
  - q2_0
  - 2-bit
  - qwen
  - bonsai
  - llama-cpp
  - lm-studio
  - apple-silicon
  - experimental
  - extreme-quantization
inference: false
---

# Vireqo-27VL-Plus-260818

> **Unofficial, experimental vision-language bundle: the unchanged Vireqo-27B-Plus-260816 ternary language model paired with Prism ML's Q8_0 Qwen3-VL vision projector.**

Vireqo-27VL-Plus-260818 adds tested image input to the 7.06 GiB Vireqo ternary branch. It is a two-file llama.cpp/LM Studio bundle, not a new monolithic GGUF and not a retrained multimodal model. The language tensor payload is byte-for-byte identical to `Vireqo-27B-Plus-260816`; visual capability comes from the separately loaded `qwen3vl_merger` projector published by Prism ML for Ternary-Bonsai-27B and Bonsai-27B.

中文摘要：本版本已经在 LM Studio 中实际通过图片颜色/形状/位置、中文 OCR、表格、柱状图读取和简单图上计算测试。它复用原 Plus 三值主 GGUF，再配套一份约 600 MiB 的官方 Q8 projector。与标准 VL 相比，它保留更大的三值语言工作点，但没有进行新的视觉训练。Thinking 必须关闭。

## Required files

Keep both files in the same model directory:

| File | Size | SHA-256 | Role |
|---|---:|---|---|
| `Vireqo-27VL-Plus-260818.gguf` | 7,585,332,288 bytes / 7.0644 GiB | `a32a8ec286a11c6534bf29d1ee20bd4c02064032b51ae8310bb1216e2de17e03` | unchanged Vireqo-27B-Plus-260816 ternary language model |
| `mmproj-Vireqo-27VL-Plus-260818-Q8_0.gguf` | 629,246,880 bytes / 0.5860 GiB | `eb561d41a7bbeb0fcf04883c8af11078ef6cae0a66862a0b68443cfca495269d` | 27-block Qwen3-VL vision tower and merger |

Logical bundle size: 8,214,579,168 bytes, approximately 7.6504 GiB. LM Studio reports 8.21 decimal GB.

The local release filename is a symbolic-link alias so the existing language weight is not duplicated. Consequently, the underlying GGUF still contains the internal `general.name` value `Vireqo-27B-Plus-260816`. The release identity comes from the bundle directory and filename. This is intentional storage deduplication, not hidden model substitution.

## What provides vision

The projector has the following verified GGUF structure:

| Item | Value |
|---|---:|
| Architecture | `clip` |
| GGUF type | `mmproj` |
| Projector | `qwen3vl_merger` |
| Output / language projection dimension | 5120 |
| Input image size | 768 |
| Patch size | 16 |
| Vision embedding length | 1152 |
| Vision FFN length | 4304 |
| Vision blocks | 27 |
| Attention heads | 16 |
| Spatial merge | 2 |
| Tensors | 334 |
| Tensor types | 224 F32, 83 Q8_0, 27 F16 |

The language GGUF has `qwen35.embedding_length = 5120`, M-RoPE sections `[11, 11, 10, 0]`, Qwen3-VL image/video special tokens, and a vision-aware Qwen3.8 chat template. The projector dimension and language interface match. Plus also preserves source-exact Ternary-Bonsai token embedding, output head, and recurrent SSM controls, making this projector pairing structurally conservative.

Prism ML publishes the same 629,246,880-byte object under both repositories:

- `prism-ml/Ternary-Bonsai-27B-gguf`, revision `abbae723028d71be674e71e1a71201a6f43fab22`, file `Ternary-Bonsai-27B-mmproj-Q8_0.gguf`;
- `prism-ml/Bonsai-27B-gguf`, revision `f10afb355f104535e3e3e98cf7ab7795c72bd292`, file `Bonsai-27B-mmproj-Q8_0.gguf`.

Both resolve to the exact same LFS SHA-256 shown above. The standard and Plus VL releases therefore share one physical projector in the tested local installation.

## Validation

Two independent runtime paths were accepted:

1. LM Studio's bundled llama.cpp server with explicit `--mmproj`;
2. LM Studio's own `lms load` automatic projector pairing followed by the native port 1234 API.

Both paths passed the same deterministic suite:

| Probe | Result |
|---|---:|
| Red circle / green triangle / blue square, including positions | PASS |
| Chinese order ID `VL-260818` | PASS |
| Amount `¥128.50` | PASS |
| Table: apples 3, oranges 2 | PASS |
| Chart title and values 2024=40, 2025=70, 2026=55 | PASS |
| Highest year 2025 and difference 15 | PASS |
| Text-only answer with projector loaded | PASS |

See [`vision-validation.md`](vision-validation.md) and [`lmstudio-native-vision-validation.md`](lmstudio-native-vision-validation.md) for complete answers. The reproducible fixtures and script are in [`eval/`](eval/).

This validates controlled synthetic images, Chinese OCR, a small table, a bar chart, spatial relations, and simple arithmetic. It is not a natural-image benchmark, MMMU score, OCR benchmark, or broad grounding claim.

## Local runtime

| Item | Result |
|---|---:|
| Backend | LM Studio `llama.cpp-mac-arm64-apple-metal-advsimd-2.28.2` |
| Context | 2048 |
| GPU offload | 100% |
| Flash attention | on |
| K/V cache | Q8_0 / Q8_0 |
| Parallel | 1 |
| Direct-server maximum RSS | 9,147,252,736 bytes / 8.5190 GiB |
| Swap | 0 |
| `lms load` reported memory | 7.65 GiB |
| LM Studio load time | 18.37 s |
| Card response generation | approximately 13 tokens/s |

The device used was an Apple M5 MacBook Pro with about 12 GiB reserved for the task. Results depend on runtime version, external-drive state, context, image resolution, and other unified-memory use.

## LM Studio

The tested local model key is:

```text
vireqo-27vl-plus-260818
```

Recommended settings:

| Setting | Value |
|---|---:|
| Context | 2048; reduce to 1024 under memory pressure |
| GPU offload | Maximum |
| Flash Attention | On |
| K cache / V cache | Q8_0 / Q8_0 |
| Parallel requests | 1 |
| Thinking / Reasoning | **Off** |
| MTP / draft model | Off |
| Image minimum tokens | default first; try 1024 only for difficult grounding |

Select `Vireqo-27VL-Plus-260818`, attach an image in a new Chat, ensure the blue **Think** control is off, and ask a direct visual question.

### Thinking warning

Thinking is not accepted. With LM Studio request defaults, the model can spend the full output allowance on reasoning tokens while returning an empty visible `content` field—including for a text-only capital-city prompt. Setting `reasoning_effort=none` and `enable_thinking=false`, equivalent to disabling Think in the UI, restored all text and image responses.

This is the known Vireqo thinking-channel issue, not a vision-projector failure.

## Local storage note

The validation setup stored the physical files on an external exFAT volume; the local release and LM Studio directories used symbolic links. No additional physical copy of the 7.59 GB language weight was created, and both VL variants shared the same projector inode.

If you use symbolic links locally, keep the external files available while loading the bundle. This Hub release contains real GGUF file contents rather than symbolic links.

## Limitations

- Thinking/reasoning must remain off.
- Natural photographs, handwriting, dense documents, fine grounding, and standard multimodal benchmarks were not comprehensively tested.
- Video input is not accepted as a release capability.
- The runtime warns that difficult Qwen-VL grounding may need at least 1024 image tokens; the tested synthetic fixtures succeeded with defaults.
- Visual answers can hallucinate even when OCR and chart probes pass.
- The language model retains all limitations of Vireqo-27B-Plus-260816, including extreme ternary compression and imperfect strict instruction following.
- This is not an official Qwen or Prism ML release and not equivalent to full-precision Qwen3.8-27B.
- Do not use it for safety-critical visual interpretation, medical images, legal evidence, or financial decisions.

## Reproducibility and attribution

- [`bundle-provenance.json`](bundle-provenance.json) records both physical sources, symlink identity, projector structure, revisions, and runtime results.
- [`runtime-validation.json`](runtime-validation.json) records accepted and rejected capabilities.
- [`SHA256SUMS.txt`](SHA256SUMS.txt) verifies both required files.
- [`NOTICE`](NOTICE) retains Qwen and Prism ML attribution.

The bundle is provided under Apache License 2.0, consistent with the pinned upstream repositories. Retained attribution: **Created using Bonsai by Prism ML.** `Vireqo-27VL-Plus-260818` is an independent experimental name; no affiliation or endorsement is claimed.
