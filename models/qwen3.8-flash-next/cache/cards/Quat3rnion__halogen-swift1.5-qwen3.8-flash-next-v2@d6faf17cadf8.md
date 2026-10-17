---
license: other
license_name: swift-open-license-1.0
license_link: https://huggingface.co/Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2/blob/main/LICENSE
base_model:
  - ukisai/Swift1.5-Qwen3.8-Flash-Next
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: halogen
inference: false
tags:
  - swift
  - efficient-thinking
  - reasoning
  - halogen
  - halogen-flash
  - qwen
  - moe
  - mixture-of-experts
  - long-context
  - quantized
  - 4-bit
  - strix-halo
  - gfx1151
  - rocm
  - amd
---

# Halogen Swift 1.5 Qwen3.8-Flash-Next v2

The current `qwen38-flash-next-v2-swift15.hgn` is a **fresh native-v2 conversion of the complete Swift 1.5 BF16 source**, not the previous 300-tensor transplant into peonist's checkpoint. The pinned source is [ukisai/Swift1.5-Qwen3.8-Flash-Next @ `0bd4fe22431372cdad1979267d3ab45aa7e6150a`](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/tree/0bd4fe22431372cdad1979267d3ab45aa7e6150a). No abliteration edit was applied. The [separate abliterated repository](https://huggingface.co/Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2-abliterated) is not changed by this release.

This is Halogen's `.hgn` format, not a transformers/vLLM/llama.cpp/Ollama/LM Studio checkpoint. The intended engine is [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server). **This newly converted file has not been loaded or inference-qualified here.** There are no new speed, perplexity, thinking-length, refusal, MTP, tool-use, vision or long-context results; the old transplant's results must not be attributed to this file. This is an unofficial derivative, not endorsed by UkisAI, Peonist or Qwen.

## Current files and conversion provenance

| File | Size | SHA-256 |
|---|---:|---|
| `qwen38-flash-next-v2-swift15.hgn` | 66,687,678,432 B | `79379640bbd1fa0ec3886d9c3c823f52022478d372dc1cb8a3581d3b257e4b25` |
| `qwen38-flash-next-ht43-swift15.hgn` (native BF16 conversion) | 57,618,253,792 B | `c4e37a0919f0fe1592d254ac0001874be5c3523c82750b8a0796e26f9944607a` |

The current v2 file was generated with `bf16-to-hgn`, profile `v2`, seed 0, two CPU threads and HT beam 128. Both HT and I4R encoders used the HIP backend; Q4CP used CPU. Conversion and source verification completed on 2026-10-13. Source verification attested provenance, checked the pinned source shard identities and unchanged source fingerprints, and verified all 1,658 source tensors against 2,349 generated output tensors across the complete local output set. The encoder SHA-256 was `397818673b3f9f2caddbe0a93b637831e9cb73eac33be39d6362a3696446c27f`. These are conversion/fidelity checks, **not inference qualification**.

Unlike the previous appended-payload transplant, this file does not use peonist's stock v2 as its weight donor: the conversion also generated Swift-source MTP, n-gram, vision and tokenizer/config assets. **Only the main checkpoint is published in this update.** The generated n-gram and vision files (roughly 52 GB combined), tokenizer/config assets and local conversion manifest are not uploaded. The main file's MTP data come from the Swift source, not an unchanged stock MTP head.

### Dependencies and limitations

A complete runtime setup needs matching dependencies, including an n-gram table and tokenizer/config assets, and a vision tower for image input. Compatibility of external peonist stock dependencies with this newly converted checkpoint has **not been validated by hash comparison or inference**. The old card's claim that these dependencies were unchanged applies to the historical transplant only. Do not combine this current file with the old stock dependency instructions and assume a qualified setup.

No runnable quick start is provided for the current file until a matching dependency set and engine have been qualified. Architecture/context descriptions from upstream are not evidence that this file's inference, MTP, vision, or long-context behavior works. Source verification does not establish runtime memory requirements or quality/speed metrics.

## Historical transplant benchmarks, tooling and instructions

The previous v2 checkpoint was 68,142,577,664 bytes, SHA-256 `a4ad500200982fcda77c3fe03367ebbbe120ec65724585e9aecda77435e55205`. Its 300-tensor transplant method, 58% shorter-thinking sample, approximately 52–57 tok/s speed, 3.181 WikiText-2 perplexity, refusal/capability tests, calibration and reproduction commands are **historical results for that different file**, not measurements of the native conversion now at the v2 filename.

See the [complete historical card and quick start pinned at `2f77baa4b1c65bf9520e8c7f2568f405d5efd210`](https://huggingface.co/Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2/blob/2f77baa4b1c65bf9520e8c7f2568f405d5efd210/README.md). Its unpinned download commands must be pinned to that old revision if reproducing the historical setup; they are not instructions for the current file. The preserved `tooling/`, `writers.txt` and transplant-related `NOTICE` describe that historical build, not the native-v2 converter. `ht43-tooling/` and `ht43-build.json` remain the historical HT43 transplant provenance/tooling. None of these files or any licenses were changed in this update.

## Current native HT43 alternative

`qwen38-flash-next-ht43-swift15.hgn` is now a **fresh, direct conversion of the original non-abliterated Swift 1.5 BF16 snapshot**, not a re-encoding of native v2 and not the previous stock-HT43/transplanted-dense-weights derivative. Its source is [ukisai/Swift1.5-Qwen3.8-Flash-Next @ `0bd4fe22431372cdad1979267d3ab45aa7e6150a`](https://huggingface.co/ukisai/Swift1.5-Qwen3.8-Flash-Next/tree/0bd4fe22431372cdad1979267d3ab45aa7e6150a). No abliteration, stock payload transplant or hidden calibration donor was used. The independently published native-v2 checkpoint is preserved unchanged.

The deterministic conversion used `bf16-to-hgn --profile ht43 --ht-backend hip --seed 0 --threads 2 --ht-beam 128`; I4R and Q4CP backend axes stayed CPU (the routed trunk experts use HT, not I4R). HIP HT encoding was directly observed on Frame. Conversion completed and full `verify --decode-backend hip` passed on 2026-10-14: 1,658 source tensors, 2,349 output tensors and 109,716,035,228 payload bytes across the complete local output set. Manifest schema 2 and source provenance were attested; all 131 source-shard hash receipts matched the pinned source and current source fingerprints remained unchanged. The encoder source SHA-256 was `397818673b3f9f2caddbe0a93b637831e9cb73eac33be39d6362a3696446c27f`; the qualified executable SHA-256 was `9eae3f4755960718d1eed132d9760e218a1a83950e604c0c1e9eeca53b5c5cfd`. These checks establish conversion fidelity, **not inference qualification**.

The main HGN file is 57,618,253,792 bytes, SHA-256 `c4e37a0919f0fe1592d254ac0001874be5c3523c82750b8a0796e26f9944607a`. The HT43 policy uses HT4 for routed trunk experts in zero-based even layers and HT3 in odd layers, with Q8G64 trunk hyper-connection mixers. Dense trunk HT and the other codec/layout policies follow the converter's documented profile. This changes live weight representation, not model architecture or context length.

The converter generated MTP, n-gram, vision, tokenizer and config assets **from this Swift BF16 source**; embedded main-file MTP is not transplanted stock data. This publication replaces **only the HT43 main checkpoint and this model card**. Newly generated sidecars, tokenizer/config assets and the local manifest are not uploaded. The existing native-v2 main file and all unrelated repository files remain unchanged. External stock-dependency compatibility has not been validated; use of historical download/dependency instructions must not be mistaken for a qualified runtime setup.

**The new native HT43 file has not been loaded into Halogen or tested for inference, MTP behavior, vision, tool use, refusal behavior, perplexity, throughput, thinking length, long context or runtime memory.** No deployment is performed. In particular, the historical transplant's 2026-10-01 GPU smoke checks and perplexity/KL results do not apply to this new file. Those results and the old HT43 identity (`598c4b031f933b74fbb2cadaa1ff8aba8149b776b80b3af0c4b6de194ffefc86`, 57,618,253,824 bytes) remain documented in the [historical model card pinned at `2f77baa4b1c65bf9520e8c7f2568f405d5efd210`](https://huggingface.co/Quat3rnion/halogen-swift1.5-qwen3.8-flash-next-v2/blob/2f77baa4b1c65bf9520e8c7f2568f405d5efd210/README.md). Preserved `ht43-tooling/` and `ht43-build.json` describe that historical transplant, not this native conversion. No comparative HT43-versus-v2 quality or runtime-memory result is claimed.

Choosing HT43 does not remove or relax the license restrictions below.

## License and attribution

This repository is **not** Apache-2.0 as a whole. Read `LICENSE`, `LICENSE-QWEN` and `NOTICE` before using or redistributing it. License files are preserved unchanged.

- **Swift Contribution** (UkisAI's fine-tune; the historical changed-tensor list is preserved in `writers.txt`): [Swift Open License v1.0](LICENSE). Commercial use is not licensed for an entity with US$1,000,000 or more in annual gross revenue (including affiliates); such entities need a Swift Enterprise License from [UkisAI](https://ukisai.com/contact).
- **Base model** ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)): [Qwen Community License 1.0](LICENSE-QWEN). Among other terms, a Model-as-a-Service or AI Work Assistant business needs a separate license from Qwen for commercial use.
- **Peonist's checkpoints and historical derivatives** ([peonist-ai/halogen-qwen3.8-flash-next](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next)), including the historical HT43 transplant and the prior v2 transplant: Apache License 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE)). The current native-v2 and native-HT43 conversions do not reuse peonist checkpoint payloads.

This summary is not legal advice; the license texts control. Historical modification notices remain in `NOTICE`; the native-v2 and native-HT43 replacements are described in this card. The historical calibration tokens are from WikiText-2 (CC BY-SA 3.0); those transplant calibration results are not the current conversion's provenance. "halogen" is a trademark of Peonist, LLC, and "Swift" and "UkisAI" are UkisAI's; the names are used here only to describe the origin of the work and the compatible engine. The halogen-flash-server engine is closed source and distributed under its own terms. The v2 format notes build on [jtsylve/hgn-spec](https://github.com/jtsylve/hgn-spec) 1.0.1.
