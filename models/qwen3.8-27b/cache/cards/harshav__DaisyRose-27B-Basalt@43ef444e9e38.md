---
base_model:
  - RadixArk/DaisyRose-27B-NVFP4
  - RadixArk/DaisyRose-27B
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
pipeline_tag: image-text-to-text
library_name: basalt
language:
  - en
  - multilingual
tags:
  - basalt
  - qwen35
  - qwen3.8
  - daisyrose
  - nvfp4
  - fp8
  - dspine
  - speculative-decoding
  - blackwell
  - sm_120
  - conversational
  - image-text-to-text
---

# DaisyRose-27B-NVFP4-DSpine for Basalt

Self-contained `.basalt` model pack combining RadixArk's [`RadixArk/DaisyRose-27B-NVFP4`](https://huggingface.co/RadixArk/DaisyRose-27B-NVFP4) (RL-trained 27B hybrid GDN/GQA multimodal reasoning model built on [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)) with a bundled 5-layer **DSpine** causal block drafter (`block_size = 16`, `sliding_window = 8192`) for [Basalt](https://github.com/jesdga95/basalt) on NVIDIA Blackwell (`sm_120`) GPUs.

> **Engine compatibility note:** This pack bundles a **DSpine** causal block drafter (`qwen35.dflash.kind = "dspine"`, CodeBridge + adjacent causal injection + serial refinement), which requires a Basalt build with DSpine support. Stock upstream [`jesdga95/basalt`](https://github.com/jesdga95/basalt) currently supports DFlash2 only; a public fork / upstream PR with DSpine + PEFT/LoRA support will be linked here once published.

## Files

| File | Size | Target Weights | Bundled Drafter | Output Head |
| --- | ---: | --- | --- | --- |
| `DaisyRose-27B-NVFP4-DSpine.basalt` | 22.36 GiB (`24,012,116,510` B) | Native ModelOpt NVFP4 FFN + FP8 E4M3 Attn/GDN | 5-layer **DSpine** (`block_size=16`, causal, `window=8192`) | NVFP4 (`type 45`) |
| `SHA256SUMS` | — | Checksum manifest | — | — |
| `LICENSE` | — | Apache License 2.0 | — | — |

SHA-256: `ac141820b5fc8821d3eef4e958a865fad9eb8b97f8ec18bf9444e8e56e1a4c87`

## What Is in the File

All target weights come directly from `RadixArk/DaisyRose-27B-NVFP4` (zero re-quantization); vision encoder, tokenizer, chat template, and DSpine drafter are packed into one `.basalt` file.

| Component | Stored Format | Source Handling |
| --- | --- | --- |
| FFN projections (`gate`, `up`, `down`) & output head (`output.weight`) | NVFP4 planes (`type 45`) | Native ModelOpt NVFP4 from `RadixArk/DaisyRose-27B-NVFP4` |
| Attention (`q`, `k`, `v`, `o`, `gate`, `qkv`) & GDN `ssm_out` | FP8 E4M3 (`type 44`) | Native ModelOpt FP8 E4M3 from `RadixArk/DaisyRose-27B-NVFP4` |
| Token embedding & GDN `ssm_alpha`, `ssm_beta` | BF16 (`type 30`) | Source BF16 as stored |
| Norms, scales, `conv1d`, `ssm_a`, `ssm_dt` | F32 (`type 0`) | Exact as stored |
| Vision encoder & projector (`vision/mmproj`) | BF16 / F32 | Whole (`--vision dynamic`) |
| DSpine causal block drafter (`dflash.*`, `kind=dspine`, 5 layers, targets `[1, 16, 31, 46, 61]`, `code_size=1024`, `bridge_rank=512`, `head_topk=16`) | Q8_0 / BF16 / F32 | Bundled from [Tencent/DSpine](https://github.com/Tencent/DSpine) (`Qwen3.8-27B-DSpine-b16`) |

## Speed

Measured on a single **NVIDIA RTX 5090 (600 W)** with `basalt-bench --kv int8 --spec 16` (16-row verify windows, DSpine drafting active, `--parallel 1`):

| Workload | Throughput (tok/s) | Draft Acceptance | Mean Draft Len |
| --- | ---: | ---: | ---: |
| `pp16384` (16K prompt read) | **14,562** | — | — |
| `tg conv` (conversation) | **227** | 42.1% | 5.69 |
| `tg struct` (structured / code) | **931** | 99.3% | 14.89 |
| `tg prose` (creative prose) | **168** | 25.0% | 6.00 |

DSpine verification is target-exact (lossless): every drafted token is verified against the target model's logits.

## Download, Verify, and Run

```bash
hf download harshav/DaisyRose-27B-Basalt \
  DaisyRose-27B-NVFP4-DSpine.basalt SHA256SUMS \
  --local-dir models

(cd models && sha256sum -c SHA256SUMS)
```

Serve with a DSpine-enabled Basalt build:

```bash
bin/basalt serve models/DaisyRose-27B-NVFP4-DSpine.basalt \
  --gpu 0 --max-context 262144 --kv int8 --spec 16 --parallel 1 --vision dynamic
```

## Sources and Attribution

- **Target model:** [`RadixArk/DaisyRose-27B-NVFP4`](https://huggingface.co/RadixArk/DaisyRose-27B-NVFP4) and [`RadixArk/DaisyRose-27B`](https://huggingface.co/RadixArk/DaisyRose-27B) by RadixArk (Apache-2.0), RL-trained on Alibaba's [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
- **DSpine drafter:** [Tencent/DSpine](https://github.com/Tencent/DSpine) (`arXiv:2609.36173`, Apache-2.0).
- **Engine & pack format:** [Basalt](https://github.com/jesdga95/basalt) (MIT), forked from [Strata](https://github.com/Niko1221/Strata) (MIT).

## License

The base target weights (`RadixArk/DaisyRose-27B-NVFP4`, derived from `Qwen/Qwen3.8-27B`) and the bundled DSpine drafter are licensed under **Apache License 2.0** (full text included in [`LICENSE`](LICENSE)). This repository distributes a repacked `.basalt` derivative combining both checkpoints for Blackwell inference. **This is not an official RadixArk, Qwen, or Tencent release.**
