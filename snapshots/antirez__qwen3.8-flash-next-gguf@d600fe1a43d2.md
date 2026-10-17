---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen4exp
- text-generation
---
# Qwen3.8 Flash Next for DwarfStar

Self-contained GGUFs for [DwarfStar](https://github.com/antirez/ds4), with
original BF16 n-grams read directly from disk. Keep the file on a local SSD.
No n-gram sidecar is needed. MTP weights are included; enable them with `--mtp`.
Requires DwarfStar's native BF16 n-gram reader. Older versions expecting
`--ple` cannot load these files.

| File | File size | Main/MTP weights |
| --- | ---: | ---: |
| Qwen3.8-Flash-Next-Q2.gguf | 137.10 GiB | 41.73 GiB |
| Qwen3.8-Flash-Next-Q4.gguf | 165.11 GiB | 69.74 GiB |

Each file contains the same 95.37 GiB BF16 n-gram table. It is not made
resident or quantized. Context and runtime buffers require additional RAM.

The main/MTP tensor bytes come unchanged from Ivan Fioravanti's
[Q2](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2) and
[Q4](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4) releases.
Q2 has imatrix-calibrated IQ2_XXS gate/up and Q2_K down routed experts,
with the down rows padded from 640 to 768 inputs. Q4 has calibrated Q4_K
gate/up and MXFP4 down. Other tensor formats and MTP weights are unchanged.

The n-grams are copied byte-for-byte from Qwen's original checkpoint at
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, including its padding rows.
The packer checks the original hash constants and verifies every copied
payload. These are quantized language-model weights, not lossless copies
of the whole original model.

Original model: Qwen. Quantized main/MTP releases: Ivan Fioravanti.
Native n-gram packaging and disk-only integration: Salvatore Sanfilippo / DwarfStar.
The original Qwen Community License is included as `LICENSE`.
