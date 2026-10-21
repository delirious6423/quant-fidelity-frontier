---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - qwen3
  - 2-bit
  - escha
  - quantized
  - research
---
# Qwen3.8-27B Escha-W2, RTX 4070 research mirror

This is an attributed mirror of [EschaLabs/Qwen3.8-27B-Escha-W2](https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2), prepared for reproducible RTX 4070 research. The weights are unchanged upstream files; this project did not train them. `opencode.json` is intentionally excluded from the publication allowlist.

## Included files

The manifest allowlists the two safetensors weight shards, the safetensors index, configuration, generation settings, tokenizer files, vocabulary/merges, chat template, and quantization metadata from `/home/trevor/inference-lab/models/escha-w2`. The model’s `LICENSE` and `THIRD_PARTY_LICENSES/Qwen-LICENSE.txt` are included here as notices. No runtime binary or custom kernel is bundled; the separate [Escha runtime model repository](https://huggingface.co/EschaLabs/escha-runtime-qwen3dense) is named in `provenance.json`.

Download the reviewed files with `hf download trevor000/Qwen3.8-27B-Escha-W2-RTX4070 --local-dir Qwen3.8-27B-Escha-W2-RTX4070`. The model source revision is `561dae0aeea1efbdafe122f2e385fdeaa99127e7`; use the runtime's own documented launcher and license rather than assuming this mirror is a standalone Transformers package.

## Historical measurements

Historical lab context includes a 148.03 tok/s median with a deliberately tiny 176-token cache control, plus a usable product fallback around 125 tok/s at 5,120 exact active tokens with a 12,288-token rolling RAM archive. Those measurements require the lab's modified SGLang runtime and separate Apathy DFlash draft, not these weights alone. See the [measured hill-climb report](https://github.com/trevor050/escha-rtx4070-lab/blob/main/outputs/ESCHA_HILLCLIMB_REPORT_2026-08-27.md) and [chat launcher](https://github.com/trevor050/escha-rtx4070-lab/tree/main/work/escha-chat). Archive recall is not equivalent to full exact attention over the archive. This mirror adds no new GPU benchmark.

## Attribution and license

The card declares Apache-2.0 and identifies Qwen/Qwen3.8-27B as the base. Retain the included Qwen third-party license and Escha Labs license. This mirror has no upstream endorsement; cite Escha Labs and Qwen, and review the separate Escha runtime license before redistributing runtime code.
