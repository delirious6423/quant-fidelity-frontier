---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
- exl3
- exllamav3
- deepseek
- deepseek_v41
- moe
- quantized
---

# DeepSeek-V4.1-Flash-1017 · EXL3 1.38 bpw

**103.8 GB, with 1.2% lower KL than [v1](https://huggingface.co/redthing1/DeepSeek-V4.1-Flash-1017-EXL3-2.87bpw) at half the size.** Other EXL3 releases spend 2.75–2.9 bpw.

KL is measured on the finished weights against the original model, over all **129,280** next tokens, on the first 131,072 tokens of complete reverse-engineering sessions. The binaries are stripped, with no source: a 67-opcode bytecode interpreter, and an asset-bundle tool audited through function-pointer tables. Neither family was used in fitting. Lower is closer.

| Session | [v1](https://huggingface.co/redthing1/DeepSeek-V4.1-Flash-1017-EXL3-2.87bpw), 207 GB | [v2](https://huggingface.co/redthing1/DeepSeek-V4.1-Flash-1017-EXL3-2.87bpw-Quantseekv2), 207 GB | **104 GB** |
| --- | ---: | ---: | ---: |
| Bytecode VM, run 1 | 0.01265 | 0.01050 | **0.01253** |
| Bytecode VM, run 2 | 0.01504 | 0.01231 | **0.01469** |
| Whole-program query, run 1 | 0.01105 | 0.00897 | **0.01071** |
| Whole-program query, run 2 | 0.01206 | 0.01038 | **0.01254** |

## Method

**Format.** The **46,080** routed matrices average **1.38 bits per weight**. **19,083** are stored as W = A·B at rank 128–512, with both factors in EXL3. B is encoded under input statistics captured inside the original model, and A is refit to the coded B. Gate and up share one B. The rest are plain EXL3 at 1–6 bits. Attention and shared-expert projections use 5–8 bits, the output head 8, and the DSpark drafter's routed experts 1.

**Calibration.** **15** complete reasoning and tool-use histories, eight of reverse engineering and seven of algorithm construction, with FP8 activation rounding included.

**Allocation.** Every candidate format is encoded. Its error on an expert's real inputs, against the original model's gradients at those positions, gives its KL. These prices come from **29** sessions, with windows as far as 540,000 tokens in. One budget covers the whole model, chosen jointly over each expert's gate/up and down projections.

**Correction.** Prices treat errors as independent. Returning each band of layers to higher precision in the complete model measures what they miss:

| Layers | 0–4 | 5–9 | 10–14 | 15–19 | 20–24 | 25–29 | 30–39 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Measured over priced KL, relative | 2.55 | 1.27 | 1.49 | 1.19 | 0.56 | 0.54 | 0.41 |

Low-rank matrices measure about **1.5×** their price against plain ones. The allocation applies both corrections.

## Download

Approximately **103.8 GB**, plus **203.1 GB** for the two unchanged Engram shards from the official checkpoint.

```sh
model_dir=./DeepSeek-V4.1-Flash-1017-EXL3-1.38bpw-Quantseekv2
hf download redthing1/DeepSeek-V4.1-Flash-1017-EXL3-1.38bpw-Quantseekv2 --local-dir "$model_dir"
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir "$model_dir"
```

Load with an ExLlamaV3 runtime supporting DeepSeek V4.1/1017 and low-rank EXL3. The index already includes Engram. Vision and drafter weights are included. `quantseek-formats.json` lists every matrix's stored format. [`lowrank/`](lowrank/LOWRANK.md) defines low-rank EXL3, with a plain PyTorch reference and test vectors.

MIT license. Base model by [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017); format and runtime by [ExLlamaV3](https://github.com/turboderp-org/exllamav3).
