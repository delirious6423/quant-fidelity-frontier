---
language:
- en
- zh
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- nowag
- big-rabbit
- quantized
- moe
- 2bit
---
# Qwen3.8-Flash-Next NoWAG 2-bit — Paper2

This repository contains the trained **NoWAG routed-expert weights**, not a standalone Transformers model. Use them with Big-Rabbit and the matching base checkpoint below. The original base-model terms apply to the underlying model.

## Contents

- Expert-only NoWAG export: 48 MoE layers, 512 experts per layer.
- D=6, 12-bit assignments (approximately 2 bits per expert weight, excluding normalizers and metadata).
- One shared BF16 codebook with shape `[4096, 6]`; packed assignments and trained BF16 normalizers.
- Paper2 export, with fixed assignments and codebook during Paper2. This is the existing restored → cross-layer → Paper2 checkpoint; it is not a restored-only checkpoint.
- 98 original export files, approximately 29 GiB. The layer indexes and manifest reference the included relative filenames. Historical absolute paths in provenance fields are not needed for serving.

## Download

```bash
hf download Samfisheryu/Qwen3.8-flash-next-nowag-2bit --local-dir ./flash-next-nowag-paper2
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --local-dir ./flash-next-base
```

The base checkpoint supplies the architecture, tokenizer and non-expert tensors; this repository replaces its routed experts. Do not pass this directory alone to `AutoModel.from_pretrained`.

## Serve with Big-Rabbit

Install [Big-Rabbit](https://github.com/ServeBig-project/Big-Rabbit) following its setup instructions. The preparation baseline is commit `bb4582d7a6c1742fca19e9844157d1a0b1437309`.

```bash
ft serve --model ./flash-next-base \
  --nowag-expert-path ./flash-next-nowag-paper2 \
  --dense-quant fp8 --kv-dtype int8 \
  --served-model-name flash-next-nowag-paper2 \
  --host 127.0.0.1 --port 8000
```

The approximately 29 GiB download is only the compressed expert export, not the complete serving memory requirement. GPU/host cache capacity and base-model tensors must also fit the deployment.

## Evaluation scope

This upload preserves the existing trained checkpoint. It does not claim a new benchmark result, bit-identical output across runtimes, or a universal accuracy guarantee.
