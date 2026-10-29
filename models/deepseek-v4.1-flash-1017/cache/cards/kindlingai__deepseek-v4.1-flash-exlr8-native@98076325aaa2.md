---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- moe
- mxfp4
- fp8
- tensor-parallel
- exlr8
---

# DeepSeek-V4.1-Flash-1017 · exlr8 native

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with its official weights,
byte for byte, in the [exlr8 v2](https://github.com/kindlingai/exlr8/blob/main/FORMAT.md) file layout. Nothing is
requantized: the routed experts keep their MXFP4 codes and scales, and every other matrix keeps its FP8 or BF16
values. The file layout is the one change.

Weight set ID: `dc566d59ad98f6a895094193753815e870849031458fa0c014f7bd1c610949b3`. Source revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`.

## Why this layout

The source checkpoint stores each expert as whole tensors in 48 shards of up to 95 GB. exlr8 cuts every expert
projection into units of 128 intermediate channels and gives each unit a fixed owner at each tensor-parallel size,
so a box fetches only its own share with HTTP range requests:

- `index-slices/tpN/<store>/rankR.safetensors` lists every unit one rank owns at TP = N: (expert, unit),
  its byte offset in the layer's store file, and its xxhash64 checksum. Slices exist for TP = 1 to 8.
- The owner of unit u of expert e in layer L at TP = N is rank (18·e + u + L) mod N. The shared expert is expert
  E (384, or 128 in the MTP layers).
- Dense matrices sit in one file per layer, with a checksum per 128-channel block of the side a TP engine shards.

## Stores

| store | used for | values | bytes per unit |
|---|---|---|---|
| `mxfp4` | routed experts | FP4 E2M1, one E8M0 scale per 32 inputs | 348,160 |
| `fp8b32` | shared expert | FP8 E4M3, one E8M0 scale per 32 × 32 block | 656,000 |

A unit is the source codes for its 128 channels followed by their scales, in HF orientation. Concatenating the
units of a projection gives the source tensors `<name>.weight` and `<name>.scale` exactly. Dense FP8 matrices use
exlr8 record format 8 (FP8 with E8M0 per 32 × 32 block); the engram tables use format 3 (MXFP8), split into
`dense/exlr8-split-LLL.engram.embed.pPPP` files of 2^24 rows.

The `native` manifest (`manifests/exlr8-manifest-native.json`) selects these stores for every unit.

## Files

```
config.json  tokenizer.json  tokenizer_config.json  LICENSE  encoding/   from the source repository
dense/exlr8-dense-LLL.safetensors              layer L's attention, router, engram and mHC tensors
dense/exlr8-split-LLL.engram.embed.pPPP...     engram tables of layers 1 and 14
dense/exlr8-vocab.safetensors                  embed, head, final norm
dense/exlr8-extra-vision.safetensors           vision tower and aligner
experts/exlr8-layer-LLL.index.safetensors      unit offsets and checksums
experts/exlr8-layer-LLL.mxfp4.safetensors      routed experts
experts/exlr8-layer-LLL.fp8b32.safetensors     shared expert
manifests/exlr8-manifest-native.json
index-slices/tpN/{mxfp4,fp8b32}/rankR.safetensors
hashes/<path>.sha256                           "<sha256> <bytes>"
```

Layers 0-39 are the main layers (`layers.L.`); layers 40-42 are the MTP stages `mtp.0.` to `mtp.2.`.

The release is 510 GB: 297.6 GB of experts and 212.9 GB of dense files. The two engram tables take 202.8 GB of the
dense files.
The experts split evenly across ranks, about 74 GB per rank at TP = 4.

## Loading

The pure-PyTorch reference loader in [kindlingai/exlr8](https://github.com/kindlingai/exlr8) reads these files:

```bash
python -m loader.exlr8 DIR --manifest native --layer 20 --raw --verify --out layer20.safetensors
```

`--raw` writes the source checkpoint's own tensors (FP4 codes, FP8 values and E8M0 scales); without it the loader
decodes to BF16 or FP32. No serving engine reads this layout yet.

## Checks

Every tensor of every layer, the vocabulary, the vision tower and every engram split part was rebuilt from these
files by the reference loader, with checksums on, and compared with the source checkpoint: names, dtypes, shapes
and bytes all match. Decoded FP32 values of sampled experts and FP8 matrices match a dequantization written from
the source's own definition.

## License

MIT, as the source model. See [LICENSE](LICENSE).
