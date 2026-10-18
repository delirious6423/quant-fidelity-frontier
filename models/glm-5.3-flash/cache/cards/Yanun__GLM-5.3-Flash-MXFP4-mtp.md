---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
license: mit
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- quantized
- mxfp4
- moe
- mtp
- custom-format
---

# GLM-5.3-Flash-MXFP4-mtp

## Model architecture

A 320B-class mixture-of-experts multimodal model with about 18B active parameters per token. This conversion retains the text backbone, vision encoder, and one multi-token prediction (MTP) layer in a custom packed binary format.

| Component | Structure |
|---|---|
| Text backbone | 45 layers; hidden size 4,096; the first three layers use dense feed-forward networks of size 12,288 |
| Attention layout | 34 KDA linear-attention layers and 11 sparse-attention layers, with sparse attention every fourth layer |
| Sparse attention | 64 heads; query/key and value head dimensions 256; compressed KV dimension 512; no rotary query/key dimensions |
| IndexPool | Groups four index keys into one pool; configured selection budget of 2,048 tokens, plus the incomplete tail |
| Routed experts | 42 main-model layers, each with 288 experts; eight selected per token; expert intermediate size 2,048; one shared expert per layer |
| Residual connections | Four-stream manifold-constrained hyper-connections (mHC), with 20 Sinkhorn iterations |
| Vocabulary | 154,880 tokens |
| Configured context limit | 1,048,576 tokens; this conversion has not been validated at that length |
| Vision encoder | 24 layers; hidden size 1,024; 16 attention heads; 14 × 14 spatial patches |
| Vision-to-text connection | Vision features are projected to the text hidden size of 4,096 |
| MTP | One additional prediction layer; its weights and routed experts are included |

## Weight precision

This checkpoint uses mixed precision rather than uniform 4-bit weights:

- All **12,384 routed experts**, including the MTP layer's experts, are converted from source FP8 to **MXFP4 E2M1**, with **32 values per group** and one **E8M0 scale byte** per group.
- Each expert contains gate, down, and up projections. Packed values occupy four bits each; scales add an effective 0.25 bits per weight, for **4.25 bits per routed-expert weight** before file metadata.
- Source FP8 weights use **FP32 scales for 128 × 128 blocks**. Conversion applies these scales in FP32, then quantizes directly to MXFP4 without an intermediate BF16 rounding step.
- The group exponent is `floor(log2(max_abs)) - 2`, clamped to the supported scale range. Values use nearest-even E2M1 rounding. This is a lossy conversion.
- Non-routed weights, including shared experts, dense feed-forward networks, attention projections, embeddings, normalization, vision weights, and MTP non-routed weights, retain their source **FP8, BF16, or FP32** representations as applicable.

The payload contains **182,191,831,203 bytes** (about **182.19 GB / 169.68 GiB**) across 46 binary files and eight source companion files. Repository documentation and conversion metadata are additional.

## Format and validation

**This is a custom binary artifact, not an MLX, GGUF, or directly loadable Transformers checkpoint. A compatible loader and inference implementation are required.** The original files under `checkpoint/` describe the source model, not a ready-to-load set of safetensors in this repository. In particular, its original shard index refers to source shards that are not included here.

| Path | Contents |
|---|---|
| `common.bin` | Main-model non-routed weights |
| `experts/layer_03.bin` through `experts/layer_44.bin` | Main-model routed experts |
| `mtp/common.bin`, `mtp/experts/layer_45.bin` | MTP weights |
| `vision/common.bin` | Vision encoder weights |
| `checkpoint/` | Original configuration, tokenizer, chat template, processor configuration, shard index, and license |
| `repack-plan.json` | Tensor shapes, source ranges, destination offsets, and conversion operations |
| `conversion.json` | Quantization settings, payload sizes, and SHA-256 digests |
| `source-info.json` | Pinned source file metadata and digests |

Each routed expert occupies **13,369,344 bytes**. Expert IDs are stored consecutively within each layer file. Within each expert, gate, down, and up projections each store packed row-major weights followed by their group scales; the low nibble represents the earlier column. Exact offsets are recorded in `repack-plan.json`.

All 62 source weight shards passed SHA-256 verification before conversion. All 54 converted payload and companion files passed read-back SHA-256 verification. Nine sampled projections from early, middle, and late layers also matched separately checked conversion results byte-for-byte.

**Full-model inference, task quality, generation speed, long-context behavior, image inference, and MTP decoding have not been validated for this artifact.** Included weights do not imply working runtime support. `qualifiedForInference` remains `false` in the conversion metadata. No benchmark scores from the source model are claimed for this conversion.

## Source and license

Source: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

Source revision: `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`.

Original model by Z.ai. This conversion retains the upstream [MIT license](LICENSE). The original license is also preserved under `checkpoint/LICENSE`.
