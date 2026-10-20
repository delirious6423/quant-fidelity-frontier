---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- exl3
- exllamav3
- deepseek
- deepseek_v41
- moe
- quantized
- dgx-spark
- tensorfold
---

# DeepSeek-V4.1-Flash-1017 — EXL3, sensitivity-allocated routed experts (2.7 bpw)

[LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) with one change: the routed experts of all 40 trunk layers (384 experts × `w1`/`w2`/`w3`) are re-quantized with a bitrate chosen per expert from measured sensitivity, using [sudoingX's V4.1 recipe](https://github.com/sudoingX/dgx-spark-deepseek-v41). Every other tensor — attention, shared experts, head, the DSpark/MTP drafter, the vision tower — is LibertAIDAI's, byte for byte.

Served on two DGX Sparks with [TensorFold](https://github.com/ashhart/TensorFold) ([bertholomus's DeepSeek-V4.1 TP2 branch](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2)):

- **Room for a 4.5M-token KV cache on two Sparks.** The pack is 11 GiB smaller, and under the same settings each Spark keeps about 3 GiB more memory free than with the 2.9 bpw pack. We spend it on a 4.5M-token KV cache pool shared by 8 concurrent streams (with the engine patches included here), and the nodes still keep 3.7 / 5.3 GiB free at the lowest. ([Memory](#memory))
- **Closer to the official model.** Mean KL divergence 0.0248 (from 0.0324), top-1 agreement 95.64% (from 94.66%) on a held-out panel. ([Quality](#quality))
- **Same speed, same accuracy.** Decode speed matches the 2.9 bpw pack, and MMLU / GSM8K accuracy matches the DeepSeek API. ([Speed](#speed))

## Pack

- **Size:** 198.8 GB (185.2 GiB) in 363 safetensors files. Engram tables are not included (see below).
- **Routed experts:** 6,237 experts at K2, 7,529 at K3, 1,559 at K4, 35 at K5 — mean 2.70 bpw (allocator's usage-weighted target 2.90). LibertAIDAI: K3 on every layer except K2 on layers 18–22.
- **All quantized trunk weights:** 2.73 bpw (LibertAIDAI: 2.9). Head 6 bpw, drafter as in LibertAIDAI.
- **Engine:** tested only with TensorFold's DeepSeek-V4.1 branch ([bertholomus/TensorFold `deepseek-v41-tp2`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2), a fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold)), which takes each tensor's bit width from its shape. The [Mia/LibertAIDAI serving kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) takes one routed-expert K per layer from `exl3_k_map.json`, so it is not expected to load this pack; not tested.

## Memory

Free memory (`MemAvailable`) on the same two Sparks with the same engine settings (`--context 1048576 --parallel 4`), median over ~90 minutes of four coding agents working against the server on 2026-10-18:

| | Spark 1 (head) | Spark 2 |
|---|---:|---:|
| **This repo, 2.7 bpw** | **9.6 GiB** | **11.9 GiB** |
| [LibertAIDAI 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | 6.9 GiB | 8.5 GiB |

TensorFold's DeepSeek KV cache takes about 1.45 GiB per node per million tokens, so the extra ~3 GiB per node is roughly 2M more tokens of cache.

Our two Sparks run this pack with `--parallel 8` and a 4.5M-token pool (`TF_DS_POOL=4718592`) shared by all streams: 5.6 / 7.7 GiB free at start-up, 3.7 / 5.3 GiB at the lowest over its first two hours of use. Stock TensorFold sizes the shared pool to one request's `--context` (at most 1M tokens, whatever `--parallel` is); a larger pool takes the engine patches in this repo ([`tensorfold-patches/`](https://huggingface.co/object-oriented-architects/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold/tree/main/tensorfold-patches), `TF_DS_POOL`), which also keep long chats' cached prompts alive when several share the pool and let one chat carry dozens of images.

## Quality

### KL divergence against the official model

Panel: 199 conversations (301,828 tokens, 239,716 scored positions) built with sudoingX's `caldata` tooling from prompts answered by DeepSeek-V4.1-Flash-1017 itself. Calibration and panel prompts are disjoint. Reference logits come from the official checkpoint (FP8 / MXFP4) in the recipe's reference forward (`p4_kld.py --mode ref`); both packs were scored by the same script on the same positions, on a rented GPU node.

| | mean KLD | KLD p99 | top-1 agreement | perplexity |
|---|---:|---:|---:|---:|
| Official weights (FP8 / MXFP4) | — | — | — | 1.5614 |
| **This repo, 2.7 bpw** | **0.0248** | **0.375** | **95.64%** | **1.5825** |
| [LibertAIDAI 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | 0.0324 | 0.433 | 94.66% | 1.5918 |

By domain (two thirds of the scored positions are code):

| Domain | positions | KLD, this repo | KLD, LibertAIDAI | top-1, this repo | top-1, LibertAIDAI |
|---|---:|---:|---:|---:|---:|
| code | 161,765 | **0.0143** | 0.0214 | **96.33%** | 95.30% |
| agentic | 16,206 | **0.0265** | 0.0395 | **95.11%** | 93.79% |
| long | 15,730 | **0.0323** | 0.0403 | **95.38%** | 94.47% |
| chat | 13,593 | **0.0724** | 0.0882 | **91.58%** | 90.82% |
| knowledge | 11,658 | **0.0563** | 0.0663 | **93.74%** | 92.65% |
| math | 9,643 | **0.0187** | 0.0259 | **95.34%** | 94.64% |
| multilingual | 8,419 | 0.0783 | 0.0669 | 94.24% | 94.39% |
| writing | 2,702 | **0.0799** | 0.0942 | **92.67%** | 91.41% |

### MMLU / GSM8K

[lucataco's recipe](https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark) eval: the same seeded subset (digest `09c2e6cc43e275af`, 200 MMLU + 100 GSM8K), greedy, thinking off and on at effort `high`. The DeepSeek API and LibertAIDAI columns are the recipe's published runs; all three were compared with the recipe's `eval/compare.py`.

| Test | This repo, 2.7 bpw (2× Spark) | DeepSeek API | [LibertAIDAI 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (2× Spark) |
|---|---:|---:|---:|
| MMLU, thinking off | 85.0% | 84.5% | 85.0% |
| MMLU, thinking on | 92.5% | 93.0% | 92.5% |
| GSM8K, thinking off | 97.0% | 96.0% | 96.0% |
| GSM8K, thinking on | 97.0% | 97.0% | 98.0% |
| Same answer as DeepSeek API | 95.0–97.5% | — | 96–98% |

Accuracy matches the DeepSeek API within noise (paired exact McNemar, p = 1.00 in every cell). A test of this size measures accuracy, not closeness to the official model; for that, see the KL divergence above.

## Speed

Same two DGX Sparks, same engine and settings ([bertholomus/TensorFold `8e0966e2`](https://github.com/bertholomus/TensorFold/tree/8e0966e2b09250abe2f6d9cc51edc20630e6eb1a), lucataco's recipe: `--context 1048576 --parallel 4 --mtp-drafts 5`), measured back to back with the engine's `kit_bench.py` (384-token replies, median of 3):

| | This repo, 2.7 bpw | [LibertAIDAI 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) |
|---|---:|---:|
| 1 stream, code | 100.7 tok/s | 95.7 tok/s |
| 1 stream, prose | 59.4 tok/s | 58.3 tok/s |
| 1 stream, structured | 144.6 tok/s | 141.1 tok/s |
| 4 streams, aggregate (2 runs) | 113.8 / 114.0 tok/s | 104.9 / 116.6 tok/s |

The lower bitrate costs no speed. (Single-stream decode moves ±3–7% between engine restarts on these machines.)

## How it was made

lucataco's [JOURNEY](https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark/blob/main/docs/JOURNEY.md) planned a 2.70 bpw per-expert allocation for 1017 with this same recipe (85.75 GiB of experts per node, as here) and parked it once a ready-made pack appeared. This pack carries that approach through.

sudoingX recipe at [`cc59c36`](https://github.com/sudoingX/dgx-spark-deepseek-v41/tree/cc59c3609e8e671418f39eb92aa16b0e4f770b77), on the official checkpoint at revision `1bae0e10`.

1. **Calibration data:** the recipe's `caldata` prompts (base and instruct sets), answered by DeepSeek-V4.1-Flash-1017 through OpenRouter (DeepInfra), plus exllamav3 standard text and random rows: 782,244 tokens.
2. **Pass 0:** Hessians and per-expert sensitivity on two DGX Sparks, merged with domain weights `agentic=1,chat=1,code=1,instruct=1.5,knowledge=1,long=1,math=1,multilingual=1,random=0.1,std=0.5,writing=1`.
3. **Allocation:** `p4_allocate.py --bpw 2.70 --decode-bpw 2.90 --importance tshare --protect-share 0.2 --floor-layers 37,38,39:3 --eps 2:0.026,3:0.0042,4:0.00058,5:0.000096` (the `eps` values were measured on V4.1 and reused).
4. **Pass 1:** routed experts quantized with exllamav3 trellis encoding (`mul1` codebook) on a rented node with 8× RTX PRO 6000.
5. **Assembly:** the routed-expert tensors (`layers.N.ffn.experts.E.w{1,2,3}.{trellis,suh,svh,mul1}`, same names as LibertAIDAI's) replace LibertAIDAI's. Shards without routed experts are copied unchanged; shards with them are rewritten without those tensors (byte copies, no re-encoding). Before writing, every expert was checked for a complete w1/w2/w3 set and every replaced key for matching `suh`/`svh` shapes and dtypes.

## Engram tables (required, not in this repo)

As with LibertAIDAI's pack, the two Engram lookup tables are read unquantized from the **1017** base checkpoint — not the older V4.1-Flash, whose tables differ:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --include model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json config.json \
  --local-dir ./DeepSeek-V4.1-Flash-1017-engram
```

[LibertAIDAI's model card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) lists the expected sha256 of both shards.

## Running

Use [lucataco's recipe](https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark) with the pack repo switched to this one. This pack has 363 `model-*.safetensors` files instead of 43, so the recipe's file-count check needs the new number:

```bash
# scripts/common.sh
PACK_REPO=object-oriented-architects/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold
PACK_REV=cdb167c4ca319fb1f74d0a76848d534e33b00081
# .env
MODEL_DIR=$HOME/Models/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold
# file count: download-weights.sh stops on anything but 43, preflight.sh warns
sed -i 's/ = 43 \]/ = 363 ]/' scripts/download-weights.sh
sed -i 's/\[ "\$v" = 43 \]/[ "$v" = 363 ]/' scripts/preflight.sh
```

`scripts/download-weights.sh` then puts this pack and the 1017 Engram shards on both nodes at the same path, and `scripts/install-chat-template.sh` adds the recipe's chat template to `MODEL_DIR`.

That serves the pack with the recipe's 1M-token pool. For the 4.5M-token pool and 8 streams, build the patched engine image and use the settings in [`tensorfold-patches/README.md`](https://huggingface.co/object-oriented-architects/DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw-TensorFold/blob/main/tensorfold-patches/README.md):

```bash
docker build -t tensorfold-dsv41:8e0966e-ooa $MODEL_DIR/tensorfold-patches/
```

- `model-000NN-of-00043.safetensors`: LibertAIDAI's shards, minus the routed experts
- `model-exl3-routed-LXX-pN.safetensors`: the routed experts, 8 files per layer
- `model.safetensors.index.json`: rebuilt for the above
- `quantization_config.json`: LibertAIDAI's, with the routed experts' entries rewritten from this pack's shard headers and `bits` 2.73
- `exl3_k_map.json`: LibertAIDAI's, with `routed` now a list of 384 per-expert K values per layer (list index = expert id)
- `config.json`: LibertAIDAI's, with `quantization_config.avg_bits` set to this pack's 2.73
- `tokenizer.json`, `tokenizer_config.json`, `encoding/`, `LICENSE`: as in LibertAIDAI's repo, from the base model
- `tensorfold-patches/`: the engine patches, a Dockerfile that applies them to the recipe's image, and their notes (Apache-2.0, as TensorFold)

## Credits

- [DeepSeek](https://huggingface.co/deepseek-ai) for DeepSeek-V4.1-Flash-1017 (MIT).
- [LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) for the base EXL3 pack (MIT): every tensor here except the routed experts.
- [Mia-AiLab](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) for the tensor layout that pack follows.
- [sudoingX](https://github.com/sudoingX/dgx-spark-deepseek-v41) for the sensitivity-allocation recipe (Apache-2.0).
- [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT): trellis quantizer and kernels.
- [TensorFold](https://github.com/ashhart/TensorFold) by ashhart (Apache-2.0), and bertholomus's DeepSeek-V4.1 TP2 port of it ([`deepseek-v41-tp2`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2)), and [lucataco's two-Spark recipe](https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark) for serving.

## License

MIT, same as the base model.
