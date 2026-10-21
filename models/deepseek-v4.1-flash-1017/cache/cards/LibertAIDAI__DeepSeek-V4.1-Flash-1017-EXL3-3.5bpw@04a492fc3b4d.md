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
---

![DeepSeek-V4.1-Flash-1017 EXL3 3.5 bpw: 240 GiB, perplexity +0.8% code / +0.1% prose vs the base model](assets/card.png)

# DeepSeek-V4.1-Flash-1017 · EXL3 3.5 bpw

**TL;DR** — EXL3 quantization of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at **3.53 bpw** (239.9 GiB). On held-out text its perplexity is **+0.8% (code) / +0.1% (prose)** above the unquantized model. It uses the same tensor layout as [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) and the 2.9 bpw sibling, with higher bitrates. Serving it with the [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) kit needs the `exl3_k_map.json` shipped here. The two Engram tables are **not** in this repo: take them from the 1017 base repo, because they changed in this release.

> **Does not fit 2× DGX Spark.** At TP=2 this build needs 121.5 GiB of weights per rank, more than a DGX Spark has available (~117 GiB). On two Sparks use the [2.9 bpw build](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (~98 GiB per rank).

> **Chat template:** this repo ships `chat_template.jinja`, a Jinja port of DeepSeek's 1017 `encoding/encoding.py` (from [jayleaton](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw)'s MIT template via a community PR by dpolist, amended for the 1017 encoder). It renders byte-identically to the reference on our 8,455-case test corpus. Runtimes that require a template file at startup need it (verified to render identically under Python Jinja and TensorFold's Zig engine); vLLM-based kits use their built-in DeepSeek encoder.

> **Agentic use / reasoning effort:** with tools in the conversation, each turn's reasoning stays in the prompt and is re-read by every later turn, so reasoning cost compounds. On our benchmark of the original 1017 weights (3,738 jobs: agentic coding, tau-bench, LiveCodeBench, AIME, chat), `reasoning_effort` **high** (75, the default) matched **max** (100) on quality at 0.55–0.8× the cost, and **low** (50) matched too at 0.4–0.6× (results in [discussion #4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/discussions/4)). Avoid max for agent loops unless you need the headroom.



| | |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT) |
| Format | EXL3 trellis, `mul1` codebook, output channel scales `always` |
| Bitrate | 3.53 bpw over the quantized trunk weights, head 6 bpw |
| Size | 257.6 GB (239.9 GiB), 44 safetensors shards |
| Perplexity vs unquantized | code 1.8895 vs 1.8749, prose 6.4341 vs 6.4277 |
| Engram tables | not included; shards 47 + 48 of the base repo |
| Vision tower | included, BF16 |
| MTP / DSpark drafter | included |
| Other releases | [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) · [LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4) |

## Tensor groups

| Group | Tensors | Stored as | Notes |
|---|---|---|---|
| Routed experts | `layers.N.ffn.experts.E.w1/w2/w3`, 384 per layer | EXL3 4 bpw on layers 0-9, 30-39; 3 bpw on 10-29 | one K per layer for all 1,152 matrices (the kit's MoE loader needs that) |
| Shared expert | `layers.N.ffn.shared_experts.w1/w2/w3` | EXL3 6 bpw |  |
| Attention | `wq_a`, `wq_b`, `wkv`, `wo_a.slice.0..7`, `wo_b` | EXL3 6 bpw | `wo_a` stored as 8 per-group slices |
| KV compressor | `attn.compressor.wkv` / `wgate` (layers 2, 8, 14, 20) | EXL3 6 bpw | `compressor.norm` BF16 |
| Indexer | `attn.indexer.wq_b` (layers 2, 8, 14, 20, 24, 28, 32, 36); `indexer.wk` (2, 8, 14, 20) | EXL3 6 bpw; `wk` 8 bpw | `weights_proj` FP16, `k_norm` BF16 |
| Engram projection | `layers.1/14.engram.wkv` | EXL3 6 bpw | `q_weight` / `k_weight` FP32 |
| Engram n-gram tables | `layers.1/14.engram.embed` | not in this repo | read unquantized from the base checkpoint, see below |
| Output head | `head` | EXL3 6 bpw |  |
| MTP / DSpark drafter | all `mtp.0-2` matrices incl. routed experts and `main_proj` | EXL3 6 bpw | kit `mtp_bits` 6 (via this repo's K map) |
| Token embedding | `embed` | BF16 |  |
| Router | `ffn.gate.weight` / `gate.bias` / `gate.bias_vl` | FP16 / FP16 / FP32 | `gate.bias_vl` = routing bias for image tokens |
| mHC streams, attention sinks | `hc_attn_*`, `hc_ffn_*`, `attn.attn_sink` | FP32 |  |
| Norms | all RMSNorm weights | BF16 |  |
| DSpark heads | `mtp.2.markov_head.*`, `mtp.2.confidence_head.proj` | BF16 / FP16 |  |
| Vision tower + aligner | `vision.*`, `aligner.*`, `image_start/end/newline` | BF16 | copied from the base checkpoint unchanged |

Every EXL3 matrix is stored as `trellis` (int16) + `suh` / `svh` (fp16) + `mul1` (int32 codebook marker).

## Perplexity

Two held-out sets of unpublished text, 32 sequences x 2048 tokens each, tokens 2-2048 of every sequence scored (65,504 tokens per set). Every row uses the same full-sequence forward (see *How it was made*).

| Model | Size | code | prose |
|---|---|---|---|
| Unquantized (base FP8 / MXFP4 weights, dequantized) | — | 1.8749 | 6.4277 |
| [EXL3 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | 196.3 GiB | 1.9094 | 6.4520 |
| **EXL3 3.5 bpw (this repo)** | 239.9 GiB | 1.8895 | 6.4341 |
| This repo with Engram switched off (sanity check) | | 4.1548 | 8.9702 |

## Engram tables (required)

DeepSeek-V4.1 adds two Engram n-gram lookup tables (`layers.1.engram.embed`, `layers.14.engram.embed`: FP8 with E8M0 block scales, ~95 GiB each). They are used unquantized and come from the base checkpoint, like in the existing V4.1 EXL3 serving stacks. **The 1017 tables are new:** the shards from the older `deepseek-ai/DeepSeek-V4.1-Flash` must not be reused.

| File from deepseek-ai/DeepSeek-V4.1-Flash-1017 | Bytes | sha256 |
|---|---|---|
| `model-00047-of-00048.safetensors` | 101,535,150,936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` | 101,537,926,640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |
| `model.safetensors.index.json`, `config.json` | | needed for `engram_layer_ids` / `engram_num_embeddings` |

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --include model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json config.json \
  --local-dir ./engram-src-1017
sha256sum engram-src-1017/model-0004[78]-of-00048.safetensors
```

## Serving

The checkpoint follows the layout of [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw), so it loads with the vLLM-based [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) kit. That kit targets 2x DGX Spark at TP=2, where this 3.5 bpw build does **not** fit (121.5 GiB of weights per rank); use the 2.9 bpw build there. The settings below come from the kit's `start.sh` / `download.sh` (commit `6f7d159`). Upstream exllamav3 has no DeepseekV41 model class yet, so it cannot load this checkpoint directly.

**Fresh install** (on the head node, from the kit checkout):

```bash
git clone https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
cd DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
cp .env.example .env
cat >> .env <<'EOF'
HF_MODEL_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw
MODEL_HOST=$PWD/model-1017-3.5bpw
EXPECTED_SHARDS=44
HF_ENGRAM_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
ENGRAM_DIR=./engram-src-1017
KMAP_HOST=$PWD/model-1017-3.5bpw/exl3_k_map.json
EOF
./download.sh   # run first: start.sh checks KMAP_HOST before it fetches the weights
./start.sh
```

`.env` is sourced by bash, so `$PWD` expands to the kit directory when the scripts are run from it.

`KMAP_HOST` is required here: the per-layer expert bitrates differ from the kit's bundled `files/exl3_k_map.json`, and the kit sizes each MoE layer's packed tensors from that map.

**Switching an existing V4.1 install** — the kit reuses anything already on disk, so make sure nothing old gets picked up:

| What | Why | Do |
|---|---|---|
| `MODEL_HOST` | weights are only downloaded when fewer than `EXPECTED_SHARDS` shards are present, so an old 39-shard tree is served as-is | point it at a new directory (as above) |
| `ENGRAM_DIR` | the Engram fetch only fills missing files, so old shards 47 / 48 would stay | new directory (as above); check the sha256 |
| Packed Engram (`./start.sh pack`, `~/dsv41-engram` on both nodes) | used automatically whenever `engram-l1-r*of2.bin` exists | delete it or set `HEAD_PACKED_DIR` / `WORKER_PACKED_DIR` to new paths, then re-run `./start.sh pack` |
| `WEIGHT_SYNC=rsync` worker copies | skipped when the `.dsv41-exl3-synced` / `.dsv41-engram-synced` markers exist | new `WORKER_MODEL_DIR` / `WORKER_ENGRAM_DIR` |

## How it was made

| Step | Details |
|---|---|
| Quantizer | [exllamav3](https://github.com/turboderp-org/exllamav3) dev branch (`ba2feb2`, 1.6.2), its own `convert.py` pipeline: per-module Hessian capture, LDLQ + trellis encoding |
| Architecture | a DeepseekV41 model definition written for this release against DeepSeek's reference `inference/model.py` / `inference/engram.py`: lagged-pre mHC residual streams, compress ratios 0 / 2 / 1 with cross-layer KV / indexer / candidate sharing, Engram lookups, sqrt-softplus routing, the reference FP8 / FP4 activation quantization inside attention |
| Forward check | the same forward on the unquantized weights gives the perplexity in the table above (code 1.8749 / prose 6.4277) |
| Calibration | exllamav3 default mix (C4, code, multilingual, technical, Wikipedia, TinyStories, random-token rows), 250 rows x 2048 tokens |
| MoE capture | every routed expert sees every calibration token (exllamav3's default for MoE models) |
| MTP / DSpark | quantized without calibration data |
| Bitrate plan | explicit per-tensor recipe, integer bitrates only, one K per routed-expert layer |
| Validation | held-out perplexity, Engram on / off, chat smoke test with the base repo's `encoding/encoding.py` (below) |

<details><summary>Smoke test outputs (greedy decoding)</summary>

**chat**: What is 17*19? Reply with the integer only.

```
323<｜end▁of▁sentence｜>
```

**chat**: Explain in three sentences why the sky is blue.

```
Sunlight contains all colors, but air molecules scatter short blue wavelengths far more than long red ones—a process called Rayleigh scattering. This scattered blue light reaches your eyes from every direction, making the sky appear blue. At sunset, light travels through more atmosphere, so the blue is scattered away and reds and oranges dominate.<｜end▁of▁sentence｜>
```

**thinking**: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?

```
We need answer classic. Need be careful. Let ball x, bat x+1, total 2x+1=1.10 => x=.05. Answer 5 cents. Need concise.</think>The ball costs **$0.05 (5 cents)**.

The bat costs $1.05, so together they total $1.10.<｜end▁of▁sentence｜>
```

</details>

## Files

| File | Content |
|---|---|
| `model-000NN-of-00044.safetensors`, `model.safetensors.index.json` | weights |
| `config.json` | base config plus `quantization_config` (same keys as the earlier V4.1 EXL3 release) and `engram_table_dir: null` |
| `exl3_k_map.json` | per-tensor K in the kit's `files/exl3_k_map.json` schema, generated from these shards |
| `quantization_config.json` | per-tensor storage map written by exllamav3 |
| `tokenizer.json`, `tokenizer_config.json`, `LICENSE`, `encoding/` | copied from the base repo (`encoding/` is DeepSeek's prompt-format reference) |

## Credits

- DeepSeek for [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).
- turboderp for [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT): quantizer, kernels and conversion pipeline.
- Mia-AiLab for [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) and the [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) kit, whose tensor layout this release follows (the per-tensor bitrates differ, so the kit needs this repo's `exl3_k_map.json`).

## License

MIT, same as the base model.
