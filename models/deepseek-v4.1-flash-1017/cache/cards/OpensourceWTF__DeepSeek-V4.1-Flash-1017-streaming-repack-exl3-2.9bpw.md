---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: mlx-serve
pipeline_tag: text-generation
tags:
- deepseek_v41
- mlx
- mlx-serve
- mlx-stream
- exl3
- eaq
- streaming
- moe
---

# DeepSeek-V4.1-Flash-1017, streaming repack, EXL3 EAQ v1 2.9 bpw experts

**This is DeepSeek-AI's model.** It is a repack of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, the refreshed V4.1-Flash weights). The weights, the
architecture, the tokenizer and the reference prompt encoder all belong to DeepSeek-AI and are released under their
MIT license. The `LICENSE` file here is theirs, byte for byte. This repack changes only how the weights are stored. It
adds no training and no fine-tuning, and it does not merge or modify any weights except for the quantized routed
experts described below.

The routed experts come from an existing EXL3 quantization:
[Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
(revision `45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`, MIT). It was made with
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) `v1.6.1` by turboderp plus Mia-AiLab's EAQ v1 patch set.
EAQ v1 changes how the weights were chosen, not the format: the tensors are plain EXL3, a trellis-coded quantization
with the mul1 codebook and 128-wide Hadamard input and output transforms (`suh` / `svh`). This repack does not
re-quantize anything. It copies those expert tensors byte for byte into a different layout.

## What changed and why

DeepSeek-V4.1-Flash has 40 MoE layers with 384 routed experts each (6 active per token), plus one shared expert per
layer. Its routed experts are too large to sit in the memory of a 128 GB Mac. This repack lays them out so that
they can be streamed from SSD:

- **Routed experts (40 layers × 384), EXL3 EAQ v1 2.9 bpw, in `experts.bin`.** These are Mia-AiLab's EXL3 tensors
  (mul1 codebook). Layers 0–17 and 23–39 are 3 bits per weight (K = 3); layers 18–22 are 2 bits per weight (K = 2).
  Each (layer, expert) pair becomes one contiguous record at a 4,096-byte-aligned offset: 13,316,096 bytes for a
  K = 3 record and 8,892,416 bytes for a K = 2 record. A record holds nine segments: trellis codes, `svh` and `suh`
  for each of gate, up and down. There are 15,360 records (13,440 at K = 3 and 1,920 at K = 2), 196.0 GB in all.
  The `expert-manifest-v2.json` file gives the codec, each layer's K and segment layout, and every record's offset
  and sha256. When the bank was built, each record was read back and checked: its sha256, its bytes against the EXL3
  source tensors, zero padding, and four decoded tiles per projection. All 15,360 records passed. The checks are in
  `receipts/`.
- **Non-expert weights, unchanged in value.** These are in `model-000NN.safetensors` (49 shards, 18.7 GB,
  indexed by `model.safetensors.index.json`). They are converted to MLX's native float formats as exact repacks:
  - The FP8 projections stored as E4M3 with UE8M0 32×32 block scales become MLX `mxfp8` with group size 32. That
    covers attention `wq_a`/`wq_b`/`wkv`/`wo_a`/`wo_b`, the indexer's `wq_b`, the shared experts and the MTP dense
    projections. The E4M3 codes are kept, and each block scale is spread over its 32 rows. The converter refuses
    any tensor whose dequantized values differ from the source.
  - The three MTP/DSpark layers' routed experts (FP4 E2M1) become MLX `mxfp4` with group size 32, also as an exact
    repack. They stay resident.
  - Everything that is BF16 or F32 in the release is copied verbatim. That includes `embed`, `head`, the norms, the
    router gates and biases, the hyper-connection tensors, attention sinks, the compressor and indexer BF16
    pieces, and the vision tower and aligner. `embed` and `head` are genuine BF16 in the release.
- **Engram n-gram tables (layers 1 and 14), lossless, in `engram/`.** `engram-L1.bin` and `engram-L14.bin` are flat
  arrays of 264-byte rows. Each row holds the release's 256 E4M3 codes and 8 UE8M0 scales, copied verbatim, so one
  n-gram lookup is one offset read. They are readable as `mxfp8` with group size 32. `engram-manifest.json` gives
  the hashing recipe and the row layout. `engram-residents.safetensors` holds each Engram layer's `wkv` (exact
  `mxfp8`) and its `q_weight`/`k_weight`, which are the release's BF16 values widened exactly to F32. The 1017
  refresh changed these tables, so they were rebuilt from the 1017 release's shards 47 and 48.
- **Engram token map.** `engram-token-map.u32` maps each of the 129,280 tokenizer ids to the Engram's compressed
  vocabulary (99,092 entries). It was built from this `tokenizer.json`, and `engram-token-map.u32.json` pins the
  sha256 of the map, the tokenizer and the Engram manifest.
- **`config.json`** is the release's config with one change: `quantization_config` (FP8/FP4) is replaced by the
  MLX `quantization` block that describes the resident codecs. `tokenizer.json`, `tokenizer_config.json`,
  `LICENSE` and `encoding/` are the 1017 release's files, unchanged (the tokenizer is byte-identical to the
  original V4.1-Flash release; `encoding/encoding.py` is the 1017 revision, which changed how tool results are
  ordered and merged). `chat_template.jinja` is Mia-AiLab's Jinja port of the 1017 `encoding.py`, taken from the
  EXL3 repo unchanged. mlx-serve renders DeepSeek-V4.1 prompts with its own built-in template and does not read
  this file.

Compared with the earlier
[DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw](https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw):
the base weights are the 1017 refresh, the expert bank is 8.5 GB smaller, and five layers are at 2 bits. Both expert
manifests ship: `expert-manifest-v2.json` (the pack format, K per layer) and `expert-manifest.json` (the runtime index
mlx-serve's bank opener requires beside it).

## What it is for

This repack is laid out for [mlx-serve](https://github.com/ddalcu/mlx-serve) with the
[mlx-stream](https://github.com/davidtai/mlx-stream) plugin (`deepseek_v41` arch). The trunk loads once. The routed
experts the router picks stream from `experts.bin` into a bounded slot cache, and EXL3 Metal kernels run them.
Engram rows are read from SSD by offset. A Mac with 128 GB of memory can therefore serve the model even though it
cannot hold all the experts.

**Runtime support, as of 2026-10-20:** mlx-stream main implements K = 3 only and refuses this bank at first boot. The
K = 2 kernels for layers 18–22 are on the plugin's `dsv41-exl3-k2` branch: every kernel passes its device self-check
there, and this pack loads, serves and generates on that build. The branch is not yet merged, and the throughput and
HumanEval numbers are still to be measured. Other loaders, such as Transformers, vLLM or
ExLlamaV3, do not read this layout. For those, use the original release or the EXL3 quantization linked above.

## How to run (on the `dsv41-exl3-k2` build)

Build mlx-serve with the mlx-stream plugin, following the plugin's README. Download this repo to a local directory,
then start:

```sh
MLX_SERVE_GPU_CEILING_MB=114688 mlx-serve --serve --model /path/to/DeepSeek-V4.1-Flash-1017-streaming-repack-exl3-2.9bpw \
  --port 8080 --ctx-size 17408 --wired-margin-gib 2
```

`MLX_SERVE_GPU_CEILING_MB=114688` is 112 GiB, which leaves macOS 16 GiB of a 128 GiB Mac. The plugin admits its expert
rows against that ceiling. `--ctx-size` sets the largest prompt the server bills for at load. The download is about
418 GB.

## Measured

One served smoke on 2026-10-20 (a 128 GiB Mac, the `dsv41-exl3-k2` build, defaults, cold file cache): a short greedy
request twice gave identical text, and a 1,664-token prompt ran at 231 prefill tok/s and 30.3 decode tok/s with 3.1
draft tokens accepted per round. These are cold single requests, not the ladder; the prompt-size ladder, the 16K
headline and HumanEval are still to run. Mia-AiLab's model card for the EXL3 source reports teacher-forced next-token fidelity of the 2.9 bpw
EAQ v1 quantization against the 1017 release (KL 0.0137 nats, top-1 agreement 96.6% on their short-context set); that
measures the expert quantization this repack copies, not this layout or any server.

## Quality

The 2.9 bpw EXL3 routed experts are lossy compared with the release's FP4 experts. They are the only lossy part of
this repack. Everything else matches the release value for value: the resident projections, MTP experts and Engram
tables are exact repacks, and the BF16/F32 tensors are verbatim.

## Files

| Path | Size | Contents |
| :--- | ---: | :--- |
| `experts.bin` | 196.0 GB | Routed experts: 15,360 EXL3 records (40 layers × 384); K = 3 records of 13,316,096 bytes, K = 2 records (layers 18–22) of 8,892,416 bytes |
| `expert-manifest-v2.json` | 5.9 MB | Expert codec, per-layer K and segment layout, per-record offset and sha256 |
| `expert-manifest.json` | 19.5 MB | Runtime index of the same records (absolute offsets, per-record sha256), which mlx-serve requires beside the v2 file |
| `model-00001…00049.safetensors` | 18.7 GB | Non-expert weights (`mxfp8` / `mxfp4` exact repacks, BF16/F32 verbatim) |
| `model.safetensors.index.json` | 0.25 MB | Tensor-to-shard index |
| `engram/engram-L1.bin`, `engram/engram-L14.bin` | 101.4 GB each | Engram n-gram tables, 264-byte rows (E4M3 codes and UE8M0 scales, verbatim) |
| `engram/engram-residents.safetensors` | 325 MB | Engram `wkv` (`mxfp8`), `q_weight`, `k_weight` |
| `engram/engram-manifest.json` | 12 KB | Engram hashing recipe and row layout |
| `engram-token-map.u32`, `.json` | 0.5 MB | Tokenizer id to Engram compressed-vocabulary map, with its pinned hashes |
| `receipts/` | 25 MB | Bank build checks for each EXL3 source shard |
| `config.json` | 122 KB | The release config with the MLX `quantization` block |
| `tokenizer.json`, `tokenizer_config.json`, `encoding/` | 6.4 MB | The 1017 release's tokenizer and reference encoder, unchanged |
| `chat_template.jinja` | 34 KB | Mia-AiLab's Jinja port of the 1017 reference encoder, unchanged |
| `LICENSE` | 1 KB | DeepSeek's MIT license, unchanged |

## Credits

- Model: [DeepSeek-AI](https://huggingface.co/deepseek-ai), DeepSeek-V4.1-Flash-1017, MIT.
- EXL3 EAQ v1 expert quantization: [Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw), MIT.
- EXL3 format and quantizer: [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp, MIT.
- Server: [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu. Plugin: [mlx-stream](https://github.com/davidtai/mlx-stream).

Repack by David Tai.
