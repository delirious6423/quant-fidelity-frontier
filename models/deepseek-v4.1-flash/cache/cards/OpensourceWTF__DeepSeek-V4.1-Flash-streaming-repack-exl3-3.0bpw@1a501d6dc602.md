---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: mlx-serve
pipeline_tag: text-generation
tags:
- deepseek_v41
- mlx
- mlx-serve
- mlx-stream
- exl3
- streaming
- moe
---

# DeepSeek-V4.1-Flash, streaming repack, EXL3 3.0 bpw experts

**This is DeepSeek-AI's model.** It is a repack of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(revision `dba1be0a40aa45a94ad051997016db3960a90277`). The weights, the architecture, the tokenizer and the
reference prompt encoder all belong to DeepSeek-AI and are released under their MIT license. The `LICENSE` file
here is theirs, byte for byte. This repack changes only how the weights are stored. It adds no training and no
fine-tuning, and it does not merge or modify any weights except for the quantized routed experts described below.

The routed experts come from an existing EXL3 quantization:
[Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw)
(revision `c5534b90602b4090980a1c3ded1eb3b4d99a38d5`, MIT). It was made with
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) `v1.4.2` by turboderp. EXL3 is turboderp's format: a
trellis-coded quantization with the mul1 codebook and 128-wide Hadamard input and output transforms (`suh` / `svh`).
This repack does not re-quantize anything. It copies those expert tensors byte for byte into a different layout.

## What changed and why

DeepSeek-V4.1-Flash has 40 MoE layers with 384 routed experts each (6 active per token), plus one shared expert per
layer. Its routed experts are too large to sit in the memory of a 128 GB Mac. This repack lays them out so that
they can be streamed from SSD:

- **Routed experts (40 layers × 384), EXL3 3.0 bpw, in `experts.bin`.** These are Mia-AiLab's EXL3 tensors
  (K = 3 for every layer, mul1 codebook). Each (layer, expert) pair becomes one contiguous record of 13,316,096
  bytes at a 4,096-byte-aligned offset. A record holds nine segments: trellis codes, `svh` and `suh` for each of
  gate, up and down. There are 15,360 records, 204.5 GB in all. The `expert-manifest-v2.json` file gives the codec,
  each layer's segment layout, and every record's offset and sha256. The `expert-manifest.json` file is the older
  index of the same records. When the bank was built, each record was read back and checked: its sha256, its bytes
  against the EXL3 source tensors, zero padding, and four decoded tiles per projection. All 15,360 records passed.
  The checks are in `receipts/`.
- **Non-expert weights, unchanged in value.** These are in `model-000NN.safetensors` (49 shards, 18.7 GB, indexed
  by `model.safetensors.index.json`). They are converted to MLX's native float formats as exact repacks:
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
  `mxfp8`) and its `q_weight`/`k_weight`, which are the release's BF16 values widened exactly to F32.
- **Engram token map.** `engram-token-map.u32` maps each of the 129,280 tokenizer ids to the Engram's compressed
  vocabulary (99,092 entries). It was built from this `tokenizer.json`, and `engram-token-map.u32.json` pins the
  sha256 of the map, the tokenizer and the Engram manifest.
- **`config.json`** is the release's config with one change: `quantization_config` (FP8/FP4) is replaced by the
  MLX `quantization` block that describes the resident codecs. `tokenizer.json`, `tokenizer_config.json`,
  `LICENSE` and `encoding/` are the release's files, unchanged. `chat_template.jinja` is a Jinja version of the
  release's `encoding/encoding.py` text path.

## What it is for

This repack is served by [mlx-serve](https://github.com/ddalcu/mlx-serve) with the
[mlx-stream](https://github.com/davidtai/mlx-stream) plugin (`deepseek_v41` arch). The plugin is proposed for
mlx-serve in [ddalcu/mlx-serve#749](https://github.com/ddalcu/mlx-serve/pull/749). The trunk loads once. The routed
experts the router picks stream from `experts.bin` into a bounded slot cache, and EXL3 Metal kernels run them.
Engram rows are read from SSD by offset. A Mac with 128 GB of memory can therefore serve the model even though it
cannot hold all the experts. Other loaders, such as Transformers, vLLM or ExLlamaV3, do not read this layout. For
those, use the original release or the EXL3 quantization linked above.

## How to run

Build mlx-serve with the mlx-stream plugin, following the plugin's README and PR #749. Download this repo to a
local directory, then start:

```sh
mlx-serve --serve --model /path/to/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw --port 8080 \
  --memory-ceiling-gb 120.259 --wired-margin-gib 2
```

`--memory-ceiling-gb 120.259` is 112 GiB, which leaves macOS 16 GiB of a 128 GiB Mac. The plugin admits its expert
rows against that ceiling. `--ctx-size` sets the largest prompt the server bills for at load. The mlx-serve app
offers prompts of 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K and 512K tokens, each plus a 1,088-token reply
allowance, and the model's limit of 1,048,576 tokens. A larger context leaves fewer expert rows resident. Without `--ctx-size`, the
server bills for 16K prompts. The download is about 426 GB.

## Measured

Machine: Apple M5 Max, 128 GiB, served through mlx-serve with the flags above. Build: mlx-stream plugin 4f42e79;
ladder measured on mlx-serve host 72b36513 (server sha256 c8cbf14e); the shipping host 3296dc0e (upstream main
84c6ee64 merged) reproduces the 16K headline token for token.

Definitions: prefill tok/s = prompt tokens ÷ prefill time; TTFT = seconds from request to first output token;
decode tok/s = generated tokens ÷ decode time; prompt rows and decode rows = the expert slot rows per layer the server
admits for the prompt pass and for decode; peak GB = the server process's peak memory, decimal GB.

16K-token prompt: 661.5 prefill tok/s, 24.77 s TTFT, 36.8 decode tok/s (135 prompt rows, 166 decode rows, peak 107.0 GB).

Prompt-size ladder, 1K to 1M, one cold request per size on a fresh server (max_tokens 1,024, temperature 0),
measured in one window on this build (memory bill derived per term; the attention selection's indices carried, and the
score chain billed at what the chunked selection holds, so long prompts admit more expert rows):

| Prompt tokens | Prefill tok/s | TTFT (s) | Decode tok/s | Prompt rows | Decode rows | Peak GB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 81.3 | 12.6 | 39.9 | 148 | 167 | 107.2 |
| 2,048 | 138.3 | 14.8 | 39.7 | 142 | 166 | 106.8 |
| 4,096 | 264.4 | 15.5 | 38.3 | 137 | 166 | 106.8 |
| 8,192 | 461.5 | 17.8 | 38.6 | 137 | 166 | 106.8 |
| 16,384 | 667.5 | 24.6 | 37.9 | 135 | 166 | 107.0 |
| 32,768 | 618.5 | 53.0 | 37.9 | 135 | 166 | 107.1 |
| 65,536 | 537.5 | 121.9 | 35.5 | 135 | 165 | 106.9 |
| 131,072 | 501.2 | 261.5 | 32.5 | 130 | 164 | 106.8 |
| 262,144 | 456.3 | 574.5 | 27.1 | 115 | 161 | 106.4 |
| 524,288 | 379.5 | 1,381.5 | 18.2 | 91 | 155 | 106.3 |
| 1,047,488 | 262.6 | 3,988.9 | 11.1 | 82 | 143 | 103.7 |

Fewer rows fit beside a larger KV, and decode slows with depth as each cycle attends over more positions. The 16K line
above is the ladder's own cell; the 16K headline cell (661.5 / 24.77 s / 36.8 at the top of this section) was measured
in its own run. The context limit is the model's 1,048,576 tokens.

### HumanEval

Served through mlx-serve with the flags above, one greedy sample per problem (temperature 0), scored offline by
running each problem's tests on the last fenced code block the model committed to.

| Quantity | Value |
| :--- | ---: |
| pass@1 | **0.9634** (158 of 164) |
| Problems failed | HumanEval/32, 83, 132, 140, 145, 163 |
| Completions truncated / errored | 0 / 0 (all 164 finished with a stop) |
| Completion tokens | 23,939 total, mean 146, max 4,073 |
| Decode tok/s | 31.8 aggregate; per request median 30.0, p10 20.2, p90 36.0 |
| Time to first token | median 7.65 s, p90 9.08 s (prompts of about 200 tokens) |
| Speculative acceptance | 4.89 tokens accepted per draft attempt |
| Wall time | 2,077 s for 164 requests; peak process memory 108.3 GB |

The same 158 of 164, with the same six failures, was scored on the previous build of the server, so the score is
stable across builds. The six failures are the model's at this quantization, not harness errors.

## Quality

The 3.0 bpw EXL3 routed experts are lossy compared with the release's FP4 experts. They are the only lossy part of
this repack. Everything else matches the release value for value: the resident projections, MTP experts and Engram
tables are exact repacks, and the BF16/F32 tensors are verbatim. The served decode uses DSpark speculative decoding
with typical acceptance and rounding-class kernels, so its output is close to the BF16 reference but not
bit-identical.

## Files

| Path | Size | Contents |
| :--- | ---: | :--- |
| `experts.bin` | 204.5 GB | Routed experts: 15,360 EXL3 3.0 bpw records (40 layers × 384), 13,316,096 bytes each |
| `expert-manifest-v2.json` | 5.9 MB | Expert codec, per-layer segment layout, per-record offset and sha256 |
| `expert-manifest.json` | 46.5 MB | Older index of the same records, plus resident tensor offsets and shard hashes |
| `model-00001…00049.safetensors` | 18.7 GB | Non-expert weights (`mxfp8` / `mxfp4` exact repacks, BF16/F32 verbatim) |
| `model.safetensors.index.json` | 0.25 MB | Tensor-to-shard index |
| `engram/engram-L1.bin`, `engram/engram-L14.bin` | 101.4 GB each | Engram n-gram tables, 264-byte rows (E4M3 codes and UE8M0 scales, verbatim) |
| `engram/engram-residents.safetensors` | 325 MB | Engram `wkv` (`mxfp8`), `q_weight`, `k_weight` |
| `engram/engram-manifest.json` | 12 KB | Engram hashing recipe and row layout |
| `engram-token-map.u32`, `.json` | 0.5 MB | Tokenizer id to Engram compressed-vocabulary map, with its pinned hashes |
| `receipts/` | 25 MB | Bank build checks for each EXL3 source shard |
| `config.json` | 122 KB | The release config with the MLX `quantization` block |
| `tokenizer.json`, `tokenizer_config.json`, `encoding/` | 6.4 MB | The release's tokenizer and reference encoder, unchanged |
| `chat_template.jinja` | 11 KB | Chat template that reproduces the reference encoder's text path |
| `LICENSE` | 1 KB | DeepSeek's MIT license, unchanged |

## Credits

- Model: [DeepSeek-AI](https://huggingface.co/deepseek-ai), DeepSeek-V4.1-Flash, MIT.
- EXL3 expert quantization: [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw), MIT.
- EXL3 format and quantizer: [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp, MIT.
- Server: [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu. Plugin: [mlx-stream](https://github.com/davidtai/mlx-stream).

Repack by David Tai.
