---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- ds4
- apple-metal
- quantized
- deepseek-v4.1
- mixture-of-experts
- imatrix
- iq2_xxs
- vision
---

# DeepSeek-V4.1-Flash DS4 IQ2 (imatrix)

DS4 conversion of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
built from the official checkpoint at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` (recorded in the GGUF as
`general.source.revision`) with the `q2` recipe of the DS4 V4.1 converter and
an importance matrix collected with DS4 itself.

This is the **smaller sibling** of
[ivanfioravanti/DeepSeek-V4.1-Flash-DS4-Q4](https://huggingface.co/ivanfioravanti/DeepSeek-V4.1-Flash-DS4-Q4).
The Q4 build keeps the released MXFP4 experts bit-exact and needs about
278 GiB resident, so it is a 512 GB Mac model. This IQ2 build requantizes the
routed experts to about 2.25 bits per weight and brings the resident model down to
about **152 GiB**, which makes DeepSeek V4.1 Flash usable on a 256 GB Mac.
**If you have the memory, use the Q4 build**: it carries no expert
quantization loss.

> **Do not mix checkpoints.** These files come from the October
> `DeepSeek-V4.1-Flash-1017` release. Do not pair them with GGUFs built from the
> September `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (for example
> `antirez/deepseek-v4.1-flash-gguf`), or the other way round: every weight
> shard differs between the two.

The GGUF uses the DS4-specific `deepseek41` architecture and tensor naming. It
loads only in the DS4 engine, not in llama.cpp.

## Quantization recipe

| tensor group | format |
|---|---|
| routed experts gate/up (40 layers × 384 experts, 362.4B values) | **IQ2_XXS with imatrix** (2.06 bpw) |
| routed experts down (40 layers × 384 experts, 181.2B values) | **Q2_K with imatrix** (2.63 bpw) |
| attention projections, shared experts, output head | Q8_0 |
| embedding, hyper-connection mixers, compressor and indexer projections, Engram KV | F16 |
| router, norms, attention sinks, hyper-connection bases/scales | F32 |
| Engram tables (layers 1 and 14) | kept raw: the original FP8 E4M3 rows with their E8M0 scales, packed losslessly at the end of the file and read on demand from disk, never resident (same packing as the Q4 build) |

Everything except the routed experts is converted exactly as in the Q4 build.
The routed experts are quantized from the released MXFP4 weights with the
imatrix described below.

| | bytes | GiB |
|---|---:|---:|
| routed experts (IQ2_XXS + Q2_K) | 152,882,380,800 | 142.4 |
| everything else resident (Q8_0 / F16 / F32) | 10,073,273,328 | 9.4 |
| **resident model** | **162,955,654,128** | **151.8** |
| Engram tables, disk-only | 202,758,032,400 | 188.8 |
| **file on disk** | **365,713,686,528** | **340.6** |

## Files

| file | size | role |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-IQ2XXSImatrix-Q2KDown.gguf` | 365,713,686,528 bytes (340.6 GiB) | **main text model**, single file |
| `DeepSeek-V4.1-Flash-1017-DSpark-support-MXFP4.gguf` | 7,966,294,016 bytes (7.4 GiB) | optional drafter for DSpark speculative decoding |
| `DeepSeek-V4.1-Flash-1017-Vision.gguf` | 970,555,552 bytes (0.9 GiB) | optional vision encoder for image input |
| `imatrix/DeepSeek-V4.1-Flash-1017-routed-moe-ds4-full.dat` | 770,707,996 bytes | importance matrix used for the routed experts |
| `imatrix/DeepSeek-V4.1-Flash-1017-routed-moe-ds4-full.dat.counts` | 122,896 bytes | per-expert sample counts for the imatrix |

The main GGUF is under the Hub's 500 GB per-file limit, so unlike the Q4
build it is **not split**: there is nothing to join after downloading. The
DSpark support GGUF and the Vision encoder are the same files published in the
Q4 repository (identical SHA-256); they work with either main model.

| file | SHA-256 |
|---|---|
| `DeepSeek-V4.1-Flash-1017-IQ2XXSImatrix-Q2KDown.gguf` | `bf3e09d6b177d32c962ce17491ff328901fa178955b1a50f8fde57fa4d340c7e` |
| `DeepSeek-V4.1-Flash-1017-DSpark-support-MXFP4.gguf` | `d3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9` |
| `DeepSeek-V4.1-Flash-1017-Vision.gguf` | `861f1097e8686111fa23dc4a6bf139fd0c4cc114db2f1b4090a267f7683c0dbf` |
| `imatrix/DeepSeek-V4.1-Flash-1017-routed-moe-ds4-full.dat` | `71bc7ffbb90c453d4ae7c41caaea41d8793b7919391db75baeb173b8c0559838` |
| `imatrix/DeepSeek-V4.1-Flash-1017-routed-moe-ds4-full.dat.counts` | `23ec3ea68ec78deee59822231fa61fb359d8c28285ceb1dac3f5cd17400349d1` |

`SHA256SUMS` covers every published file except itself, this README and the
Hub's `.gitattributes`: `shasum -a 256 -c SHA256SUMS` from the download directory.

## Running

Requires the DS4 runtime from
[`ivanfioravanti/ds4-metal`](https://github.com/ivanfioravanti/ds4-metal),
branch `main` at commit **`3e92959` or later** (`make`, run from the repo
root). That branch carries the V4.1 Metal decode and prefill work and pairs
the `-1017` vision encoder with this language GGUF; older builds reject it (see
[Attribution](#attribution-and-license)).

Run the binaries from the ds4 checkout directory: the Metal sources are
loaded from the current working directory at startup.

```sh
git clone https://github.com/ivanfioravanti/ds4-metal
cd ds4-metal && make

# Download the model (keep it on a fast internal SSD: Engram rows are read from it at run time).
DS41_DIR="$HOME/models/deepseek-v4.1-flash-ds4-iq2"
hf download ivanfioravanti/DeepSeek-V4.1-Flash-DS4-IQ2 --local-dir "$DS41_DIR"
(cd "$DS41_DIR" && shasum -a 256 -c SHA256SUMS)

# Interactive CLI.
./ds4 -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-IQ2XXSImatrix-Q2KDown.gguf" --ctx 65536

# OpenAI-compatible server: 1M-token context, DSpark and vision, reachable on the LAN.
./ds4-server --ctx 1048576 \
  -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-IQ2XXSImatrix-Q2KDown.gguf" \
  --dspark --mtp-model "$DS41_DIR/DeepSeek-V4.1-Flash-1017-DSpark-support-MXFP4.gguf" \
  --vision "$DS41_DIR/DeepSeek-V4.1-Flash-1017-Vision.gguf" \
  --host 0.0.0.0 --port 8000
```

To fetch only the text model, pass its filename to `hf download`:
`hf download ivanfioravanti/DeepSeek-V4.1-Flash-DS4-IQ2 DeepSeek-V4.1-Flash-1017-IQ2XXSImatrix-Q2KDown.gguf --local-dir "$DS41_DIR"`.

`--nothink` disables thinking. `--think`, `--think-max` and `--think-level N`
(1–100) select the V4.1 reasoning effort. `--temp 0` gives greedy decoding.

### DSpark speculative decoding

DSpark proposes up to five future tokens from the separate support GGUF in this
repository, built from the same `-1017` checkpoint revision. Pass it with:

```sh
./ds4 -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-IQ2XXSImatrix-Q2KDown.gguf" \
  --dspark --mtp-model "$DS41_DIR/DeepSeek-V4.1-Flash-1017-DSpark-support-MXFP4.gguf"
```

The same flags work in `ds4-server` for non-batched requests. Session-batched
serving (`--batched-session N`) uses ordinary target decoding instead. Gains
depend on the prompt: code and structured output accept long drafts, while
free-form prose gains little. The drafter adds about 7.4 GiB to memory.

### Vision (image input)

The language GGUF has no vision tensors by design. For images, pass the
encoder from this repository with `--vision`. ds4 pairs the encoder with the
language GGUF by `general.source.revision`, so both must come from the same
`-1017` revision.

> **Do not use antirez's `DeepSeek-V4.1-Flash-Vision.gguf`** with this model:
> it is built from the September checkpoint. Use `DeepSeek-V4.1-Flash-1017-Vision.gguf` from this repository.

```sh
# CLI: then type `/read image.png` and ask about the image.
./ds4 -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-IQ2XXSImatrix-Q2KDown.gguf" \
  --vision "$DS41_DIR/DeepSeek-V4.1-Flash-1017-Vision.gguf"
```

`--vision` adds about 0.9 GiB of memory. Text-only requests keep working.

Over HTTP, send images as inline data URIs in OpenAI chat `image_url` parts:

```sh
IMG=$(base64 -i image.png)
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d @- <<JSON
{"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "Describe this image."},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,$IMG"}}
]}]}
JSON
```

PNG and JPEG only; remote URLs and server-side file paths are rejected. Up to
16 images per request and a 64 MiB HTTP body. Base64 image blocks of the form
`{"type": "image", "source": {"type": "base64", ...}}` are accepted too.

### Two Macs

The two-Mac tensor-parallel mode over Thunderbolt (each Mac holds the full
file, the routed experts are split 50/50) is documented in the
[Q4 model card](https://huggingface.co/ivanfioravanti/DeepSeek-V4.1-Flash-DS4-Q4#two-macs-tensor-parallelism-over-thunderbolt-rdma)
and in [`docs/DISTRIBUTED.md`](https://github.com/ivanfioravanti/ds4-metal/blob/main/docs/DISTRIBUTED.md).
It has not been measured with this IQ2 file yet. If you have two large-memory
Macs, the Q4 build is usually the better choice.

## Requirements

- Apple Silicon with Metal. Built and measured on a Mac Studio M3 Ultra, 512 GB.
- **Memory.** The resident model is about **152 GiB**. With the full
  1,048,576-token context, ds4 plans about **171 GiB** in total (KV cache
  6.3 GiB + graph buffers 12.7 GiB), plus about 7.4 GiB with the DSpark
  drafter and 0.9 GiB with `--vision`. For comparison, the Q4/MXFP4 build
  needs about 278 GiB resident before any context.
  - **256 GB Mac:** comfortable, including 1M context with DSpark and vision.
    Raise the GPU wired-memory limit if macOS refuses the allocation, for
    example `sudo sysctl iogpu.wired_limit_mb=235000` (resets at reboot).
  - **192 GB Mac:** the model should fit at a reduced context with a raised
    wired limit and little else running, but this is **untested**.
- **Disk: 341 GiB** for the GGUF, on a fast local SSD. The ~189 GiB Engram
  table is read on demand at run time and never enters Metal residency.

## How the imatrix quant was made

**Importance matrix.** Collected with DS4 on the MXFP4 GGUF of the same
revision (the released expert precision), using a layer-major batched
collector that records the routed-expert MoE inputs during prefill:

- **4,690 prompts / 2,961,212 tokens / 710,690,880 expert routes**, at an 8K
  context, in **3 h 02 min** on one M3 Ultra.
- **Calibration set:** the DS4 imatrix corpus tracked in
  [`gguf-tools/imatrix/dataset`](https://github.com/ivanfioravanti/ds4-metal/tree/main/gguf-tools/imatrix/dataset),
  re-rendered into the V4.1 chat layout (system turn, and the
  reasoning-effort prefix at level 75 for thinking prompts) and shuffled
  across categories with a fixed seed, so it is a mixed general / code / math /
  multilingual / chat set. Every prompt appears once in thinking and once in
  non-thinking mode. By prompt count:
  - 46% code: review and explanation of real C, Objective-C, Metal and
    documentation excerpts, plus programming, Bash and algorithm prompts;
  - 24% agentic chat: tool-use conversations with DSML tool schemas, tool calls
    and tool results, including multi-turn transcript replays;
  - 22% language: rewriting, summarization, copy-editing and extraction;
  - 4% multilingual translation between English and Chinese, Dutch, French,
    German, Italian, Polish, Portuguese, Romanian, Spanish and Swedish;
  - 3% reasoning: AIME 2025 math, GPQA and SuperGPQA questions;
  - plus general-knowledge explanations and long-context code synthesis.

  The rendered file has SHA-256
  `b7fc775aa18d6f1e05fd4e1c67bc1f8cb715e4c45d62547a9f5bf0242008a1e7`.
- **Coverage:** 15,339 of 15,360 routed experts (99.86%) received samples. The
  21 experts that were never routed fall back to a deterministic weight-energy
  importance. The `.dat.counts` sidecar records the per-expert sample counts
  for gate/up and down.

The layer-major collector is development work at the time of writing; the
`.dat` file here lets you rebuild the GGUF without collecting again.

**Conversion.** With the DS4 V4.1 converter in `gguf-tools` (its `q2` recipe),
reading the official FP8/MXFP4 checkpoint:

```sh
python gguf-tools/deepseek41_quantize.py \
  --hf /path/to/DeepSeek-V4.1-Flash-1017 \
  --source-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --quant q2 --imatrix imatrix/DeepSeek-V4.1-Flash-1017-routed-moe-ds4-full.dat \
  --out DeepSeek-V4.1-Flash-1017-IQ2XXSImatrix-Q2KDown.gguf
```

**Validation.** `gguf-tools/deepseek41_validate_gguf.py --payload`, against the
same checkpoint, revision and imatrix, passed:

```text
PASS: 1046 tensor layouts; complete 365713686528-byte GGUF
PASS: source payload checks 1046/1046
```

That checks the complete tensor layout and file size, re-encodes every
non-expert tensor and three sampled experts of every routed-expert tensor from
the source (with the same imatrix) and compares them byte for byte with the
file, and samples about 130 Engram rows per table against the source FP8 rows
and scales. It is artifact
validation, not inference quality testing.

## Quality

Evaluations against the MXFP4 (Q4) build, which shares every non-expert
tensor with this file, are in progress: full-vocabulary KL divergence of
the logits, the official V4.1 scoring sets and a hard reasoning suite. Results
will be added here. Until then, expect the usual 2-bit tradeoff: some drift from
the released weights, larger on hard reasoning than on everyday chat and code.

## Speed

Benchmarks on the reference M3 Ultra will follow. With 142 GiB of experts
instead of about 269 GiB, each decoded token reads less memory than with the
Q4 build.

## Attribution and license

Derived from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(thanks to the DeepSeek team), and run by the DS4 inference engine:

- [antirez/ds4](https://github.com/antirez/ds4): the original DS4 engine by Salvatore Sanfilippo (antirez).
- [kernelpool/ds4](https://github.com/kernelpool/ds4), branch `ds41-perf`: V4.1 Metal performance work, the V4.1 and DSpark converters, and tensor-parallel decode by Tarjei Mandt (kernelpool).
- Mark Shank ([@IngeniousIdiocy](https://github.com/IngeniousIdiocy)): V4.1 decode and prefill speedups, [kernelpool/ds4#1](https://github.com/kernelpool/ds4/pull/1) and [#2](https://github.com/kernelpool/ds4/pull/2).
- nazeshinjite ([@nazeshinjite](https://github.com/nazeshinjite)): pairing V4.1 vision encoders with the language GGUF by source revision, [antirez/ds4#1225](https://github.com/antirez/ds4/pull/1225), which makes the `-1017` encoder work.
- [ivanfioravanti/ds4-metal](https://github.com/ivanfioravanti/ds4-metal): integration, the V4.1 imatrix collection and calibration used for this build, and the two-Mac Thunderbolt RDMA fixes.

This derivative inherits the **MIT License** of the base model. The full text is in [`LICENSE`](LICENSE).
