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
- mxfp4
- vision
---

# DeepSeek-V4.1-Flash DS4 Q4 (MXFP4)

DS4 conversion of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
built from the official checkpoint at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` with the `mxfp4` recipe of the
DS4 V4.1 converter. The GGUF metadata's `general.source.url` reads
`deepseek-ai/DeepSeek-V4.1-Flash` without `-1017` (a converter hardcode), but
its `general.source.revision` is the correct `-1017` revision above.

> **Do not mix checkpoints.** These files come from the October
> `DeepSeek-V4.1-Flash-1017` release. Do not pair the DSpark support GGUF here
> with GGUFs built from the September `deepseek-ai/DeepSeek-V4.1-Flash`
> checkpoint (for example `antirez/deepseek-v4.1-flash-gguf`), or the other way
> round: every weight shard differs between the two.

The routed experts are the **released MXFP4 weights, unchanged**: the
converter repacks them into GGUF MXFP4 blocks without requantization or
calibration, so every expert is bit-exact with the checkpoint. Only the dense
projections are converted (to Q8_0).

The GGUF uses the DS4-specific `deepseek41` architecture and tensor naming. It
loads only in the DS4 engine, not in llama.cpp.

## Quantization recipe

| tensor group | format |
|---|---|
| routed experts gate/up/down (40 layers × 384 experts, 543.6B values) | **MXFP4, the released weights repacked bit-exact** |
| attention projections, shared experts, output head | Q8_0 |
| embedding, hyper-connection mixers, compressor and indexer projections, Engram KV | F16 |
| router, norms, attention sinks, hyper-connection bases/scales | F32 |
| Engram tables (layers 1 and 14, 768M rows) | original FP8 E4M3 rows with their E8M0 scales, packed losslessly at the end of the file and read on demand from disk, never resident |

No importance matrix is involved: the experts are taken as released, and the
remaining tensors are converted directly from the FP8/BF16 source.

## Files

| file | size | role |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part1` | 480,000,000,000 bytes | **main text model**, first part |
| `DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part2` | 21,609,136,128 bytes | **main text model**, second part |
| `DeepSeek-V4.1-Flash-1017-DSpark-support-MXFP4.gguf` | 7,966,294,016 bytes (7.4 GiB) | optional drafter for DSpark speculative decoding |
| `DeepSeek-V4.1-Flash-1017-Vision.gguf` | 970,555,552 bytes (0.9 GiB) | optional vision encoder for image input |

The Hub limits single files to 500 GB, so the 501.6 GB (467.2 GiB) main GGUF is
split in two at byte 480,000,000,000, the same layout `download_model.sh` uses
for the V4.1 Q4 file. Join the parts once after downloading (below). Of the
joined file, about 278 GiB is resident; the remaining ~189 GiB is the
disk-only Engram table.

| file | SHA-256 |
|---|---|
| `DeepSeek-V4.1-Flash-1017-MXFP4.gguf` (joined) | `4d16135d76ef1a71ae3109ab5fc02d3014fa806c2caed7a4cbe84e2f695f4344` |
| `DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part1` | `682ae7e12397da12f9b4d295ba593307bb434dffa5ade5cf80e87129d73a8fac` |
| `DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part2` | `64555a7c27c21c53b81fa0b7ca035ef1a605a382472ac67e24558f7c4b11479c` |
| `DeepSeek-V4.1-Flash-1017-DSpark-support-MXFP4.gguf` | `d3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9` |
| `DeepSeek-V4.1-Flash-1017-Vision.gguf` | `861f1097e8686111fa23dc4a6bf139fd0c4cc114db2f1b4090a267f7683c0dbf` |

All three GGUFs come from the same `-1017` revision. The language GGUF has no
vision tensors by design; image input uses the separate encoder (see
[Vision](#vision-image-input)).

## Running

Requires the DS4 runtime from
[`ivanfioravanti/ds4-metal`](https://github.com/ivanfioravanti/ds4-metal),
branch `main` at commit **`3e92959` or later** (`make`, run from the repo root).
That branch carries the V4.1 Metal decode and prefill work and the two-Mac
tensor-parallel fixes this model was measured with, and pairs the `-1017`
vision encoder with this language GGUF; older builds reject it (see
[Attribution](#attribution-and-license)).

```sh
git clone https://github.com/ivanfioravanti/ds4-metal
cd ds4-metal && make

# Download the model (keep it on a fast internal SSD: Engram rows are read from it at run time).
DS41_DIR="$HOME/models/deepseek-v4.1-flash-ds4-q4"
hf download ivanfioravanti/DeepSeek-V4.1-Flash-DS4-Q4 --local-dir "$DS41_DIR"

# Join the two parts in place (appends part 2 to part 1; no second full copy).
cd "$DS41_DIR"
cat DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part2 >> DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part1
mv DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part1 DeepSeek-V4.1-Flash-1017-MXFP4.gguf
rm DeepSeek-V4.1-Flash-1017-MXFP4.gguf.part2
shasum -a 256 DeepSeek-V4.1-Flash-1017-MXFP4.gguf   # compare with the table above
cd -

# Interactive CLI.
./ds4 -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-MXFP4.gguf" --ctx 65536

# OpenAI-compatible server: 1M-token context, four resident sessions, reachable on the LAN.
./ds4-server -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-MXFP4.gguf" \
  --ctx 1048576 --batched-session 4 --host 0.0.0.0 --port 8000
```

`--nothink` disables thinking. `--think`, `--think-max` and `--think-level N`
(1–100) select the V4.1 reasoning effort. `--temp 0` gives greedy decoding.

### DSpark speculative decoding

DSpark proposes up to five future tokens from the separate support GGUF in this
repository, built from the same `-1017` checkpoint revision. Pass it with:

```sh
./ds4 -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-MXFP4.gguf" \
  --dspark --mtp-model "$DS41_DIR/DeepSeek-V4.1-Flash-1017-DSpark-support-MXFP4.gguf"
```

The same flags work in `ds4-server` for non-batched requests. Session-batched
serving (`--batched-session N`) uses ordinary target decoding instead. Gains
depend on the prompt: code and structured output accept long drafts, while
free-form prose gains little.

### Vision (image input)

The language GGUF has no vision tensors by design. For images, download the
encoder from this repository (`hf download` above already fetches it) and pass
it with `--vision`. ds4 pairs the encoder with the language GGUF by
`general.source.revision`, so both must come from the same `-1017` revision.

> **Do not use antirez's `DeepSeek-V4.1-Flash-Vision.gguf`** with this model:
> it is built from the September checkpoint. Use `DeepSeek-V4.1-Flash-1017-Vision.gguf` from this repository.

```sh
# CLI: then type `/read image.png` and ask about the image.
./ds4 -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-MXFP4.gguf" \
  --vision "$DS41_DIR/DeepSeek-V4.1-Flash-1017-Vision.gguf"

# Server: add --vision to the usual command.
./ds4-server -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-MXFP4.gguf" \
  --vision "$DS41_DIR/DeepSeek-V4.1-Flash-1017-Vision.gguf" \
  --ctx 1048576 --batched-session 4 --host 0.0.0.0 --port 8000
```

`--vision` adds about 0.9 GiB of memory. Text-only requests keep working.
On two Macs, pass the same `--vision` file to both the coordinator and the worker.

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
16 images per request and a 64 MiB HTTP body. Anthropic-style base64 image
sources are accepted too.

### Two Macs: tensor parallelism over Thunderbolt RDMA

Each Mac holds the full file. The pair splits the routed experts 50/50 and
exchanges partial results over a Thunderbolt cable. Set up the link as in
[`docs/DISTRIBUTED.md`](https://github.com/ivanfioravanti/ds4-metal/blob/main/docs/DISTRIBUTED.md):
put an IPv4 address on the cabled member interface of each Mac (not only on
the Thunderbolt bridge), and check `ibv_devinfo` for an active verbs device.

On **both** Macs, before starting (these reset at reboot):

```sh
sudo sysctl iogpu.wired_limit_mb=460000 iogpu.disable_wired_collector=1
```

Then start the worker first; it retries while the coordinator loads:

```sh
# Mac B (worker).
./ds4 --tensor-parallel --role worker --coordinator 10.9.0.1 9911 --transport rdma \
  --ctx 1048576 -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-MXFP4.gguf"

# Mac A (coordinator, here the server).
./ds4-server --tensor-parallel --role coordinator --listen 10.9.0.1 9911 --transport rdma \
  --ctx 1048576 --batched-session 4 --host 0.0.0.0 --port 8000 \
  -m "$DS41_DIR/DeepSeek-V4.1-Flash-1017-MXFP4.gguf"
```

Run the same commit on both Macs, keep the worker in a held terminal session,
and use `--transport tcp` on both sides only if RDMA is unavailable. For image
input, add the same `--vision` file to both commands.

## Requirements

- Apple Silicon with Metal. Developed and measured on Mac Studio M3 Ultra, 512 GB.
- **Single Mac: 512 GB.** The resident model is about **278 GiB**, plus the
  context-dependent KV cache and graph buffers (about 9 GiB at a 128K context),
  plus about 0.9 GiB with `--vision`.
- **Disk: 467 GiB** for the joined GGUF (plus 20 GiB while joining), on a fast local SSD. The ~189 GiB Engram
  table is read on demand at run time and never enters Metal residency.
- Two-Mac tensor parallelism was tested on two 512 GB M3 Ultras connected by
  Thunderbolt, with the sysctls above on both.

## Measured on the reference M3 Ultra

`ds4-bench`, one Mac Studio M3 Ultra 512 GB, ds4-metal `main`:

| context | prefill | decode |
|---:|---:|---:|
| 8K | 760 tok/s | 37.6 tok/s |
| 63K | 854 tok/s | 36.6 tok/s |
| 98K | 830 tok/s | 36.2 tok/s |

Two M3 Ultras, tensor parallel over Thunderbolt RDMA (measured before the
kernelpool/ds4 #1 and #2 speedups were merged):

| context | prefill | decode |
|---:|---:|---:|
| 8K / 16K / 32K | 700 / 682 / 725 tok/s | ~41 tok/s |
| 64K / 128K | 767 / 724 tok/s | ~41 tok/s |

Through `ds4-server --batched-session 4`, the two-Mac pair processed a
1.2K-token prompt at 380 tok/s (181 tok/s on one Mac) and decoded at
40–42 tok/s (34–35 tok/s on one Mac).

## Comparison with antirez/ds4 main

![ds4-metal main vs antirez/ds4 main: prefill and decode on one M3 Ultra](assets/antirez_vs_ds4_metal.png)

Single Mac Studio M3 Ultra 512 GB, this GGUF, `ds4-bench` sweep from 2K to
128K, average of 2 runs. Prefill is new tokens per second while growing the
context to each point (2K→4K→…→128K), so it differs from the one-shot prefill
figures above. Decode is 64 tokens at that context.
[antirez/ds4](https://github.com/antirez/ds4) `main` at `fc80bd6`,
ds4-metal `main` at `24e30c0`.

| context | prefill, antirez → ds4-metal | decode, antirez → ds4-metal |
|---:|---:|---:|
| 2K | 354 → 438 tok/s (1.24×) | 20.4 → 38.3 tok/s (1.88×) |
| 4K | 360 → 427 tok/s (1.19×) | 20.4 → 38.1 tok/s (1.87×) |
| 8K | 353 → 531 tok/s (1.50×) | 20.4 → 37.6 tok/s (1.84×) |
| 16K | 628 → 733 tok/s (1.17×) | 20.2 → 37.2 tok/s (1.84×) |
| 32K | 687 → 792 tok/s (1.15×) | 19.9 → 37.0 tok/s (1.86×) |
| 64K | 697 → 790 tok/s (1.13×) | 19.3 → 36.5 tok/s (1.89×) |
| 128K | 635 → 724 tok/s (1.14×) | 18.2 → 35.9 tok/s (1.98×) |

On average ds4-metal is **1.22× faster at prefill and 1.88× faster at decode**.

## Quality

The routed experts, which hold 98.5% of the non-Engram parameters, are
bit-exact with the release, and the Engram tables keep their original FP8
values and scales. The only conversion loss is in the dense projections,
which are stored as Q8_0, and the F16 copies of BF16 tensors.

Build-to-build exactness was checked when merging the speedups: ds4-metal
`main` produces byte-identical full-vocabulary logits at 8K, 63K and 98K
context, and byte-identical greedy decode logprobs on 8K- and 105K-token
prompts, compared with the build before the merge.

## Attribution and license

Derived from [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(thanks to the DeepSeek team), and run by the DS4 inference engine:

- [antirez/ds4](https://github.com/antirez/ds4): the original DS4 engine by Salvatore Sanfilippo (antirez).
- [kernelpool/ds4](https://github.com/kernelpool/ds4), branch `ds41-perf`: V4.1 Metal performance work, the V4.1 and DSpark converters, and tensor-parallel decode by Tarjei Mandt (kernelpool).
- Mark Shank ([@IngeniousIdiocy](https://github.com/IngeniousIdiocy)): V4.1 decode and prefill speedups, [kernelpool/ds4#1](https://github.com/kernelpool/ds4/pull/1) and [#2](https://github.com/kernelpool/ds4/pull/2).
- nazeshinjite ([@nazeshinjite](https://github.com/nazeshinjite)): pairing V4.1 vision encoders with the language GGUF by source revision, [antirez/ds4#1225](https://github.com/antirez/ds4/pull/1225), which makes the `-1017` encoder work.
- [ivanfioravanti/ds4-metal](https://github.com/ivanfioravanti/ds4-metal): integration, wider V4.1 short-prefill tiles ([kernelpool/ds4#4](https://github.com/kernelpool/ds4/pull/4)), and the two-Mac Thunderbolt RDMA fixes.

This derivative inherits the **MIT License** of the base model. The full text is in [`LICENSE`](LICENSE).
