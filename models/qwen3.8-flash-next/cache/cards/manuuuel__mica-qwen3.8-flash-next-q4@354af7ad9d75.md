---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
---

# mica-qwen3.8-flash-next-q4

Qwen3.8-Flash-Next at 4 bits, in one GGUF file, converted from the authors'
release by the converter of [Mica](https://github.com/manuuuel/mica), an
inference engine for Apple silicon written in C, which runs it. Its vision
tower is a second file, kept as the release has it ([below](#the-vision-tower)).

| | |
| --- | --- |
| Source | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision [`de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) |
| Converter | `mica-convert` at [`47cf31d65b0c96fbdada889ab69ca8f9a2855634`](https://github.com/manuuuel/mica/commit/47cf31d65b0c96fbdada889ab69ca8f9a2855634) for the checkpoint, at [`abe658701ae65f2ba85f3616bb63d4d8b389fcfa`](https://github.com/manuuuel/mica/commit/abe658701ae65f2ba85f3616bb63d4d8b389fcfa) for the vision tower |
| Licence | the authors', the Qwen Community License 1.0: [LICENSE](LICENSE), as in the release |

## Files

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `mica-qwen3.8-flash-next-q4.gguf` | 177,280,263,200 | `961012a2c2e2e809e6ce55f4b754bbbe1fb041d043a9e2f2c1c21c0da3e9f68e` |
| `mica-qwen3.8-flash-next-q4.importance.gguf` | 323,621,632 | `5e7a53a5496a97bf4b0244d6cda8219b99d3e4502c14ec8693ac8498fe1dee4c` |
| `mica-qwen3.8-flash-next-vision.gguf` | 897,881,920 | `ab0e4fb9f8a15936d809e0cdf94196c4b22ec448edaf42a057aaa72f19c00eac` |

`SHA256SUMS` lists all three, for `shasum -a 256 -c SHA256SUMS`.

The checkpoint is the language model and its prediction head, without the
release's vision tower: 1,256 tensors and 56 metadata keys. The second file
is the importance statistics it was converted with, its other input beside the
release (below). The third is the vision tower.

## Recipe

| Tensors | Format |
| --- | --- |
| experts' gate and up | Q4_K |
| experts' down | MXFP4 |
| dense projections, output head, shared expert, n-gram key/value, head's `eh_proj` | Q8_0 |
| gated-residual pairs and injections | F16 |
| norms, `ssm_a`, `ssm_dt.bias`, conv weights, `ssm_alpha`/`ssm_beta`, router, shared-expert gate | F32 |
| token embedding, n-gram table | BF16, bit for bit |

By count: F32 472, F16 298, Q8_0 337, Q4_K 98, MXFP4 49, BF16 2.

### Calibration by importance

The Q8_0, Q4_K and MXFP4 tensors are calibrated by the importance of their
inputs ([ADR 16](https://github.com/manuuuel/mica/blob/47cf31d65b0c96fbdada889ab69ca8f9a2855634/docs/adr/0016-the-checkpoint-is-calibrated-by-importance.md)).

- **The statistics.** For every product stored in those formats, the mean
  over the calibration texts of each input element squared; for each routed
  expert, over the tokens routed to it, for its input and its hidden state,
  with its token count.
- **How they are used.** Each value goes to its nearest level, and each
  block's scales (Q8_0's step, MXFP4's exponent, Q4_K's scales and minimums)
  are searched for the least error weighted by the statistics. An expert
  reached by few tokens has its statistic drawn towards a prior, with a weight
  of 96 tokens for its input and 512 for its hidden state.
- **The calibration texts.** 252 texts of 512 tokens, 129,024 tokens in all,
  63 of each of four kinds:
  - English prose from WikiText-2's train split;
  - code from CPython's standard library at v3.14.7;
  - chat from No Robots' train split, in the model's chat template;
  - Wikipedia in nine languages, each article as of a pinned revision.

  Every source is pinned in Mica's
  [`tools/sources.json`](https://github.com/manuuuel/mica/blob/47cf31d65b0c96fbdada889ab69ca8f9a2855634/tools/sources.json),
  and so is the set's SHA-256, `e8e2cc124f98afc84569b0c6c3d3e8d22133a631772ef585e607c11b15b57734`.
  No calibration text comes from a document the quality texts below are taken
  from.
- **Who collected them.** Mica, on the GPU (`make importance`), reading the
  release converted without statistics. The file records the calibration
  texts' SHA-256, their token count and the fingerprint of the checkpoint it
  was read on.

A GPU's sums may differ from another's in their last bits, so the statistics
are published here, and they are what the conversion is made again from.

## Quality

Measured against the authors' unquantised weights on Mica's 100 quality
texts: 25 of English prose (WikiText-2's test split), 25 of code (CPython's
standard library at v3.14.7), 25 of chat (No Robots' test split, in the
model's chat template) and 25 of Wikipedia in nine languages. Each text is
512 tokens, 51,100 predicted positions in all.

Mica on the GPU (`make quality`): its surprise on each text's actual next
tokens, and how often its top token is the original's, against the original
model run in the authors' code:

| Text | Nats a token, this file | Nats a token, original | Perplexity change | Same top token |
| --- | ---: | ---: | ---: | ---: |
| English prose | 1.3147 | 1.3062 | +0.85% | 92.67% |
| Code | 0.2041 | 0.2052 | -0.11% | 99.14% |
| Chat | 3.2928 | 3.2977 | -0.48% | 91.49% |
| Other languages | 1.3926 | 1.3903 | +0.23% | 92.62% |
| **All** | **1.5510** | **1.5498** | **+0.12%** | **93.98%** |

The authors' code with this file's weights, both models in single precision
on the processor (`tools/quality/quality.py`). KL is the divergence of this
file's next-token distribution from the original's, in nats, over the whole
248,320-token vocabulary at every position:

| Text | KL mean | KL 99th percentile | Same top token | Perplexity change |
| --- | ---: | ---: | ---: | ---: |
| English prose | 0.0542 | 0.738 | 92.61% | +0.76% |
| Code | 0.0080 | 0.181 | 99.06% | -0.08% |
| Chat | 0.0567 | 0.731 | 91.66% | -0.37% |
| Other languages | 0.0407 | 0.511 | 92.67% | +0.26% |
| **All** | **0.0399** | **0.585** | **94.00%** | **+0.14%** |

## The vision tower

`mica-qwen3.8-flash-next-vision.gguf` is the release's vision tower: 27
blocks of width 1,152 with 16 heads and a feed-forward width of 4,304, patches
of 16 pixels over two frames, a 48 x 48 grid of learned positions, and the
merger of 2 x 2 patches into tokens of the language model's width, 2,560. 333
tensors, 448,931,056 values.

- **Stored as the release stores it.** Every tensor is BF16, bit for bit the
  release's: read back, all 333 are byte for byte those of
  `model-00001-of-00131.safetensors`, the one shard of the release that holds
  them. The patch projection, a convolution in the release whose kernel is
  its stride, is the same bytes as a 1,536 by 1,152 product.
- **Its metadata** holds every number its arithmetic takes: the
  configuration's; the layer norms' epsilon, 1e-6, and the rotary base,
  10,000, which the authors' code fixes; the image's bounds, 65,536 to
  16,777,216 pixels after resizing, and its normalisation, from the release's
  `preprocessor_config.json`; the image, start and end tokens, 248056, 248053
  and 248054, checked against the release's vocabulary; the language model's
  rotary sections for images, 11, 11 and 10, interleaved, which the checkpoint
  does not record; and the release it was made from, in `general.source`.
- **Made** by `mica-convert --vision` at
  [`abe6587`](https://github.com/manuuuel/mica/commit/abe658701ae65f2ba85f3616bb63d4d8b389fcfa)
  from that shard and the release's `config.json`, `tokenizer.json` and
  `preprocessor_config.json` at the revision above, in under a second. The
  same inputs give the same bytes.

It is published ahead of Mica's reading of images, and is loaded only where
`--vision` names it; Mica reads text without it.

## Getting it, and making it again

With Mica, whose `tools/sources.json` pins these files by revision, bytes and
SHA-256, `make venv fetch` downloads the checkpoint and the vision tower into
`checkpoints/` and checks them, with the authors' files and the quality texts.
No token is needed.

It can also be made again from the release. The same release, recipe and
statistics give the same bytes on any machine and at any thread count:

```sh
make
python3 tools/fetch.py authors checkpoints    # the authors' config, tokenizer and weight index, at the pinned revision
python3 tools/fetch.py original checkpoints   # every shard of the release, checked against the Hub's SHA-256: 360 GB
make checkpoint                               # the statistics from here, the conversion, its SHA-256 checked
```

`make checkpoint` fetches the statistics from this repository and checks
them. It then writes `checkpoints/mica-qwen3.8-flash-next-q4.gguf`, with a
report of every tensor's error against the release beside it, and checks
the file's SHA-256 against the pinned one. On an M5 Max this takes about 13
minutes on 18 threads. `mica-convert` writes over no file, so a copy of this
file already at that path is moved away first.

At the converter's commit itself, `47cf31d`, `tools/sources.json` pins
neither file yet. Put `mica-qwen3.8-flash-next-q4.importance.gguf` in
`checkpoints/` by hand first; `make checkpoint` then prints the SHA-256, to
compare with the one above.

The vision tower needs 1.0 GB of the release, not 360:

```sh
make
make vision   # the authors' files, the one shard that holds it, the conversion, its SHA-256 checked
```

`make vision` fetches the authors' files and that shard into `checkpoints/`,
or checks them there, writes `checkpoints/mica-qwen3.8-flash-next-vision.gguf`
in under a second, and checks its SHA-256 against the pinned one. At the
converter's commit, `abe6587`, neither `make vision` nor the pin exists yet:
`python3 tools/fetch.py authors checkpoints` and
`python3 tools/fetch.py tower checkpoints` get them, and
`./mica-convert checkpoints/original checkpoints/mica-qwen3.8-flash-next-vision.gguf --vision --source Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`
prints the SHA-256, to compare with the one above.

## Running it

Mica runs it on Apple silicon with 128 GB of unified memory, under macOS.
Building and running are in its
[README](https://github.com/manuuuel/mica#readme):

```sh
make
./mica checkpoints/mica-qwen3.8-flash-next-q4.gguf                       # a conversation in the terminal
./mica-server checkpoints/mica-qwen3.8-flash-next-q4.gguf --port 8080    # an OpenAI-compatible API
```
