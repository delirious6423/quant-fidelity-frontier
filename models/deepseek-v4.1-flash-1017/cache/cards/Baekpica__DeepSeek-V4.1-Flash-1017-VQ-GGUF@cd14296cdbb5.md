---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- deepseek
- deepseek-v4.1
- gguf
- vector-quantization
- vq
- mixture-of-experts
- engram
- mtp
- speculative-decoding
- vision
- 1m-context
- dgx-spark
- gb10
- ds4
- ds4-dfm-rs
---

# DeepSeek-V4.1-Flash-1017 VQ GGUF

> [!WARNING]
> **Experimental low-bit release.** The routed experts, which hold most of the
> weights, are stored at about 1.6 bits per weight, far below common 4-bit
> GGUF quants. You may see failures the source model rarely shows: reasoning
> that loops or never reaches an answer, text that breaks mid-character (shown
> as `�`) or switches language mid-word, and weaker answers in general. This
> is exploratory quantization research, not a drop-in replacement for the
> source model.
>
> **The files can change.** This line of conversions is still being researched,
> so any file here may be replaced by a new revision. The weighted-Lloyd fitting of
> the codebooks that was being tested here was released separately as
> [iVQ](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF) on
> 2026-10-22; this repository keeps the weights-only build. Replacements so far:
> none since the first release on 2026-10-17. To keep a fixed build, download
> a pinned commit:
> `hf download Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF --revision <commit>`.
>
> **Problems or odd output?** Feel free to post them in the
> [Community tab](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF/discussions),
> even as rough notes. The prompt, sampling settings and runtime commit help
> most.

> [!NOTE]
> **New, 2026-10-22: [DeepSeek-V4.1-Flash-1017-iVQ-GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF).**
> It is this release with the routed-expert codebooks, code assignments and row gains
> refitted to activation statistics from 3.65M calibration tokens, an importance
> matrix for VQ as in llama.cpp's imatrix. The bit layout (13-bit on layers 0–13,
> 12-bit on 14–39), the file sizes and the runtime stay the same, so its files drop
> into a setup that serves this one. Measured against the original FP4/FP8 model on
> 58,803 held-out tokens, the routed experts' output error falls in every layer (by
> 7% for a typical token). End to end the gain is small: KL 0.369 → 0.350
> (−4.9%) with next-token agreement unchanged. Chat, reasoning and tool-call
> text improve most; Chinese and Japanese web text does not. Details are on its card.

**DeepSeek's October 17 update of V4.1 Flash, converted the day it shipped and sized for one 128 GB DGX Spark.**
The 510 GB official checkpoint becomes a ~113.6 GB base GGUF that carries the
model's own three-tower MTP drafter, plus a ~1 GB vision sidecar and the
**1017** Engram tables. Context stays at the native 1,048,576 tokens.

The conversion uses YoungAi's published `vq8sh14-q4k-mtpnative` recipe without
changes: the same bit allocation, codebooks, skeleton and drafter treatment as
[wenzhouwu/YoungAi-DeepSeek-V4.1-Flash](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash),
with the tools pinned to fixed [YoungAi](https://github.com/fodelf/YoungAi)
commits. Only the input changes, to the 1017 weights. The target runtime is
[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs), which already serves the
V4.1 Flash artifact built with this recipe on DGX Spark.

> **Status, 2026-10-17:** the conversion is finished and uploaded. The base has the
> V4.1 reference's exact tensor inventory and byte layout (1,000 tensors,
> 113,556,639,424 bytes) and passes the ds4-dfm-rs DeepSeek V4.1 loader check
> (`bind: slots=1000 bound=1000 required-missing=0`). Serving on DGX Spark has
> not been measured for this artifact yet; the speed and fidelity figures below
> are V4.1 Flash numbers for the same recipe and are labelled as such.

## At a glance

| | |
|---|---|
| Source | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) at `1bae0e1088993b9b82de08f59de5be5a34e42170`; all 48 input shards SHA-256 checked |
| Recipe | YoungAi `vq8sh14-q4k-mtpnative`, unchanged |
| Routed experts | 8-dim VQ, one E4M3 codebook per layer shared by all 384 experts and gate/up/down, F16 row gains. Layers 0–13 use 13-bit indices (K=8192), layers 14–39 use 12-bit (K=4096). No calibration data. |
| Attention, shared experts, embedding, head | q4_K |
| MTP drafter | The model's three DSpark draft towers, embedded in the base GGUF. Tower experts are 12-bit VQ; other tower weights keep their native precision. |
| Vision | DeepSeek-ViT tower, aligner and per-layer `bias_vl` router biases from 1017, original BF16, in a separate sidecar GGUF |
| Engram | The 1017 n-gram tables (2 × 101.5 GB) in `deepseek-engram/`, unmodified, read in place through `--engram-dir` |
| Context | 1,048,576 tokens, from the model metadata. YoungAi measures the full 1M-token KV cache of this architecture at 0.89 GB. |
| Domain sidecar (zchain / GR-RB) | None |
| Runtime | ds4-dfm-rs on DGX Spark (CUDA). YoungAi `ds4-server` reads the same files. |

## Use the 1017 Engram, not the V4.1 one

1017 changed all 48 weight shards, including the two that hold the Engram
n-gram tables. Their names and sizes did not change, so an engine pointed at
the older `deepseek-ai/DeepSeek-V4.1-Flash` tables would load them without
any error and run a mismatched model. YoungAi's V4.1 install steps fetch
exactly those older shards. With this GGUF, use the copies in
`deepseek-engram/`, which are the official 1017 files copied unmodified, or
the same files from your own download of the 1017 checkpoint.

| Shard | DeepSeek-V4.1-Flash (SHA-256) | DeepSeek-V4.1-Flash-1017 (SHA-256) |
|---|---|---|
| `model-00047-of-00048.safetensors` | `824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f` | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` | `976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed` | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |

The vision tower changed too. This release's `DeepSeek-V4.1-Flash.vision.gguf`
has the same 306 tensors and size as the V4.1 sidecar, but 532,623,100 of its
971,066,016 bytes differ, so an older V4.1 sidecar copied into this directory
would also load silently and see images with the old tower.

The same applies to YoungAi's V4.1 domain sidecars. They were solved against
the V4.1 Flash base and do not fit these weights, so do not mount them here.

## Files

The layout and file names are the same as the V4.1 Flash VQ artifact, so a
directory or script set up for that artifact serves this one after only the
directory path is changed. The "1017" is in the repository name, not in the
file names. Do not mix files between the two directories; `SHA256SUMS`
catches it if that happens.

| Path | Contents | Size |
|---|---|---:|
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf.part01-of-40` … `part40-of-40` | Base GGUF, including the MTP drafter, as 40 byte-range parts with the V4.1 reference's sizes (24 × 2,838,915,986 B, then 16 × 2,838,915,985 B) | 113,556,639,424 B |
| `DeepSeek-V4.1-Flash.vision.gguf` | Vision sidecar for `--vision-gguf` | 971,066,016 B |
| `deepseek-engram/model-00047-of-00048.safetensors` | 1017 Engram shard (official file, unmodified) | 101,535,150,936 B |
| `deepseek-engram/model-00048-of-00048.safetensors` | 1017 Engram shard (official file, unmodified) | 101,537,926,640 B |
| `SHA256SUMS` | SHA-256 of the assembled base, every part and every other file | |
| [`RECIPE.md`](RECIPE.md) | Exact commands, flags and pinned tool commits | |
| [`SOURCE.md`](SOURCE.md) | Provenance and licenses | |
| `LICENSE`, `LICENSE-DeepSeek` | YoungAi tools (MIT, with the GGML notice) and DeepSeek weights (MIT) | |
| `docs/img/` | The ds4-dfm-rs DGX Spark throughput graph shown below (V4.1 Flash, same recipe) | |

SHA-256 of the assembled base and the vision sidecar (all parts are in `SHA256SUMS`):

```
9f3afe6a46ff8de711911dfad5994519a4e5dd57b163f88a5bf4111c308d0e6f  DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
3a8b194a011e630e8e49a51b5f4e9821e48a9e5674e678d52c0302d6b164698c  DeepSeek-V4.1-Flash.vision.gguf
```

The download is about 318 GB. Plan for about 114 GB more while the parts are
assembled, or use the append-and-delete loop below. At run time the base is
resident (about 106 GiB loaded for this recipe on V4.1 Flash). The Engram
tables are read in rows from disk, so keep them on a local SSD.

## Quick start on DGX Spark

```sh
# 1. Download and verify (parts, vision sidecar, Engram, checksums)
hf download Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF --local-dir ds41-1017
cd ds41-1017
sha256sum -c --ignore-missing SHA256SUMS

# 2. Assemble the base. This loop needs only one extra part of free disk.
B=DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
for i in $(seq -w 1 40); do cat "$B.part$i-of-40" >> "$B" && rm "$B.part$i-of-40"; done
sha256sum -c --ignore-missing SHA256SUMS      # the assembled .gguf must report OK

# 3. Build ds4-dfm-rs and serve
git clone https://github.com/Baekpica/ds4-dfm-rs.git && (cd ds4-dfm-rs && make cuda-spark)
./ds4-dfm-rs/ds4-server --cuda -m "$B" \
    --engram-dir deepseek-engram \
    --vision-gguf DeepSeek-V4.1-Flash.vision.gguf \
    --mem-budget-mb 110000 --host 0.0.0.0 --port 8000
```

If you already have the official 1017 checkpoint, add
`--exclude "deepseek-engram/*"` to the download and point `--engram-dir` at that
checkpoint's directory. The engine takes only the shard file names from the
GGUF.

The server provides OpenAI-style `/v1/chat/completions`, `/v1/completions`
and `/v1/responses`, and Anthropic-style `/v1/messages`. The embedded drafter
is controlled with `--mtp-mode off|auto|on`. DeepSeek recommends
`temperature` 1.0 and `top_p` 1.0 for 1017; send them explicitly in each
request, because sampling on this path follows the request. See the
[ds4-dfm-rs DeepSeek V4.1 guide](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/deepseek41-flash.md)
for the full flag set and serving limits. Standard llama.cpp, Ollama and
Transformers loaders do not read this format.

## Recipe

All of the following comes from YoungAi's V4.1 recipe and tools. See
[RECIPE.md](RECIPE.md) for the exact commands.

- **Routed experts (40 layers × 384 experts × gate/up/down).** Every 8
  consecutive weights of a row become one index into a codebook, scaled by one
  F16 gain per row (the row RMS). One codebook per layer is shared by all
  384 experts and all three matrices. Codewords are stored as E4M3.
  Layers 0–13 use K=8192 (13-bit indices, stored as a 12-bit stream plus a
  separate 1-bit plane). Layers 14–39 use K=4096 (12-bit). That is 1.625 or
  1.5 index bits per weight. Codebooks come from 8 Lloyd iterations on the
  GPU over the weights alone, with no calibration text.
- **Skeleton.** Every FP8 projection outside the routed experts (attention,
  shared experts, FP8 indexer and compressor projections) and the BF16
  embedding and output head are requantized to q4_K (144 bytes per 256
  weights, no importance matrix).
- **MTP drafter.** The three DSpark draft towers (128 experts each) use the
  same VQ at K=4096, with one shared codebook per tower. The towers' other
  weights are copied from the original checkpoint: FP8 projections with
  their 32×32 scales and BF16 router, Markov and confidence heads stay as
  shipped, and norms are widened losslessly to F32.
- **Unchanged.** Norms, routers and router biases, hyper-connection maps,
  attention sinks, compressor and small projections, and the Engram
  projections keep their shipped precision. The Engram tables are not in
  the GGUF.
- **Vision.** The tower, aligner, image delimiter rows and the 40 per-layer
  `bias_vl` router biases are copied from the 1017 checkpoint into the
  sidecar GGUF: matrices stay BF16, small tensors are widened losslessly to
  F32.

How YoungAi's base is made up, as YoungAi measured it on V4.1 Flash:

| Part | Format | GB |
|---|---|---:|
| Routed-expert indices | VQ-8, 14 layers × 13-bit + 26 layers × 12-bit | 104.90 |
| Row gains | F16 | 0.30 |
| Codebooks | 40 × E4M3 | 0.003 |
| Attention, shared experts, output head | q4_K | 4.96 |
| MTP draft towers | VQ-8, 12-bit | 2.56 |
| Everything else | as shipped | 0.85 |

What this 1017 base contains, read from the file with the ds4-dfm-rs inspector.
The tensor counts and byte totals per type are identical to the V4.1 reference:

| Storage type | Tensors | GB |
|---|---:|---:|
| f32 (GGUF type 0) | 531 | 0.172 |
| q4_k (GGUF type 12) | 330 | 4.413 |
| i32 (GGUF type 26) | 1 | 0.001 |
| i64 (GGUF type 27) | 3 | 0.000 |
| bf16 (GGUF type 30) | 65 | 0.333 |
| vqblob (GGUF type 42) | 43 | 107.753 |
| fp8_32x32 (GGUF type 44) | 27 | 0.880 |

Weight-space fidelity of the routed experts, as reported by YoungAi's quantizer
on the 1017 weights (cosine between original and reconstructed expert weights;
this is not a model-quality measurement):

| Layers | Index bits | Mean cosine | Range |
|---|---:|---:|---|
| 0–13 | 13 | 0.9253 | 0.9250–0.9253 |
| 14–39 | 12 | 0.9107 | 0.9101–0.9109 |

## Reference numbers from V4.1 Flash (not this artifact)

These numbers were measured on the V4.1 Flash artifact built with the same
recipe. They show what the recipe and runtime have delivered so far. 1017
measurements will be added here once they exist.

**Speed, one DGX Spark, ds4-dfm-rs**

![DeepSeek V4.1 Flash VQ prefill and decode on one DGX Spark, 2K–64K](docs/img/v41-flash-vq-spark-2k-64k-2026-10-13.png)

*DeepSeek V4.1 Flash with this recipe and YoungAi's code GR sidecar, not this
1017 artifact. Three fresh 2K–64K full-prefill sweeps on one DGX Spark,
temperature 0, automatic embedded MTP; curves are per-frontier medians and
bands the min–max of the three runs. Graph and record from
[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs/blob/a51ec2c0b4f8d55c5fee0e794977d95ea277481f/docs/benchmarks/ds41-spark-64k-2026-10-13/README.md);
DeepSeek V4.1 Flash was ported to ds4-dfm-rs by [JLFN](https://github.com/JLFN)
([#80](https://github.com/Baekpica/ds4-dfm-rs/pull/80)) using YoungAi's VQ
artifacts.*

| 2K–64K sweep, 2026-10-13 | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Median of the three runs' means | 1,039.65 | 38.17 |
| 2K frontier (2,149 tokens) | 775.7 | 42.6 |
| 16K frontier (16,546 tokens) | 1,120.0 | 41.3 |
| 64K frontier (66,608 tokens) | 969.5 | 39.8 |

A later ds4-dfm-rs build (`ae40aa1d`,
[record](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/benchmarks/ds41-spark-head-2026-10-14/README.md),
same artifact and sidecar) reached 1,222.4 tok/s prefill at 16K and, on short
prompts, 54.10 tok/s decode for code and 65.85 tok/s for a table.

**Fidelity to the full-precision original**, by YoungAi on V4.1 Flash
([technical notes §5.1](https://github.com/fodelf/YoungAi/blob/7fb26cc332d740416792204a29c7d13d2986caf5/docs/TECHNICAL.md)).
"Same top-1" is how often the quantized model's next token matches the
original model's, teacher-forced on held-out text. This is the bare base
without any sidecar, which is how this release ships:

| Held-out text | Same top-1 | Mean KL |
|---|---:|---:|
| Code (30 languages) | 78.61% | 0.421 |
| English (WikiText-2) | 78.52% | 0.694 |
| Law | 72.90% | 0.578 |
| Finance | 71.73% | 0.612 |
| Science | 71.65% | 0.537 |
| Medicine | 69.08% | 0.602 |

Speed should be close to these figures, because decoding is bound by
memory bandwidth and this artifact reads the same bytes per token. Drafter
acceptance depends on the generated text and can differ. Fidelity has to be
measured again on 1017.

## What 1017 changes upstream

DeepSeek's own scores for the original FP8 checkpoints, from the
[1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
These describe the source model, not this quantization:

| Benchmark | V4.1-Flash | V4.1-Flash-1017 |
|---|---:|---:|
| Terminal-Bench 4.0 | 31.2 | **42.3** |
| ProgramBench | 20.3 | **29.1** |
| Apex Bench v1.1 | 64.6 | **72.1** |
| HLE w/o tools | 36.8 | **42.6** |
| Automation-Bench v1.0.6 | 54.8 | **60.5** |
| DeepSWE v1.1 | 74.2 | **77.0** |

## Limits

- This is lossy, roughly 1.6 bits per routed-expert weight. On V4.1 Flash
  the recipe agrees with the original's next token 69–79% of the time
  (table above). It is not a drop-in equal of the FP8 model.
- No 1017-specific quality or GB10 speed measurement has been published yet.
- YoungAi reports that this weight recipe under-weights stop decisions
  (`</think>`, EOS), so long thinking segments can fail to close. Its server
  therefore defaults to non-thinking mode. The same caveat should be assumed
  here until 1017 is measured.
- No domain sidecar is shipped. YoungAi's V4.1 sidecars do not apply to
  1017.
- The custom VQ format needs ds4-dfm-rs or YoungAi's engine.

## Credits and sources

This release runs other people's work on a new checkpoint. The method is
theirs.

- **[YoungAi](https://github.com/fodelf/YoungAi)** by Wenzhou Wu (吴文周,
  [fodelf](https://github.com/fodelf) /
  [wenzhouwu](https://huggingface.co/wenzhouwu)), MIT. Taken from it without
  modification:
  - the `vq8sh14-q4k-mtpnative` recipe: 8-dim VQ with per-layer shared E4M3
    codebooks, 13-bit shallow layers in the 12+1 bit-plane layout, q4_K
    skeleton and VQ draft towers (YoungAi
    [`docs/TECHNICAL.md` §4.1](https://github.com/fodelf/YoungAi/blob/7fb26cc332d740416792204a29c7d13d2986caf5/docs/TECHNICAL.md),
    [`docs/113.md`](https://github.com/fodelf/YoungAi/blob/7fb26cc332d740416792204a29c7d13d2986caf5/docs/113.md));
  - the quantizer `gguf-tools/quantize/v41_quantize` and the GGUF converter
    `v41_to_gguf` at
    [`7fb26cc`](https://github.com/fodelf/YoungAi/tree/7fb26cc332d740416792204a29c7d13d2986caf5),
    the commit ds4-dfm-rs's V4.1 port is pinned to;
  - the vision sidecar mode `v41_to_gguf vision` at
    [`e4ea078`](https://github.com/fodelf/YoungAi/tree/e4ea07889144d85acd33b2490ec4f44032644fb0);
  - the Engram and tokenizer constant generators
    `gguf-tools/scripts/v41_engram_consts.py` and `v41_tokenizer_consts.py`;
  - the DQVL v3 VQ payload format, and the file layout and serving flags of
    the reference artifact
    [wenzhouwu/YoungAi-DeepSeek-V4.1-Flash](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash).

  Its notice is kept in [`LICENSE`](LICENSE).
- **[antirez/ds4](https://github.com/antirez/ds4)** (DwarfStar) by Salvatore
  Sanfilippo, the DeepSeek V4 engine YoungAi started from.
- **[llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)**: the GGUF
  container and the q4_K layout. The GGML copyright notice is part of
  `LICENSE`.
- **[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)**: the runtime. Its
  DeepSeek V4.1 support was ported by [JLFN](https://github.com/JLFN)
  ([#80](https://github.com/Baekpica/ds4-dfm-rs/pull/80), vision in
  [#83](https://github.com/Baekpica/ds4-dfm-rs/pull/83)).
- **[DeepSeek-AI](https://huggingface.co/deepseek-ai)**: the model, its
  Engram tables and the reference inference code, MIT
  ([`LICENSE-DeepSeek`](LICENSE-DeepSeek)).

Done here: running the YoungAi pipeline on the 1017 checkpoint with the 1017
Engram and vision weights, verifying the output and packaging it. No
algorithm or format is changed. [RECIPE.md](RECIPE.md) records the build
environment.

## License

MIT. The weights derive from DeepSeek-V4.1-Flash-1017 (MIT,
[`LICENSE-DeepSeek`](LICENSE-DeepSeek)). The quantization tools and recipe
are YoungAi's (MIT, [`LICENSE`](LICENSE), which includes the GGML notice).
This is the same license file layout as the V4.1 reference artifact.
