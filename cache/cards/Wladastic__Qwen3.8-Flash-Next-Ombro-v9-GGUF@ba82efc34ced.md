---
base_model: Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - qwen
  - qwen3.8
  - moe
  - quantized
  - ombro
---

# Qwen3.8 Flash Next — Ombro V9 GGUF

Ombro V9 is an experimental, measurement-guided mixed-precision GGUF
quantization of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Where the earlier Ombro Q1 release traded precision for size and speed, V9 is
quality-first: every routed-expert precision choice had to hold a strict local
error floor before any bytes were saved.

**Orientation:** V9 is built around Unsloth's UD quants of this model. It takes
[UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (111.3 GB)
as its quality floor and beats it: no routed-expert layer is less accurate than
UD-Q4_K_XL on any of the local error measures below, and the extra bytes go where
the measurements show they matter. At 114.5 GB it stays far below UD-Q5_K_XL
(158.3 GB).

Ombro is a community quantization recipe, not an official GGUF quant type. This
repository is not affiliated with Qwen, Unsloth, Bartowski, or llama.cpp.

## Download

Download all nine files. Point llama.cpp at the first shard; the remaining
shards are discovered automatically.

| Variant | Total size | Effective BPW | Shards |
| --- | ---: | ---: | ---: |
| Ombro V9 | 114,493,489,696 bytes (114.49 GB / 106.63 GiB) | 5.18 | 9 |

Checksums are in [`SHA256SUMS`](SHA256SUMS).

## Quantization approach

The official BF16 Safetensors checkpoint was converted to a source-faithful
BF16 GGUF and verified tensor-by-tensor before quantization.

V9 then chooses the storage type of the routed-expert `gate`, `up`, and `down`
projections independently in each of the 48 layers. For every layer, up to 108
gate/up/down type combinations were measured against BF16 expert outputs on
recorded real activations and router decisions, across two separate calibration
partitions, using 13 error measures (expert and routed NMSE, tail and worst-token
NMSE, output direction and output norm, and worst-expert NMSE).

A combination was only eligible if **each of those 13 measures, individually and
on both partitions,** was no worse than the corresponding layer's
Unsloth UD-Q4_K_XL type combination measured the same way. No average was allowed
to hide a single regression. (One layer, for which the UD combination was not
measured, used an earlier Ombro reference as its floor instead.) Among eligible combinations, an exact byte-budget
optimizer distributed the remaining precision.

Routed experts per layer (`gate` / `up` / `down`), compared with Unsloth's
UD-Q4_K_XL and UD-IQ3_XXS. Only layers that differ from the common pattern are
listed:

| Layer | UD-Q4_K_XL | UD-IQ3_XXS | Ombro V9 |
| --- | --- | --- | --- |
| **all other 39 layers** | Q4_K / Q4_K / Q5_1 | IQ2_S / IQ2_S / IQ4_NL | Q4_K / Q4_K / Q5_1 |
| 2 | Q5_K / Q5_K / Q8_0 | IQ3_S / IQ3_S / IQ4_NL | **Q6_K / Q6_K** / Q8_0 |
| 4 | Q4_K / Q4_K / Q8_0 | IQ2_S / IQ2_S / IQ4_NL | **Q6_K / Q6_K** / Q8_0 |
| 17 | Q4_K / Q4_K / Q5_1 | IQ2_S / IQ2_S / IQ4_NL | **Q6_K / Q6_K** / Q5_1 |
| 30 | Q4_K / Q4_K / Q8_0 | IQ2_S / IQ2_S / IQ4_NL | **Q6_K / Q6_K** / Q8_0 |
| 33 | Q4_K / Q4_K / Q5_1 | IQ2_S / IQ2_S / IQ4_NL | IQ4_XS / **Q6_K** / Q5_1 |
| 35 | Q4_K / Q4_K / Q5_1 | IQ2_S / IQ2_S / IQ4_NL | **Q6_K / Q6_K / Q8_0** |
| 38 | Q4_K / Q4_K / Q5_1 | IQ2_S / IQ2_S / IQ4_NL | Q4_K / **Q6_K** / Q5_1 |
| 46 | Q4_K / Q4_K / Q8_0 | IQ2_S / IQ2_S / IQ4_NL | Q4_K / **Q5_K** / Q8_0 |
| 47 | Q4_K / Q4_K / Q8_0 | IQ2_S / IQ2_S / IQ4_NL | **Q6_K / Q6_K** / Q8_0 |

| Part | UD-Q4_K_XL | UD-IQ3_XXS | Ombro V9 |
| --- | --- | --- | --- |
| Attention / SSM projections | Q8_0 | Q6_K | Q8_0 |
| Per-layer token embedding (PLE) | IQ4_NL | IQ4_NL | IQ4_NL |
| Token embedding / output | Q8_0 | Q6_K | Q8_0 |
| Shared experts | Q8_0 | Q6_K / Q8_0 | Q8_0 |
| **Routed experts** | **77.0 GB** | **48.6 GB** | **80.2 GB** |
| **Total** | **111.3 GB** | **82.0 GB** | **114.5 GB** |

### Experimental follow-ups (not released)

Built from the same BF16 source and the same importance matrix, to test single
changes against V9. Layers not listed are as in V9 (`Q4_K / Q4_K / Q5_1`).

| Layer | V10 | V10-b | V11 | V12 |
| --- | --- | --- | --- | --- |
| **other 39 layers** | as V9 | as V9 | as V9 | **IQ3_XXS / IQ3_XXS / Q8_0** |
| 2, 4, 30, 35, 47 | as V9 | as V9 | as V9 | as V9 |
| 17 | as V9 | Q6_K / Q6_K / **Q8_0** | as V9 | Q6_K / Q6_K / **Q8_0** |
| 33 | as V9 | IQ4_XS / Q6_K / **Q8_0** | as V9 | IQ4_XS / Q6_K / **Q8_0** |
| 38 | as V9 | Q4_K / Q6_K / **Q8_0** | as V9 | Q4_K / Q6_K / **Q8_0** |
| 46 | as V9 | as V9 | as V9 | as V9 |

| Part | V10 | V10-b | V11 | V12 |
| --- | --- | --- | --- | --- |
| Attention / SSM projections | Q8_0 | Q8_0 | **BF16** | **BF16** |
| Per-layer token embedding (PLE) | **Q5_1** | **Q5_1** | **Q5_1** | IQ4_NL |
| **Routed experts** | 80.2 GB | 81.0 GB | 80.2 GB | 79.4 GB |
| **Total** | **124.1 GB** | **124.9 GB** | **126.6 GB** | **116.3 GB** |

The `down` projections have 640-wide rows, which is not divisible by the
256-element blocks of K-quants, so only 32-element block types are possible
there. V9 uses `Q5_1` as the floor everywhere, with `Q8_0` on six layers.
There is no `Q5_0` in this model.

The per-layer token embedding table (28.8 GB, a quarter of the model) is
`IQ4_NL`; its 160-wide rows admit no smaller block format. Small control tensors,
for example `ssm_alpha`, `ssm_beta`, the hyper-connection inject weights and the
layer-1 PLE convolution, are kept at `F32`. Token embeddings, output weights, and
unlisted always-active tensors use `Q8_0` where their source type permits it.
The complete per-tensor override map is included as
[`quantization-recipe.txt`](quantization-recipe.txt).

The importance matrix combines the public Qwen3.8-Flash-Next matrices from
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) and
[Bartowski](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF) by raw
accumulator summation. The merged matrix contains 926 entries and represents
2,234 calibration chunks of 512 tokens. Its SHA-256 is
`7e8c06b000b3a4146997a5db81d651709aee12694b256be0832507460af287ff`.

The quantization used llama.cpp commit
`bbdd9f246e9667f9aeb7ad11cca269466f981184`. The final split contains 1,224
unique tensors.

## What the local measurements do and do not say

The selection rule guarantees a *local* property: per layer, the chosen routed
expert types reproduce BF16 expert outputs at least as well as the
UD-Q4_K_XL type combination, on every one of the 13 measures, on the recorded
activations. It does not by itself guarantee end-to-end behaviour, because
errors propagate between layers and the comparison uses the UD *type
combination* measured locally rather than the published UD file.

## Performance

Throughput figures will be added after a controlled sweep. A first local run
was not measured under comparable conditions and is intentionally not
reported here.

## Running with llama.cpp

Use a recent CUDA-enabled llama.cpp build with Qwen3.8-Flash-Next support. A
starting point for a two-GPU system is:

```bash
llama-server \
  --model Qwen3.8-Flash-Next-Ombro-v9-00001-of-00009.gguf \
  --jinja \
  --ctx-size 131072 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k bf16 \
  --cache-type-v bf16 \
  --load-mode mmap \
  --fit on \
  --batch-size 2176 \
  --ubatch-size 768 \
  --cache-reuse 256 \
  --no-context-shift \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --repeat-penalty 1.0
```

Reduce batch sizes if your available VRAM is lower. For this model's chat
template, valid reasoning efforts are `low`, `medium`, and `xhigh`; unsupported
labels such as `high` are rejected by llama-server.

`--cache-reuse` helps agentic and tool-calling workloads, where tool output
inserted between turns makes the prompt prefix diverge. In image conversations,
the cache is invalidated from the first image onward regardless.

## Vision and MTP

These shards contain the main language model only. The base model's vision
encoder is available separately as the Unsloth
[`mmproj-F16.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
which has been used with these shards locally (`--mmproj`).

The base model has an MTP architecture, but these shards contain no
`nextn`/MTP tensors. Speculative decoding needs a compatible external MTP draft
model and a llama.cpp build that supports MTP for this architecture.

## Validation status

- Official BF16 source conversion verified across all 1,224 tensors.
- Final size matches the pre-computed byte plan to within 304 bytes.
- Every entry of the published override map was confirmed in the quantizer log;
  no quantizer fallback occurred.
- Loads and generates successfully with llama.cpp, including image input with
  the Unsloth mmproj.
- No formal perplexity, KL-divergence, long-context accuracy, coding, agentic,
  or broad source-equivalence evaluation has been completed yet.

## License and attribution

The original model is released under the Qwen Community License 1.0. The
included `LICENSE` is copied from the official model repository. Review the
original [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), intended
use, limitations, and license terms before using or redistributing this model.
