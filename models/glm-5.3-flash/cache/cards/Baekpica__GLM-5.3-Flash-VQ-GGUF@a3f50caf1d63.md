---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags:
  - gguf
  - vector-quantization
  - residual-vq
  - glm5-next
  - q4_k
  - dgx-spark
  - experimental
pipeline_tag: image-text-to-text
---
# GLM-5.3-Flash VQ GGUF (R13+8 experts, Q4_K nonexpert v3)

> [!WARNING]
> **Experimental low-bit release.** The routed experts, which hold most of the
> weights, are stored at 1.625 or 2.625 bits per weight depending on the layer
> (about 2.1 on average), far below common 4-bit GGUF quants. You may see
> failures the source model rarely shows: reasoning that loops or never
> reaches an answer, text that breaks mid-character (shown as `�`) or switches
> language mid-word, and weaker answers in general. This is exploratory
> quantization research, not a drop-in replacement for the source model.
>
> **The files can change.** This line of conversions is still being researched
> (as of 2026-10-21, weighted Lloyd fitting of the codebooks is being tested),
> so any file here may be replaced by a new revision. Replacements so far:
> three since the first release on 2026-10-15. The non-expert weights were
> requantized (Q4_K v3, 2026-10-16), then the 8-bit residual stage that
> protects sensitive routed layers was added to layers 3–16 and widened to 19
> layers (2026-10-18). The earlier files remain at the tags `vq-original`,
> `q4k-v3` and `r13p8-l3-16`. To keep a fixed build, download a pinned commit:
> `hf download Baekpica/GLM-5.3-Flash-VQ-GGUF --revision <commit>`.
>
> **Problems or odd output?** Feel free to post them in the
> [Community tab](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF/discussions),
> even as rough notes. The prompt, sampling settings and runtime commit help
> most.

> **Status, 2026-10-18:** the files are the 19-layer R13+8 revision: the
> residual stage now covers routed layers 3–17, 22, 27, 41 and 44 (before:
> 3–16). Loading needs a ds4-dfm-rs build with GVQ1 container v2 support
> (`main` since [PR #96](https://github.com/Baekpica/ds4-dfm-rs/pull/96),
> merge `34c46d18`); the runtime binds the residual stage on any routed layer,
> so no runtime change is needed. Speed and memory of this revision on DGX
> Spark are not measured yet. Previous revisions are kept at the tags
> [`r13p8-l3-16`](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF/tree/r13p8-l3-16)
> (R13+8 on layers 3–16) and
> [`q4k-v3`](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF/tree/q4k-v3)
> (all routed layers R13).

Experimental custom GVQ1 / GGUF v3 conversion of the official
[Z.AI GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda)
(`91b69215804f2d2036a8867f6ce44848e278dcda`) for DGX Spark (GB10), run by
[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs). Standard
llama.cpp/Ollama/Transformers loading is not claimed.

## What this revision changes

The routed experts' 1.625-bit VQ was the main source of quality loss (the
original card measured held-out prose PPL 4.8555 → 8.0797 from the experts
alone). The R13+8 revision spends more bits on the routed layers where they
lower the model's output error most. With R13+8 on layers 3–16 the collapse
was reduced but not removed, so this revision upgrades the next five layers
of the same ranking as well:

- **19 of 42 routed layers are R13+8:** layers 3–17, 22, 27, 41 and 44.
  Layers 17, 22, 27, 41 and 44 are new; the parts of layers 3–16 are
  byte-identical to the previous revision. Each R13+8 layer keeps the 13-bit
  stage (K8192 E4M3 book, F16 row gain) and adds an 8-bit residual stage:
  one K=256 E4M3 book per layer and one index byte per 8-weight vector,
  2.625 bits per weight in total. Stage-1 and stage-2 indices are chosen
  jointly. Weight relative MSE on these layers falls from 0.1438 to about
  0.041.
- **The other 23 routed layers stay R13** (1.625 bits per weight), with
  their published bytes unchanged.
- **Unchanged:** the 316 Q4_K nonexpert matrices of v3, native-precision
  tensors (embedding, DSA kv_b, routers, mHC, KDA gates, indexer, norms),
  vision and the MTP layer 45.
- **Size:** about 94.49 GB (88.0 GiB) of GGUF parts, against 89.96 GB
  (83.8 GiB) with 14 R13+8 layers and 77.27 GB (71.97 GiB) with none; each
  upgraded layer part grows by 0.906 GB (0.844 GiB).
- **Format:** GVQ1 container v2 ([FORMAT.md](FORMAT.md)). Only the 19
  upgraded layer parts declare version 2; every other part keeps version 1.

## How the layers were chosen

Measured end to end. Upgrading one routed layer at a time (all others R13)
and measuring KL(original || model) on held-out text ranks the bottom
layers far ahead: layer 5 lowers KL by 0.066, 8 by 0.064, 6 by 0.064, 14 by
0.056, 3 by 0.045, and layers 3–16 take the first 14 places (68% of the
summed single-layer gains). The next five are 17 (0.021), 41 (0.017),
44 (0.015), 27 (0.013) and 22 (0.011); with them the 19 upgraded layers hold
77% of the summed gains. Below them the gains flatten (42, 33 and
24: 0.010–0.011), and the middle and upper-middle layers are last
(26: 0.001).

A local screen that scored each layer by the routed-FFN output error per
extra GiB, normalized by that layer's FFN output energy, pointed the other
way and lost end to end; the FFN output energy differs by three orders of
magnitude between layers, so a normalized local error does not measure a
layer's effect on the model. Held-out comparison of the 14-layer choices
made for the previous revision (5,435 tokens, original-precision
non-experts; R13+8 rows use the published stage 1 and a greedy stage 2
unless marked):

| Routed layers upgraded | PPL | KL vs original |
|---|---:|---:|
| none (`q4k-v3`) | 4.858 | 0.607 |
| local-screen top 14 (4, 5, 9, 25–40) | 4.470 | 0.498 |
| bottom 7 + top 7 (3–9, 38–44) | 4.220 | 0.437 |
| local-screen bottom 14 (3, 6, 7, 10–14, 24, 34, 35, 42–44) | 4.058 | 0.390 |
| layers 3–16 | 3.987 | 0.354 |
| **layers 3–16, released parts (joint stages, `r13p8-l3-16`)** | **3.969** | **0.342** |
| all 42 (reference, does not fit the budget) | 3.595 | 0.151 |
| original model | 3.538 | 0 |

## Measured on this revision's parts

The same reference forward (Transformers 5.19.0 modules, original-precision
non-experts) on the 5,195 held-out and fixture tokens available in this run;
the original card's nine cases were not available here. On these tokens the
original model and the 14-layer parts reproduce the earlier run exactly.

| Routed layers R13+8 | PPL | KL vs original | top-1 agreement |
|---|---:|---:|---:|
| none (`q4k-v3`, earlier run) | 4.808 | 0.605 | 73.9% |
| 3–16 (`r13p8-l3-16`) | 3.936 | 0.343 | 80.1% |
| **3–17, 22, 27, 41, 44 (this revision)** | **3.789** | **0.288** | **81.4%** |
| original model | 3.508 | 0 | 100% |

KL falls on every case: Korean prose 0.523 → 0.461, Korean chat 0.194 →
0.156, English prose 0.522 → 0.413, code 0.455 → 0.413, tool calls 0.294 →
0.250, and on the runtime fixture's Korean summary (garbled under R13)
0.118 → 0.042. These isolate the routed experts; they are not ds4 engine
numbers, since the release stores the non-experts as Q4_K.

## Memory on DGX Spark

On GB10, `ds4-server --check-config` quoted 112.0 GiB for the 14-layer
revision at the native 1,048,576-token context with two MTP drafts (weights
83.8, KV bank 12.8, scratch 10.7, MTP 0.7, floor 4 GiB) of 118.7 GiB
available. This revision adds 4.2 GiB of weights, about 116.2 GiB by the
same quote; the five extra layers were sized for that headroom. The quote
has not been re-run on this revision.

## Speed and quality

The 14-layer revision measured on GB10 (2K–64K sweep, two MTP drafts):
1,113.7 prefill / 31.81 decode tok/s against 1,175.4 / 35.00 for all-R13;
mean target NLL on the runtime's quality fixture 0.313 → 0.285, and the
Korean summary that R13 garbles is answered. This revision reads about
7.98 GB of weights per decoded token instead of 7.85 GB (+0.125 GB of
residual indices), and five more layers take the second prefill MMA, so it
is expected to be slightly slower again. DGX Spark measurements of this
revision follow; the reference-forward numbers are in the section above.
This is not a quality-equivalent replacement of the source model.

## Integrity and licensing

[files.json](files.json) and [SHA256SUMS](SHA256SUMS) give file sizes and
hashes; [tensor-manifest.json](tensor-manifest.json) gives per-tensor
precision, shapes and payload hashes. Verify them before loading.
Weights: MIT, see [LICENSE](LICENSE). YoungAi algorithm attribution: see
[LICENSE-YoungAi](LICENSE-YoungAi).
