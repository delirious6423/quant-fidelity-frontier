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

> **Status, 2026-10-18:** the files are the R13+8 revision. Loading it needs
> a ds4-dfm-rs build with GVQ1 container v2 support (branch
> [`feat/glm-vq-r13p8`](https://github.com/Baekpica/ds4-dfm-rs/tree/feat/glm-vq-r13p8)
> until it merges); older builds reject the 14 upgraded parts. Speed and
> memory on DGX Spark are not measured yet. The previous revision (all routed
> layers R13) is kept at the tag
> [`q4k-v3`](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF/tree/q4k-v3).

Experimental custom GVQ1 / GGUF v3 conversion of the official
[Z.AI GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda)
(`91b69215804f2d2036a8867f6ce44848e278dcda`) for DGX Spark (GB10), run by
[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs). Standard
llama.cpp/Ollama/Transformers loading is not claimed.

## What this revision changes

The routed experts' 1.625-bit VQ was the main source of quality loss (the
original card measured held-out prose PPL 4.8555 → 8.0797 from the experts
alone). This revision spends more bits on the 14 routed layers where they
reduce the experts' output error most:

- **14 of 42 routed layers are R13+8:** layers 3 to 16, the first 14
  routed layers. Each keeps the 13-bit stage (K8192 E4M3 book,
  F16 row gain) and adds an 8-bit residual stage: one K=256 E4M3 book per
  layer and one index byte per 8-weight vector, 2.625 bits per weight in
  total. Stage-1 and stage-2 indices are chosen jointly. Weight relative
  MSE on these layers falls from 0.1438 to about 0.041.
- **The other 28 routed layers stay R13** (1.625 bits per weight), with
  their published bytes unchanged.
- **Unchanged:** the 316 Q4_K nonexpert matrices of v3, native-precision
  tensors (embedding, DSA kv_b, routers, mHC, KDA gates, indexer, norms),
  vision and the MTP layer 45.
- **Size:** about 89.96 GB (83.8 GiB) of GGUF parts, against 77.27 GB
  (71.97 GiB) before; each upgraded layer part grows by 0.906 GB.
- **Format:** GVQ1 container v2 ([FORMAT.md](FORMAT.md)). Only the 14
  upgraded layer parts declare version 2; every other part keeps version 1.

## How the 14 layers were chosen

Measured end to end. Upgrading one routed layer at a time (all others R13)
and measuring KL(original || model) on held-out text ranks the bottom
layers far ahead: layer 5 lowers KL by 0.066, 8 by 0.064, 6 by 0.064, 14 by
0.056, 3 by 0.045, and layers 3–16 take the first 14 places (68% of the
summed single-layer gains). The top layers come next (17, 41, 44), the
middle and upper-middle layers last (26: 0.001).

A local screen that scored each layer by the routed-FFN output error per
extra GiB, normalized by that layer's FFN output energy, pointed the other
way and lost end to end; the FFN output energy differs by three orders of
magnitude between layers, so a normalized local error does not measure a
layer's effect on the model. Held-out comparison of 14-layer choices
(5,435 tokens, original-precision non-experts; R13+8 rows use the
published stage 1 and a greedy stage 2 unless marked):

| Routed layers upgraded | PPL | KL vs original |
|---|---:|---:|
| none (previous revision) | 4.858 | 0.607 |
| local-screen top 14 (4, 5, 9, 25–40) | 4.470 | 0.498 |
| bottom 7 + top 7 (3–9, 38–44) | 4.220 | 0.437 |
| local-screen bottom 14 (3, 6, 7, 10–14, 24, 34, 35, 42–44) | 4.058 | 0.390 |
| layers 3–16 | 3.987 | 0.354 |
| **layers 3–16, this release's parts (joint stages)** | **3.969** | **0.342** |
| all 42 (reference, does not fit the budget) | 3.595 | 0.151 |
| original model | 3.538 | 0 |

## Memory on DGX Spark

At the native 1,048,576-token context with two MTP drafts, the runtime's own
quote needs about 112 GiB: 83.8 GiB of weights, 22.5 GiB of KV and
4,096-row workspace, 1.7 GiB of MTP state and the 4 GiB floor. That leaves
roughly 4.5 GiB of the ~116.5 GiB a 128 GB Spark gives the process. The
byte budget was capped there on purpose, well below DeepSeek V4.1 Flash VQ
(113.6 GB).

## Speed and quality

Measured on this release's parts with the reference forward above
(Transformers 5.19.0 modules, original-precision non-experts): held-out KL
against the original falls from 0.607 to 0.342 and top-1 agreement rises
from 73.6% to 80.1%; the original card's held-out prose PPL is 5.905
(original 4.856, previous revision 8.080). These isolate the routed
experts; they are not ds4 engine numbers, since the release stores the
non-experts as Q4_K.

Decode reads about 7.85 GB of weights per token instead of 7.5 GB (+0.35 GB
of residual indices), so decode is expected to be a few percent slower;
R13+8 layers prefill with 32-token items. DGX Spark measurements on this
revision follow. This is not a quality-equivalent replacement of the source
model.

## Integrity and licensing

[files.json](files.json) and [SHA256SUMS](SHA256SUMS) give file sizes and
hashes; [tensor-manifest.json](tensor-manifest.json) gives per-tensor
precision, shapes and payload hashes. Verify them before loading.
Weights: MIT, see [LICENSE](LICENSE). YoungAi algorithm attribution: see
[LICENSE-YoungAi](LICENSE-YoungAi).
