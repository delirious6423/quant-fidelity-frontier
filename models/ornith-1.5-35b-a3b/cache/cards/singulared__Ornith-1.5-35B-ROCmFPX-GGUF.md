---
license: apache-2.0
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - rocm
  - rocmfpx
  - rocmfp4
  - rocmfp6
  - amd
  - strix-halo
  - gfx1151
  - mtp
  - speculative-decoding
  - moe
---

# Ornith-1.5-35B-A3B — ROCmFPX builds for Strix Halo

ROCmFPX quantisations of [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
for AMD Strix Halo (`gfx1151`), with the MTP head kept live for speculative decoding.

| file | bpw | size | pick it for |
| --- | ---: | ---: | --- |
| `Ornith-1.5-35B-HYBRID-fp6.gguf` | 4.41 | 18.21 GiB | **prefill-dominated work** — best quality |
| `Ornith-1.5-35B-ROCMFP4-FAST.gguf` | 4.27 | 17.65 GiB | **generation-dominated work** — fastest decode |

More variants may be added later.

## HYBRID: class-aware assignment

Every stock ROCmFP4 preset leaves the obvious lever unused on a 256-expert MoE: they apply **one
type to every tensor**. The hybrid splits them:

| tensor class | count | type |
| --- | ---: | --- |
| routed experts | 123 | `Q4_0_ROCMFP4_FAST` (4.25 bpw) |
| attention | 104 | **`Q6_0_ROCMFPX`** (FP6) |
| shared expert | 123 | **`Q6_0_ROCMFPX`** (FP6) |
| token embedding / output | 2 | **`Q6_0_ROCMFPX`** (FP6) |
| MTP (`nextn`) head | 1 | `Q4_0_ROCMFP4_FAST` |

**4.41 bpw · 18.21 GiB.** Routed experts are sparse (8 of 256 fire per token) and tolerate 4-bit;
attention and the shared expert are on every token's critical path and get 6-bit.

## Perplexity

wikitext-2, 145 chunks @ ctx 2048, identical corpus, Vulkan, all measured here:

| build | bpw | size | PPL |
| --- | ---: | ---: | --- |
| **HYBRID (this)** | 4.41 | 18.21 GiB | **7.3991** ±0.0506 |
| `ROCMFP4_FAST` | 4.27 | 17.65 GiB | 7.7749 ±0.0539 |
| `ROCMFP4_COHERENT` | 4.55 | 18.81 GiB | 7.8233 ±0.0550 |
| `ROCMFP4_STRIX` | 4.31 | 17.81 GiB | 7.8307 ±0.0547 |

The three stock presets cluster within **0.8%** of each other — preset choice barely matters on this
architecture, because none of them differentiate by tensor class. Class-aware assignment moves
**4.8%** for +0.14 bpw.

Perplexity measures prose next-token prediction, not agentic capability. Use it to compare
quantisations of the same weights, not to rank models.

## Speed (Radeon 8060S, gfx1151, Vulkan, MTP `n4`, `-ub 2048`)

| build | 8.5K pp / tg | 34K pp / tg | 69K pp / tg |
| --- | --- | --- | --- |
| HYBRID | 990.9 / **63.0** | 815.9 / 55.6 | 488.4 / 45.2 |
| `FAST` | 993.9 / **87.7** | 813.3 / 67.3 | 478.9 / 56.3 |

**Prefill is identical** (within 0.5%) — it is compute-bound, so the FP6 weights cost nothing there.
Decode pays the whole price: −28%, because FP6 attention means more bytes per generated token.

⇒ **Pick HYBRID for prefill-dominated work** (digesting repos/documents, long context, short
answers). **Pick `FAST` for generation-dominated work.** The recipe is a quality/decode dial, not a
free win.

Needle-in-a-haystack retrieval passes at **8.5K, 34.5K and 69.5K** on both.

## Backend: use Vulkan

Same build, same model, same flags — only `-dev` changes:

| backend | 8.5K pp / tg | 34K pp / tg |
| --- | --- | --- |
| **Vulkan** | 993.9 / **87.7** | 813.3 / **67.3** |
| HIP · ROCm 7.2.4 | 968.1 / 72.7 | 675.3 / 64.1 |
| HIP · ROCm 10.1 nightly | **1087.0** / 58.2 | **834.6** / 55.1 |

The ROCm nightly is a **prefill-for-decode trade**: +12% prefill over HIP 7.2 but −20% decode, and
−34% decode against Vulkan. Vulkan wins overall and needs no container.

## MTP head at FP4 is safe here

The `nextn.eh_proj` head is often kept at Q8_0 on the theory that it determines draft acceptance.
Measured on this model, dropping it to FP4 **did not hurt** — identical perplexity to 4 decimals
(7.7749 both) and slightly *better* acceptance:

| MTP head | acceptance |
| --- | --- |
| Q8_0 | 0.73–0.77 |
| **FP4** | **0.78–0.80** |

## Usage

```bash
llama-server -m Ornith-1.5-35B-HYBRID-fp6.gguf \
  -ngl 99 -c 131072 -dev Vulkan0 --jinja -fa on -b 2048 -ub 2048 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.6
```

Requires a [ROCmFPX](https://github.com/charlie12345/ROCmFPX) build — mainline llama.cpp does not
know the `Q4_0_ROCMFP4_*` / `Q6_0_ROCMFPX` tensor types. The MTP head is native to Ornith 1.5
(`blk.40.nextn.*`, `nextn_predict_layers=1`); no graft is needed, unlike 1.0.

Reproduce the recipe with:

```
attn_.*=q6_0_rocmfpx
ffn_(gate|up|down)_shexp=q6_0_rocmfpx
token_embd.weight=q6_0_rocmfpx
output.weight=q6_0_rocmfpx
nextn.*=q4_0_rocmfp4_fast
```
`llama-quantize --tensor-type-file <rules> Ornith-1.5-35B-BF16.gguf out.gguf Q4_0_ROCMFP4_FAST`

## Honest caveat

On wikitext perplexity, **Ornith 1.0 scores far better** — 6.19 (ROCmFP4-COHERENT) against 7.40
here, and the gap is present at BF16, so it is a property of the 1.5 weights and not of this
quantisation. 1.0 also decodes faster (86.7 t/s) with higher draft acceptance (0.88).

Ornith 1.5 is chosen here for its reported agentic/SWE gains, which wikitext does not measure. If
your workload is prose modelling rather than agentic coding, 1.0 may serve you better.
