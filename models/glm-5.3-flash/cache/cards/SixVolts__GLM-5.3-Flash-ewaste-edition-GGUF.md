---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - glm5-next
  - moe
  - imatrix
  - llama.cpp
---

# GLM-5.3-Flash — "e-waste edition" GGUF

This is a quantization of GLM-5.3-flash that is optimized to work better on older hardware. This quant was built 
using i-matrix GGUF quantizations with the same principle as the [GLM-5.2](https://huggingface.co/SixVolts/GLM-5.2-ewaste-edition-GGUF) and
[GLM-5.3](https://huggingface.co/SixVolts/GLM-5.3-ewaste-edition-GGUF) e-waste editions: **K-quants for the routed
experts, no codebook i-quants**, because on pre-AVX-512 CPUs and previous-generation datacenter GPUs (MI100 / gfx908)
expert dequantization, not bandwidth, is the decode bottleneck.

**Format: mainline llama.cpp `glm5-next`** (merged 2026-09-30, ggml-org/llama.cpp#27773). These files do **not** load in
builds that only know the earlier `glm5next` arch from the unmerged PR #27754 (unsloth's September files are that
format). Any llama.cpp from 2026-10 on loads them. The NextN/MTP head (blk.45) is in every file for when mainline's MTP
support (#27917) lands.

## Available quantizations

| Variant | Size | Routed experts | Non-experts | Target |
|---|---|---|---|---|
| **`GLM-5.3-Flash-Q4_K_XL`** | 173.7 GiB | Q4_K | Q8_0 | Highest quality; 8×24 GB, 192 GB unified memory |
| **`GLM-5.3-Flash-Q3_K_XL`** | 136.1 GiB | Q3_K | Q8_0 | 6×24 GB |
| **`GLM-5.3-Flash-Q3_K_M`** ⭐ | 129.4 GiB | Q3_K/Q4_K hot + Q2_K cold (blk.4–20) | Q6_K, KDA q/k/v/o Q5_K | 6×24 GB with room for context |
| **`GLM-5.3-Flash-Q2_K_XL`** | 104.4 GiB | Q2_K | Q5_K | 128 GB unified memory, 5×24 GB; lowest quality |
| `mmproj/mmproj-GLM-5.3-Flash-F16.gguf` | 1.1 GB | — | — | vision tower (`glm5v`), use with `--mmproj` |

All are imatrix-quantized with [unsloth's GLM-5.3-Flash importance matrix](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
(88 chunks × 9216 tokens) from an **F32 intermediate** of the FP8 checkpoint (`fp8 × block scale` in F32; a BF16
intermediate re-rounds most weights because the scales are arbitrary floats). Tensors that mainline's quantizer keeps
unquantized for this arch (hyper-connection mixers, DSA indexer, KDA gates/betas, `attn_kv_a_mqa`) are F32 in every
file; `attn_q_a/q_b` and `nextn.eh_proj` are Q8_0 (mainline's floor). The MTP head's experts are Q4_K in every variant.

## Why this isn't just the GLM-5.3 recipe scaled down

Two things about Flash are different from the 745B model, measured on this box:

1. **The cold tail is short.** On GLM-5.2/5.3 the imatrix activation energy of the expert layers spans six orders of
   magnitude, so 32 early layers could take Q2_K almost for free. On Flash the spread is ~50× and the cost of Q2_K rises
   steeply after blk.20: Q2_K on blk.4–20 costs +0.037 KL in total; extending it to blk.28 costs another +0.075 for only
   7 GiB saved (see the table below). The `Q3_K_M` cut is therefore blk.4–20.
2. **Per-token bandwidth is dominated by the non-expert weights.** The 9 B non-expert parameters (mostly the 34 layers of
   KDA q/k/v/o projections) are read every token: 9.6 GB/token at Q8_0 against 4.9 GB/token for the eight active experts.
   `Q3_K_M` and `Q2_K_XL` therefore carry Q6_K / Q5_K non-experts — a ~2 GiB file-size difference that is worth ~20% of
   decode bytes. Measured twice (two expert layouts): KDA projections at Q5_K instead of Q6_K change KL by <0.001
   (0.2628 vs 0.2632; 0.188 vs 0.189), so `Q3_K_M` and `Q2_K_XL` use Q5_K there.

## Quality (perplexity, KL divergence)

wikitext-2 test, 100 chunks × 512 tokens, f16 KV cache, `-fa off`, mainline llama.cpp `4ebdf2c`, 10× MI100.
KL divergence is against our `Q4_K_XL` (4096-token chunks, 49k tokens).

| Quant | Size | PPL ↓ | KL vs Q4_K_XL (mean / median) | top-1 agreement |
|---|---|---|---|---|
| unsloth `UD-Q4_K_XL` (reference) | 185.9 GiB | 3.3038 ± 0.042 † | — | — |
| **`Q4_K_XL`** | 173.7 GiB | **3.3105 ± 0.043** | — | — |
| **`Q3_K_XL`** | 136.1 GiB | 3.6015 ± 0.047 | 0.152 / 0.030 | 87.7% |
| **`Q3_K_M`** | 129.4 GiB | 3.7038 ± 0.049 | 0.188 / 0.042 | 86.0% |
| `Q3_K_M` with Q6_K KDA projections (not published) | 129.9 GiB | 3.6855 ± 0.049 | 0.189 / 0.041 | 85.9% |
| `Q3_K_M` with Q2_K to blk.28 (not published) | 123 GiB | 4.0427 ± 0.055 | 0.263 / 0.061 | 83.1% |
| **`Q2_K_XL`** | 104.4 GiB | 4.6112 ± 0.065 | 0.378 / 0.095 | 79.5% |

† measured on a different llama.cpp build (the `glm5next` fork that can load that file); our `Q4_K_XL` measures
3.3099 on that same build, so the two are equal within error at 12 GiB less.

## Speed on the author's box (10 × MI100, gfx908)

Measured with a fork of llama.cpp that carries this model under the earlier `glm5next` arch name (header re-keyed, same
tensors), `llama-bench -p 2048 -n 128`, no speculative decoding: `Q4_K_XL` 780 t/s prefill / 33.8 t/s decode,
`Q3_K_M` 730 / 32.2. The 3-bit file is not faster than the 4-bit one here even though it reads fewer bytes — 2/3-bit
K-quant dequantization costs more per block on this GPU class, which is exactly the trade the e-waste editions make
(and why there are no i-quants). Mainline llama.cpp runs this architecture at roughly 11 s per 512-token perplexity
chunk on the same cards.

## Usage

```bash
# Q3_K_M — six 24 GB cards (first shard loads the set)
llama-server -m GLM-5.3-Flash-Q3_K_M-00001-of-00004.gguf -sm layer -fit on -fa off -c 32768

# with images
llama-server -m GLM-5.3-Flash-Q4_K_XL-00001-of-00005.gguf --mmproj mmproj-GLM-5.3-Flash-F16.gguf -fa off
```

`-fa off` is required for this architecture at the time of writing. Sampling per Z.ai: temperature 1.0, top-p 0.95.

## Credits / provenance

- Base model: **GLM-5.3-Flash** by [zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash) (FP8 checkpoint, revision `eb9eb208`).
- Importance matrix: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) `imatrix_unsloth.gguf_file`
  (not redistributed here).
- llama.cpp `glm5-next` support: [ggml-org/llama.cpp#27773](https://github.com/ggml-org/llama.cpp/pull/27773) (timkhronos).
- Quantized with [llama.cpp](https://github.com/ggml-org/llama.cpp) `convert_hf_to_gguf.py --outtype f32` and
  `llama-quantize --imatrix --tensor-type-file`.
