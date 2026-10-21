---
license: mit
base_model: zai-org/GLM-5.3
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - glm-dsa
  - moe
  - imatrix
  - speculative-decoding
  - llama.cpp
---

# GLM-5.3 — "e-waste edition" GGUF

This is a quantization of GLM-5.3 that is optimized to work better on older hardware. This quant was built 
using i-matrix GGUF quantizations with the same principle as the [GLM-5.2 e-waste edition](https://huggingface.co/SixVolts/GLM-5.2-ewaste-edition-GGUF): **K-quants for the
routed experts, not codebook i-quants**, because on pre-AVX-512 CPUs and previous-gen datacenter GPUs (MI100 /
gfx908) expert dequantization, not bandwidth, is the decode bottleneck. GLM-5.3 is GLM-5.2 with further
post-training (identical architecture and config), so the 5.2 recipes carry over tensor for tensor.

## Available quantizations

| Variant | Size | Routed experts | Non-experts | Target |
|---|---|---|---|---|
| **`GLM-5.3-Q4_K_XL`** | 403.8 GiB | Q4_K | Q8_0 | Highest quality; big-RAM CPU-expert boxes (≈512 GB) |
| **`GLM-5.3-Q3_K_XL`** | 314.1 GiB | Q3_K | Q8_0 | Max quality on GPU; experts spill to CPU on <11×32 GB |
| **`GLM-5.3-Q3_K_M`** ⭐ recommended | 297.5 GiB | Q3_K + Q2_K (cold layers) | Q6_K | Fits **0-spill on 10×32 GB** (320 GB HBM) |
| **`GLM-5.3-Q2_K_XL`** | 228.7 GiB | Q2_K + IQ2_XXS (13 coldest layers) | Q4_K | Fits **0-spill on 8×32 GB**; fastest decode, lowest quality |
| `GLM-5.3-MTP-Q3_K_M-sc` | 7.2 GB | — | — | self-contained NextN/MTP draft head (see *Speculative decoding*) |

All are imatrix-quantized with [unsloth's GLM-5.3 importance matrix](https://huggingface.co/unsloth/GLM-5.3-GGUF)
(209 chunks × 3584 tokens). The per-tensor types are the 5.2 files' types read from their GGUF headers, so each
variant is the exact 5.3 counterpart of its 5.2 namesake, with one deliberate change (the MTP head, below).

## What changed from the 5.2 build

- **F32 intermediate, not BF16.** GLM-5.3 ships as FP8 e4m3 with 128×128 block scales, and the scales are arbitrary
  floats (194 of 95,040 in the first shard are powers of two). A BF16 intermediate re-rounds 88% of the dequantized
  weights (typically 0.1%, up to 2⁻⁸ relative). The conversion here writes `fp8 × scale` in F32 (3.0 TB) and
  `llama-quantize` works in f32 anyway, so the quantizer sees exactly the dequantized checkpoint.
- **Fewer tensors (1,524 vs 1,809).** The checkpoint only carries DSA indexer weights on the 22 `indexer_types: full`
  layers; the other layers reuse the previous full layer's selection (what the reference implementation does). The
  5.2 Q3_K_M / Q2_K_XL files were made by an older converter that duplicated the indexer into every layer; these files
  record `indexer_types` instead. The 5.2 `_XL` files already had the 1,524-tensor layout.
- **The MTP head (blk.78) is usable.** The 5.2 files kept the NextN/MTP block's experts at Q2_K because mainline did
  not use it. llama.cpp now drafts with it (`--spec-type draft-mtp`), so here its experts are **Q4_K** and its
  `nextn.eh_proj` is Q6_K/Q8_0 in every variant (+2.1 GiB vs the 5.2 recipe). Measured on the head's own weights: the
  MoE block's time on an MI100 is within 30 µs between every 4-bit type (and q3_K is the slowest at 413 µs vs q4_K
  253), while relative RMS error is q2_K 0.30, q3_K 0.15, q4_K 0.072, q5_K 0.036 — so the head costs nothing in speed
  and Q4_K is the best of the equal-cost options.

## `GLM-5.3-Q3_K_M` — composition

| Tensor group | Quant | Notes |
|---|---|---|
| Routed experts gate/up, **blk.35–77** (43 hottest MoE layers) | **Q3_K** | |
| Routed experts down, **blk.35–77** | **Q4_K** | every hot `ffn_down_exps` |
| Routed experts, **blk.3–34** (32 coldest MoE layers) | **Q2_K** | the cold cut, see below |
| Routed experts, **blk.78** (MTP head) | **Q4_K** | changed from the 5.2 recipe's Q2_K |
| Attention (`q_a`, `q_b`, `kv_a_mqa`, `v_b`, `output`), indexer `attn_q_b` | **Q6_K** | |
| `attn_k_b` | Q4_0 | `ncols=192` not divisible by 256 → automatic fallback |
| Indexer `attn_k`, `proj` | Q3_K | |
| Shared expert (`ffn_{gate,up,down}_shexp`) | **Q6_K** | fires every token |
| Dense FFN blk.0–2 | gate/up Q3_K, down Q5_K | |
| `token_embd`, `output` | **Q6_K** | |
| norms, router (`ffn_gate_inp`, `exp_probs_b`) | F32 | |

**Total: 297.45 GiB, 3.39 bpw**, 1,524 tensors.

### The cold cut is the same in 5.3

"Cold" is read from the imatrix (`in_sum2`, the sum of squared input activations per channel). The mean
`ffn_down_exps` importance is monotonic in depth and spans six orders of magnitude in 5.3 exactly as in 5.2:

```
blk.3   ≈ 0.003      ← coldest
blk.8   ≈ 2.3        (the one outlier, as in 5.2)
blk.20  ≈ 2.0
blk.34  ≈ 70
blk.35  ≈ 84
blk.77  ≈ 14860      ← hottest
```

so the 32 coldest MoE layers (blk.3–34) take Q2_K experts and the hot tail (blk.35–77) stays Q3_K/Q4_K.

## `GLM-5.3-Q2_K_XL`, `Q3_K_XL`, `Q4_K_XL` — composition

Tensor for tensor the 5.2 files' types (`Q2_K_XL`: Q2_K experts with IQ2_XXS on blk.3–7 and 9–16, Q4_K
everything else, 6 shards; `Q3_K_XL` / `Q4_K_XL`: Q3_K / Q4_K experts, Q8_0 everything else, 8 / 11 shards), plus
the MTP head change above. All four were quantized from the same F32 intermediate with the same imatrix.

## Quality (perplexity)

wikitext-2 test, 100 chunks × 512 tokens, f16 KV cache, `-fa off` (the DSA path), identical settings for all:

| Quant | Size | PPL ↓ | 5.2 counterpart (q8_0 KV) |
|---|---|---|---|
| `GLM-5.3-Q4_K_XL` | 403.8 GiB | 2.6537 ± 0.032 | 2.6733 |
| `GLM-5.3-Q3_K_XL` | 314.1 GiB | 2.7910 ± 0.034 | 2.8176 |
| **`GLM-5.3-Q3_K_M`** | 297.5 GiB | **2.8092 ± 0.034** | 2.8348 |
| `GLM-5.3-Q2_K_XL` | 228.7 GiB | 3.6057 ± 0.047 | 3.6129 |

The 5.2 column is the previous README's measurement (q8_0 KV cache); the two models and the KV setting differ, so
compare within the 5.3 column.

## Performance — 10 × MI100 (gfx908), 0-spill, `Q3_K_M`

Mainline-equivalent (dense DSA attention, no draft head): **11.5 tok/s** decode, single stream, 16k context.

With this build's llama.cpp branch (fused sparse top-k attention for `glm-dsa`, the MTP head as a self-contained
drafter on its own GPU, per-tensor placement across the ten cards, pre-scaled int8 MMQ tiles for Q2_K/Q3_K on
gfx908): **19.6 tok/s** decode at temperature 1.0 with 82–84% of drafted tokens accepted, **~110–140 tok/s** prefill
at ubatch 512–1024, and 112k of context with the drafter still resident. The branch is
[sixvolts/llama.cpp `glm53/bringup`](https://github.com/sixvolts/llama.cpp); the pieces that matter for this model
are the sparse attention path (`LLAMA_DSA_SPARSE=1`), the Q2_K/Q3_K MMQ kernels (2–2.2× faster per MoE block on
gfx908), and a batch-size gate on HIP graphs that fixes a prefill corruption on this GPU class.

## Speculative decoding with the MTP head

`GLM-5.3-MTP-Q3_K_M-sc.gguf` is blk.78 plus `token_embd` / `output_norm` / `output`, copied byte for byte from the
`Q3_K_M` file, so it can run as a separate draft model on a device the main model does not use:

```bash
llama-server -m GLM-5.3-Q3_K_M-00001-of-00008.gguf \
  -md GLM-5.3-MTP-Q3_K_M-sc.gguf --spec-type draft-mtp --spec-draft-n-max 2 \
  -sm layer -fa off -c 16384
```

Two drafted tokens per step is the sweet spot here (three loses on prose: acceptance falls to 0.56).

## Usage

```bash
# Q3_K_M — 0-spill on ten 32 GB cards (first shard loads the set)
llama-server -m GLM-5.3-Q3_K_M-00001-of-00008.gguf -sm layer -fit on -fa off -c 16384

# Q4_K_XL — big-RAM CPU-expert box (≈512 GB): experts in system RAM, the rest on one GPU
llama-server -m GLM-5.3-Q4_K_XL-00001-of-00010.gguf -ngl 999 -ot 'ffn_.*_exps=CPU' -fa off -c 16384
```

Use `-sm layer`, not `-sm tensor`. Sampling per Z.ai: temperature 1.0, top-p 0.95, min-p 0.01.

## Credits / provenance

- Base model: **GLM-5.3** by [zai-org](https://huggingface.co/zai-org/GLM-5.3) (FP8 checkpoint, revision `aca966e4`).
- Importance matrix: [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF) `imatrix_unsloth.gguf_file`
  (not redistributed here). Thanks to the unsloth team for the calibration data.
- Quantized with [llama.cpp](https://github.com/ggml-org/llama.cpp) `convert_hf_to_gguf.py --outtype f32` and
  `llama-quantize --imatrix --tensor-type-file`.
