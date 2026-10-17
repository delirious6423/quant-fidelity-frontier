---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [qwen3.8, w8a8, int8, smoothquant, gptq, compressed-tensors, vllm, dflash2, ampere]
---

# Qwen3.8-27B — SmoothQuant + activation-aware GPTQ, W8A8 INT8 (v3)

**v3 (`main`, 2026-09-03):** mean full-vocab KLD vs BF16 **0.00556** — lower than
the official `Qwen/Qwen3.8-27B-FP8` on the same harness (0.00584) — and the stock
**DFlash2 drafter works unchanged** (acceptance length 4.34, identical to a W8A16
target). No residual-stream rotation. 288 of 400 Linear modules run INT8×INT8 on
Ampere tensor cores; prefill is 1.42× a W8A16 checkpoint of the same model.

Previous revisions stay available: `v2-rotated` (KLD 0.01098, rotation-based;
**breaks DFlash2**, MTP only) and `v1-smoothquant-rtn` (0.01414, DFlash2-compatible).
If you are here because DFlash2 gave 0% acceptance on v2: `main` fixes it.

## Results (our harness, same for every row)

Teacher-forced full-vocab KLD, BF16 reference captured offline (transformers), candidates
served through vLLM (so INT8 *activation* rounding is included). 240 × 512-token
WikiText-2 windows = 122,640 positions, top-K 512 with reported tail bound. "Floor-free"
is the paired estimator that removes the truncation floor (see `eval/` in the repo).

| Checkpoint | KLD | floor-free | top-1 agree | PPL Δln | DFlash2 AL |
|---|---:|---:|---:|---:|---:|
| Minachist W8A16 AutoRound gs128 (weight-only) | 0.00230 | 0.00138 | 97.9% | +0.0002 | 4.33 |
| **this, v3 W8A8** | **0.00556** | **0.00371** | 96.9% | +0.0030 | **4.34** |
| Qwen official FP8 (block-128) | 0.00584 | 0.00397 | 96.9% | — | — |
| this, `v2-rotated` W8A8 | 0.01098 | 0.00810 | 95.9% | +0.0040 | 1.00 |
| this, `v1-smoothquant-rtn` W8A8 | 0.01414 | — | 95.3% | — | ~3.5 |
| lokeshe09 INT8 W8A8 (264 modules) | 0.02110 | — | 94.0% | — | — |

Paired v3 vs official FP8: Δ = −0.00021 ± 0.00005 (t = −3.9), v3 better at 54% of positions.
Paired v3 vs v2: Δ = −0.0044, t ≈ −60.

### Where the INT8 activation error lives (why this recipe)
Ablation on the same smoothed model, RTN, one module class at a time moved to W8A16:

| Class kept weight-only | modules | KLD | share of error | 8K prefill cost |
|---|---:|---:|---:|---:|
| none (all 400 W8A8) | 0 | 0.01385 | — | — |
| `mlp.down_proj` | 64 | 0.00746 | **46%** | −16% |
| `linear_attn.out_proj` (GDN) | 48 | 0.01227 | 11% | −5% |
| `self_attn.o_proj` | 16 | 0.01345 | 3% | −2% |
| `gate_proj`+`up_proj` | 128 | 0.01351 | 2.5% | −28% |
| q/k/v + GDN `in_proj_qkv/z` | 144 | 0.01348 | 2.7% | −15% |

The `down_proj` error is position-dynamic (1% of positions carry 32% of it; concentrated
in the first tokens of a sequence), so per-channel transforms — SmoothQuant α, outlier
channel splitting, or the residual rotation used in v2 — cannot remove it; only keeping
that input in 16-bit does. Per-layer bisection is additive; keeping the 32 worst layers
(0–7, 16–23, 48–63) recovers 83% of the gain for +7.6% prefill (KLD 0.00679; recipe flag
available, not published).

## Recipe
1. **SmoothQuant folds, all four GEMM input classes**, per-hook α chosen offline by a
   quantized-output proxy; median-of-window-maxima scales, capped at 16. Exact pre-quant
   (zero-centered norms absorbed; GDN gated norm and attention output gate are elementwise
   so the folds commute). **No rotation.**
2. **Mixed precision** via compressed-tensors `config_groups`:
   `group_1` W8A8 — INT8 per-channel symmetric weights, INT8 per-token dynamic activations
   (`q/k/v/o_proj, gate/up_proj, linear_attn.in_proj_qkv/in_proj_z`; 288 modules);
   `group_0` W8A16 — INT8 per-channel weights, BF16 activations
   (`mlp.down_proj`, `linear_attn.out_proj`; 112 modules, `pack-quantized`).
   Kept BF16: `embed_tokens`, `lm_head`, `mtp.*`, vision tower, `linear_attn.in_proj_a/b`,
   norms, GDN conv/A_log/dt_bias.
3. **Activation-aware GPTQ** (llm-compressor 0.13, sequential pipeline): Hessians and
   layer-to-layer propagation computed with the W8A8 modules' inputs fake-quantized
   per-token to INT8, so weights compensate the rounding vLLM will actually apply.
   256 × 2048-token windows, mixed wiki/code/tool-JSON corpus. Worth ≈ −0.0005 over RTN.

## Serving (vLLM ≥ 0.28, Ampere SM86 or newer)
Kernel routing: W8A8 modules → `CutlassInt8ScaledMM`; W8A16 modules → Marlin.
```
vllm serve Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8 \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 131072 \
  --kv-cache-dtype fp8_e4m3 --attention-backend FLASHINFER \
  --mamba-cache-mode align --enable-prefix-caching \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN","draft_sample_method":"probabilistic"}'
```
Notes: `--dtype bfloat16` (fp16 breaks DFlash acceptance on this hybrid target on some
builds); `num_speculative_tokens` must equal the drafter's `dflash_config.block_size − 1`;
fp8 KV on Ampere requires FlashInfer (FlashAttention 2 cannot read quantized KV);
`--language-model-only` if you do not need vision. MTP head (`mtp.*`, BF16, shipped in
`model-mtp.safetensors`) loads for `{"method":"mtp","num_speculative_tokens":3}` users: AL 3.12 at K=3
(keep K≤3 on hybrid-GDN targets — vllm#37035). DFlash2 is the faster drafter on this checkpoint.

Measured, 2×RTX 3090 @ 420 W, TP2, DFlash2 K=7, N=1:

| | prefill 2K / 8K / 16K (tok/s) | decode narrative / code (tok/s) | AL |
|---|---|---|---:|
| this (v3) | 2820 / 2846 / 2747 | 82 / 232 | 4.34 |
| W8A16 AutoRound (same drafter) | 1991 / 1996 / 1931 | 94 / 267 | 4.33 |
| `v2-rotated` (MTP K=3, no DFlash2) | 3500 / 3585 / 3438 | 67 / 106 | 2.70 |

**Which one should I use?** Single-stream decode: a weight-only W8A16 checkpoint is ~13% faster (Marlin is the better small-batch kernel); prefill / batched throughput / long-prompt agentic use: this W8A8 (INT8 tensor cores, 1.4× prefill).

Prefix caching + DFlash2: 9/9 correct on a 9-turn arithmetic chain, 67% warm hit-rate.
Long-context agentic soak (10.9K→28.5K accumulated context): decode and acceptance flat.

VRAM: 29 GB on disk. TP2 on 24 GB cards: ≈14.5 GB/rank weights + 1.9 GB/rank drafter;
we ran 131K context with a 4 GB/rank KV cache. Community reports for v2 on 4×3090 apply.

## Known limitations
- Divergence is 2.4× a weight-only W8A16 checkpoint; if you do not need INT8 prefill
  throughput, `Minachist/Qwen3.8-27B-INT8-AutoRound` is the higher-fidelity choice.
- vLLM issue #50021 (GDN + speculative decoding illegal-access class) is open upstream;
  the "decode collapse past ~12K context" reported by club-3090 (#1096) on vLLM 0.27.1 with
  a backported DFlash2 did **not** reproduce on upstream-native DFlash2 (0.28.1rc1) in our soak.
- `v2-rotated`: rotated residual basis; any DFlash2 drafter gets 0% acceptance. MTP only.

## Reproduction
All scripts are in this repo under `recipe/`: `apply-smoothing-v2.py` (folds), `alpha-proxy.py` (α selection), `build-attrib.py` (mixed-precision ablation builds), `build-gptq-v3.py` (activation-aware GPTQ), `multigpu_fix.py` (multi-GPU calibration fixes for llm-compressor 0.13 / compressed-tensors 0.18), `split-outliers.py` + `verify-split.py` (the outlier-splitting experiment, negative result), and the KLD harness `kld-fullvocab.py` / `kld-paired-floorfree.py`. Quantization ran in a container built from `recipe/Dockerfile` (vLLM image + llm-compressor 0.13.0 + transformers 5.15.1).

## Credits
@Neiko2002 (vision-tower packaging report), @water258 and @ghostnavy (DFlash2-on-v2 report
that motivated v3), @todiadiyatmo (4×3090 FP8-comparison benchmarks), z-lab / incoai (DFlash2
drafter), Intel AutoRound and Minachist (W8A16 reference), vLLM / llm-compressor teams.
