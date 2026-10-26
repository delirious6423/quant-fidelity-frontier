---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen4exp
- text-generation
---

# Qwen3.8 Flash Next — Q4-D6, Q4 and Q2 (GGUF, `qwen4exp`)

Quantizations of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(source revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`).

| File | Bytes | SHA-256 |
|---|---|---|
| `Qwen3.8-Flash-Next-Q4-D6.gguf` | 177,280,286,720 | `6bc0986afb1e5ec4a4cc4ab49dff82099da85113e3e1092e0d1c3b8d7d158287` |
| `Qwen3.8-Flash-Next-Q4.gguf` | 177,280,286,720 | `be134f2db59261c002fd79d4bc6bccac5ba0e23d2980d8c4db520036deaaa380` |
| `Qwen3.8-Flash-Next-Q2.gguf` | 147,207,127,040 | `db8dbf8891122feb22df1dbb7b3caf430b4814552563219b35dd8b2ad065a1da` |

Recipes:

- **Q4-D6 (recommended for Inferno):** the Q4 file requantized for decode
  speed. The name states the recipe: **Q4** routed experts (unchanged Q4_K /
  MXFP4) and **~6-bit dense** weights (6.2 bits per weight on average across
  every non-expert tensor, down from 10.0 in Q4). GDN projections are graded
  by layer sensitivity (Q5_K/Q6_K in layers 0–23, Q4_K in 24–47); attention,
  head, PLE and NextN projections are Q6_K; the shared-expert gate/up is
  Q4_K; the hyper-connection mixers are Q8_0 (up) and Q6_K (down); the router
  is BF16 (lossless: the F32 values are BF16-exact). The re-quantized tensors
  are importance-weighted with the unsloth imatrix (SHA-256
  `a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`). The
  precision map, source header SHA-256 and imatrix SHA-256 are recorded in the
  GGUF metadata (`qwen4exp.requant.*`). Decode reads 4.62 GB of weights per
  token instead of 6.70 GB: on an Apple M5 Max, single-session decode goes
  from 51.9 to 68.0 tok/s (+31%) and NextN speculative decode from 87.6 to
  96.5 tok/s (+10%). Teacher-forced KL divergence from Q4 is 0.022 (Q2:
  0.172); GSM8K, HumanEval, MGSM, IFEval and 32k/100k long-context retrieval
  are within noise of Q4. Rebuild byte-identically with
  `inferno convert qwen-q4-d6 --source Qwen3.8-Flash-Next-Q4.gguf --imatrix imatrix_unsloth.gguf_file`.
  Full study: [docs/quantization.md](https://github.com/flaviodelgrosso/inferno/blob/main/docs/quantization.md).
- **Q4:** Q4_K routed gate/up + MXFP4 routed down (640 inputs, unpadded); Q8_0
  dense; embedded NextN layer; original BF16 n-gram table copied from the
  source revision.
- **Q2:** IQ2_XXS routed gate/up + Q2_K routed down, padded 640 → 768 inputs,
  on trunk layers 0–47; the NextN layer 48 keeps Q4_K/MXFP4 at 640; imatrix
  SHA-256 `a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`
  (recorded in the GGUF metadata as `quantize.imatrix.sha256`); original BF16 n-gram table.

## Use

- [Inferno](https://github.com/flaviodelgrosso/inferno) on Apple Silicon
  (resident, about 69 GiB for Q4-D6 and 72 GiB for Q4):
  `inferno download qwen-q4-d6`, then
  `inferno serve --model models/Qwen3.8-Flash-Next-Q4-D6.gguf`
  (`--speculation opportunistic|exact` uses the embedded NextN head). Q4-D6
  needs an Inferno build with its K-quant dense kernels (the release that
  adds `qwen-q4-d6`). The Q2
  file is admitted by its header layout; Inferno's Q2 execution is not yet
  validated.

## License

Qwen Community License 1.0 (see `LICENSE`; commercial-use conditions apply).
