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

# Qwen3.8 Flash Next — Q4 and Q2 (GGUF, `qwen4exp`)

Quantizations of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(source revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`).

| File | Bytes | SHA-256 |
|---|---|---|
| `Qwen3.8-Flash-Next-Q4.gguf` | 177,280,286,720 | `be134f2db59261c002fd79d4bc6bccac5ba0e23d2980d8c4db520036deaaa380` |
| `Qwen3.8-Flash-Next-Q2.gguf` | 147,207,127,040 | `db8dbf8891122feb22df1dbb7b3caf430b4814552563219b35dd8b2ad065a1da` |

Recipes:

- **Q4:** Q4_K routed gate/up + MXFP4 routed down (640 inputs, unpadded); Q8_0
  dense; embedded NextN layer; original BF16 n-gram table copied from the
  source revision.
- **Q2:** IQ2_XXS routed gate/up + Q2_K routed down, padded 640 → 768 inputs,
  on trunk layers 0–47; the NextN layer 48 keeps Q4_K/MXFP4 at 640; imatrix
  SHA-256 `a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`
  (recorded in the GGUF metadata as `quantize.imatrix.sha256`); original BF16 n-gram table.

## Use

- [Inferno](https://github.com/flaviodelgrosso/inferno) on Apple Silicon
  (resident, about 70 GiB for Q4): `inferno serve --model Qwen3.8-Flash-Next-Q4.gguf`
  (`--speculation opportunistic|exact` uses the embedded NextN head). The Q2
  file is admitted by its header layout; Inferno's Q2 execution is not yet
  validated.

## License

Qwen Community License 1.0 (see `LICENSE`; commercial-use conditions apply).
