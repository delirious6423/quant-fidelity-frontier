---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- gguf
- deepseek
- lucebox
- rocmfp
---

# DeepSeek V4.1 Flash for the Lucebox engine (ROCMFP block formats)

Lucebox builds of DeepSeek V4.1 Flash for the Lucebox DS4.1 engine
(AMD R9700 + Strix Halo). Routed experts in the engine's ROCMFP block
formats; attention, shared experts and head in Q8_0; both Engram tables
embedded. Calibrated with the published 262,144-token imatrix
(`smalinin/DeepSeek-V4.1-Flash-GGUF`, converted in `imatrix/`).

| file | routed experts | expert bytes | KL top-256 vs teacher | top-1 | ppl |
|---|---|---|---|---|---|
| teacher (fp4 checkpoint) | | 269 GiB | 0 | 100% | 8.45 |
| antirez Q2 (IQ2_XXS / Q2_K) | | 142 GiB | 0.260 | 78.3% | 9.71 |
| `DeepSeek-V4.1-Flash-ROCMFP23.gguf` | fp2 gate/up, fp3 down | 179 GiB | 0.272 | 77.9% | 9.80 |
| `DeepSeek-V4.1-Flash-ROCMFP2S.gguf` | fp2s everywhere | 158 GiB | 0.266 | 77.7% | 9.59 |
| `DeepSeek-V4.1-Flash-ROCMFP2S-MIX10.gguf` | fp2s, fp3 down on layers 0-9 | 164 GiB | 0.246 | 78.3% | 9.45 |

Formats: `Q2_0_ROCMFP2` (ggml type 107, 10 bytes per 32 weights, 2.5 bits,
codebook {-1, 0, 1, 2} x a ue4m3 scale per 16 weights), `Q3_0_ROCMFPX`
(type 104, 14 bytes per 32, 3.5 bits). **fp2s** files use type 107 with bit
7 of a half-block's scale byte meaning "mirror the codebook"; a reader that
ignores that bit produces garbage, so use an engine build that honors it.
KL is measured on 8,184 held-out fineweb-edu tokens against the unquantized
model through the same reference forward pass (`eval/` holds the teacher
log-probs and the tokens). Recipe, converter patches and harness:
Lucebox repo, `lucebox_training/quantization`.

## IQ2 / IQ3 + MXFP8 builds

`ds41-lucebox-final.gguf` (recommended) and `ds41-lucebox-minus10gib.gguf` (10 GiB smaller): routed experts in
`IQ2_XXS` / `IQ3_XXS`, dense projections in MXFP8 (type 117, needs an engine with MXFP8, see Luce-Org/lucebox#800).
Correct answers on our internal hard bench; exact = plain greedy, no drafter.

| internal hard bench | final, Lucebox profile + DSpark | final, exact | minus10gib, exact | antirez Q2 |
|---|---|---|---|---|
| file size | 366.1 GB | 366.1 GB | 355.4 GB | 365.7 GB |
| 96 questions (8K tokens max) | 94 | 93 | 91 | 86 |
| 32 long-reasoning questions (16K tokens max) | 27 | 30 | 30 | 26 |
| 49 long-reasoning questions (16K tokens max) | 37 | 41 | 34 | 29 |
| **total (177 questions)** | **158** | **164** | **155** | **141** |
