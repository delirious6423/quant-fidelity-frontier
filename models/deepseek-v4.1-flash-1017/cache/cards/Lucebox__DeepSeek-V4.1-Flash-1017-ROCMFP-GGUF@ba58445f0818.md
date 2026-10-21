---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- gguf
- deepseek
- lucebox
- rocmfp
---

# DeepSeek V4.1 Flash 1017 for the Lucebox engine (ROCMFP block formats)

Lucebox builds of DeepSeek's October 17 update of V4.1 Flash
(`deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e10`) for the
Lucebox DS4.1 engine (AMD Radeon AI PRO R9700 + Ryzen AI Max). Routed experts
in the engine's ROCMFP block formats; attention, shared experts and head in
Q8_0; both Engram tables embedded, read in place. Same recipes and the same
quality harness as our September V4.1 Flash files, re-measured against the
1017 model.

| file | routed experts | expert bytes | size | KL top-256 vs teacher | top-1 | ppl |
|---|---|---|---|---|---|---|
| teacher (fp4 checkpoint, 1017) | | 269 GiB | 510 GB | 0 | 100% | 8.56 |
| `DeepSeek-V4.1-Flash-1017-ROCMFP2S.gguf` | fp2s everywhere | 158 GiB | 383 GB | 0.265 | 78.6% | 9.77 |
| `DeepSeek-V4.1-Flash-1017-ROCMFP2S-MIX10.gguf` | fp2s, fp3 down on layers 0-9 | 164 GiB | 388 GB | 0.244 | 78.8% | 9.59 |
| `DeepSeek-V4.1-Flash-1017-ROCMFP23S.gguf` | fp2s gate/up, fp3 down everywhere | 179 GiB | 405 GB | 0.195 | 81.5% | 9.42 |
| `DeepSeek-V4.1-Flash-1017-ROCMFP2S-IM1017.gguf` | fp2s everywhere, imatrix collected on 1017 | 158 GiB | 383 GB | 0.253 | 78.7% | 9.78 |
| `DeepSeek-V4.1-Flash-1017-ROCMFP2S-MIX10-IM1017.gguf` | MIX10, imatrix collected on 1017 | 164 GiB | 388 GB | 0.238 | 79.2% | 9.64 |
| `DeepSeek-V4.1-Flash-1017-ROCMFP23S-IM1017.gguf` | ROCMFP23S, imatrix collected on 1017 | 179 GiB | 405 GB | 0.194 | 81.3% | 9.41 |

For reference, the September files scored KL 0.266 (ROCMFP2S), 0.246
(MIX10) and 0.204 (fp2s with fp3 down everywhere) against the September model
(teacher ppl 8.45) on the same tokens.

**Which one:** ROCMFP2S is what the 128 GB Lucebox installs (the hottest
experts on the R9700, the rest on the Ryzen AI Max and the SSD). MIX10 is the
better file where it fits resident (192 GB boards). ROCMFP23S is the best
quality of the three, for two-box setups and boards with room for 179 GiB of
experts. The `-IM1017` files are the same recipes calibrated on the 1017
weights; prefer them (Lucebox Manage installs `ROCMFP2S-IM1017`).

## Drafter

`DeepSeek-V4.1-Flash-1017-DSpark-draft-MXFP4-Q8.gguf` (7.4 GiB, sha256 in `drafter/`): the
draft block DeepSeek ships inside the 1017 checkpoint (`mtp.*`, three stages,
128 routed experts top-3), routed experts repacked bit-exactly to MXFP4, dense
projections Q8_0. Use it with these targets; a drafter from the September
checkpoint proposes tokens the 1017 target accepts less often.

`DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf` (4.6 GiB): the same
drafter with Q2_K routed experts and Q4_K dense weights, for the 192 GB
layout (`--profile ds41-gorgon`), where it leaves the R9700 room for larger
prefill chunks.

## Image input

`DeepSeek-V4.1-Flash-1017-mmproj-BF16.gguf` (0.90 GiB): the 1017
checkpoint's vision encoder (32-layer ViT and the aligner into the 5120-wide
model), its 266 tensors copied byte for byte in BF16 by the engine's
`server/tools/export_ds4v_mmproj.py`, with the checkpoint revision recorded.
The 1017 update retrained the encoder with the language weights: pair it only
with these 1017 targets (the engine refuses an encoder from another
checkpoint). Add `--mmproj DeepSeek-V4.1-Flash-1017-mmproj-BF16.gguf` to the
command below; images then go in as OpenAI `image_url` parts.

## Run

```bash
luce_server DeepSeek-V4.1-Flash-1017-ROCMFP2S.gguf \
  --profile ds41-lucebox \
  --draft DeepSeek-V4.1-Flash-1017-DSpark-draft-MXFP4-Q8.gguf
```

With 192 GB of unified memory use `--profile ds41-gorgon` with the Q2K-Q4K
drafter, where every expert stays resident. Lucebox Manage installs the files and picks the layout for the
box. Engine and guide: https://github.com/Luce-Org/lucebox (`server/docs/DS41.md`).

## Formats and measurement

`Q2_0_ROCMFP2` (ggml type 107, 10 bytes per 32 weights, 2.5 bits, codebook
{-1, 0, 1, 2} times a ue4m3 scale per 16 weights). **fp2s** files use type
107 with bit 7 of a half-block's scale byte meaning "mirror the codebook"; a
reader that ignores that bit produces garbage, so use an engine build that
honors it. `Q3_0_ROCMFPX` (type 104, 14 bytes per 32, 3.5 bits) on the down
projections of the MIX10 file's first ten layers.

Every block's scale is chosen by an exhaustive search minimizing the
activation-weighted squared error, weighted by an importance matrix (262,144
calibration tokens; the September V4.1 imatrix for the three plain files,
an imatrix collected on the 1017 weights for the `-IM1017` files: 196,608
tokens for ROCMFP2S-IM1017, the completed 262,144 tokens in `imatrix/` for
MIX10-IM1017 and ROCMFP23S-IM1017). KL is measured on 8,184
held-out tokens against the unquantized 1017 model through the same reference forward pass; `eval/` holds
the teacher log-probs and the tokens, `results/` the raw scores and the file
headers, `imatrix/` the calibration statistics, so anyone can rescore a file.
Toolkit: the Lucebox repo, `lucebox_training/quantization`.
