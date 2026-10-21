---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
tags:
- imatrix
- calibration
- quantization
- gguf
- qwen
---

# Importance matrix (imatrix) for Qwen3.8-27B

`imatrix-qwen38.gguf` is an importance matrix for the **Qwen3.8-27B** dense model
(GGUF arch `qwen35`, the same conversion used by unsloth/BF16 GGUFs), in the
standard GGUF imatrix format — usable with **any llama.cpp** via
`llama-quantize --imatrix`, no fork required.

It is the matrix used to produce
[pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN](https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN)
(4.34 BPW effective), where it passed our full quality gate.

## Calibration data

Generated from a ~1.7 MB interleaved composite corpus (~131k tokens consumed,
`--chunks 256`), mixed on purpose so that every chunk window sees all three
components:

| Share | Content | Source |
|---|---|---|
| ~55% | agentic-coding traces (chat format, `<think>` traces, tool calls) | **[ProCreations/grug-think-v3-10k](https://huggingface.co/datasets/ProCreations/grug-think-v3-10k)** — public 10k-conversation dataset (see its card for license terms), originally used to calibrate the *grug* finetune quants and reused here as the agentic-coding component |
| ~28% | Italian prose | our own documents |
| ~17% | real source code | our own |

Credit where due: the agentic-coding component descends from the **grug**
calibration work — thanks to the grug community and to
[ProCreations](https://huggingface.co/ProCreations) for publishing
`grug-think-v3-10k`. Only aggregate per-block statistics are published here
(sum of squares / counts); no calibration text is included in this repo.

## Coverage

- 496/496 quantizable tensors covered (importance-collection gate: PASS).
- Not collectable by design (left at defaults by `llama-quantize`): the MTP
  draft layers (`blk.*.nextn`), `ssm_conv1d`, `token_embd`, `output.weight`.

## Regenerate it

```bash
llama-imatrix -m Qwen3.8-27B-BF16.gguf \
  -f qwen38-calibration.txt \
  -o imatrix-qwen38.gguf \
  --chunks 256 --no-ppl --parse-special \
  --output-frequency 10 --save-frequency 0 \
  -ngl 999
```

A dense 27B needs the GPU path (`-ngl 999`); CPU collection would take hours.
Note: `llama-imatrix` saves silently at the end of the run.

## File details

- `imatrix-qwen38.gguf` — 13,642,656 bytes (13.0 MiB)
- SHA256: `b5e681d65e726415f899fa8fd4f49e56c3f1d4fd83e3bb8963abce68f881a418`

## See also

- Quants built with this matrix: [pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN](https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN) (AMD Strix Halo ROCmFP4 preset, includes DFlash2/MTP speculative decoding notes)
- Quantization fork with the ROCmFP4 presets: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)
- Calibration dataset: [ProCreations/grug-think-v3-10k](https://huggingface.co/datasets/ProCreations/grug-think-v3-10k)

---

## Software

- **Runtime**: [`charlie12345/ROCmFPX`](https://github.com/charlie12345/ROCmFPX) (fork of llama.cpp) —
  our experimental build with per-request drafter routing, reasoning budget and
  spec-boundary cache salvage: **full source included in the lab repo under
  [`rocmfpx/`](https://github.com/pugant/strix-nebulosa/tree/main/rocmfpx)**
- **Lab repo** (patches, benchmarks, full replication guide):
  [`pugant/strix-nebulosa`](https://github.com/pugant/strix-nebulosa)

> The quant built with this matrix and everything linked here is experimental and provided as-is, at your own risk.
