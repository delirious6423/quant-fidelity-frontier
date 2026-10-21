---
license: apache-2.0
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
base_model_relation: quantized
library_name: gguf
tags:
  - gguf
  - rocmfp4
  - llama.cpp
  - strix-halo
  - gfx1151
  - rocm
  - amd
  - ryzen-ai-max
---

# DeepSeek-V4-Flash-180B — ROCmFP4 (STRIX_LEAN) GGUF — AMD Ryzen AI Max+ 395 / Strix Halo / gfx1151

ROCmFP4 quantizations of DeepSeek-V4-Flash-180B, built for AMD Strix Halo
(gfx1151) unified-memory systems. Two variants are provided: one retaining the
MTP (multi-token prediction) tensors, one without.

These were converted from the MXFP4 release rather than from BF16, so the
quantization is MXFP4 → `Q4_0_ROCMFP4_STRIX_LEAN`.

## Files

Each variant is sharded to stay under the 50 GB per-file limit. Point
`--model` at the **first** shard; llama.cpp loads the rest automatically.

| variant | total | notes |
|---|---|---|
| `DeepSeek-V4-Flash-180B-MTP-...-STRIX_LEAN` | ~92 GB | MTP tensors retained |
| `DeepSeek-V4-Flash-180B-...-STRIX_LEAN` | ~90 GB | MTP tensors removed |

## Requirements

ROCmFP4 is **not** in upstream llama.cpp. You need a build that supports the
`Q4_0_ROCMFP4_*` quantization types (the ROCmFPX fork), compiled for gfx1151:

```
-DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON -DLLAMA_BUILD_WEBUI=OFF
```

A stock llama.cpp binary will not load these files.

## Usage

```
llama-server \
  --model DeepSeek-V4-Flash-180B-MXFP4-to-ROCmFP4-STRIX_LEAN-00001-of-0000N.gguf \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 999 --flash-attn on -dio \
  --ctx-size 32768
```

`-dio` (direct I/O) matters on unified memory — without it the initial load is
substantially slower.

> ### ⚠️ If the MTP draft command crashes on your build
> The separate-model `draft-mtp` path has known bugs in the legacy `charlie12345/ROCmFPX`
> line — reported on **Windows 11 / gfx1151 / HIP SDK 7.2** (five stacked bugs, ending in an
> h-row width mismatch in the draft's embedding buffer).
>
> **Fix: build the official repo instead — no patches needed.**
>
> ```bash
> git clone https://github.com/ROCmFPX/ROCmFPX.git
> ```
>
> The MTP path was reworked there (unified `n_embd_out` row widths, `t_h_nextn` reset in
> `llm_graph_result::reset()`, `ctx_other` wired centrally), which covers the whole chain.
> Reported and verified on that configuration: 24–31 tok/s, coherent output, tool calling working.
> Legacy-line patch: [PR #109](https://github.com/charlie12345/ROCmFPX/pull/109).
>
> Linux builds on the legacy line are not known to be affected.


## On the MTP variant

The MTP tensors are included so speculative decoding can be evaluated, not
because it is recommended by default. On a sibling DeepSeek-V4-Flash ROCmFP4
build, enabling MTP measured a **net loss (~0.78×)** against the same model with
speculation off — the draft acceptance rate did not pay for the extra work.

Measure it on your own hardware and workload before enabling it, and gate on
byte-identical greedy output. If you see no gain, the non-MTP variant is the
smaller file and the simpler thing to run.

## Memory

At ~90 GB these need a machine that can actually place them. On a 128 GB Strix
Halo, load them onto an otherwise-idle GPU; large models that follow other large
models can fail to fit if the previous allocation has not been released.

<!-- CREDITS:START -->

## Acknowledgements

This build would not exist without the work below. Please star and follow these
projects — the quantisation format used here is their engineering, not mine.

**[ROCmFPX](https://github.com/charlie12345/ROCmFPX) — maintained by
[`charlie12345`](https://github.com/charlie12345) / `caf`**
The ROCmFP4 / ROCmFPX tensor formats (ggml types 100–106) exist only in this fork.
Every ROCmFP4 file in this repository was produced with its `llama-quantize`, and
runs on its runtime. The fork also credits collaborators **ciru-ai**, **Tom Turney**,
**PlunderStruck** and **Aydan S.**, and acknowledges AMD for hardware support.
Licensed MIT, based on upstream llama.cpp.

**[llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml-org and contributors**
The inference engine, GGUF format and conversion tooling everything here is built on.

**[AMD ROCm](https://github.com/ROCm/ROCm)**
The compute platform these builds target — ROCm 7.2.4 on gfx1151 / Radeon 8060S.

**Base model authors** — see `base_model` in the metadata above; all model weights,
licences and capabilities are theirs. This repository contributes quantisation and
measurement only.

If you use these files, please credit ROCmFPX alongside this repository.

<!-- CREDITS:END -->
