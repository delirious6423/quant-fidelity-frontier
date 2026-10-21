---
license: other
license_name: deepseek
license_link: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: kingjones777
pipeline_tag: text-generation
library_name: gguf
language:
  - en
tags:
  - deepseek
  - deepseek-v4
  - rocmfp4
  - rocmfpx
  - strix-halo
  - gfx1151
  - amd
  - vulkan
  - gguf
  - moe
  - rocm
  - ryzen-ai-max
---

### Quants in this repo

| build | size | notes |
|---|---|---|
| `...-Q3-ROCmFP4-*` (4 shards) | **100.45 GiB** | lower-bit build |
| `...-Q4_0_ROCMFP4_STRIX-*` (4 shards) | **140.98 GiB** | ftype 105 `STRIX` — Strix Halo attn-K/V recipe |

Both target **AMD gfx1151 (Ryzen AI MAX+ 395 / Strix Halo)** and require a llama.cpp with the
`Q4_0_ROCMFP4_*` quant types — see [`charlie12345/ROCmFPX`](https://github.com/charlie12345/ROCmFPX).

⚠️ Note on the STRIX (105) tier: it protects attention K/V but **not** the LM head. On
large-vocabulary models we now prefer tier **102 `COHERENT`** plus explicit
`--output-tensor-type q6_K --token-embedding-type q6_K`. This STRIX build predates that finding.


# DeepSeek-V4-Flash-0731 — ROCmFP4 (Strix Halo) GGUF

This is a ROCmFP4 quant of `deepseek-ai/DeepSeek-V4-Flash-0731`, built to fit a **single AMD Strix Halo box (128 GB unified memory)** with full GPU offload. As far as I can tell it's the first ROCmFP4 quant of this model. I made it with the [ROCmFPX fork](https://github.com/charlie12345/ROCmFPX) of `llama.cpp` for the `gfx1151` (Radeon 8060S / Ryzen AI MAX+ 395) Vulkan/ROCm stack.

| | |
|---|---|
| Base model | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) |
| Quant | Q3 — mixed ROCmFP4, experts **~3.14 bpw**, **2.92 BPW** overall |
| Size | **~101 GB** (fits 128 GB unified memory with headroom) |
| Arch | `deepseek4` (sparse MoE, 256 experts, indexer/DSA attention) |
| Target HW | AMD Strix Halo `gfx1151` iGPU (Ryzen AI MAX+ 395), Vulkan RADV |
| Loader | **ROCmFPX fork** — stock `llama.cpp` cannot load ROCmFP4 tensors |

## Why I made it

A standard 4-bit GGUF of this model comes out around 141 GB, which overflows a 128 GB Strix Halo's shared pool and spills to CPU. I wanted the largest-quality quant that still **fully offloads on a single box** and stays coherent, so I mixed the expert tensors down to land it at ~101 GB.

Recipe (quantized from the F16 with the fork's `llama-quantize`):

- base type `Q2_0_ROCMFPX`
- `ffn_down_exps` → `q3_0_rocmfpx` (3.5 bpw)
- `ffn_gate_exps`, `ffn_up_exps` → `q2_0_rocmfpx` (2.5 bpw)
- attention / embeddings → ROCmFPX; norms kept in fp32

The ROCmFP4 (`_ROCMFPX`) types hold quality better than equivalent-bit k-quants on this hardware while using the FP4 paths on `gfx1151`.

## Running it

Build the [ROCmFPX fork](https://github.com/charlie12345/ROCmFPX) (`llama-server` / `llama-cli`) for `gfx1151`, then:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1
export AMD_VULKAN_ICD=RADV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json

./llama-server \
  -m DeepSeek-V4-Flash-0731-Q3-ROCmFP4-00001-of-00004.gguf \
  -dev Vulkan0 -ngl 999 -fa on -fit off --no-mmap \
  -c 8192 -n 2048 -np 1 -b 1024 -ub 512 -t 16 --poll 50 --jinja \
  --reasoning-format deepseek \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --host 0.0.0.0 --port 8084
```

Notes from getting it stable on my box:

- `-fit off` — the fork's auto-fit step crashed on this arch for me; pin `-ngl 999` and turn it off.
- `--no-mmap` — important for MoE speed. With mmap, experts page-fault per token and throughput roughly halves.
- `-c 8192` with `-b 1024 -ub 512` keeps the graph pool under its limit; larger context can overflow it.
- `-n 2048` caps runaway generations so one request can't hold the single slot forever.
- `--chat-template-kwargs '{"enable_thinking":false}'` gives fast, direct answers. Drop it (or pass `enable_thinking:true` per request) for the model's reasoning mode.
- Expect roughly **5–8 tok/s** — it's a 101 GB model on one iGPU. Use streaming for a usable feel.

## A note on MTP

This checkpoint ships a multi-token-prediction (nextn) head, and I kept those tensors in this quant. I got a working MTP inference path running on this arch and tested it thoroughly, but on this hardware/loader combination MTP nets out slightly **slower** than plain decoding — the draft head's acceptance is low and the sparse-MoE verify step can't amortize its weight reads across draft tokens. I ran it against draft depth, the probability threshold, and draft-head precision; none of them turned it into a win here. **So I ship it with MTP off.** If you want the model's advertised MTP speedup, run it on a CUDA/vLLM stack instead of this one.

## License

Derived from `deepseek-ai/DeepSeek-V4-Flash-0731`; the original model's license applies (see `license_link`). This upload is only a quantization — all capabilities and limitations are the base model's.

<!-- PEER-TABLE:START -->

## Other public builds of this model

Compiled from Hugging Face repository metadata — file sizes, shipped files, quant variant as named by each repo. **No third-party build was run or benchmarked here, so this table makes no speed or quality claim about any of them.** It is here so you can see the size and format options at a glance and pick what fits your hardware.

| Repository | Largest model file | Variant | Ships | Downloads | Likes |
| --- | ---: | --- | --- | ---: | ---: |
| [`drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32`](https://huggingface.co/drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32) | 3.44 GiB | — | safetensors | 2144 | 73 |
| [`dealignai/DeepSeek-V4-Flash-0731-CRACK-NVFP4`](https://huggingface.co/dealignai/DeepSeek-V4-Flash-0731-CRACK-NVFP4) | 3.44 GiB | NVFP4 | safetensors | 0 | 1 |
| [`auroter/DeepSeek-V4-Flash-0731-NVFP4`](https://huggingface.co/auroter/DeepSeek-V4-Flash-0731-NVFP4) | 3.54 GiB | NVFP4 | safetensors | 1504 | 3 |
| [`sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4`](https://huggingface.co/sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4) | 3.54 GiB | NVFP4 | safetensors | 952 | 11 |
| [`mmangkad/DeepSeek-V4-Flash-0731-NVFP4`](https://huggingface.co/mmangkad/DeepSeek-V4-Flash-0731-NVFP4) | 3.54 GiB | NVFP4 | safetensors | 300 | 1 |
| [`nmitchko/DeepSeek-V4-Flash-0731-Latent-Reasoning`](https://huggingface.co/nmitchko/DeepSeek-V4-Flash-0731-Latent-Reasoning) | 3.54 GiB | — | safetensors | 196 | 4 |
| [`MJPansa/DeepSeek-V4-Flash-0731-NVFP4`](https://huggingface.co/MJPansa/DeepSeek-V4-Flash-0731-NVFP4) | 3.54 GiB | NVFP4 | safetensors | 110233 | 10 |
| [`Rarri/DeepSeek-V4-Flash-0731-NVFP4`](https://huggingface.co/Rarri/DeepSeek-V4-Flash-0731-NVFP4) | 3.63 GiB | NVFP4 | safetensors | 208 | 3 |
| [`Ralii/DeepSeek-V4-Flash-0731-DSpark-Selective-Q4_K_M-GGUF`](https://huggingface.co/Ralii/DeepSeek-V4-Flash-0731-DSpark-Selective-Q4_K_M-GGUF) | 9.85 GiB | — | single model file | 23 | 0 |
| [`kingjones777/DeepSeek-V4-Flash-0731-ROCmFP4`](https://huggingface.co/kingjones777/DeepSeek-V4-Flash-0731-ROCmFP4) **(this repo)** | 31.66 GiB | ROCmFP4 | 4 model files | 217 | 1 |
| [`bullerwins/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32-GGUF`](https://huggingface.co/bullerwins/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32-GGUF) | 45.70 GiB | — | 5 model files | 2344 | 4 |
| [`julianmb/DeepSeek-V4-Flash-0731-IQ2XXS-STRIX`](https://huggingface.co/julianmb/DeepSeek-V4-Flash-0731-IQ2XXS-STRIX) | 80.76 GiB | STRIX | single model file | 423 | 1 |
| [`otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF`](https://huggingface.co/otheru/DeepSeek-V4-Flash-Strix-Halo-GGUF) | 85.26 GiB | STRIX | drafter | 2667 | 12 |
| [`Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3`](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3) | 95.29 GiB | ROCmFP3 | 2 model files | 5372 | 14 |
| [`Geometric-AI/DeepSeek-V4-Flash-0731-ROCMFPX`](https://huggingface.co/Geometric-AI/DeepSeek-V4-Flash-0731-ROCMFPX) | 95.29 GiB | ROCmFPX | single model file | 50 | 3 |
| [`Geometric-AI/DeepSeek-V4-Flash-0731-ROCmFP3-MIX`](https://huggingface.co/Geometric-AI/DeepSeek-V4-Flash-0731-ROCmFP3-MIX) | 95.29 GiB | ROCmFP3 | 2 model files | 1551 | 4 |
| [`Kevletesteur/DeepSeek-V4-Flash-0731-StrixHalo-Verified-GGUF`](https://huggingface.co/Kevletesteur/DeepSeek-V4-Flash-0731-StrixHalo-Verified-GGUF) | 96.07 GiB | STRIX | single model file | 51 | 1 |
| [`bertholomus/DeepSeek-V4-Flash-0731-DSpark-Graph8`](https://huggingface.co/bertholomus/DeepSeek-V4-Flash-0731-DSpark-Graph8) | — | — | single model file | 0 | 4 |
| [`JasonW2025/DeepSeek-V4-Flash-0731-CB-C16-NVFP4`](https://huggingface.co/JasonW2025/DeepSeek-V4-Flash-0731-CB-C16-NVFP4) | — | NVFP4 | single model file | 0 | 3 |
| [`bertholomus/DeepSeek-V4-Flash-0731-DSpark-Graph8-4xGB10`](https://huggingface.co/bertholomus/DeepSeek-V4-Flash-0731-DSpark-Graph8-4xGB10) | — | — | single model file | 0 | 1 |

*Base model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). Generated from Hub metadata; download counts move over time.*

<!-- PEER-TABLE:END -->

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
