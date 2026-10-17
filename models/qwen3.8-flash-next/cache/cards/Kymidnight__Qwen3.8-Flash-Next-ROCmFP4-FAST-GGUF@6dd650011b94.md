---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- rocmfp4
- rocmfpx
- vulkan
- strix-halo
- qwen4exp
---

# Qwen3.8-Flash-Next ROCmFP4-FAST GGUF

ROCmFP4 quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
sized for the 96 GiB VRAM carve-out of a Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S).
83.65 GiB across 5 shards, 4.06 bpw.

**This is an experimental build, made for speed on one machine rather than for quality.**
ROCmFPx is an experimental quant family, it is carried in a fork rather than upstream, and
this file is quantized without an importance matrix. It measures 4.6785 perplexity against
4.0068 for the unquantized model, which is a wider gap than a good 4-bit quant should have.
If you want quality, use a mainstream quant; if you want ROCmFP4 kernels on RDNA3.5, this
is what it is for.

## Setup

`qwen4exp` and the ROCmFPx quant types are not in upstream llama.cpp yet, so build this branch:

```bash
git clone https://github.com/LaurentZuijdwijk/llama.cpp
cd llama.cpp && git checkout vulkan/qwen4exp-rocmfpx
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

Point at the first shard; the rest follow automatically.

```bash
./build/bin/llama-cli \
  -m Qwen3.8-Flash-Next-ROCmFP4-FAST-00001-of-00005.gguf \
  -ngl 99 -c 32768
```

Runs fully on the GPU: **~85 GiB VRAM**, negligible host RAM. KV is roughly 24 KiB
per token.

The n-gram table ships as 16 per-head tensors of ~1.3 GiB rather than one 20.9 GiB
tensor. A single tensor that size is past `maxStorageBufferRange` (4 GiB on most Vulkan
devices), so joined it can only ever live on the host - and on a machine whose VRAM
carve-out leaves less than 21 GiB for the host, that means swap.

## Quantization

| tensors | type | bpw |
|---|---|---|
| MoE experts, attention, GDN, hyper-connections | `Q4_0_ROCMFP4_FAST` | 4.25 |
| n-gram PLE table (51.2 B params) | `Q3_0_ROCMFPX` | 3.50 |
| `token_embd`, `output` | `Q6_K` | 6.56 |

## Perplexity

wikitext-2 raw, 145 chunks at `-c 2048`.

| build | PPL |
|---|---|
| unquantized reference (as reported in PR 27742) | 4.0068 +/- 0.02271 |
| **this file** | **4.6785 +/- 0.02780** |

Quantized without an importance matrix. An imatrix build may follow and should close
much of that gap at the same size.

## Converting other qwen4exp GGUFs

Files built the upstream way carry the table joined. This fork reads them, but the
table stays host-side. To split it per head:

```bash
python gguf-py/gguf/scripts/gguf_split_ple_heads.py in-00001-of-000NN.gguf out.gguf
```

Head bounds come from the file's own KV, and the quantized bytes are copied through
untouched - no dequantize, no requantize, no quality change. Works on any quant and on
split inputs. Verified on unsloth's UD-IQ4_XS, which goes from OOMing a 30 GB host to
88.6 GiB fully resident on the GPU.

## MTP draft head

`mtp/` holds the model's own multi-token-prediction head, 2.27 GiB, exported from the same
checkpoint. Qwen trains it jointly with the target, so it drafts better than a separate
small model would.

```bash
./build/bin/llama-server \
  -m Qwen3.8-Flash-Next-ROCmFP4-FAST-00001-of-00005.gguf \
  -md mtp/Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST.gguf \
  -ngl 99 --n-gpu-layers-draft 99 \
  --spec-type draft-mtp --spec-draft-n-max 3 -c 32768
```

Measured on a Radeon 8060S, 250 tokens at temp 0, each config warmed up first:

| draft | t/s | acceptance |
|---|---|---|
| none | 28.1 | -- |
| n-max 2 | 31.8 | 0.695 |
| n-max 3 | 32.4 | 0.612 |

It is quantized to match the target rather than above it. A Q8_0 draft measured *worse* on
both throughput and acceptance and cost 1.5 GiB more: acceptance is the draft agreeing with
the target, and two models quantized the same way are wrong in the same places.

Adds ~2.3 GiB to the ~85 GiB the target uses.

## Credits

`qwen4exp` support is the work of **Daniel Han**
([@danielhanchen](https://github.com/danielhanchen)), from
[ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742) - an
unmerged draft. If it lands upstream, prefer upstream.

Quant formats hand-ported from [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX).
Base model by the Qwen team.

## Not included

Vision tower.

## License

Qwen Community License 1.0, included as `LICENSE`.
