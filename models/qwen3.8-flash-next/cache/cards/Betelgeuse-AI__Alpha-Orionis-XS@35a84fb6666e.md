---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Betelgeuse-AI/Alpha-Orionis-XS/blob/main/QWEN-LICENSE
tags:
  - betelgeuse
  - llama-cpp
  - gguf
  - moe
  - iq2-xs
  - gsq
  - rco
---

<p align="center"><img src="assets/betelgeuse.svg" width="560" alt="Betelgeuse block star"></p>

# Alpha-Orionis-XS

**Qwen3.8-Flash-Next, GSQ-RCO IQ2_XS, repackaged for Betelgeuse's demand-paged MoE engine.** Betelgeuse is an independent custom fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), built to run frontier-scale sparse models on a **12 GB VRAM GPU and 16 GB system RAM**, with local NVMe storage.

This is a storage-layout conversion, not a new trained model, a distilled model, or a new quantization. All 24,576 routed experts remain present. Their quantized bytes and the dense weights are copied unchanged from [ISTA-DASLab's IQ2_XS release](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ2_XS).

> **Experimental and partly broken under active development. Use `bel-server` in preference to `bel-cli`.** This custom stripped GGUF is not a standalone model and is not compatible with stock llama.cpp, Ollama, or LM Studio. Use the Betelgeuse branch containing the stripped-expert loader. No vision projector or MTP sidecar is included; this bundle is for text generation.

## Download all four runtime files

```bash
hf download Betelgeuse-AI/Alpha-Orionis-XS --local-dir ./Alpha-Orionis-XS
```

| File | Contents |
| --- | --- |
| `Qwen3.8-Flash-Next-Dense.gguf` | Model metadata, tokenizer, dense attention/recurrent weights, shared experts, routers, embeddings and LM head; routed-expert descriptors without their data payloads |
| `experts.bin` | Original gate/up/down bytes, layer-major and expert-major, with 4 KiB-aligned expert blobs for this model |
| `Qwen3.8-Flash-Next-PLE.gguf` | Unchanged second shard, containing the disk-backed per-layer n-gram embedding table |
| `profile-qwen4exp.bin` | Small hot-expert profile for slot seeding |

Exact sizes and SHA-256 digests are in [bundle-manifest.json](bundle-manifest.json). Keep these four files together. The dense filename does **not** mean that the MoE was converted into a dense model. The file retains expert shapes and types, marks `betelgeuse.stripped_experts = true`, and refers to the adjacent PLE file. The expert pack has no duplicate copy inside the dense GGUF and needs no external index file.

## Build the native fork

From the Betelgeuse source checkout on its `betelgeuse` branch, with a C++17 compiler, CMake, CUDA and liburing development files installed:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DLLAMA_BUILD_TOOLS=ON
cmake --build build -j6 --target llama-cli llama-server
```

The native binaries are `build/bin/bel-cli` and `build/bin/bel-server`. CMake target and library names remain upstream-compatible; these are not wrapper programs. This release targets Linux x86-64 and the RTX 3060 (`sm_86`), not every llama.cpp backend.

## Run the server (recommended)

```bash
export BUNDLE="$PWD/Alpha-Orionis-XS"
export MOE_HOT_K=0 MOE_PREKICK=0 MOE_SEED_BUDGET_MS=0
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 \
    build/bin/bel-server -m "$BUNDLE/Qwen3.8-Flash-Next-Dense.gguf" \
    -ngl 99 --moe-direct-io --moe-pack-dir "$BUNDLE" --moe-slots 5000 \
    --moe-profile "$BUNDLE/profile-qwen4exp.bin" --cache-warmup 0 \
    -ub 64 -np 1 -c 512 -t 6 --host 127.0.0.1 --port 8080
```

```bash
curl http://127.0.0.1:8080/completion \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Explain demand paging.","n_predict":8,"temperature":0,"cache_prompt":false}'
```

Do not expose this development server to an untrusted network without authentication and a deployment review. One paged context and one server slot per process are supported in the measured configuration.

## Run the CLI

```bash
build/bin/bel-cli -m "$BUNDLE/Qwen3.8-Flash-Next-Dense.gguf" \
    -ngl 99 --moe-direct-io --moe-pack-dir "$BUNDLE" --moe-slots 5000 \
    --moe-profile "$BUNDLE/profile-qwen4exp.bin" --cache-warmup 0 \
    -ub 64 -c 512 -t 6 -st -p "Explain demand paging." -n 8
```

The CLI shares the native inference path, but the server is the recommended interface during development.

## Paging and warmup controls

| Flag | What it controls |
| --- | --- |
| `--moe-direct-io` | Keep routed expert weights off full-model device buffers and use the demand-paged VRAM slot pool. The optimized packed demand worker uses buffered `io_uring` reads despite the historical flag name. |
| `--moe-slots 5000` | Total resident expert slots shared across layers; fewer slots save VRAM but increase misses. |
| `--moe-pack-dir DIR` | Directory containing `experts.bin`; required for this stripped bundle. |
| `--moe-profile FILE` | Hot-expert ordering for slot seeding, not a promise of a warmed RAM cache or 22 tok/s on arbitrary prompts. |
| `--auto-warm` | Opt in to up to 256 generated startup tokens plus one replay before accepting work. The historical measured startup cost was about 42 seconds. |
| `--cache-warmup N` | Set that warmup budget explicitly; `0` disables it, maximum 8192. |
| `--cache-warmup-prompt TEXT` | Choose representative startup text. Warmup clears KV/recurrent state and resets sampling state while retaining reusable expert caches. |
| `--no-warmup` | Disable ordinary graph warmup, not the separate cache-warmup option. |

The host victim cache defaults to 8 GiB within the tested process envelope; it is not extra memory outside that limit. `MOE_CACHE_MIB=N` changes its requested size and `MOE_VICTIM_CACHE=0` disables it. The dense weights stay resident, PLE rows are staged on demand, and an in-stream CUDA doorbell resolves expert misses into the VRAM pool. CUDA Graph replay and a RAM victim cache reduce repeated I/O without pruning experts or changing the router's top-10 selections.

## Historical performance, not a new bundle benchmark

RTX 3060 12 GB, 16 GB DDR4 RAM, local NVMe; Betelgeuse process constrained to 12 GiB RAM with swap disabled, context 512, 5,000 slots, one server slot.

| Configuration | Generation (tok/s) | Evidence |
| --- | ---: | --- |
| Stock llama.cpp | 0.9 | User-reported corrected baseline on the same hardware |
| Strata | 8.0 | User-reported corrected baseline on the same hardware |
| Betelgeuse cold request | 16.80 | Verified prior engine run, with automatic warmup enabled before requests |
| Betelgeuse warm repeated workload | 22.03 | Verified prior victim-cache repeat, a different run from the 16.80 result |

“Cold request” here is the historical first measured request, not a zero-warmup cold process. The 16.80 run paired with 21.85 warm; the 22.03 run paired with 16.29 first-request throughput. These values are not a single controlled pair, and the baseline measurements are not independently re-run for this publication. Cache locality and workload change performance. The supplied profile alone does not guarantee these speeds. No full benchmark suite was run as part of this storage conversion.

## Provenance, changes and licensing

- Original model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Quantization: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`; credit to Qwen and ISTA DASLab for the model and GSQ/RCO quantization.
- Betelgeuse changes: removed routed-expert payloads from the first GGUF, retained descriptors, added bundle metadata, copied those payloads into an expert-major pack, renamed the unchanged PLE shard, and supplied a hot-expert profile. No training, pruning or requantization.
- **License discrepancy:** the quantization repository declares Apache-2.0, while the base model's distributed license is **Qwen Community License 1.0**. Both [QWEN-LICENSE](QWEN-LICENSE) and [APACHE-LICENSE](APACHE-LICENSE) are supplied. The base license includes commercial Model-as-a-Service and AI Work Assistant restrictions and attribution thresholds. Do not assume unrestricted Apache rights; review the notices and resolve commercial licensing with the rights holders.
- The Betelgeuse engine is a separately licensed MIT fork of llama.cpp. Its MIT license does not relicense these weights. This project is not an official release of, or endorsed by, Qwen, ISTA DASLab, llama.cpp or Strata.

The model and software are provided without warranty. The inherited model's capabilities and safety limitations still apply. Storage-layout parity is not a new quality, safety or production-readiness evaluation.
