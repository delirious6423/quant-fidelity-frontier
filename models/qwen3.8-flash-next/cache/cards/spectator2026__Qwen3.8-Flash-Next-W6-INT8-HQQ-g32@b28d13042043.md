---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- vllm
- int6
- int8
- w6a16
- hqq
- smoothquant
- quantized
- moe
- multimodal
- a100
pipeline_tag: image-text-to-text
library_name: vllm
---

# Qwen3.8-Flash-Next — W6 experts + int8 dense, HQQ g32, A100-ready

A data-light quantization of **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** packaged to serve on **NVIDIA A100 (SM80)** under vLLM: **6-bit routed experts**, **int8 dense and shared-expert layers**, and the MTP draft layers quantized the same way. It replaces our earlier experts-only W6 release; same expert files, smaller dense half, more KV room.

- BF16 ≈ 360 GB → **≈ 219 GB**; fits **TP2 on 2× A100 80GB** with fp8 KV and MTP
- **+20% KV capacity** over the experts-only build at the same memory setting (1,832,462 vs 1,524,507 tokens per TP2 engine, GMU 0.90, fp8 KV)
- KL-divergence vs BF16, as served (fp8 KV): **0.0238** wikitext / **0.0782** agentic multi-turn — see [Quality](#quality)
- Text + vision, 262,144-token context, tool calling + reasoning, MTP speculative decoding

## What is quantized

| part | format | SmoothQuant fold |
|---|---|---|
| routed experts, 48 main layers (512 × gate/up/down) | **int6**, g32, HQQ | yes (α = 0.65, up ↔ down) |
| attention / linear-attention projections | **int8**, g32, HQQ | no — no exact fold partner (the hyper-connection mixer sits in front) |
| shared experts, 48 main layers | **int8**, g32, HQQ | yes, exact through SwiGLU |
| MTP: routed experts / attention / shared expert | int6 / int8 / int8 | experts and shared expert yes |
| embeddings, `lm_head`, router, hyper-connection mixers, norms, vision tower, n-gram embedding table | BF16 | — |

## Use this model (vLLM)

Image: `vllm/vllm-openai:v0.31.0` (pin `@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`; use the CUDA 13 tag, the `-cu129` variant fails at import). The patches are fuzz-free unified diffs against that image's files.

```bash
cd serving
./make_overlays.sh      # copies the target files out of the image and applies patches/ -> ./overlays
MODEL=/path/to/this/repo CACHE=/path/to/vllm-cache GPUS=0,1 PORT=8000 ./serve.sh
```

`serve.sh` bind-mounts the patched files and the kernel library over the image and starts one TP2 engine:

```
--tensor-parallel-size 2 --dtype bfloat16 --kv-cache-dtype fp8
--mamba-cache-mode align --prefix-match-unit 8 --max-num-batched-tokens 6144
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--max-num-seqs 32 --max-model-len 262144 --enable-prefix-caching --gpu-memory-utilization 0.90
--engram-config '{"cpu_offload": true}'
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --trust-remote-code
env: OSC_W6_SO=/w6/osc_w6.so OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=<pair>
```

Set `KV_OFFLOAD_GIB=<n>` to add a host-RAM KV tier (`--kv-offloading-size`). Sampling defaults: `temperature 1.0, top_p 0.95, top_k 20` (from `generation_config.json`).

## Files

| file | what |
|---|---|
| `model-*.safetensors` | the checkpoint: int6 experts (`weight_packed` 4-bit low plane + `weight_packed_hi` 2-bit high plane + fp32 `weight_scale` per 32-group), int8 dense, BF16 rest |
| `config.json` | model config + compressed-tensors `quantization_config`: group 0 = int6 experts, group 1 = int8 dense |
| `SHA256SUMS` | `sha256sum -c SHA256SUMS` verifies every file |
| tokenizer, chat template, preprocessor and generation configs | unchanged from the base model |
| `serving/patches/` | the vLLM patches (below) |
| `serving/make_overlays.sh`, `serving/serve.sh` | build the patched files from the image; launch one engine |
| `serving/kernel/` | 6-bit Marlin MoE kernel: source, `build.sh`, prebuilt `osc_w6.so` (SM80) |
| `quantize/build_w6_group.py`, `quantize/build_best.py` | the two quantizer passes that produced this checkpoint |

## Method

Data-light: activation maxima only, no Hessians, no calibration fine-tuning. Same core for every quantized tensor:

1. **SmoothQuant fold, α = 0.65**, where an exact partner exists: per channel `s = amax|h|^α / max|W_down|^(1−α)`, `up /= s`, `down *= s`.
2. **Symmetric grid**, `QMAX = 2^(b−1) − 1` (int6: −31…31; int8: −127…127), **group size 32** along K, fp32 scales.
3. **MSE-clip init**: per group, the clip ratio (0.70…1.00, 13 points) that minimises reconstruction error.
4. **HQQ proximal refit** of the scales: lp loss, p = 0.7, 20 iterations.
5. **Pack**: int6 as a standard 4-bit plane plus a 2-bit high plane (so the stock loader and Marlin tile layout carry the low plane and only the kernel knows about the second); int8 as standard compressed-tensors int8.

Two passes reproduce it from the BF16 base: `build_w6_group.py G=32 QMAX=31 MODE=hqq ALPHA=0.65` (experts), then `build_best.py VARIANT=w6` (int8 dense, shared experts, MTP). The `STATS` inputs (per-layer activation maxima) are not included.

## A100 serving patches (vLLM v0.31.0)

| patch | why |
|---|---|
| 0002 | 6-bit expert scheme: 4-bit packing geometry plus a `*_weight_packed_hi` parameter per projection |
| 0003 | 6-bit Marlin MoE path: high planes through the call chain into `osc_w6.so` |
| 0004, 0005 | **fp8 KV on SM80**: upstream views the cache as `float8_e4m3fn`, which Triton cannot compile below SM89; the cache stays `uint8` and the attention kernel converts e4m3 bytes to bf16 exactly |
| 0006 | lets the n-gram embedding load unquantized from a compressed-tensors checkpoint |
| 0007 | host-RAM KV offloading on this model's attention ring buffer: without it the engine refuses `--kv-offloading-size`, or writes the tier and never reads it back |
| 0008 | optional: several engines on one box share ONE host copy of the n-gram table instead of one each |
| 0010 | optional: a per-process PyTorch memory cap and allocator telemetry; inert unless its env vars are set |
| 0012 | keeps FlashInfer's per-step host sync under async scheduling when speculative decoding is off (vllm#57214 removed it; without it decode can read freed pinned memory) |

## Serving notes (A100) — learned the hard way

- **Host RAM:** the BF16 n-gram embedding table (~95 GiB) is pinned in host memory **per engine**. With several engines on one box, patch 0008 makes them share one copy; without it, budget ~128 GiB of host shared memory per TP2 engine.
- **Prefix caching on this hybrid model** needs `--mamba-cache-mode align`. Recurrent state is saved only where a prefill chunk ends on a 3,200-token block, so `--max-num-batched-tokens 6144` (every block) beats 8192 (every second block: no reuse below ~10.5k-token prompts).
- **`--prefix-match-unit 8`** is required with KV offloading (it must divide the 8-token ring block); no measured decode cost versus 64.
- **MTP k = 3** is the measured optimum: k = 4 accepted more tokens yet lost 25% aggregate throughput at depth; k ≥ 5 does not boot.
- **fp8 KV was faster here, not just bigger**: the smaller BF16 KV pool thrashed the prefix cache under deep multi-turn load.
- **GMU 0.90** boots; **0.95** failed in CUDA-graph memory profiling (`cudaErrorNotPermitted`).
- **Cold start is slow**: the first boot of a fresh cache compiles for 25+ minutes. Boot engines **one at a time**, each with its own cache directory (`serve.sh` does).
- **KV offload check**: after enough load to evict, `curl -s localhost:8000/metrics | grep -E 'external_prefix_cache_hits_total|CPU_to_GPU'` must show both non-zero.
- **Thread counts on big hosts:** set `OMP_NUM_THREADS` (16 here); left unset, torch sizes its CPU pools to every core on the host.
- **2× TP2 beat 1× TP4** by 15–20% throughput per 4 GPUs in our runs.
- `osc_w6.so` matches the pinned image's torch/CUDA; for another image run `serving/kernel/build.sh` (CPU-only build inside the image).

## Quality

Mean KL-divergence of the next-token distribution against the BF16 model, as served (fp8 KV), same image for both rows:

| build | wikitext | agentic multi-turn conversations |
|---|---|---|
| **this release** (int6 experts + int8 dense) | **0.0238** | **0.0782** |
| previous release (int6 experts, BF16 dense) | 0.0223 | 0.0875 |

At GMU 0.90 one TP2 engine holds a **1,832,462-token** fp8 KV pool (+20.2%); with MTP k = 3 it averages **2.58 tokens per decode step**, and throughput at 8 concurrent sessions matched the previous release. These figures were measured on the vLLM nightly `0bfc7a15`; the serving files here target v0.31.0, the version we serve this layout on.

## License & credits

- Base model: **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** © Qwen, [Qwen Community License 1.0](./LICENSE).
- Serving patches and kernel source are derived from **[vLLM](https://github.com/vllm-project/vllm)** (Apache-2.0); see [`serving/NOTICE`](./serving/NOTICE).
- Quantization, kernel and A100 patches by **[@spectator2026](https://huggingface.co/spectator2026)**.
