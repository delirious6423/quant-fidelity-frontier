---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- vllm
- int6
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

# Qwen3.8-Flash-Next — W6 HQQ g32 (int6 routed experts), A100-ready

6-bit quantization of the **routed experts** of **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, packaged to serve on **NVIDIA A100 (SM80)** under vLLM. A100 has no FP4 hardware and the BF16 checkpoint does not fit two cards, so this repo is the A100 path: int6 experts, a 6-bit Marlin MoE kernel, and seven small vLLM patches (fp8 KV cache on SM80 included).

- BF16 ≈ 360 GB → **≈ 224 GB** (209 GiB), fits **TP2 on 2× A100 80GB** with fp8 KV and MTP
- KL-divergence vs BF16: **0.0228** (wikitext) / **0.0749** (agentic multi-turn), with fp8 KV — see [Quality](#quality)
- Text + vision, 262,144-token context, tool calling + reasoning, MTP speculative decoding

## Use this model (vLLM)

Pinned image: `vllm/vllm-openai:nightly-0bfc7a15d095fe83ecc82b50561a93c177fece2d` (vLLM commit `0bfc7a15`). The patches are fuzz-free unified diffs against that image's files.

```bash
cd serving
./make_overlays.sh      # copies 7 files out of the image and applies patches/ -> ./overlays
MODEL=/path/to/Qwen3.8-Flash-Next-W6-HQQ-g32 CACHE=/path/to/vllm-cache GPUS=0,1 PORT=8000 ./serve.sh
```

`serve.sh` bind-mounts the patched files and the kernel library over the image and starts one TP2 engine:

```
--tensor-parallel-size 2 --dtype bfloat16 --kv-cache-dtype fp8
--mamba-cache-mode align --prefix-match-unit 8 --max-num-batched-tokens 6144
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--max-num-seqs 32 --max-model-len 262144 --enable-prefix-caching --gpu-memory-utilization 0.90
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --trust-remote-code
env: OSC_W6_SO=/w6/osc_w6.so VLLM_PLE_CPU_OFFLOAD=1 OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=<pair>
```

Set `KV_OFFLOAD_GIB=<n>` to add a host-RAM KV tier (`--kv-offloading-size`). Sampling defaults: `temperature 1.0, top_p 0.95, top_k 20` (from `generation_config.json`).

## Files

| file | what |
|---|---|
| `w6g-layer-{00..47}.safetensors` | int6 routed experts, one file per layer: `weight_packed` (4-bit low plane), `weight_packed_hi` (2-bit high plane), `weight_scale` (fp32, per 32-group) |
| `model-*.safetensors`, `base-model-*.safetensors` | everything left in BF16 (see Method), incl. the ~95 GiB n-gram embedding table |
| `config.json` | model config + compressed-tensors `quantization_config` (`num_bits: 6`, `group_size: 32`) |
| `tokenizer*`, `vocab.json`, `merges.txt`, `chat_template.jinja`, `*preprocessor_config.json`, `generation_config.json` | unchanged from the base model |
| `serving/patches/0001…0007` | the vLLM patches (below) |
| `serving/make_overlays.sh`, `serving/serve.sh` | build the patched files from the image; launch one engine |
| `serving/kernel/` | 6-bit Marlin MoE kernel: source, `build.py` / `build.sh`, and a prebuilt `osc_w6.so` (SM80, built in the pinned image) |
| `quantize/build_w6_group.py` | the quantizer that produced the expert files |

## Method

Only the **routed experts** of the 48 main layers are quantized (512 experts × gate/up/down). Attention, shared experts, router, hyper-connection mixers, the n-gram embedding table, embeddings, `lm_head`, the vision tower and the whole MTP layer stay **BF16** (`targets: re:.*\.mlp\.experts\..*`, `ignore: [lm_head, re:^mtp\..*]`).

Per expert, data-light (activation maxima only, no Hessians):

1. **SmoothQuant fold, α = 0.65** — per-channel `s = amax|h|^α / max|W_down|^(1−α)` from per-expert activation maxima at the down-projection input; `up /= s`, `down *= s`. Exact through SwiGLU.
2. **int6 symmetric**, values −31…31, **group size 32** along K, fp32 scales.
3. **MSE-clip init** — per group, the clip ratio (0.70…1.00) that minimises reconstruction error.
4. **HQQ proximal refit** of the scales — lp loss, p = 0.7, 20 iterations.
5. **4+2-plane pack** — value + 32 stored as a standard 4-bit plane plus a 2-bit high plane, so the stock compressed-tensors loader and Marlin tile layout carry the low plane and only the kernel knows about the second one.

`quantize/build_w6_group.py` reproduces this from the BF16 base with `G=32 QMAX=31 MODE=hqq ALPHA=0.65`. Its `STATS` input (per layer, `h_amax` of shape `[experts, moe_intermediate]`) is not included.

## A100 serving patches

| patch | why |
|---|---|
| 0001 | registers the 6-bit weight type |
| 0002 | 6-bit expert scheme: 4-bit packing geometry plus a `*_weight_packed_hi` parameter per projection |
| 0003 | 6-bit Marlin MoE path: high planes through the call chain into `osc_w6.so` |
| 0004, 0005 | **fp8 KV on SM80**: upstream views the cache as `float8_e4m3fn`, which Triton cannot compile below SM89; the cache stays `uint8` and the attention kernel converts e4m3 bytes to bf16 exactly, branch-free |
| 0006 | lets the n-gram embedding load unquantized from a compressed-tensors checkpoint |
| 0007 | host-RAM KV offloading on this model's attention ring buffer: without it the engine refuses to start with `--kv-offloading-size`, or writes the tier and never reads it back |

## Serving notes (A100) — learned the hard way

- **Host RAM:** the BF16 n-gram embedding table (~95 GiB) is pinned in host memory **per engine** (`VLLM_PLE_CPU_OFFLOAD=1`); we measured ~128 GiB of host shared memory per TP2 engine. Budget it before running several engines on one box.
- **Prefix caching on this hybrid model** needs `--mamba-cache-mode align`. Recurrent state is saved only where a prefill chunk ends on a 3,200-token block, so `--max-num-batched-tokens 6144` (every block) beats 8192 (every second block: no reuse below ~10.5k-token prompts).
- **`--prefix-match-unit 8`** is required with KV offloading (it must divide the 8-token ring block); we measured no decode cost versus 64.
- **MTP k = 3** is the measured optimum: k = 4 accepted more tokens yet lost 25% aggregate throughput at depth; k ≥ 5 does not boot.
- **fp8 KV was faster here, not just bigger**: in our runs the smaller BF16 KV pool thrashed the prefix cache under deep multi-turn load.
- **GMU 0.90** boots; **0.95** failed in CUDA-graph memory profiling (`cudaErrorNotPermitted`).
- **Cold start is slow**: the first boot of a fresh cache compiles for 25+ minutes. Boot engines **one at a time** — three cold compiles at once killed two of them. Give each engine its own cache directory (`serve.sh` does).
- **KV offload check**: after enough load to evict, `curl -s localhost:8000/metrics | grep -E 'external_prefix_cache_hits_total|CPU_to_GPU'` must show both non-zero. The tier lives in `/dev/shm` (`--ipc=host`); size it against host RAM.
- **Do not set `VLLM_USE_RUST_FRONTEND=1`** on this image: it silently drops tool calling and structured outputs and rejects `--mm-processor-kwargs`.
- **2× TP2 beat 1× TP4** by 15–20% throughput per 4 GPUs in our runs.
- `osc_w6.so` matches the pinned image's torch/CUDA; for another image run `serving/kernel/build.sh` (CPU-only build inside the image).

## Quality

Mean KL-divergence of the next-token distribution against the BF16 model:

| set | weights only (BF16 KV) | as served (fp8 KV) |
|---|---|---|
| wikitext | 0.0210 | **0.0228** |
| agentic multi-turn conversations | 0.0672 | **0.0749** |

Measured on an earlier preview build of vLLM for this architecture; on the pinned nightly image the wikitext figure re-measured at 0.0223. On the pinned image one TP2 engine at GMU 0.90 holds a **1,524,507-token** fp8 KV pool and, with MTP k = 3, averages 2.57–2.58 tokens per decode step at 8 concurrent sessions.

## License & credits

- Base model: **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** © Qwen, [Qwen Community License 1.0](./LICENSE).
- Serving patches and kernel source are derived from **[vLLM](https://github.com/vllm-project/vllm)** (Apache-2.0); see [`serving/NOTICE`](./serving/NOTICE).
- int6 quantization, kernel and A100 patches by **[@spectator2026](https://huggingface.co/spectator2026)**.
