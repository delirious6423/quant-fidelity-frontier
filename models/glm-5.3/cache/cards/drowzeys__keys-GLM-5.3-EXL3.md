---
license: mit
base_model: zai-org/GLM-5.3
tags:
- exl3
- exllamav3
- glm-5.3
- dgx-spark
- gb10
- vllm
- 1m-context
- mtp
library_name: vllm
pipeline_tag: text-generation
---

# keys-GLM-5.3-EXL3 — full GLM-5.3 (753B MoE) as a 3-bit EXL3, quantized *for* a serving container

**What this is:** the complete `zai-org/GLM-5.3` (78 layers, 256 routed experts, DeepSeek-style sparse MLA attention with indexer, one MTP block) with the **routed experts quantized to 3-bit EXL3 trellis** and everything else kept bf16/fp16. 308 GB in 41 safetensors shards. It serves on **four NVIDIA DGX Spark (GB10) with TP=4 + decode-context-parallel 4 at a 1,000,000-token context**, inside Mia AI Lab's vLLM container for the GLM-5.3-*Flash* EXL3 (`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`) plus the small patch set shipped in `serving/`.

| | |
|---|---|
| Source | `zai-org/GLM-5.3` (FP8 block-scaled, 756 GB, 141 shards) |
| Quantizer | exllamav3 **1.4.5** (`convert.py`, `-hb 16 -mb 16 -cpi 0`, calibration 250 rows × 2048) |
| Routed experts | 57,600 matrices (75 MoE layers × 256 experts × gate/up/down) at **3.00 bpw**, `out_scales: always` |
| Codebook | layer 3 = `mcg`, layers 4–77 = `mul1` (see *Codebook note*); the serving overlay handles both per layer |
| Everything else | attention (q_a/q_b/kv_a/o_proj/indexer), dense MLP (layers 0–2), shared experts, router, norms, embed, lm_head: **bf16/fp16 (unquantized)** |
| MTP block (layer 78) | **bf16**, kept — usable as an MTP speculator |
| Per-rank footprint (TP=4) | ~75 GiB weights (+~5 GiB with the MTP block loaded) |
| Context verified | needle-in-haystack PASS at 8.5K / 25K / 50K / 105K / 263K / **979K** tokens (DCP=4 sharded KV) |

## Why we made it this way

Every previous EXL3 GLM-5.3 we tried to serve on DGX Spark cost hours-to-days of patching: the published quants were made for TabbyAPI/exllamav3 (full-model EXL3, 2-bit heads, mul1 everywhere) and vLLM has no full-model EXL3 — Mia's container only implements EXL3 for the **routed experts**, with attention/dense/head native. So this time we quantized **to the container's ABI first**:

* scope = routed experts only (the only thing the overlay's `Exl3Config` quantizes), uniform integer bits in the overlay's allowed set {3,4,5,6} → **3-bit**, because 4-bit experts (≈101 GB/rank) do not fit four 128 GB GB10s next to a 1M KV cache; 3-bit ≈ 78 GB/rank does.
* `-hb 16 -mb 16`: head and MTP stay bf16 so the stock vLLM paths load them.
* suffix layout `trellis/suh/svh/<codebook marker>` exactly as the overlay's loader expects, `quantization_config.json` with `quant_method=exl3, bits=3, scope=glm53_routed_experts_only`.

It still needed a handful of patches (all in `serving/`), but they are small, mechanical, and now documented — the point of this repo.

## How it was made (4× DGX Spark "encode farm")

One 753B model does not fit a single 128 GB Spark for quantization, and exllamav3 has no multi-GPU convert. We built a rolling pipeline (`quant/orchestrate.py`):

1. **Rolling source shards.** The FP8 source is 756 GB and not layer-ordered; the orchestrator keeps a 5-shard download window (sha-checked), runs `convert.py` for **two layers at a time** in a fresh container (`--max_module L+1`, then `-r` resume), and after each step rewrites finished shards **dropping only `model.layers.L.mlp.experts.*`** (norms/attention stay — the compile step needs them). Peak disk ≈ 660 GB; permanent shards = embed/lm_head, final norm, and the layer-78 MTP shard (must exist at every resume).
2. **Remote expert-encoding farm** (`quant/exl3_farm.py` + `quant/patch_farm.py`). exllamav3's per-layer H (Hessian) capture is serial, but the 768 expert matrices per layer are independent once H is final. The patch adds remote work threads to `quantize_linears_parallel`: it ships each matrix + its finalized H over TCP (length-prefixed `torch.save`) to worker containers on the other three Sparks, which return `trellis/suh/svh/marker`. Verified bit-identical to local encoding. **One MoE layer: 49 min single-GPU → ~20–22 min on four.** Whole model: ~27 wall-clock hours including a power-cut restart.
3. **Resume-safe.** A power interruption at 60/78 lost nothing: exllamav3 checkpoints after every layer; the in-flight layer is simply redone.
4. **Compile.** `compile.py` needs `marisa_trie` (not in the base image) — `pip install marisa-trie` in the quant container.

`quant/recipe.yaml` (58,167 tensor entries) is included; `quant/build.sh` + `quant/patch_aarch64.py` build exllamav3 1.4.5 on aarch64/CUDA 13 inside the Mia container image (AVX all-reduce/CPU-MoE stubs + pause intrinsic).

### Codebook note (read before you quantize with exllamav3 ≥ 1.4)

`convert.py -r` (resume) lets the **command-line default win over the stored args** for the codebook (`("codebook", True, "mul1")` in the arg table). We passed `-cb mcg` on the first run only; every resumed 2-layer run silently quantized with `mul1`. Result: layer 3 is `mcg`, layers 4–77 are `mul1`. Quality is identical (both are 3-bit trellis codebooks), but Mia's overlay was **mcg-only** — which is why `serving/exl3.py` now detects the codebook per layer and drives the kernels' `(mcg, mul1)` flag pairs. **If you resume an exllamav3 job, re-pass every quant flag.**

## Serving on 4× DGX Spark (what `serving/` contains and why)

Base: `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` + its `start-tp4.sh` launcher (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks). Image layers we add (Dockerfiles included, build order `dcp → mul1 → v145 → keys → keys3`):

| Patch | Why |
|---|---|
| `exl3.py` (overlay) | + `mul1` marker tensors and per-layer codebook detection; markers may be 0-d scalars; `NullConfig/InferParams` stub for exllamav3 ≥ 1.x's `LinearEXL3`; **MTP block exempt from EXL3** so `SPEC_METHOD=mtp` loads the bf16 MTP experts through the stock path |
| exllamav3 **1.4.5** swap (`Dockerfile.v145`) | the image's exllamav3 0.0.43 `exl3_moe` kernel hard-rejects `mul1` (`TORCH_CHECK`); 1.4.5 has the same 29-arg signature (+`num_active`) and accepts it. Built against the same base torch, so the `.so` drops in |
| `sparse_attn_indexer.py` top-k fallback (`Dockerfile.keys`) | prefill `persistent_topk` for top-k 2048 oversubscribes the GB10's 48 SMs; Mia's runtime patch only covers the decode ("kpool") path |
| `sm120_dcp.py` + `mla_attn_dcp.py` (`Dockerfile.dcp`) | decode-context-parallel port of the SM120 sparse-MLA backend (KV sharded across TP ranks → 1M context on four GB10s; the stock stack replicates KV and tops out ~300K) |
| `patch_ablit_dsv2.py` + `ablit_runtime.py` | optional load-time o_proj transplant hook for the full-model classes (`DeepseekV2Model`/`DeepSeekMTP`); off by default |
| `patch_tp4_launcher.py` | adds `LOCAL_MODEL_DIR`/`LOCAL_DFLASH_DIR` (serve from an NFS dir) and `ABLIT_HOST_DIR` to `start-tp4.sh` |
| `keys-launch.sh`, `env.tp4.example` | our launch wrapper (drops page caches on all ranks first) and the working `.env.tp4` |

Launcher-level gotchas we hit (all in `env.tp4.example` comments): `--ulimit nofile=1048576` (NCCL "Too many open files" at 4 ranks), a rank whose fabric IP sits on the *second* CX7 port needs its own `WORKER_CX7_IF/IB`, the NFS server node must bind-mount its own export at the shared path, `gpu_memory_utilization` above ~0.82 is refused by vLLM's startup guard (the worker already holds ~14 GiB of CUDA/NCCL state at check time) — use `--kv-cache-memory-bytes` to size the KV cache explicitly.

### Measured (4× GB10, TP=4, DCP=4, CUDA graphs, fp8 KV, **MTP k=3**)

Production config: MTP k=3 speculator (the checkpoint's own bf16 MTP block), 1M context, idle serve, 256-token outputs, temperature 0.6 / top_p 0.95, thinking off. tok/s is decode only (completion tokens ÷ (wall − TTFT)); aggregate is summed across streams.

| Task | C1 per-stream | C2 agg | C4 agg | C8 agg | accepted tok/step |
|---|---|---|---|---|---|
| prose | 11.1 | 15.4 | 21.0 | 30.5 | 1.8 |
| code  | 15.0 | 24.4 | 28.4 | 41.7 | 2.5 |
| list  | 13.5 | 21.7 | 28.2 | 42.9 | 2.4 |
| essay | 15.0 | 20.6 | 26.5 | 37.9 | 2.3 |

TTFT 0.6–2.0 s across C1–C8. Decode rises with depth as the fixed per-step DCP cost amortizes (8→12 tok/s from short context to 1M). Prefill ≈ 320–410 tok/s is the limiter at depth: a 979K prompt is ~52 min to first token. Speculative gain is acceptance-bound — code/list/essay accept 2.3–2.5 draft tokens/step, prose ~1.8 — so 25–30 tok/s single-stream is not reachable on prose with this MTP head.

**DFlash2 draft (parked):** the full-model DFlash2 draft (`incoai/GLM-5.3-DFlash2`) can be made to *boot* at DCP=4 (see `serving/` patches: replicate the sliding-window KV per rank, `--prefix-match-unit 16`), but vLLM has no correct decode-context-parallel path for sliding-window *attention*, so it generates garbage. Full-model DFlash2 at DCP needs a DCP-aware SWA attention kernel (upstream), not a config change. MTP k=3 is the shipping speculator.

## Quick start (4 Sparks)

```bash
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks glm53-mia && cd glm53-mia
python3 serving/patch_tp4_launcher.py start-tp4.sh          # LOCAL_MODEL_DIR + ABLIT_HOST_DIR knobs
# build the image chain on EVERY rank (see serving/Dockerfile.* ; ~1 min each on top of the Mia base)
cp serving/env.tp4.example .env.tp4                          # set rank IPs / NIC pins / LOCAL_MODEL_DIR=<this repo on shared storage>
SKIP_PULL=1 SKIP_SHIP=1 SKIP_BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 SKIP_OVERLAY_VERIFY=1 ./start-tp4.sh
```
OpenAI-compatible API on `:8888`, served model name `GLM-5.3-EXL3`. Sampling: temperature 1.0, top_p 0.95, thinking on (GLM defaults).

## Lessons for "quantize with the serving container in mind"

1. Read the serving overlay's **loader** first: allowed scope, bits set, tensor suffixes, marker dtype/shape, codebook — and quantize to that ABI, not to the quantizer's defaults.
2. Keep anything the container serves natively (attention, dense, head, MTP) unquantized; the memory cost is small next to the experts and it removes whole classes of patches.
3. Verify tensor suffixes after the **first resumed** step, not at the end (see the codebook note).
4. Budget memory per rank for weights + KV + the ~14 GiB the process holds before vLLM measures anything; size KV explicitly.
5. A per-layer encode farm turns a multi-day single-GPU convert into an overnight job on the same boxes that will serve the model.

## Credits

Zhipu / zai-org for GLM-5.3 · turboderp for exllamav3 · Mia AI Lab for the GB10 EXL3 vLLM container and launcher · Blackfrost-Research for the DERISKED GLM-5.3 (source of the optional ablit donors; **not applied** in these weights) · Quantized and served by @drowzeys on a 4× DGX Spark cluster, 2026-09.

License: the weights inherit the `zai-org/GLM-5.3` license (MIT). Patches in `serving/` are MIT; they modify Apache-2.0 vLLM/SGLang/exllamav3 code — see their headers.
