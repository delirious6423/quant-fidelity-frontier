---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - qwen3.8
  - qwen
  - 27b
  - gguf
  - amd
  - rocm
  - gfx1151
  - strix-halo
  - iu4
  - w4a4
  - kairic-edge
  - promptforge
  - dualview
  - mtp
  - local-inference
---

![Kairic Edge — Qwen3.8-27B IU4](assets/kairicedge.png)

# Qwen3.8-27B-IU4-KAIRIC-EDGE

<!-- kairic-edge-v1.3-patch-note -->
## September 24, 2026 — v1.3 runtime patch

The improved [Kairic Edge v1.3 runtime](https://github.com/ciru-ai/ROCmFPX/releases/tag/kairic-edge-qwen38-27b-v1.3) is available.
**The GGUF and all three sidecars are unchanged.** This update preserves
v1.2's strict M65 verification and adds selected Ciru, charlie12345/ROCmFPX,
and upstream llama.cpp correctness fixes.

The runner now defaults to MCOL8 and CK1. Separate same-binary cold-prefill
screens measured **+6.47%** for MCOL8 and **+2.85%** for CK1 at MCOL8 on a
3,917-token prompt, with identical text and all 128 generated token IDs.
These workload-specific gains are not a compounded speed claim. The final
build passed 360 focused backend cases; baseline/candidate cached generation
throughput remained within 0.3% on two frozen prompts.

Use the [v1.3 source and build guide](https://github.com/ciru-ai/ROCmFPX/blob/kairic-edge-qwen38-27b-v1.3/docs/kairic-edge-gfx1151.md)
and [updated runner](https://huggingface.co/jcbtc/Qwen3.8-27B-IU4-Kairic-Edge/resolve/main/runtime/v1.3/run-kairic-edge-gfx1151.sh).
Set `CONTEXT=32768` to match the new qualification on Radeon 8060S / gfx1151.
The existing 256K and full-suite results below remain historical; neither
was rerun for v1.3. The downloadable binary snapshot requires the documented
NixOS/TheRock dependencies; source builds are recommended for other systems.

See [v1.3 patch notes](https://huggingface.co/jcbtc/Qwen3.8-27B-IU4-Kairic-Edge/blob/main/RELEASE_NOTES_v1.3.md) for settings,
compatibility limits, source credits, checksums, and downloads. The original
card is preserved below, including its earlier version references.
<!-- /kairic-edge-v1.3-patch-note -->

**Qwen3.8-27B-IU4-KAIRIC-EDGE** is a performance-focused Qwen3.8-27B model for AMD Strix Halo `gfx1151`. To our knowledge, this is the world's first use of an accelerated IU4 lane in an LLM on AMD `gfx1151`, and it is the first public introduction to **[Kairic.ai](https://kairic.ai/)**, an AI hardware and software optimization company building performance inference infrastructure.

Kairic Edge combines **Prompt Forge**, **Dual View**, and the accelerated IU4 lane. It connects selected 4-bit activations and weights to AMD's native unsigned/signed 4-bit matrix instruction for prompt and multi-token verification shapes. The important result is not simply a smaller model: it is a working, end-to-end native IU4 compute route inside a served 27B language model.

> **Custom runtime required.** Standard llama.cpp does not understand the Kairic sidecars or `--kairic-edge`. Build and run the immutable [Kairic Edge v1.2 source release](https://github.com/ciru-ai/ROCmFPX/tree/kairic-edge-qwen38-27b-v1.2). The bundled runner enables Kairic Edge, prompt caching, the qualified 256K configuration, and native MTP4 by default.

## v1.2 correctness update

Kairic Edge v1.2 keeps 24/64/64 n-gram drafting but changes exact 65-row verification to the compact authoritative path. The earlier native IU4 M65 verifier could select a different greedy token from M1/no-spec decoding on a reproduced low-margin case. Speculative decoding must change speed, not the target model's answer, so the native M65 specialization is no longer a production default.

On the frozen 1,967-token structured-generation repro, v1.2 produced the exact target response hash in all six runs. Five warm runs completed in **22.31–22.38 seconds** (mean **22.34 seconds**), accepted **9,255/9,280 drafted tokens (99.73%)**, and retained a mean accepted length of **64.83 tokens**. The strict route measured approximately **5–8% below** the unsafe native M65 verifier while remaining about **7.19× faster** than speculation off on this workload.

The model and three `.pfs` files are unchanged. v1.2 is a runtime correctness release. PromptForge prefill, M1 target decode, M2–M5 MTP, prompt caching, context size, and API behavior are unchanged.

Release source: [`kairic-edge-qwen38-27b-v1.2`](https://github.com/ciru-ai/ROCmFPX/tree/kairic-edge-qwen38-27b-v1.2), commit [`205a3e5f40e5542e2f2eb68e3d3f81f918b1d895`](https://github.com/ciru-ai/ROCmFPX/commit/205a3e5f40e5542e2f2eb68e3d3f81f918b1d895). Full details are in [`RELEASE_NOTES_v1.2.md`](https://huggingface.co/jcbtc/Qwen3.8-27B-IU4-Kairic-Edge/blob/main/RELEASE_NOTES_v1.2.md).

## Release highlights

- **v1.2 makes exact-M65 verification target-equivalent by default** while retaining 64-token n-gram drafts and 99.73% acceptance on the reproduced stability gate.
- **47.73 generated tokens/s** across the 164-task coding suite, **85.0% above Unsloth Dynamic Q4** and **88.6% above Unsloth Dynamic Q6** in the collected configured-system runs.
- **158/164 HumanEval Base** and **152/164 HumanEval Plus**: Base tied the Q4 comparison and Plus passed four more tasks; it also passed one more Base and two more Plus tasks than the Q6 comparison.
- The native IU4 instruction harness reached **104.66 TOPS**, or **1.94×** the matched FP16 matrix result and **1.93×** the matched IU8 result on the same device and run.
- The inclusive IU4 feed-forward operator was **2.52–3.48× faster** across tested prompt shapes; pooled prompt processing increased **56.03%** in the controlled lane A/B.
- The complete native companion set is **9.13 GiB / 46.3% smaller** than the matched 8-bit companion inventory, despite including an additional projection family.
- The validated prompt cache cut repeated-prefix prompt time by **98.39–99.87%** from 2K through 32K tokens.

## Prompt Forge and Dual View

Kairic Edge is a **Dual View** model. Its GGUF is the authoritative view for model storage, quality-sensitive selection, target decode, and unsupported shapes. Three `.pfs` companions provide phase-specialized execution views for eligible feed-forward, recurrent-projection, and output-projection work.

**Prompt Forge** is the runtime layer that loads those companion views, identifies the physical request shape, and routes only qualified prompt or verification operations through the faster view. It fails closed to the authoritative path when a shape or operation is outside the validated envelope. This lets the model favor a compact authoritative representation for bandwidth-sensitive work while using a hardware-native representation where wider matrix operations can exploit it.

Dual View does have a memory cost: the accelerated companions add 10.57 GiB beside the 15.48 GiB GGUF. The gain is a phase-specialized compute path without making the accelerated view authoritative for every operation.

## Why the IU4 lane matters

Many “4-bit” releases use four bits for storage but expand weights into wider arithmetic before the expensive matrix operation. Kairic Edge routes supported shapes through AMD RDNA 3.5's documented `V_WMMA_I32_16X16X16_IU4` instruction: packed unsigned 4-bit activations × signed 4-bit weights, with 32-bit integer accumulation and explicit scale/zero-point reconstruction.

That distinction matters because it turns low precision into a compute advantage, not just a capacity advantage. On Radeon 8060S / `gfx1151`, the arithmetic advantage survived packing, transforms, corrections, output conversion, model routing, and serving. The release is deliberately hybrid: unsupported or behavior-sensitive operations stay on their qualified fallback paths.

To our knowledge, this is **the world's first production-facing use of an accelerated IU4 lane in an LLM on AMD `gfx1151`**. This is not a claim that every operation in the model runs natively at four bits, that M1 decode is native IU4, or that this is a vendor-wide or energy-efficiency result.

## Coding evaluation and throughput

The table reports our collected 164-task, chat-adapted HumanEval/EvalPlus harness from the preceding v1/v1.1 configured system. Every arm ran on the same AMD Ryzen AI Max+ 395 / Radeon 8060S `gfx1151` host with one slot, F16 target/draft KV, batch 2048, ubatch 512, 16 target threads, 32 batch threads, native MTP4, deterministic sampling, one trajectory per task, and no repair retry.

| Release | Base | Plus | Aggregate PP | Aggregate TG | Peak TG | Generation time |
|---|---:|---:|---:|---:|---:|---:|
| **Kairic Edge IU4 v1/v1.1** | **158/164 (96.34%)** | **152/164 (92.68%)** | **358.45 tok/s** | **47.73 tok/s** | **106.68 tok/s** | **950.45 s** |
| Unsloth Dynamic Q4 | 158/164 (96.34%) | 148/164 (90.24%) | 314.14 tok/s | 25.80 tok/s | 30.00 tok/s | 1,778.27 s |
| Unsloth Dynamic Q6 | 157/164 (95.73%) | 150/164 (91.46%) | 260.29 tok/s | 25.31 tok/s | 27.99 tok/s | 1,732.38 s |

Relative to the collected Q4 and Q6 runs, Kairic Edge delivered:

| Comparison | Aggregate TG | Generation time | Peak TG | Base delta | Plus delta |
|---|---:|---:|---:|---:|---:|
| vs. Unsloth Dynamic Q4 | **+85.03%** | **−46.55%** | **3.56×** | tie | **+4 tasks** |
| vs. Unsloth Dynamic Q6 | **+88.55%** | **−45.14%** | **3.81×** | **+1 task** | **+2 tasks** |

These are **configured-system comparisons**, not a one-variable quantization experiment. Kairic Edge used its release configuration at 262,144 context with an 8 GiB prompt cache and 32 context checkpoints; the comparison runs used 65,536 context without that cache allocation. Generation throughput is the most useful cross-run signal, but the different complete configurations must remain visible. This chat-adapted harness is also not directly comparable to a published EvalPlus leaderboard score.

The 164-task Kairic run was collected before the v1.2 strict-M65 correction and is retained as historical configured-system evidence, not relabeled as a fresh v1.2 speed run. v1.2 preserves the target model's greedy output, but exact-M65-heavy workloads can be approximately 5–8% slower than the unsafe native verifier. The v1.2 release gate instead freezes target equivalence, draft acceptance, and the bounded performance cost on the reproduced failure.

### Comparison artifact identity

The acquired comparison files are from Unsloth Dynamic 3.0 at revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`. We use short public labels in the tables and publish exact identity here.

| Public label | Source file | Bytes | SHA-256 |
|---|---|---:|---|
| Unsloth Dynamic Q4 | `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16,464,440,224 | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| Unsloth Dynamic Q6 | `Qwen3.8-27B-UD-Q6_K_XL.gguf` | 25,299,061,664 | `701d8fa9ed214ab21bfc130cd2a7df19ca89bbef7713e2dfb19f3c63696aa917` |
| Shared MTP companion | `Qwen3.8-0.6B-MTP-BF16.gguf` | 1,369,590,656 | `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |

Sources: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [Unsloth Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), and [EvalPlus](https://github.com/evalplus/evalplus).

## Kairic Edge sweeps

### Native matrix instruction

Best of three HIP-event timings in a short, dependency-chain instruction harness. These are GPU instruction rates, not NPU marketing figures, sustained application throughput, or power measurements.

| Dependency chains | IU4 TOPS | IU8 TOPS | FP16 TOPS |
|---:|---:|---:|---:|
| 2 | 99.87 | 50.65 | 52.35 |
| 4 | 100.09 | 52.94 | 52.47 |
| 8 | **104.66** | **54.30** | **53.87** |

At the strongest measured point, IU4 was **1.93× IU8** and **1.94× FP16**. AMD documents the instruction in the [RDNA 3.5 ISA](https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture).

### Inclusive feed-forward operator

The measured native route includes input packing, native gate/up, activation-and-pack, native down projection, and BF16-to-F32 output. Each native value combines mirrored five-run HIP-event medians after warmup.

| Physical rows | Kairic IU4 | Compact control | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|
| 96 | 1.162 ms | 2.928 ms | **2.52×** | 60.3% |
| 128 | 1.219 ms | 3.700 ms | **3.04×** | 67.1% |
| 256 | 1.999 ms | 6.706 ms | **3.35×** | 70.2% |
| 512 | 3.801 ms | 13.222 ms | **3.48×** | 71.3% |

### Prompt-processing route sweep

Same binary, model, sidecars, and target-only protocol; the Kairic IU4 feed-forward route was the lane delta. One excluded warmup and one scored request were used at each shape.

| Physical rows | Control PP | Kairic IU4 PP | Gain |
|---:|---:|---:|---:|
| 96 | 234.64 tok/s | 325.56 tok/s | **+38.75%** |
| 160 | 260.34 tok/s | 384.17 tok/s | **+47.56%** |
| 256 | 310.97 tok/s | 488.72 tok/s | **+57.16%** |
| 512 | 321.60 tok/s | 529.22 tok/s | **+64.56%** |
| **Pooled** | **297.42 tok/s** | **464.06 tok/s** | **+56.03%** |

A follow-on recurrent projection route retained the Kairic IU4 feed-forward lane in both arms and increased pooled PP from **446.65 to 526.16 tok/s (+17.80%)**, with shape-level gains from **+6.72% to +25.16%**.

### Repeated-prefix TG128 comparison sweep

All three systems processed identical cold prompt files and generated 128 tokens. There was one request per point. The synthetic text intentionally repeats a prefix, so this measures that serving shape rather than general model speed. No Kairic cache entry was restored in the cold rows.

| Prompt | Kairic PP / TG | Unsloth Q4 PP / TG | Unsloth Q6 PP / TG | Kairic TG uplift vs. Q4 / Q6 |
|---:|---:|---:|---:|---:|
| 2K | 328.62 / 23.58 | 316.92 / 14.61 | 217.67 / 16.42 | +61.37% / +43.58% |
| 4K | 477.70 / 21.77 | 326.92 / 17.94 | 280.00 / 16.57 | +21.38% / +31.40% |
| 8K | 450.35 / **54.00** | 311.73 / 15.82 | 268.43 / 13.05 | +241.29% / +313.87% |
| 16K | 400.23 / 37.62 | 285.45 / 14.25 | 246.50 / 13.17 | +163.91% / +185.58% |
| **Token-weighted pooled** | **415.51 / 29.98** | **299.22 / 15.53** | **253.84 / 14.61** | **+93.00% / +105.19% TG** |

Pooled PP was **+38.87% versus Q4** and **+63.69% versus Q6** in this sweep. Kairic used its 262K release context; both comparison runners used 32K context.

### Served verification A/B

The original exact-M65 controlled serving A/B at 65,536 context measured **48.73 tok/s** on compact verification and **52.57 tok/s** on native IU4 verification, a **7.89%** native advantage. A later deterministic trace found that the native path could change a target greedy token despite passing the ten-task screen. v1.2 therefore promotes compact verification as the correctness-preserving default. The old native path is retained only for diagnostics through `KAIRIC_UNSAFE_NATIVE_M65_VERIFY=1`; do not enable it for correctness-sensitive serving.

The v1.2 six-run gate produced one target-identical response hash. Cold target-reference generation took **161.26 seconds**; five warm strict-M65 rows took **22.31–22.38 seconds**, with **99.73%** draft-token acceptance and mean accepted length **64.83**.

### Generation-throughput peaks

These rows come from different workloads and show the validated envelope; they must not be averaged together.

| Sweep | Aggregate / hot TG | Peak TG | Output gate |
|---|---:|---:|---|
| 164-task coding suite (v1/v1.1) | **47.73** | **106.68** | 158 Base / 152 Plus |
| Repeated-prefix cold sweep | 29.98 pooled | **54.00** | fixed 128-token generations |
| Repeated-prefix warm-cache sweep | — | **99.41** | single 8K point |
| Release-runner qualification | **123.19** | — | byte-identical 512-token output to predecessor |
| Structured-generation hot slice | 144.74–145.10 | **145.10** | byte-identical across matched arms |
| HumanEval 0–9 hot slice | 48.78 | — | 10/10 Base / 10/10 Plus; identical raw output |
| Natural prose slice | 34.88 | — | parity screen |
| Forced-512 prose slice | 30.87 | — | matched 512-token output |

### Prompt-cache sweep

The recommended cache is `--cache-ram 8192 --cache-prompt --cache-idle-slots -ctxcp 32`. The cache peaked at **7,816.823 MiB** and completed without swap, device loss, or cache-cap errors.

| Prefix | Cold prompt time | Warm prompt time | Reduction | Cached tokens |
|---:|---:|---:|---:|---:|
| 2K | 6,232.11 ms | 100.60 ms | **98.39%** | 2,044 |
| 4K | 8,574.45 ms | 102.23 ms | **98.81%** | 4,092 |
| 8K | 18,190.28 ms | 106.38 ms | **99.42%** | 8,188 |
| 16K | 40,936.04 ms | 113.34 ms | **99.72%** | 16,380 |
| 32K | 100,044.68 ms | 127.91 ms | **99.87%** | 32,764 |

Context checkpoints are required for this hybrid recurrent model. A zero-checkpoint control could store entries but could not restore the recurrent state. The cache sweep validates reuse performance, not quality equivalence: 2K and 32K generated identically, while intermediate shapes diverged later because cache restore changed batch shape; all sampled completions remained coherent.

## Files

Download all four release artifacts into one directory. The sidecars are accelerator-specific, prepacked copies of selected weights; they are not optional for Kairic Edge.

| File | Role | Bytes | GiB | SHA-256 |
|---|---|---:|---:|---|
| `Qwen3.8-27B-IU4-Kairic-Edge.gguf` | authoritative model | 16,617,792,672 | 15.48 | `360caf7381907c3eca7ac0afd1228efc016af747f3f38637fb1c7f94daabac2a` |
| `Qwen3.8-27B-Kairic-IU4-FFN.pfs` | feed-forward sidecar | 8,576,856,064 | 7.99 | `adcbb90a7b429a30a2a39043366d68320d72e8b4816a0f498e882b2f80a2ba2b` |
| `Qwen3.8-27B-Kairic-IU4-GDN.pfs` | recurrent projection sidecar | 2,019,569,664 | 1.88 | `82f931316f1c895da104915dec4697163808d06f0e6b2dc027cee7aa3afc0f0e` |
| `Qwen3.8-27B-Kairic-IU4-GDN-Output.pfs` | output projection sidecar | 756,953,088 | 0.70 | `3b07e7b176559e4402924ba0c368532fa6f02118a33c71e70974c809bf6208a3` |
| **Total** | model + native companions | **27,971,171,488** | **26.05** | — |

Artifact size is not peak runtime memory. The validated 256K, one-slot profile used unified host/GPU memory and an additional 8 GiB prompt-cache budget; reduce context and cache size if your system has less headroom.

## Build the Kairic runtime

The immutable source release and known-best runner live in the pinned [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) repository. The release was certified with TheRock `7.15.0a20260718`, AMD clang 23.0.0, GCC 13.3.0, CMake 4.4.2, Ninja 1.13.0, and `gfx1151`. The following is the portable ROCm source-build path; it is not a claim of bit-identical reproduction of the certified toolchain.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git ninja-build pkg-config libssl-dev

git clone https://github.com/ciru-ai/ROCmFPX.git
cd ROCmFPX
git checkout kairic-edge-qwen38-27b-v1.2

git clone https://github.com/ROCm/composable_kernel.git third_party/composable_kernel
git -C third_party/composable_kernel checkout fdf4bb7fcc984811cef48ce817d89aac064b984a
git -C third_party/composable_kernel apply ../../patches/composable-kernel-gfx1151-iu4.patch

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$ROCM_PATH/llvm/bin:$PATH"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
if [[ -x "$ROCM_PATH/bin/amdclang++" ]]; then
  HIP_COMPILER="$ROCM_PATH/bin/amdclang++"
else
  HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++"
fi

cmake -S . -B build-kairic -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_HIP_COMPILER="$HIP_COMPILER" \
  -DCMAKE_HIP_FLAGS='-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=5 -DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=4' \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON -DGGML_HIP=ON -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF -DGGML_HIP_FORCE_MMQ=ON -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_NATIVE=ON \
  -DAMDGPU_TARGETS=gfx1151 -DGPU_BUILD_TARGETS=gfx1151 \
  -DPROMPTFORGE_CK_ROOT="$PWD/third_party/composable_kernel" \
  -DLLAMA_BUILD_WEBUI=OFF \
  -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON

cmake --build build-kairic --target llama-server -j"$(nproc)"
./build-kairic/bin/llama-server --version
./build-kairic/bin/llama-server --help | grep -A1 -- '--kairic-edge'
```

See the release repository's [build and verification guide](https://github.com/ciru-ai/ROCmFPX/blob/kairic-edge-qwen38-27b-v1.2/docs/kairic-edge-gfx1151.md) for dependency checks, exact compiler identity, compatibility mode, and smoke tests.

## Recommended launch

The pinned runner binds to localhost by default. Give it the four artifact paths and the built server:

```bash
export LLAMA_SERVER="$PWD/ROCmFPX/build-kairic/bin/llama-server"
export MODEL_PATH="$PWD/models/Qwen3.8-27B-IU4-Kairic-Edge.gguf"
export KAIRIC_FFN_SIDECAR="$PWD/models/Qwen3.8-27B-Kairic-IU4-FFN.pfs"
export KAIRIC_GDN_SIDECAR="$PWD/models/Qwen3.8-27B-Kairic-IU4-GDN.pfs"
export KAIRIC_GDN_OUTPUT_SIDECAR="$PWD/models/Qwen3.8-27B-Kairic-IU4-GDN-Output.pfs"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

./ROCmFPX/scripts/run-kairic-edge-gfx1151.sh
```

The runner owns the qualified defaults:

```text
AMD device         ROCm0 / gfx1151
context            262,144
batch / ubatch     2,048 / 512
threads            16 target / 32 batch
slots              1
KV                  F16 target and draft
prompt cache        8,192 MiB, enabled, idle-slot persistence
context checkpoints 32
Kairic Edge         enabled
native MTP          depth 4, explicit
ngram               match/min/max 24/64/64
M65 verifier        strict compact/reference
sampler             temperature 0, top-p 1, top-k 0, min-p 0
reasoning            off
metrics              enabled
```

The default is the highest-throughput fast greedy mode. It accepts one unmodified greedy completion (`temperature: 0`, `top_p: 1`, `top_k: 0` or `1`, `min_p: 0`) and rejects sampling, penalties, probabilities, grammar-constrained tool calls, logit bias, LoRA, and reasoning budgets because those features require full target logits.

For sampling, penalties, DSH, or tool calling, restart the runner with compatibility mode enabled:

```bash
export KAIRIC_EDGE_COMPATIBILITY_MODE=1
./ROCmFPX/scripts/run-kairic-edge-gfx1151.sh
```

Then use Qwen's recommended live values at request level:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "main",
    "messages": [{"role": "user", "content": "Write a tiny ray tracer in Rust."}],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "cache_prompt": true,
    "stream": true
}'
```

Compatibility mode disables only the target greedy argmax fast path. Kairic Edge, native MTP4, prompt caching, context checkpoints, and the rest of the recommended live configuration stay enabled. In our quick release gate, sampled chat and a forced tool call both returned HTTP 200; HumanEval 0–9 passed 10/10 Base and 10/10 Plus in both modes with byte-identical raw output. Compatibility measured 41.87 versus 46.37 generated tokens/s on that short coding subset, a 9.70% reduction, so it remains opt-in. This ten-task run is a compatibility smoke test, not a leaderboard score or a universal throughput estimate.

For benchmark reproduction, keep the launcher's deterministic sampler and use the first trajectory only. Do not compare results collected with repair retries or a different chat template.

## Scope and limitations

- Validated on one AMD Ryzen AI Max+ 395 system with Radeon 8060S, RDNA 3.5, `gfx1151`, and unified memory. Other AMD targets are unvalidated.
- Kairic Edge accelerates selected prompt and multi-token verification shapes. It is not whole-model native four-bit execution and does not accelerate the current M1 target-decode path with native IU4 sidecars.
- The 104.66 TOPS result is a short instruction harness. It is not sustained model throughput, an NPU number, or a performance-per-watt claim.
- Full-suite comparisons are configured-system measurements with different context/cache allocations. They are not proof that quantization alone caused every delta.
- One slot was qualified. Multi-user concurrency, sustained power, and cross-vendor performance remain future work.
- Prompt caching needs context checkpoints for correct recurrent-state restoration. Keep `-ctxcp 32` with the recommended runner.
- Exact-M65 native IU4 verification is intentionally disabled in v1.2 because it changed a reproduced greedy output. `KAIRIC_UNSAFE_NATIVE_M65_VERIFY=1` exists only for controlled diagnostics and is not a production setting.
- The release preserves strong task results, but no finite benchmark establishes universal quality equivalence to BF16.

## Model lineage, license, and credits

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache 2.0.
- Runtime base: [llama.cpp](https://github.com/ggml-org/llama.cpp) through [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX).
- Native kernel framework: [ROCm/Composable Kernel](https://github.com/ROCm/composable_kernel).
- Comparison artifacts: [Unsloth Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
- Evaluation: [EvalPlus](https://github.com/evalplus/evalplus), adapted for the model's chat endpoint and template.
- Engineering and release validation: **Kairic.ai** and **Ciru Inference Lab**.

The model artifacts inherit the Apache 2.0 license from Qwen3.8-27B. Runtime source and third-party components retain their respective upstream licenses.

## Citation

```bibtex
@software{kairic_edge_qwen38_27b_iu4_2026,
  title        = {Qwen3.8-27B IU4 Kairic Edge},
  author       = {{Kairic.ai} and {Ciru Inference Lab}},
  year         = {2026},
  month        = {8},
  url          = {https://huggingface.co/jcbtc/Qwen3.8-27B-IU4-Kairic-Edge},
  note         = {Kairic Edge accelerated IU4 release for AMD Strix Halo gfx1151}
}
```
