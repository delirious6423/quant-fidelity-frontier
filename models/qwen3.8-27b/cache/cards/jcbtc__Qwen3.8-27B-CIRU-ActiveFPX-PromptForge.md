---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - qwen
  - qwen3.8
  - gguf
  - rocm
  - amd
  - activefpx
  - promptforge
  - mtp
  - multimodal
  - vision
---

# Qwen3.8-27B CIRU ActiveFPX + PromptForge v3

![Qwen3.8-27B CIRU ActiveFPX PromptForge](assets/qwen3827bactivefpx.png)

V3 is Ciru's consolidated quality-and-speed build of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for AMD ROCm systems.
It combines the best compatible work from the non-IU4 development lane:

- an FFN-equalized compact language-model body;
- 50 selectively retained late-writer Q6 tensors;
- the full Q8 output projection from v2;
- full-W8 PromptForge FFN and Gated DeltaNet prompt views;
- the small-row W8 route;
- draft-only Output-K8 shortlist generation followed by authoritative Q8 reranking;
- native depth-4 MTP, with no ngram subsystem; and
- the original BF16 Qwen3.8 vision tower and 5120-wide merger as a separate projector.

The recommended model is
`Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-v3-Q8-FFNEQ-LateQ6.gguf`.
The GGUF can run by itself. The `.pfs` companions require the included CIRU
runtime patch and trade additional memory for faster prompt and draft-output
paths.

> **Runtime requirement:** the accelerated configuration requires the CIRU
> ROCmFPX v2.3 base plus `runtime/qwen38-v3-output-k8-runtime.patch`. The
> performance work was developed on `gfx1151` with TheRock ROCm 7.15. A stock
> llama.cpp binary can load the GGUF alone, but it cannot use the PromptForge
> companions or the Output-K8 proxy.

## What improved over v2

The new tensor allocation strictly improved all three retained BF16-relative
numerical quality measures:

| Matched measure | Public v2 | V3 | Change |
|---|---:|---:|---:|
| Perplexity ratio vs BF16 | 1.027876 | **1.016361** | better |
| Mean KLD vs BF16 | 0.036162 | **0.031457** | better |
| Same-top token agreement | 90.637% | **91.152%** | +0.515 pp |

On the matched serving screen, v3 improved 128-token chat generation throughput
by **4.79%** and end-to-end chat wall throughput by **6.17%**. Prompt throughput
was effectively even at 2,048 tokens (-0.28%) and improved at 3,524 tokens
(+1.20%). The short 512-token prompt route regressed by 34.24%; that is a named
optimization target rather than a hidden result.

A locked, no-ngram, deterministic 20-task HumanEval/EvalPlus panel improved in
aggregate from 14/20 to **15/20** base and from 6/20 to **7/20** Plus. V3 gained
on tasks 25, 83, and 132, but introduced one genuine semantic regression on
task 130 for odd-length input. Accordingly, the frozen "no new base failure"
sub-gate did not pass even though aggregate base and Plus scores improved.

These are local matched measurements on one Ryzen AI Max+ 395 / Radeon 8060S
system. They are evidence for this release decision, not hardware-independent
leaderboard claims.

## 262K context and vision status

The canonical serving profile defaults to **262,144 tokens**. Context length is
not reduced to disguise short-prompt or allocation overhead. A 245,760-token
prompt plus generation passed without truncation or OOM on the preceding v2
artifact and the same model architecture. That long screen was not repeated on
the v3 file at publication time so the GPU could be returned to active work.

The repository includes the matching BF16 vision projector converted unchanged
from official Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its 5120-wide merger matches the
language model. Vision is enabled in the release profile on that structural and
provenance basis; an additional v3-specific image-plus-MTP runtime screen was
not run during publication.

## Release files

| File | Bytes | SHA-256 | Purpose |
|---|---:|---|---|
| `Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-v3-Q8-FFNEQ-LateQ6.gguf` | 16,578,061,472 | `32e16a2daa8b185f4308b2f59a1548986bcdf3270a18d3d1b5708b0fa58e554c` | Recommended v3 language model |
| `Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-FFN.pfs` | 17,123,004,416 | `98d4dc803914314f9b4769cb5bcd075226209803c468bfbac87b21695cea971f` | Full-W8 FFN prompt view |
| `Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-GDN.pfs` | 4,029,685,760 | `e9ccbc152492eb881026bd6adbfd1e6b7c8398b1b163319d9858d080f556a1b7` | Full-W8 Gated DeltaNet prompt view |
| `Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-Output-K8.pfs` | 715,165,696 | `f6a2d9d1c33aebfe0ec9b92838193fe885fd3cdda0c9e8b9fb5acdfde4a942b7` | Draft-only top-8 proxy; final token is reranked by Q8 |
| `Qwen3.8-27B-mmproj-BF16.mmproj` | 931,145,984 | `765e805687953f5b3625f0231b947ca630d9c67412372207b7c32eaa747c351d` | GGUF-format BF16 vision projector |
| `runtime/qwen38-v3-output-k8-runtime.patch` | 30,098 | `8ba7d721ea380b060aeb0c7595b8b05f1582c0f7bf7a4b89e0910bc8e72465d5` | Patch from public v2.3 runtime to the qualified Output-K8 runtime |

The v3 language-serving payload with all three `.pfs` companions is
38,445,917,344 bytes (35.81 GiB). The GGUF alone is 15.44 GiB. The vision
projector adds 0.87 GiB when multimodal serving is enabled. Previous v1 and v2
artifacts remain in the repository for reproducibility, but v3 is the default.

The Output-K8 proxy was extracted from the retained v1 Q4 output tensor. It is
safe with v3 because v3's authoritative Q8 output payload is byte-identical to
the qualified public-v2 payload. The proxy proposes only a shortlist; it never
replaces final Q8 token selection.

## Build the required runtime

The accelerated path is a custom ROCm build. The commands below provide a
copyable Ubuntu/Debian-family source-build path. The release itself was
validated on NixOS with a pinned TheRock ROCm 7.15 toolchain; the Ubuntu package
combination is a portable build path, not a separately benchmarked environment.

Install ordinary build dependencies and a ROCm 7.15 installation that supports
your AMD target:

```bash
sudo apt update
sudo apt install -y build-essential cmake git ninja-build python3 libcurl4-openssl-dev

export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$ROCM_PATH/bin/rocminfo" | grep -m1 gfx1151
```

Clone the public v2.3 base, apply this repository's v3 patch, and pin
Composable Kernel:

```bash
git clone https://github.com/ciru-ai/ROCmFPX.git
cd ROCmFPX
git checkout qwen3.8-activefpx-promptforge-v2.3
git am /models/runtime/qwen38-v3-output-k8-runtime.patch

git clone https://github.com/ROCm/composable_kernel.git ../composable_kernel
git -C ../composable_kernel checkout fdf4bb7fcc984811cef48ce817d89aac064b984a
```

Configure and build for the validated `gfx1151` target:

```bash
cmake -S . -B build-promptforge -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DGGML_HIP=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DGGML_HIP_FORCE_MMQ=ON -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_NATIVE=ON \
  -DAMDGPU_TARGETS=gfx1151 -DGPU_BUILD_TARGETS=gfx1151 \
  -DPROMPTFORGE_CK_ROOT="$PWD/../composable_kernel" \
  -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-promptforge --target llama-server -j"$(nproc)"
```

Change the GPU target only if your ROCm toolchain and the required kernels
support it. The advertised measurements are specific to `gfx1151`.

## Serving: full v3 configuration

Download the v3 GGUF, three `.pfs` companions, and the projector into
`/models`. Then launch with the exact non-IU4/no-ngram profile:

```bash
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH="$PWD/build-promptforge/bin:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HIP_VISIBLE_DEVICES=0
export GGML_CUDA_GRAPH_OPT=0

export PROMPTFORGE_SIDECAR=/models/Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-FFN.pfs
export PROMPTFORGE_GDN_SIDECAR=/models/Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-GDN.pfs
export PROMPTFORGE_MODE=m2048_fused_tail1476
export PROMPTFORGE_ENABLE_SMALLM_W8=1
export PROMPTFORGE_MTP_OUTPUT_K8=1
export PROMPTFORGE_MTP_OUTPUT_K8_VALIDATE=0
export PROMPTFORGE_MTP_OUTPUT_K8_PROXY=/models/Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-Output-K8.pfs

unset LLAMA_MTP_CPU_ARGMAX_FASTPATH LLAMA_MTP_BACKEND_GREEDY
unset PROMPTFORGE_IU4_SIDECAR PROMPTFORGE_GDN_IU4_SIDECAR
unset PROMPTFORGE_ENABLE_NGRAM_MOD PROMPTFORGE_ENABLE_NGRAM_M65_IU4

./build-promptforge/bin/llama-server \
  -m /models/Qwen3.8-27B-CIRU-ActiveFPX-PromptForge-v3-Q8-FFNEQ-LateQ6.gguf \
  --mmproj /models/Qwen3.8-27B-mmproj-BF16.mmproj \
  --alias main --host 127.0.0.1 --port 8080 --jinja \
  -fit off -dev ROCm0 -ngl 999 \
  -c 262144 -b 2048 -ub 2048 -fa on \
  -ctk f16 -ctv f16 -t 16 -tb 16 -np 1 \
  -ctxcp 0 --cache-ram 0 --no-cache-prompt --no-cache-idle-slots \
  --timeout 3600 --metrics \
  --spec-type draft-mtp --spec-draft-device ROCm0 --spec-draft-ngl 999 \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --spec-draft-n-max 4 --spec-draft-n-min 0 \
  --spec-draft-p-min 0.0 --spec-draft-p-split 0.10 \
  --spec-draft-backend-sampling \
  --temp 0.7 --top-p 0.95 \
  --reasoning off --reasoning-format none --reasoning-budget -1
```

At 262K, request latency and memory use are workload-dependent. The long
context ceiling is intentional; optimize short-prompt allocation and routing
without silently lowering the configured maximum.

## GGUF-only fallback

The v3 GGUF is independently usable without the three `.pfs` files. Remove all
`PROMPTFORGE_*` environment variables and use a compatible GGUF runtime. This
retains the v3 tensor allocation and Q8 output head but not DualView prompt
acceleration or Output-K8 draft shortlisting.

## Intended use and limitations

This release targets local inference, coding, agents, general assistants,
long-context work, and multimodal experimentation on AMD systems. It is a
hardware-sensitive custom runtime release. Users remain responsible for
application-specific evaluation, safety controls, and compliance.

Known release-specific follow-ups are the 512-row prompt route, the observed
HumanEval/130 odd-length failure, and a dedicated v3 vision-plus-MTP runtime
screen. These are disclosed so downstream users can choose appropriate tests.

## Credits

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [CIRU ROCmFPX](https://github.com/ciru-ai/ROCmFPX)
- [ROCmFPX](https://github.com/charlie12345/ROCmFPX)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [AMD Composable Kernel](https://github.com/ROCm/composable_kernel)
- CIRU Inference Lab

## License

The derived model artifact follows the Qwen3.8-27B Apache-2.0 license. Runtime
source and third-party components retain their respective licenses.
