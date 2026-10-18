---
license: mit
library_name: cortiq
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q2tp
  - q4tp
  - mixed-precision
  - moe
  - hybrid-attention
  - 2-bit
  - 4-bit
language:
  - en
  - zh
---

# GLM-5.3-Flash Q2TP and Q4TP — universal CPU/GPU CMF

This repository contains the **text tower** of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
converted directly from the official FP8 checkpoint to one memory-mapped CMF
file. The same artifact is mmap-backed for CPU and Cortiq's portable GPU
backends; inference does not require Python, PyTorch, a CUDA toolkit, or
full-model GPU residency. This release was validated on CPU and Vulkan;
Metal validation is still pending.

The upstream model is a 320B/18B-active native multimodal MoE. This release is
text-generation only: it intentionally omits the vision tower and the optional
speculative MTP block. `cortiq info` reports 313.33B retained text parameters,
45 trunk layers (34 linear KDA and 11 full DSA), a 154,880-token vocabulary,
and 37,534 payload tensors.

```bash
hf download infosave/GLM-5.3-Flash-cmf glm-5.3-flash-q2tp.cmf --local-dir .
cortiq verify glm-5.3-flash-q2tp.cmf
cortiq run glm-5.3-flash-q2tp.cmf \
  --prompt "What is 2+2? Answer briefly." --no-think
```

Replace `q2tp` with `q4tp` in these commands for the larger,
quality-oriented profile.

The file requires a GLM-5.3-compatible Cortiq build containing the dedicated
`glm5_next` runtime. Older binaries reject this architecture instead of
silently treating it as a generic Transformer. A packaged version will be
listed here after the remaining GPU release gate.

## File and integrity

| file | profile | exact size | status |
|---|---|---:|---|
| `glm-5.3-flash-q2tp.cmf` | mixed q2tp + q4tp + q8_2f + f16 | 116,254,068,322 bytes (108.27 GiB) | 37,534/37,534 tensor hashes pass |
| `glm-5.3-flash-q4tp.cmf` | mixed q4tp + q8_2f + f16 | 167,164,530,274 bytes (155.68 GiB) | 37,534/37,534 tensor hashes pass |

SHA-256:

```text
818d997b7c19dd0da1b3424ed12d869e8ff6fc4a187154c23b08f2efc4198cde  glm-5.3-flash-q2tp.cmf
6bd350591e1efd9dd098a189abf603467eed8387ee38d28f805e0891cedf36c6  glm-5.3-flash-q4tp.cmf
```

Neither file is a uniform low-bit dump. In Q2TP, routed/shared expert gate and
up projections use q2tp while their down projections remain q4tp. In Q4TP,
all large routed/shared expert projections use q4tp. Both profiles preserve
always-active KDA/DSA attention projections and both vocabulary edges as
q8_2f to protect recurrent error and input-channel outliers. Small norms,
routers, convolution taps, mHC controls, and other sensitive nonlinear control
tensors remain f16.

The source checkpoint's fine-grained E4M3 weights are decoded with their
F32 `weight_scale_inv` planes using the official 128×128 tile orientation,
then quantized independently. The CMF embeds the tokenizer, exact chat
template, tensor directory, architecture descriptor, and per-tensor hashes.

## Exact GLM-5.3 runtime

Cortiq executes the release architecture explicitly rather than approximating
it with a generic decoder:

- four-stream Manifold-Constrained Hyper-Connections and final stream mixing;
- 34 KDA linear-attention layers and 11 DSA/MLA full-attention layers;
- DSA indexer pooling, selected-token cache policy, and ordinary-context exact
  full-attention fallback;
- three dense SwiGLU layers followed by 288-expert routed MoE layers, top-8
  sigmoid routing, correction bias, scaling, and the unconditional shared
  expert;
- untied input embedding and `lm_head`, partial RoPE, and the native
  one-million-token architecture limit;
- the upstream Jinja chat template, including `reasoning_effort` and the
  direct-answer `<think></think>` path used by `--no-think`.

The release oracle compares real source FP8 tensors with the CMF runtime. For
the first routed layer, Q4TP output has source correlation `0.997998` and
relative RMS error `6.36%`. The deterministic CPU gate renders the official
24-token direct-answer prompt and emits token 19 (`4`), then the model stop
marker. `cortiq verify` independently checks every payload hash.

## Models larger than VRAM

Neither the 108.27 GiB nor the 155.68 GiB file needs to fit in VRAM. CMF keeps weights mmap-backed
in host memory and detects the available adapter budget. When dynamic pooling
is enabled, routed experts use a bounded global GPU pool: resident experts run
on the GPU, while cache misses are completed exactly on CPU and accumulated
into the same MoE result. Automatic mode and systems without a supported
adapter use the same file on CPU.

Useful controls:

```bash
cortiq gpu
CMF_GPU=0 cortiq run glm-5.3-flash-q2tp.cmf --prompt "CPU check" --no-think
CMF_GPU_VRAM_MB=16000 CMF_GLM_DYNAMIC_MOE=1 \
  cortiq run glm-5.3-flash-q2tp.cmf \
  --prompt "bounded forced GPU pool" --no-think
CMF_GPU_ADAPTER=0 cortiq run glm-5.3-flash-q2tp.cmf \
  --prompt "select adapter 0" --no-think
CMF_GLM_DYNAMIC_MOE=0 cortiq run glm-5.3-flash-q2tp.cmf \
  --prompt "force exact CPU MoE" --no-think
```

Automatic mode deliberately keeps both profiles on the exact CPU MoE path. On
the measured host, bounded Vulkan expert pooling was slower for both Q2TP and
Q4TP, so it is not selected automatically. Explicit `CMF_GLM_DYNAMIC_MOE=1`
remains available for compatibility and experimentation; `CMF_GLM_DYNAMIC_MOE=0`
always disables it. The GLM-specific pool is limited to 20–40% of the configured
physical-memory envelope, and an explicit `CMF_GLM_EXPERT_SLOTS` value cannot
bypass that cap.

The final runtime pass also reduces dispatch overhead without changing model
bytes or arithmetic. Q4TP combines compatible shared-input projection groups
into one tiled dispatch (only when every member uses Q4TP), while retaining
independent routed and shared-expert accumulation. Q2TP batches all selected
expert gate/up rows and the weighted down rows through the existing bounded
worker path, reusing thread-local scratch. The Q2TP CPU MoE cadence fell from
about 1,577 serial jobs per token to 527–530; greedy output and the official
template oracle remained unchanged.

Measured on an RTX 3090 with dual AMD EPYC 7H12 CPUs and 32 CPU worker threads:

| execution path | steady decode | notes |
|---|---:|---|
| Q2TP CPU32, explicit CPU | **2.955 tok/s** | 20-token clean run, 4,071 allocations/token, 530 pool dispatches/token, zero GPU submissions |
| Q2TP automatic (warm CPU32) | **3.063 tok/s** | 20-token clean run, exact CPU path, zero GPU submissions/uploads |
| Q2TP forced dynamic Vulkan, normal budget | 1.240 tok/s | 752 slots / 6,392 MB pool, 42 submissions/token, 7,564 MiB peak |
| Q2TP forced dynamic Vulkan, 16,000-MB target | 1.302 tok/s | 512 slots / 4,352 MB pool, 42 submissions/token, 6,563 MiB peak |
| Q4TP CPU32, final runs A/B | **2.685 / 2.648 tok/s** | mean **2.666 tok/s**, 5,248 allocations/token, zero GPU submissions |
| Q4TP automatic | same exact CPU path | no dynamic pool, zero GPU submissions/uploads |
| Q4TP forced dynamic Vulkan, normal budget | 1.278 tok/s | 512 slots / 6,400 MB pool, 42 submissions/token, 6,731 MiB peak |
| Q4TP forced dynamic Vulkan, 16,000-MB target | 1.430 tok/s | 344 slots / 4,300 MB pool, 42 submissions/token, 6,475 MiB peak |

The Q4TP CPU result varies with NUMA placement; the final A/B mean is reported
instead of presenting one noisy maximum. A warm Q4TP regression run measured
2.614 tok/s and remained within that accepted spread. Forced Vulkan remains an
explicit compatibility path: it is slower than exact CPU for both profiles,
which is why neither profile selects it automatically.

For both profiles, bounded compatibility runs with
`CMF_GPU_VRAM_MB=16000 CMF_GLM_DYNAMIC_MOE=1` completed successfully with the
official-template answer `4`. The Q2TP 4,352-MB/512-slot pool peaked at 6,563
MiB, and the Q4TP 4,300-MB/344-slot pool peaked at 6,475 MiB, both safely below
the simulated 16,000-MB target. These samples were taken on the physical
24,576-MiB (24-GB) RTX 3090; the 16,000-MB value is an explicit software
budget, not the card's capacity. The automatic Q2TP normal-budget run also
emitted `4`, stayed on exact CPU, and peaked at 331 MiB of VRAM. CPU-only
systems and smaller adapters remain valid through the same exact fallback.
Metal will be tested from these immutable artifacts.

## Scope and license

This CMF is a redistribution/quantization of the official
[GLM-5.3-Flash checkpoint](https://huggingface.co/zai-org/GLM-5.3-Flash).
The upstream model card is MIT-licensed. Refer to the upstream repository for
model limitations, intended use, citation, and the authoritative training and
benchmark description.
