---
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
tags:
  - mxfp4
  - fp4
  - pytorch
  - triton
  - cuda
  - mixture-of-experts
  - quantized
  - weight-only-quantization
---
# LFM2.5-24B-A2B — Packed MXFP4 Inference

A custom packed MXFP4 checkpoint and inference runner for LFM2.5-24B-A2B, supporting both fully GPU-resident execution and adaptive GPU/CPU expert offloading.

The runner uses PyTorch, Triton packed-weight kernels, FlashAttention-2, static KV/convolution state, and CUDA Graphs. Weights remain quantized during inference; a full BF16 copy of the model is not required.

**Thinking is enabled by default.** All inference and benchmark commands below preserve it.

## Model and quantization

| Property | Configuration |
|---|---|
| Architecture | LFM2.5-24B-A2B MoE |
| Layers | 40 total, including 38 MoE layers |
| Experts | 64 per MoE layer |
| Routing | Top-4 experts per token |
| Weight values | Packed E2M1 FP4 |
| Packing | Two FP4 values per byte |
| Scales | E8M0, one scale per 32 weight values |
| Activations | BF16 |
| Packed GEMV accumulation | FP32, with BF16 outputs |

This checkpoint uses a custom packed layout. Use the included `lfm25_mxfp4_auto_vram.py` runner.

Triton kernels decode packed weights during computation. They do not permanently expand the expert matrices to BF16. The current implementation uses packed GEMV kernels; it does not claim native MXFP4 tensor-core GEMM acceleration.

## Performance

Measured on an **RTX 5070 Ti Laptop GPU with 12 GB VRAM and 32 GB system RAM**:

| Engine | Warm decode throughput |
|---|---:|
| Original serial expert execution | Approximately 10.1 tok/s |
| Updated grouped execution | Approximately 25–29 tok/s |

The best measured grouped warm median was **29.15 tok/s**, compared with **10.11 tok/s** for the matched serial benchmark. A later installed-version verification measured **24.88 tok/s**.

The selected hybrid configuration had approximately:

- **1,854 GPU-resident experts**
- **578 CPU-backed experts**
- **9.8 GiB peak allocated GPU memory**
- **1.9 GiB of packed expert data in the pinned RAM cache**

These measurements used a thinking-enabled chat prompt, context capacity 128, and repeated 64-forward decode runs. Prefill was timed separately. Warm runs replayed the same token inputs after resetting model state.

**These are warm-cache benchmark results, not a guaranteed rate for every prompt.** Cold cache population, routing changes, available VRAM, PCIe bandwidth, context length, and system load affect performance.

Updated-engine cloud throughput has not yet been measured.

## What was optimized

### Grouped top-4 expert execution

The original runner executed each selected expert separately. With 38 MoE layers and four selected experts per layer, that required **152 expert graph replays per token**.

The updated engine processes all four selected experts together:

- Grouped packed GEMV for the gate projection.
- Grouped packed GEMV for the up projection.
- Activation and multiplication.
- Grouped packed GEMV for the down projection.
- A fused routing-weighted output combination.

The combination preserves the original sequential BF16 multiply/add rounding.

This reduces MoE graph launches from **152 to 38 per token**.

A grouped graph contains multiple GPU kernels; “38 graph launches” does not mean only 38 kernels execute.

### Direct reads from GPU-resident weights

Previously, even GPU-resident expert weights were copied into a staging workspace before every expert invocation.

Fully resident layers now read their existing packed GPU weights directly through persistent device pointer tables. This removes repeated expert-weight GPU-to-GPU staging copies.

GPU-resident experts selected in a partially resident layer also use their existing addresses directly.

### GPU routing inside resident-layer graphs

For a fully GPU-resident MoE layer, its graph includes:

- Router projection.
- Sigmoid scores.
- Expert bias, where configured.
- Dynamic top-k selection.
- Routing coefficients.
- Grouped expert computation and output combination.

**Expert selection is recomputed on every replay.** Capture does not freeze the selected experts.

Normal execution of these layers does not copy expert IDs back to CPU.

### Bounded pinned RAM expert cache

CPU-backed experts use a separate packed RAM cache.

Each expert triplet is stored as one contiguous byte buffer. A selected CPU expert can then be transferred with **one H2D copy instead of six separate packed-weight/scale copies**.

Pinned cache hits avoid repeatedly rebuilding that contiguous payload. The cache uses LRU eviction and has a configurable memory limit.

A pinned RAM cache hit still requires transferring the selected expert to GPU. This cache is distinct from optional GPU expert caching.

### Shared capture stream

Resident-layer capture and warmup reuse one CUDA stream, avoiding a separate retained cuBLAS workspace for every layer.

## Hybrid mode

Hybrid mode is selected automatically when the available GPU budget cannot hold every expert.

The runner first loads non-expert weights and allocates its static execution state. It then measures available VRAM, applies the configured reserve, and loads as many complete packed expert triplets as safely fit.

Placement proceeds layer by layer. This produces as many fully resident MoE layers as possible, followed by a partially resident layer when space permits.

During inference:

1. Fully resident layers route and execute entirely on GPU.
2. Partially resident layers obtain their selected expert IDs on CPU.
3. Selected GPU-resident experts are read directly.
4. Selected CPU-backed experts are retrieved from the packed RAM cache or checkpoint.
5. CPU payloads are transferred into persistent GPU staging buffers.
6. The grouped expert graph executes the selected experts together.

**CPU selection, cache management, and CPU-to-GPU copies remain outside graph capture.** Changing CPU source addresses are never captured.

Only routed CPU experts are transferred. The runner does not transfer every expert in an offloaded layer or move the entire model for each token.

The offloaded percentage alone does not determine transfer volume or throughput: actual routing and cache locality matter.

## Full GPU mode

When every expert fits within the GPU budget:

- All packed expert weights remain on GPU.
- Every MoE layer can use GPU routing and grouped execution.
- No CPU expert transfers are required.
- Resident expert weights are read directly rather than copied into staging buffers.

Full GPU execution retains the same packed format and grouped kernels.

The previous serial engine still incurred substantial overhead in full GPU mode: repeated resident-weight staging, CPU routing synchronization, and separate expert dispatch. The updated full GPU path removes those mechanisms.

Full residency requires more than the raw checkpoint size because activations, static caches, workspaces, allocator overhead, and graphs also consume memory.

## CUDA Graph lifetime and state management

CUDA Graphs remain enabled with:

```text
--compile --compile-mode max-autotune --cuda-graphs
```

The runner addresses graph-managed tensor lifetimes through:

- `torch.compiler.cudagraph_mark_step_begin()` at token-forward boundaries.
- Separate persistent buffers for hidden, query, and key normalization outputs.
- Immediate output copies outside compiled RMS functions.
- Functional compiled convolution cells, with state committed once by the caller.
- Persistent expert staging and output buffers.
- In-place reset of static KV and convolution state.

Attention continues to use the static KV cache through FlashAttention-2. Attention/cache updates remain outside the manual MoE graphs.

Profiling the updated engine demonstrated **169 CUDA Graph launches per token**:

- 38 grouped MoE launches.
- 131 compiled RMS/convolution launches.

The original path used 283 launches per token.

## Requirements

The local validated environment used:

| Component | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.14.0+cu132 |
| Triton | 3.8.0 |
| FlashAttention | 2.8.3 |
| OS | Windows |
| GPU | RTX 5070 Ti Laptop GPU, 12 GB |

Additional dependencies include:

- `safetensors`
- A working Transformers tokenizer, or the local `tokenizers` + `jinja2` fallback

Activate the Python environment containing your CUDA stack before running the script.

Windows requires compatible Windows builds of Triton and FlashAttention. Match PyTorch, CUDA, Triton, and attention packages to your environment.

FlashAttention-2 is the default. An SDPA fallback is available:

```text
--attention sdpa
```

The placement policy accommodates different VRAM capacities. The updated engine was hardware-tested on the 12 GB GPU; allocation-policy checks also covered simulated 8, 16, 24, and 32 GiB configurations. Additional hardware performance is not claimed.

## Download

```bash
git lfs install
git clone https://huggingface.co/FP8Enjoyer/LFM2.5-24B-A2B-MXFP4
cd LFM2.5-24B-A2B-MXFP4
git lfs pull
```

Ensure `model.safetensors` is the downloaded checkpoint rather than a Git LFS pointer.

The runner prefers a checkpoint beside the script. An explicit `--model` path overrides that selection.

## Run inference with thinking

### Linux / Bash

```bash
python lfm25_mxfp4_auto_vram.py \
  --model "$PWD" \
  --compile \
  --compile-mode max-autotune \
  --cuda-graphs \
  --expert-backend grouped \
  --context 1024 \
  --max-new-tokens 512 \
  --prompt "Explain how GPU and CPU expert offloading works."
```

### Windows / PowerShell

```powershell
python .\lfm25_mxfp4_auto_vram.py `
  --model . `
  --compile `
  --compile-mode max-autotune `
  --cuda-graphs `
  --expert-backend grouped `
  --cpu-pinned-cache-mib 2048 `
  --context 1024 `
  --max-new-tokens 512 `
  --prompt "Explain how GPU and CPU expert offloading works."
```

Thinking consumes generated tokens. Allow a sufficient generation budget for both reasoning and the final answer.

The context allocation is static. Increasing `--context` reserves more VRAM and may reduce expert residency. The prompt and generation budget must fit within it.

## Benchmark decode separately from prefill

```bash
python lfm25_mxfp4_auto_vram.py \
  --model "$PWD" \
  --compile \
  --compile-mode max-autotune \
  --cuda-graphs \
  --expert-backend grouped \
  --cpu-pinned-cache-mib 2048 \
  --benchmark \
  --benchmark-runs 3 \
  --context 128 \
  --max-new-tokens 64
```

Benchmark mode:

- Reports prefill and decode separately.
- Generates a decode token trace on the first run.
- Resets state and replays those inputs on subsequent runs.
- Excludes the first run from the reported warm median.
- Disables token streaming and ignores EOS for consistent forward counts.

For comparison with the original expert execution path, use:

```text
--expert-backend serial
```

Do not calculate isolated decode throughput by dividing generated tokens by a wall time that also includes prefill, validation, or model loading.

## Memory controls

| Option | Purpose |
|---|---|
| `--gpu-weight-budget-gib auto` | Select residency from measured free VRAM; default |
| `--gpu-weight-budget-gib N` | Set a total model allocation target, constrained by available memory |
| `--gpu-reserve-gib N` | Override the free-memory safety reserve |
| `--cpu-pinned-cache-mib auto` | Bound the RAM cache to at most 2 GiB and one quarter of available host RAM |
| `--cpu-pinned-cache-mib 2048` | Explicitly select the cache limit used in the laptop benchmark |
| `--cpu-pinned-cache-mib 0` | Disable pinning and use pageable staging |
| `--expert-cache-mib N` | Optional GPU LRU for nonresident experts, subject to the global GPU budget |

The default GPU reserve is the larger of **0.75 GiB** and **8% of total VRAM**.

Pinned cache allocation is lazy. Lower RAM cache limits remain functional but may reduce throughput through more frequent payload reconstruction.

## Validation

Test packed decoding and GEMV:

```bash
python lfm25_mxfp4_auto_vram.py --self-test
```

Test graph lifetimes, grouped kernels, dynamic expert IDs, and staging:

```bash
python lfm25_mxfp4_auto_vram.py --graph-self-test
```

Check checkpoint inventory:

```bash
python lfm25_mxfp4_auto_vram.py --model "$PWD" --audit
```

Validate repeated model execution before ordinary generation:

```bash
python lfm25_mxfp4_auto_vram.py \
  --model "$PWD" \
  --compile \
  --compile-mode max-autotune \
  --cuda-graphs \
  --validate \
  --context 128 \
  --max-new-tokens 64
```

Completed checks include:

- Exact comparison against the serial runner for 32 consecutive decode logits, routed expert IDs, and final KV/convolution state.
- Exact captured/uncaptured-expert comparisons across two multi-token requests.
- Exact repeated graph outputs and state after reset.
- All experts CPU-backed with forced pinned-cache eviction.
- All experts CPU-backed with pinning disabled.
- Mixed CPU/GPU payloads and dynamic GPU pointer updates.
- Actual graph-launch profiling.

Exact comparisons use the same compiled RMS/convolution arithmetic. They do not establish equivalence to a fully eager or unquantized reference model; numerical differences can affect nearly tied MoE routing scores.

## Scope

The runtime optimizations do not modify the checkpoint, quantization parameters, or chat-template file.

Current performance results demonstrate the hybrid laptop path. The updated fully GPU-resident path needs a fresh cloud benchmark.

See [LICENSE](LICENSE) for licensing terms.