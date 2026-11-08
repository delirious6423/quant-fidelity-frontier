---
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
- betelgeuse
- llama-cpp
- gguf
- moe
license: apache-2.0
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
---

<p align="center"><img src="betelgeuse.svg" width="560" alt="Betelgeuse block star"></p>

# DeepSeek-V4.1-Flash (1017) DSpark Drafter (FLE / Betelgeuse Edition)

This repository provides the official **DSpark speculative-decoding draft head** (`mtp.0..2`) extracted from DeepSeek-V4.1-Flash-1017 and converted to a specialized GGUF format for use with the **Betelgeuse** (`bel-server` / `bel-cli`) and **FLE** out-of-core MoE inference engines.

> **Important:** This is an auxiliary draft sidecar, not an independent language model. It executes 5-token speculative drafts verified by the 552B base model. Every accepted token is **100% mathematically verified** against the target model with zero quality loss.

---

## File Specifications

| File | Size | Format & Precision | Target Role |
| :--- | :---: | :--- | :--- |
| **`dspark-DeepSeek-V4.1-Flash-1017-FLE-Q2K-Q8.gguf`** | **5.2 GB** | Routed MoE: `Q2_K`<br>Dense Projections: `Q8_0`<br>Markov Head: `F16`<br>Hyper-Connections (`hc_*`): `F32`<br>Confidence Head (`conf_proj`): `F32` | 3-stage, 128-expert top-3 draft head tailored for consumer VRAM envelopes. |

### Why This Specific Quantization Recipe?
Standard quantization tools fail on the DSpark head:
1. Compressing the **Markov Head (`markov_w1`, `markov_w2`)** to K-quants introduces divergence in token-transition probabilities, dropping acceptance rates below 30%. Keeping them in **`F16`** preserves the full multi-token probability distribution.
2. The **Hyper-Connections (`hc_*`)** and **Confidence Head (`conf_proj`)** must remain in **`F32`** to avoid numerical instability and memory access violations during parallel draft passes.
3. Quantizing the routed MoE layers to **`Q2_K`** while keeping dense projections in **`Q8_0`** compresses the file from 16 GB down to **5.2 GB**, fitting comfortably inside a 12GB VRAM consumer envelope.

---

## Complete Setup & Download Guide

To run DeepSeek-V4.1-Flash on consumer hardware (e.g., RTX 3060 12GB with 16GB system RAM), you will assemble three components into a single directory:
1. **The MoE Backbone:** Choose your preferred quant (`UD-Q2_K_XL`, `UD-Q4_K_XL`, or `UD-Q8_K_XL`) from Unsloth.
2. **The Engram Table:** The modular `IQ4_NL` Engram shard (110.6 GB) from Unsloth.
3. **The DSpark Drafter:** The `FLE-Q2K-Q8` sidecar from this repository.

### 1. Set Your Local Model Directory
Define where your model files will live on your fastest NVMe mount:

```bash
export MODEL_PATH="/path/to/your/fast_nvme/DeepSeek-V4.1-Flash-1017"
mkdir -p "$MODEL_PATH"
```

### 2. Download the Model Shards

#### Step A: Download the Base Model Backbone
Choose **one** of the following quantizations from [`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF):

```bash
# Option 1 (Recommended for 16GB RAM systems): UD-Q2_K_XL (~199.6 GB across 6 shards)
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  --include "UD-Q2_K_XL/*" \
  --local-dir "$MODEL_PATH"

# Option 2 (Higher bit-depth): UD-Q4_K_XL (~297.9 GB)
# hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-Q4_K_XL/*" --local-dir "$MODEL_PATH"

# Option 3 (Maximum precision): UD-Q8_K_XL (~305.6 GB)
# hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-Q8_K_XL/*" --local-dir "$MODEL_PATH"
```

#### Step B: Download the Swappable `IQ4_NL` Engram Shard
The Engram conditional-memory table is identical across quants and only runs on Layers 1 and 14. Pair your backbone with the compact 110.6 GB `IQ4_NL` table:

```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  --include "*engram-IQ4_NL.gguf" \
  --local-dir "$MODEL_PATH"
```

#### Step C: Download the DSpark Draft Head
Download the 5.2 GB sidecar from this repository:

```bash
hf download Betelgeuse-AI/DeepSeek-V4.1-Flash-DSpark-FLE \
  --include "dspark-DeepSeek-V4.1-Flash-1017-FLE-Q2K-Q8.gguf" \
  --local-dir "$MODEL_PATH"
```

All downloaded files should now sit side-by-side in `$MODEL_PATH`.

---

## Launch Instructions

### 1. Host Preparation & Environment Variables

```bash
# Unlock GPU power ceiling to prevent thermal/power down-clocking during heavy GEMM passes
sudo nvidia-smi -pl 200

# Kernel & Out-of-Core MoE Engine Environment Flags
export MOE_PROFILE=1
export MOE_PLE_STAGING=1
export GGML_CUDA_GRAPH_CHUNK_SIZE=0
export MOE_CPU_VALIDATE=0

# Launch inside an enforced 12GB RAM container (guaranteeing zero Linux swap thrashing)
systemd-run --user --scope --unit=betelgeuse-bench.scope \
    -p MemorySwapMax=0 \
    -p MemoryHigh=12032M \
    -p MemoryMax=12G \
    -- ./run_server.sh
```

### 2. Execution Command (`bel-server`)

Inside your execution script (`run_server.sh`), start the engine with native direct-IO and DSpark speculation enabled:

```bash
./build/bin/bel-server \
    -m "$MODEL_PATH/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL-00001-of-00007.gguf" \
    -md "$MODEL_PATH/dspark-DeepSeek-V4.1-Flash-1017-FLE-Q2K-Q8.gguf" \
    --spec-type draft-dspark \
    --spec-draft-n-max 5 \
    --spec-draft-p-min 0.50 \
    -ngl 99 \
    -ngld 99 \
    -b 64 \
    -ub 16 \
    -c 512 \
    -np 1 \
    -fit off \
    --moe-direct-io \
    --no-warmup \
    --no-cache-prompt \
    --cache-ram 0 \
    -t 6 \
    -tb 6 \
    --threads-http 2 \
    --ctx-checkpoints 0 \
    --host 127.0.0.1 \
    --port 8091 \
    -lv 4
```

---

## Parameter Breakdown: What Each Flag Does

### Hardware & Power Tuning
* `sudo nvidia-smi -pl 200`: Unlocks the board power limit to 200W (on cards that support it), preventing transient voltage drops during intense tensor-core compute.
* `systemd-run ... -p MemoryMax=12G -p MemorySwapMax=0`: Runs the process in an isolated cgroup that strictly limits host RAM usage to 12 GB and bans swap. This guarantees that unbuffered reads never induce `kswapd` thrashing.

### Engine Environment Variables
* `export MOE_PROFILE=1`: Activates internal latency tracking across the MoE doorbell, NVMe I/O worker, and PCIe DMA engines.
* `export MOE_PLE_STAGING=1`: Routes the 110 GB Engram table through an active-row CUDA staging buffer. Only the 48 active 144-byte rows required per token are read via Direct-IO into GPU memory, eliminating CPU graph splits and preventing full-tensor page faults.
* `export GGML_CUDA_GRAPH_CHUNK_SIZE=0`: Disables sub-graph chunking, forcing the entire forward pass to evaluate under a single, unified CUDA Graph replay instance.
* `export MOE_CPU_VALIDATE=0`: Bypasses redundant CPU activation readbacks, preventing host synchronization stalls during decode.

### Server CLI Flags
* `-m "$MODEL_PATH/...-00001-of-00007.gguf"`: Points to the first split of the base model. The loader automatically discovers the remaining shards and the companion Engram file.
* `-md "$MODEL_PATH/dspark-...-FLE-Q2K-Q8.gguf"`: Loads this speculative draft sidecar.
* `--spec-type draft-dspark`: Engages the native DSpark drafting engine, utilizing its parallel 5-token prediction path.
* `--spec-draft-n-max 5`: Sets the speculative window to draft 5 candidate tokens per cycle.
* `--spec-draft-p-min 0.50`: **The confidence gate.** Draft positions whose confidence falls below 50% are discarded before verification. This prevents low-probability tokens from widening the union of active experts and saturating the NVMe bus.
* `-ngl 99` & `-ngld 99`: Offloads all available dense layers for both the base model and the draft model to the GPU.
* `-b 64` & `-ub 16`: Configures the batch size to 64 and micro-batch size to 16. This minimizes the allocation size of temporary CUDA scratch buffers from over 2.3 GB down to ~40 MB, leaving maximum VRAM available for the MoE slot pool.
* `-c 512`: Sets the active context window. (Increase according to your KV cache budget).
* `-np 1`: Allocates exactly one processing slot, eliminating duplicate scratchpad and KV-cache overhead in low-memory environments.
* `-fit off`: Disables automatic VRAM fitting, allowing the engine's custom memory map to strictly govern tensor placement.
* `--moe-direct-io`: Activates the Betelgeuse out-of-core MoE paging pipeline. Expert weights are bypassed from standard CUDA memory and streamed on-demand using asynchronous `io_uring` Direct-IO and pinned VRAM slot pools.
* `--no-warmup`: Skips default startup generation loops, allowing immediate initialization.
* `--no-cache-prompt`: Ensures reproducible benchmarking by evaluating fresh prompt paths without relying on cached activations.
* `--cache-ram 0`: Prevents standard CPU layer allocation, reserving host memory for the 8 GB pinned victim cache.
* `-t 6` & `-tb 6`: Allocates 6 CPU worker threads to align with the physical core count of the host CPU.
* `--ctx-checkpoints 0`: Disables full-context checkpointing, utilizing in-VRAM recurrent state updates instead.
* `-lv 4`: Sets verbose logging to output real-time expert hit rates, DMA latencies, and speculative acceptance rates.

---

## Verified Benchmarks (RTX 3060 12GB + 16GB Single-Channel DDR4)

On an Intel i5-12400 / RTX 3060 12GB running under a strict **12 GB RAM systemd cgroup**:

| Metric | Target Baseline | Betelgeuse + FLE-DSpark | Performance Delta |
| :--- | :---: | :---: | :---: |
| **Warm Prompt Processing (PP)** | ~1.2 tok/s | **61.80 tok/s** | **~51× speedup** |
| **Warm Token Generation (TG)** | ~0.5 tok/s | **18.15 – 18.92 tok/s** | **~37× speedup** |
| **Peak TG Burst Rate** | — | **Up to 19.92 tok/s** | Near theoretical compute floor |
| **DSpark Acceptance Rate** | — | **70.6% – 85.7%** | Mean length: 4.43 – 5.00 tokens/sweep |
| **Active ColdSSD Reads (Warm)** | 100% | **0.00%** | Zero disk stall (`disk_wait = 0.0 ms`) |
| **Peak Host RAM Utilization** | OOM | **10.61 GiB** | Fully stable inside 12 GB envelope |

---

## License & Attribution
* **Base Architecture:** [DeepSeek AI](https://github.com/deepseek-ai)
* **GGUF Conversion & Quantization:** Prepared by the Betelgeuse-AI team for the Betelgeuse / FLE runtime.
* **Usage Terms:** Follows the DeepSeek-V4.1 Community License Agreement.