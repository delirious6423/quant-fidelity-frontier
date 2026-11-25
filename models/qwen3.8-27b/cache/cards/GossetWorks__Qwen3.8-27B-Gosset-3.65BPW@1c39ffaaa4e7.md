---
license: other
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - gosset-packed
  - quantization
  - e8-lattice
  - 16gb-gpu
  - vision-language
---

# Qwen3.8-27B-Gosset-3.65BPW

**The current-generation flagship, resident on a 16 GB GPU.** Qwen3.8-27B
(27.78 billion parameters including its vision encoder) packed with
Gosset Packing to **3.6577 bits per weight** (all-in), 13.82 GB on disk
against ~54.66 GB for the BF16 original, and verified **100% resident**
on a single RTX 5080: **12.06 GiB** of GPU weight allocation, 3.94 GiB to
spare, zero spill to system RAM. A standard 4-bit build of this model
(GGUF Q4_K_M ≈ 17.77 GB) does not fit on a 16 GB card at all.

A pack is a compiled output, the encoder never
ships.

**Full benchmark record:** every measured run behind this pack, with the protocol printed, is published on the [Gosset Works benchmarks page](https://www.gossetworks.com/benchmarks.html).

## Why Hugging Face shows a smaller model

Hugging Face counts the elements stored in a repo's safetensors files and reports that number as the parameter count. A Gosset pack does not store weights one per element: it stores bit-packed lattice codes, each code standing for a group of eight weights, so the count comes out far below the model's real size.

This repo reads as about 5.24 billion elements. The model is Qwen3.8-27B, all 27.78 billion parameters present, encoded at 3.6577 bits per weight. Nothing is missing; the count is what compression looks like from the outside.

## Perplexity, stated plainly

WikiText-2 test, texts joined, the model's own tokenizer; BF16 anchors
measured in the same protocol, same harness:

| Protocol | This pack | BF16 | Delta |
|---|---|---|---|
| Long-chunk (24 × 2048 tokens, 49,128 scored) | 6.7379 | 6.4830 | +3.93% |
| Short-window (128 × 128 tokens, 16,256 scored) | 12.4171 | 11.6057 | +6.99% |

The hybrid architecture, 48 of 64 layers carrying recurrent
linear-attention state, is less forgiving to quantize than a plain
transformer; the encoder's calibration was built to exercise that state
path, and the long-chunk figure above is the result. The numbers are
the pack's fidelity record; judge it alongside the certification below
and the residency, which are its claims.

## Fidelity certification

Per-tensor relative reconstruction error, ‖W − W′‖F / ‖W‖F, computed
from the stored artifacts against the BF16 source tensors. Code storage
is lossless: reconstructing from the stored codes reproduces the
encoder's floating-point reconstruction to within 0.0137%.

| Component | Codec | Mean rel. error | Norm-weighted | Tensors |
|---|---|---|---|---|
| Linear attention projections | 4.25 bpw | 0.1008 | 0.0971 | 240 |
| Full attention projections | 4.25 bpw | 0.1008 | 0.0991 | 64 |
| MLP projections | 3.25 bpw | 0.1682 | 0.1685 | 192 |
| Vision attention | 4.25 bpw | 0.0901 | 0.0920 | 56 |
| Vision MLP | 3.25 bpw | 0.1708 | 0.1711 | 54 |
| LM head | 4.25 bpw | 0.0933 | 0.0933 | 1 |
| Embedding | int8 | 0.0079 | 0.0079 | 1 |
| Multi-token prediction head | BF16, unquantized | 0.0000 | 0.0000 | 15 |

## Speed (RTX 5080, fully resident)

The bundled runtime carries a fused decode engine built for this
hybrid architecture and this GPU:

- Decode: **54.64 tok/s sustained** (18.30 ms per token) at the
  default settings (`run_e8_chat.py` Step Zero v2 SM120 CUDA-Graph runtime).
- Time to first token: 146.7 ms.
- Peak VRAM: **13.48 GiB** at the default 32,768-token FP8 KV context
  (2.52 GiB free headroom on 16 GB, zero spill to system RAM).
- Streaming output verified against the reference implementation,
  outputs aligned to a relative error of 1.82e-7.
- Serving: Ollama- and OpenAI-compatible endpoints, verified.

## Speed in vLLM (stock vllm serve, plugin v2)

This pack also runs in unmodified `vllm serve` 0.22.1 through the
out-of-tree `gosset_e8_vllm` plugin (v2 `e8_mma_fast` kernel). No fork,
no patched vLLM build, and the pack files are loaded bit-for-bit as
shipped. Measured on the same RTX 5080 with `--max-model-len 8192
--gpu-memory-utilization 0.9 --max-num-seqs 16
--max-num-batched-tokens 512`, `VLLM_USE_FLASHINFER_SAMPLER=0`:

- Single-stream: **55.55 tok/s** steady-state SSE (128-token
  generations), 52.37 tok/s on the 8-warmup + 32-eval SSE measure,
  TTFT 61.0 ms.
- Batched throughput (128-in / 128-out, aggregate): **217.62 tok/s at
  c=8, 277.21 tok/s at c=16, 278.60 tok/s at c=32**. Concurrency in
  this configuration is residency-bound: the KV pool held 14,677
  tokens (1.07 GiB), about 10 concurrent slots at 8,192 context.
- Residency in this configuration: 12.15 GiB weights, peak VRAM
  15.24 GiB. Zero Gosset/E8 JIT warnings.
- **Plugin access:** `gosset_e8_vllm` is distributed by Gosset Works for enterprise and cloud evaluation. Inquiries: gossetworks.com or X @GossetWorks.

## Context window and memory scaling

Architectural ceiling: **262,144 tokens** (rope_theta 10,000,000),
preserved unchanged from the base model. The runtime defaults to a
**32,768-token FP8 KV context** and **2,048 max reply tokens** out of the box.

48 of the 64 layers are DeltaNet recurrent linear attention with O(1)
state: they consume **0 bytes of KV cache at any sequence length**.
Only the 16 full-attention layers store KV, at **64 KB per token**,
against 256 KB per token for a full-attention layout of the same
model: a 75% reduction in KV memory growth.

On a single 16 GB GPU, measured residency is 12.06 GiB static and
12.80 GiB peak, leaving about 3.2 GiB of on-card headroom. At the
64 KB/token rate above, that headroom is a budget of roughly
**48,000 tokens of context, 100% resident in FP16**. (Budget computed
from the measured residency and the KV rate; the perplexity figures
above were measured on standard short benchmark prompts.)

## Why Gosset E8 Fits

Gosset Packing replaces independent 1D scalar rounding with sphere-packing projection onto the 8-dimensional Gosset E8 lattice (the densest packing in 8D by Viazovska's Theorem) and decodes packed blocks directly in GPU registers with zero shared-memory codebook lookup tables:

- **-61.24% quantization damage at matched bitrate:** Cuts short-window WikiText-2 perplexity degradation on `Qwen3-14B` (`3.4301 BPW`) vs. matched scalar (`p = 1.21e-25`) at `42.84 tok/s` resident on an 8 GB laptop GPU vs. `Q4_K_M` PCIe spill (`3.2x - 4.0x` speedup).
- **-64.2% lower matrix condition number:** Under Born-Quantized E8 training, constraining weights inside E8 cuts weight condition number `k(W)` by `64.2%` vs. scalar INT4 QAT (`1,362` vs. `3,810`) with zero post-training shock.
- **Robotics proven:** The same codec wins `394/394` linear projections on NVIDIA `GR00T-N1.5-3B` (`1.20 GiB` for 8 GB Jetson Orin Nano, `-52.3%` physical 6-DoF servo drift).

This pack: attention projections
(linear and full) and the language-model head at 4.25 bpw, MLP
projections in both the language and vision towers at 3.25 bpw,
per-group scales, int8 embedding, and the multi-token prediction head
preserved in BF16. The encoder is activation-aware, calibrated on data
disjoint from the evaluation set.

## What's in the bundle

Packed weights (4 SafeTensors shards), the Gosset reference runtime,
an interactive chat CLI, and the tokenizer files.

## How to use it

You need an NVIDIA GPU with at least 16 GB of VRAM (verified on an RTX 5080) and about 15 GB of free disk space.

### Option 1: Gosset Runner (Recommended on Windows, No Python Setup)

Download and install the standalone Windows x64 app from [`GossetWorks/GossetRunner`](https://huggingface.co/GossetWorks/GossetRunner):

- **Direct Installer:** [`GossetRunnerSetup-1.0.3-win64.exe`](https://huggingface.co/GossetWorks/GossetRunner/resolve/main/GossetRunnerSetup-1.0.3-win64.exe?download=true) (`2.65 GiB`, SHA-256 `8924dffb003fb27b74d4b396bd0b9e8f1bc14c70802fe9558542ab1dd86d5740`)

Gosset Runner bundles its own isolated Python 3.11 + CUDA PyTorch + NVRTC environment, verifies your GPU VRAM fit, downloads and SHA-256 checks this pack with pause/resume, launches chat (`32,768 FP8 KV` context, `2,048` max reply tokens by default), and starts the local OpenAI/Ollama endpoint on `http://127.0.0.1:11434/v1`.


### Option 2: Manual Python CLI

If running directly from Python 3.11+:

1. Download every file in this repo (or clone it with Git LFS).
2. Install the dependencies into a clean virtual environment (`huggingface-hub>=1.5.0,<2.0` if using `transformers 5.x`):

   `pip install -r requirements.txt`

3. Start chatting (`2,048` max tokens default):

   `python run_e8_chat.py --interactive`

   On Windows, double-clicking `start_e8_chat_cli.bat` does the same thing.
4. Optional: serve as an Ollama and OpenAI compatible endpoint on `http://localhost:11434`:

   `python run_e8_chat.py --serve-ollama`

## Access and terms

Open download, no approval step. The pack is provided under the
Gosset Evaluation License (LICENSE.md in this repo): non-commercial
evaluation only, no redistribution, provided as-is. The base model,
Qwen3.8-27B by Alibaba, is Apache-2.0; this pack is a derived
artifact of Gosset Works. The packing encoder, its calibration data,
and its source are not included and are not licensed.

## Limitations

- A residual perplexity gap of +3.93% (long-chunk) and +6.99%
  (short-window) against full precision remains. It is printed
  above, not rounded away.
- No task-benchmark table is published for this pack. Its published
  fidelity record is the certification and perplexity tables above.
- Verified on RTX 5080 (SM120). Evaluation contexts are those of the
  harnesses above; no claim is made beyond measured configurations.

Patent pending.
