---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
- gemma4
- mxfp8
- w8a8
- blackwell
- sm121
- dgx-spark
- gb10
- quantized
- vllm
language:
- en
---

# Gemma 4 26B-A4B MXFP8 W8A8

OCP Microscaling FP8 (MXFP8) W8A8 quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

**First public MXFP8 W8A8 checkpoint for Gemma 4.**

This is a fully quantized version of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
using NVIDIA's MX (microscaling) FP8 format — purpose-built to saturate the native
MX tensor core compute units on **NVIDIA Blackwell (SM100+)** hardware.

Quantized and served on an **NVIDIA DGX Spark (GB10, SM121, 128 GB unified memory)**
using a CUTLASS-enabled vLLM engine to extract maximum throughput from Blackwell's
new FP8 MX GEMM paths.

## Quantization details

| Property | Value |
|----------|-------|
| **Base model** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **Architecture** | Gemma 4 MoE — 26B total / 4B active parameters |
| **Quantization** | MXFP8 W8A8 — OCP MX FP8 E4M3, block-32 scaling |
| **Weights** | FP8 E4M3 with block-32 MX (E8M0) scales |
| **Activations** | FP8 E4M3 with block-32 MX (E8M0) scales, dynamic |
| **Tool** | `nvidia-modelopt 0.44.0` |
| **Calibration** | `HuggingFaceH4/ultrachat_200k` — 512 samples × 1024 tokens |
| **Hardware** | NVIDIA DGX Spark (GB10 · SM121 Blackwell · 128 GB unified memory) |
| **License** | Apache 2.0 |
| **Context length** | 131,072 tokens |
| **Language** | English |

## Why MXFP8 W8A8?

Standard FP8 quantization uses a single per-tensor or per-channel scale factor, which
struggles with outliers common in large MoE models. **MX (microscaling) FP8** uses
**block-32 scaling** — one E8M0 scale per 32 elements — giving dramatically finer
quantization granularity that preserves model quality while enabling full FP8 tensor
core throughput.

Blackwell GPUs (SM100+) have **native hardware support for MXFP8 block-32 GEMMs**,
meaning this format maps directly to silicon without emulation overhead. On non-Blackwell
hardware this format either won't load or will fall back to BF16 — this checkpoint is
specifically optimized for the GB10/GB200 family.

## Quantization Choices

Certain layers are intentionally excluded from quantization to preserve accuracy:

- **`lm_head`** — output projection kept in BF16
- **MoE router/gate layers** — all `model.language_model.layers.N.router.proj` kept in BF16
- **Vision tower** — `model.vision_tower.*` kept in BF16
- **Vision embedding projection** — `model.embed_vision.*` kept in BF16

This follows best practices: routing logic and output softmax are numerically sensitive
and represent a negligible fraction of total compute.

## Usage with vLLM

```bash
vllm serve tgmerritt/gemma-4-26B-A4B-it-MXFP8W8A8 \
    --quantization modelopt_mxfp8 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.80 \
    --port 11437
```

▎ Note: Requires vLLM with --quantization modelopt_mxfp8 support and a
▎ Blackwell GPU (SM100+). The CUTLASS-backed MXFP8 GEMM kernels are what make
▎ this format fast — ensure your vLLM build includes CUTLASS FP8 MX support.

Python (via OpenAI-compatible endpoint)

```
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11437/v1", api_key="token")

response = client.chat.completions.create(
    model="tgmerritt/gemma-4-26B-A4B-it-MXFP8W8A8",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Excluded from quantization

- `lm_head` (output projection)
- All MoE router/gate layers (`model.language_model.layers.N.router.proj`)
- Vision tower (`model.vision_tower.*`)
- Vision embedding projection (`model.embed_vision.*`)

# Hardware Requirements

| Requirement | Details |
|---|---|
| GPU Architecture | NVIDIA Blackwell (SM100 / SM121) required for native MXFP8|
| Memory | ~14–16 GB for W8A8 quantized weights (vs ~52 GB BF16)|
| Tested on | NVIDIA DGX Spark (GB10, SM121, 128 GB unified) |
| vLLM | Requires CUTLASS-enabled build with modelopt_mxfp8 backend|

# GB10 Benchmark: MXFP8 W8A8 vs FP8-Dynamic — Gemma 4 26B-A4B + MTP

**Hardware:** NVIDIA DGX Spark (GB10, SM121 Blackwell, 128 GB unified memory)
**vLLM:** v0.1.dev50+g4438b6e7d.d20260527 (SM121 FlashInfer + cutlass-dsl 4.4.2 patch)
**Speculative decoding:** MTP, num_speculative_tokens=4, draft=google/gemma-4-26B-A4B-it-assistant
**Benchmark:** 3 × {short=128tok, medium=512tok, long=768tok}, `/v1/completions`, temperature=0

## Results (wall-clock approximate tok/s, warm runs only)

| Prompt length | FP8-Dynamic + MTP | MXFP8 W8A8 + MTP | Delta |
|---|---|---|---|
| Short (128 tok) | 81.5 / 82.6 / 81.5 → **avg 81.9 t/s** | 92.1 / 92.1 → **avg 92.1 t/s** ¹ | **+12%** |
| Medium (512 tok) | 87.4 / 90.8 / 91.1 → **avg 89.8 t/s** | 109.2 / 110.3 / 107.8 → **avg 109.1 t/s** ² | **+22%** |
| Long (768 tok)  | 83.3 / 84.1 / 89.0 → **avg 85.5 t/s** | 90.0 / 90.8 / 90.9 → **avg 90.6 t/s** | **+6%** |

¹ Run 1 discarded (40 t/s — Triton JIT cold-start penalty, consistent with prior load tests)
² Medium prompt exhibits intermittent early-stop at 2 tokens (~1/5 probability at temperature=0); full-completion runs used here

## Configuration

| Setting | FP8-Dynamic | MXFP8 W8A8 |
|---|---|---|
| Checkpoint | RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic | ~/models/gemma-4-26B-A4B-it-MXFP8W8A8 (local) |
| Quant format | FP8 E4M3, per-channel weight scales | MXFP8 E4M3, block-32 MX scales (OCP) |
| Linear kernel | FP8 CUTLASS | FlashInferCutlassMxfp8LinearKernel (native Blackwell MX) |
| MoE kernel | FP8-Dynamic W8A8 | MARLIN MXFP8 W8A16 |
| KV cache dtype | fp8 | auto (bf16) |
| GPU mem util | 0.30 | 0.70 ³ |
| Max model len | 131072 | 32768 |
| Benchmark date | 2026-05-27 | 2026-05-28 |

³ 0.70 used instead of 0.30 because `--kv-cache-dtype fp8` was not specified; adding it would allow MXFP8 to run at 0.30 as well

## Notes

- MXFP8 W8A8 linear layers use Blackwell-native MX tensor core path (`tcgen05.mma` MX format descriptors), vs FP8-Dynamic which uses standard per-channel FP8 CUTLASS
- MXFP8 MoE falls back to MARLIN W8A16 because Gemma 4's expert `down_proj` inner_dim (704) is not divisible by 128 as required by the FlashInfer TRTLLM MX kernel — this is a ceiling on MoE speedup
- MTP acceptance rate and speculative throughput not measured in this run (completions API doesn't expose spec-decoding metrics)
- Intermittent medium-prompt early-stop warrants follow-up quality evaluation (MMLU/GPQA)

## Acknowledgements

Quantized using NVIDIA ModelOpt on an NVIDIA DGX Spark (GB10). Served via a CUTLASS-optimized vLLM build targeting
Blackwell's native MXFP8 tensor core paths.
  
## License

Apache 2.0 (inherited from google/gemma-4-26B-A4B-it)
