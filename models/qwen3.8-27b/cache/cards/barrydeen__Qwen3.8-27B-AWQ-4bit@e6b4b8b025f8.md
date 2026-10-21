---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - qwen
  - qwen3_5
  - awq
  - w4a16
  - compressed-tensors
  - int4
  - 4-bit
  - 4bit
  - vllm
  - quantized
  - multimodal
---

# Qwen3.8-27B-AWQ-4bit

AWQ **W4A16** quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) in
`compressed-tensors` `pack-quantized` format — the best-optimised Marlin path on Ampere (SM 8.6).

Built for **2× RTX 3090** running vLLM, but nothing here is 3090-specific.

| | |
|---|---|
| Size | **27.8 GB** (from 55.6 GB BF16) |
| Format | `compressed-tensors` / `pack-quantized` |
| Scheme | W4A16 asymmetric, group size 128 |
| Quantized modules | 256 Linear layers |
| Vision tower | **preserved**, BF16 |
| Gated DeltaNet | **preserved**, BF16 |
| MTP head | **preserved**, BF16 |
| Architecture string | `Qwen3_5ForConditionalGeneration` |

## Why this one is bigger than other 4-bit builds

Qwen3.8-27B is a **hybrid** model. Only **16 of its 64** decoder layers use full attention; the
other **48 use Gated DeltaNet** (`linear_attn.*`). Those DeltaNet projections are extremely
sensitive to quantization, so they are left at full BF16 precision here, along with the vision
tower and the multi-token-prediction head.

That means only **256 Linear modules** are quantized — 16 layers × 4 attention projections plus
64 layers × 3 MLP projections — rather than the ~448 you would get by assuming a uniform
architecture. Keeping 48 layers of DeltaNet in BF16 costs roughly 7 GB versus builds that
quantize everything, and that is the entire difference in file size.

Everything protected here matches the exclusion set of the official FP8 release's
`modules_to_not_convert`.

## Usage

### vLLM

```bash
vllm serve barrydeen/Qwen3.8-27B-AWQ-4bit \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.91 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Two things worth knowing:

- **Use tensor parallelism, not pipeline parallelism, if you enable MTP.** The MTP draft model
  does not implement `SupportsPP`, and startup fails with
  `NotImplementedError: Pipeline parallelism is not supported for this model`.
  Without MTP, `--pipeline-parallel-size 2` is fine.
- If your host has no CUDA toolkit, set `VLLM_USE_FLASHINFER_SAMPLER=0`. FlashInfer's sampler
  JIT-compiles kernels and will fail with `Could not find nvcc`.

### Transformers

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "barrydeen/Qwen3.8-27B-AWQ-4bit", device_map="auto", trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    "barrydeen/Qwen3.8-27B-AWQ-4bit", trust_remote_code=True,
)
```

## Measured performance

2× RTX 3090, vLLM 0.26.0, TP=2, MTP speculative decoding with 3 draft tokens:

| Metric | Value |
|---|---|
| Generation | 85–95 tok/s |
| MTP acceptance | 67.1% |
| Tokens per forward step | 3.01 |
| KV cache @ 131k context | 178,667 tokens |

Speculative acceptance by draft position: 85% / 69% / 47%.

## Quantization details

- **Source**: `Qwen/Qwen3.8-27B` BF16 — not re-quantized from the FP8 release.
- **Method**: AWQ via llm-compressor, `W4A16_ASYM`, group size 128.
- **Calibration**: 512 samples from
  [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
  (`train_sft`), chat-template formatted, `max_seq_length=1024`.
- **Pipeline**: sequential, one decoder layer at a time
  (`sequential_targets=["Qwen3_5DecoderLayer"]`), so each layer is calibrated against the actual
  quantized output of the layers before it.

Excluded from quantization: `linear_attn.*` (Gated DeltaNet), the full `visual.*` tower,
`mtp.*`, `lm_head`, `embed_tokens`, all norms, and the MoE gates.

## Limitations and honest notes

- **Calibration used 1024-token sequences, not 2048.** Quantizing a 27B model alongside the
  sequential pipeline's activation cache exceeded available memory at 2048. Halving sequence
  length was chosen over halving sample count, on the reasoning that activation-statistic
  diversity across 512 distinct conversations matters more for AWQ than per-sample length. It
  does mean activations from the 1024–2048 token range were not observed during calibration.
- **Calibration data is text-only.** The vision tower is not quantized, so it is unaffected, but
  no image data was used.
- **No formal benchmark suite has been run.** The performance figures above are measured; quality
  has been verified only by inspection (coherent generation, correct arithmetic, working code
  output, no degradation under tool-calling). If you benchmark it, please open a discussion —
  particularly a comparison against builds that quantize the DeltaNet layers, since that is the
  central claim here and it deserves numbers rather than reasoning.

## License

Apache 2.0, inherited from the base model.
