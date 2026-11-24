---
library_name: transformers
pipeline_tag: image-text-to-text
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- fp8
- vllm
- llm-compressor
- compressed-tensors
- qwen3_5
---

# Qwen3.8-27B-FP8-dynamic

FP8 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
in the `compressed-tensors` format:

- Weights: FP8 (E4M3), **per-channel** scales
- Activations: FP8 (E4M3), **per-token dynamic** scales
- Gated DeltaNet: the `in_proj_qkv`, `in_proj_z` and `out_proj` projections
  are FP8. The small `in_proj_a` / `in_proj_b` gate projections stay in BF16,
  the same split as the official `Qwen/Qwen3.8-27B-FP8`.
- Kept in BF16: the vision encoder, `embed_tokens`, `lm_head`, and the MTP
  head (`mtp.*` tensors are carried over unchanged so speculative decoding
  stays possible)

Only `Linear` modules inside the transformer blocks are quantized, with
[LLM Compressor](https://github.com/vllm-project/llm-compressor) 0.13.0.
No calibration data is needed for this scheme.

Size on disk: ~30 GB. The BF16 source is 55.6 GB. The weights, the FP8 KV
cache and the full 262,144-token context fit on one 48 GB GPU.

**Revisions.** The `v1` tag holds the first release, which kept all Gated
DeltaNet projections in BF16 (~37 GB). `main` is v2. Both score the same on
HumanEval+ (table below).

This is a community quantization, not an official Qwen release.

## Why per-channel FP8 (and not block-128)

The official `Qwen/Qwen3.8-27B-FP8` uses block-128 FP8 (`weight_block_size
[128, 128]`). In vLLM, the CUTLASS block-scaled FP8 GEMM requires SM90+
(Hopper) and DeepGEMM requires SM90/SM100, so on Ada (SM 8.9, e.g. RTX 6000
Ada, L40S, RTX 4090) block-128 FP8 falls back to a Triton kernel. Per-channel
FP8 with dynamic per-token activations uses the CUTLASS FP8 GEMM on SM 8.9
(CUDA >= 12.4) and also runs on Hopper and Blackwell.

Source: vLLM v0.19.0, `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu`
(`cutlass_scaled_mm_supports_fp8` vs `cutlass_scaled_mm_supports_block_fp8`).

## Deployment with vLLM

Requires vLLM >= 0.17.0.

One 48 GB GPU, full context (vLLM 0.31.0):

```bash
vllm serve AzatAI/Qwen3.8-27B-FP8-dynamic \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Keep `--max-num-seqs` low on one GPU. Each sequence slot reserves a fixed
Gated DeltaNet state. With the default slot count, the CUDA graph memory
profiler asks for about 13 GiB more and startup runs out of memory. With 32
slots the KV cache holds 354,406 tokens.

The MTP head is included (`model_mtp.safetensors`), so `--speculative-config`
with `method: mtp` works out of the box; vLLM resolves the drafter as
`Qwen3_5MTP` and shares the target's embeddings and `lm_head`. Under
speculative decoding vLLM ignores `min_p` and `logit_bias`.

Thinking is on by default in the chat template; the flag above turns it off
by default, and clients re-enable it per request with
`chat_template_kwargs={"enable_thinking": true}` (vLLM >= 0.19 returns it in
`message.reasoning`). Drop the flag to keep the upstream default.

## Measured on one RTX 6000 Ada (SM 8.9), vLLM 0.31.0, v2

| Metric | Value |
|---|---|
| HumanEval / HumanEval+ (EvalPlus, greedy, thinking off) | 96.3 / 91.5 (v1: 96.3 / 91.5) |
| decode, code / prose / 16K-token input, MTP k=3 | 55.9 / 39.6 / 32.5 tok/s |
| 224,071-token prompt, retrieve one line | correct, 161 s |
| vision, tool call | pass |

## Measured on 2x RTX 6000 Ada (Ada, SM 8.9), vLLM 0.19.0, TP=2, v1

Single stream, greedy unless noted, `max_tokens=512`:

| Metric | without MTP | with MTP k=3 |
|---|---|---|
| decode, prose | 38.7 tok/s | 63.3 tok/s |
| decode, code (pytest task) | - | 103 tok/s |
| decode, sampled chat (temperature 0.7) | - | 50-71 tok/s |
| 36k-token prompt, time to 16 output tokens, uncached | 10.2 s | 9.6 s (chunk 32768) |
| same prompt, prefix-cached | 0.7 s | 0.6 s |
| GPU memory per card at 0.9 utilization | 45.4 GB | 45.4 GB |
| KV cache free per card at start | 23.9 GiB | 20.9 GiB |

MTP acceptance: mean 2.56 tokens per step on prose (per-position 0.75 / 0.49 /
0.33), 3.6 on code. Decode without MTP is memory-bandwidth-bound on Ada
(~18.4 GB of weights per GPU per token). vLLM selects
`CutlassFP8ScaledMMLinearKernel` for the linear layers on SM 8.9.

## Creation

```python
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3.8-27B"
SAVE_DIR = "Qwen3.8-27B-FP8-dynamic"

model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.in_proj_a$",
        "re:.*linear_attn.in_proj_b$",
        "re:.*linear_attn.in_proj_ba$",
    ],
)
oneshot(model=model, recipe=recipe)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
save_mtp_tensors_to_checkpoint(MODEL_ID, SAVE_DIR)
```

Run on CPU (no GPU needed for `FP8_DYNAMIC`); ~250 GB of RAM headroom is
comfortable for the 55.6 GB BF16 source.

## License

Apache-2.0, inherited from the base model. The upstream `LICENSE` file is
included unchanged.
