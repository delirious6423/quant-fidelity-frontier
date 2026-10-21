---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: vllm
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - 4-bit
  - 4bit
  - int4
  - quantized
  - autoround
  - w4a16
  - compressed-tensors
  - marlin
  - qwen3
  - qwen3.8
  - vllm
---

# Qwen3.8-27B · 4-bit (INT4)

**A 4-bit quant of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).**  
Weights are **INT4**. Activations stay **BF16**. **19.5 GB** on disk (BF16 is ~54 GB).

This is **AutoRound**, not AWQ, not GGUF, not NVFP4. Packed as `compressed-tensors` for **vLLM Marlin**. Runs on Ampere and newer (3090, 4090, A6000, A100). You do not need a 5090.

| | |
|---|---|
| Quant | **4-bit / INT4** weights, 16-bit activations (W4A16) |
| Method | AutoRound 0.14.2, group size 128 |
| Format | compressed-tensors / Marlin |
| Not | AWQ · GGUF · NVFP4 |
| Size | 19.5 GB |
| Context | **262,144** native, up to **1M** with YaRN |
| Proven | 2× RTX 3090, full 262k context |

Use this on 3090 / 4090 / A100 class cards.  
Use [cyankiwi's AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) if you want their AWQ.  
Use an NVFP4 build only on Blackwell (50-series / B200).

## Serve (vLLM)

The model can take **262,144 tokens** of context. `--max-model-len` is how much of that you reserve in GPU memory. Set it to `262144` if you have the VRAM (we do, on two 3090s). Set it lower if you run out of memory, or if you want more than one request in flight — leftover VRAM goes to the KV cache.

```bash
pip install "transformers>=5.15"

vllm serve MIRALABS/Qwen3.8-27B-W4A16-AutoRound \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code
```

Need `transformers>=5.15` for `Qwen3_5Config`. vLLM 0.19 still asks for `<5`; install 5.15 anyway.

### One 3090

The weights fit in 24 GB. Context does not — not at 262k. Start around 8k and raise `--max-model-len` until it OOMs.

```bash
vllm serve MIRALABS/Qwen3.8-27B-W4A16-AutoRound \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code
```

For long context on a single 3090, use a GGUF in llama.cpp and offload what does not fit.

### 1M context (YaRN)

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve MIRALABS/Qwen3.8-27B-W4A16-AutoRound \
  --max-model-len 1000000 \
  --hf-overrides '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}' \
  --trust-remote-code
```

## Thinking

Thinking is on by default (`reasoning_effort=xhigh`). The model writes inside `<think>…</think>`, then the answer. `--reasoning-parser qwen3` puts the think in `reasoning` and the answer in `content`.

Give it room. A tiny `max_tokens` cuts it off mid-thought and you get an empty answer.

- `xhigh` — default  
- `medium` / `low` — shorter think  
- `enable_thinking=false` — skip thinking  

Qwen's sampling: thinking `temperature=1.0, top_p=0.95, top_k=20`. No-think: `temperature=0.7, top_p=0.8, presence_penalty=1.5`.

## Recipe

| | |
|---|---|
| Base | Qwen/Qwen3.8-27B |
| Bits | 4-bit weights (INT4), activations BF16 |
| Algorithm | Intel AutoRound 0.14.2 |
| Group | 128 |
| Packing | compressed-tensors, pack-quantized |
| Kernel | Marlin (Ampere+) |
| Calibration | NeelNanda/pile-10k, 128 samples, seq 2048, 200 iters |
| Left in BF16 | recurrence (`in_proj_a` / `in_proj_b`), vision, MTP, lm_head |

## FAQ

**Is this 4-bit?**  
Yes. INT4 weights. W4A16 means 4-bit weights, 16-bit activations.

**Is this AWQ?**  
No. AutoRound. AWQ is a different 4-bit method — see cyankiwi.

**3090 or 5090?**  
3090 / 4090 / A100 → this. Blackwell → NVFP4 will be faster on those tensor cores.

## License

Apache 2.0, same as the base.
