---
license: apache-2.0
library_name: transformers
base_model: google/gemma-4-26B-A4B-it
language:
  - multilingual
tags:
  - nvidia
  - nvfp4
  - modelopt
  - quantized
  - gemma4
  - moe
  - dgx-spark
  - blackwell
  - W4A16
  - post-training-quantization
pipeline_tag: text-generation
model-index:
  - name: Gemma-4-26B-A4B-it-NVFP4A16
    results: []
---

# Gemma-4-26B-A4B-it-NVFP4A16

**First community NVFP4 quantization** of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) — the Mixture-of-Experts variant of Gemma 4 with 25.2B total parameters and only 3.8B active per token.

**W4A16** — weights in FP4, activations in FP16 (weight-only quantization).

## Key Specs

| | Original (BF16) | NVFP4A16 (this) |
|---|---|---|
| **Size on disk** | ~49 GB | ~16.5 GB |
| **Compression** | — | 3.0x |
| **Total parameters** | 25.2B | 25.2B |
| **Active parameters** | 3.8B | 3.8B |
| **Architecture** | MoE: 128 experts, 8 active/token | same |
| **Context window** | 256K tokens | 256K tokens |
| **Modalities** | Text, Image, Video | Text, Image, Video (all verified) |
| **Quantization** | — | W4A16 (FP4 weights, FP16 activations) |

## Serving with vLLM

### Requirements

- vLLM build with `transformers >= 5.4` (for Gemma 4 architecture support)
- On DGX Spark / SM 12.1: [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) built with `--tf5` flag
- Included `gemma4_patched.py` for NVFP4 MoE scale key loading (see [vLLM Patch](#vllm-patch))

### Quick Start

```bash
docker run -d \
  --name vllm-gemma-4 \
  --gpus all --ipc=host --network host \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -v /path/to/Gemma-4-26B-A4B-it-NVFP4A16:/model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/gemma4_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py \
  <your-vllm-image> \
  vllm serve /model \
    --served-model-name gemma-4 \
    --host 0.0.0.0 --port 8888 \
    --quantization modelopt \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536 \
    --max-num-seqs 4 \
    --moe-backend marlin \
    --trust-remote-code
```

### Key Flags

| Flag | Why |
|------|-----|
| `--quantization modelopt` | modelopt NVFP4 checkpoint format |
| `--moe-backend marlin` | Marlin kernel for MoE expert layers |
| `--kv-cache-dtype fp8` | Saves memory for longer contexts |
| `-e VLLM_NVFP4_GEMM_BACKEND=marlin` | Marlin for non-MoE layers (needed on SM 12.1) |
| `--trust-remote-code` | Required for Gemma 4 |

### Testing

This is an instruct model — use the **chat completions** endpoint:

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [{"role": "user", "content": "Hello! Tell me a joke."}],
    "max_tokens": 200
  }'
```

### DGX Spark

Tested on NVIDIA DGX Spark (128GB unified memory, GB10 Blackwell, SM 12.1). Model loads at 15.7 GiB — plenty of headroom for 65K+ context with FP8 KV cache.

## How this was made

### The Problem

Gemma 4 MoE stores expert weights as **fused 3D tensors** (`nn.Parameter` of shape `[128, dim, dim]`) instead of individual `nn.Linear` modules. NVIDIA Model Optimizer (modelopt) only quantizes `nn.Linear` — it silently skips the 3D expert parameters, which are 91% of the model.

### The Solution

We wrote a `_QuantGemma4TextExperts` modelopt plugin that unfuses the 3D expert tensors into 128 × 3 individual `nn.Linear` layers before quantization. This follows the same pattern modelopt uses for Qwen3.5, Llama4, and DBRX MoE models. After quantization, a post-processing step renames the exported keys to match vLLM's expected format.

### Calibration

- **Tool:** NVIDIA Model Optimizer v0.43, `_nvfp4_selective_quant_cfg(["*"], weight_only=True)`
- **Data:** 4096 samples from CNN/DailyMail, batch 16, seq_len 1024
- **Expert routing:** Natural (router decides which experts see which data)
- **Vision encoder:** Excluded from quantization (stays BF16)
- **Hardware:** NVIDIA DGX Spark

## vLLM Patch

vLLM's Gemma 4 `expert_params_mapping` doesn't correctly map NVFP4 scale keys (`.weight_scale`, `.weight_scale_2`, `.input_scale`) to FusedMoE parameter names. The included `gemma4_patched.py` fixes this. A PR to upstream vLLM is forthcoming.

## Reproduce

```bash
pip install torch transformers>=5.4 accelerate datasets
git clone https://github.com/NVIDIA/Model-Optimizer.git
pip install -e Model-Optimizer[all]
pip install --force-reinstall transformers>=5.4 huggingface_hub>=1.5

python quantize_gemma4_moe.py --qformat nvfp4_w4a16
```

Full quantization script included as `quantize_gemma4_moe.py`.

## Limitations

- Requires vLLM with `transformers >= 5.4` and the included `gemma4_patched.py`
- `--moe-backend marlin` required for correct MoE computation
- Community quantization, not an official NVIDIA or Google release

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) — inherited from the base model.

## Credits

Quantized by [Mario Iseli](https://huggingface.co/marioiseli) on an NVIDIA DGX Spark. Built and validated with AI-engineering assistance from Anthropic.

Shout-out to [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) for the DGX Spark-optimized vLLM build.

📬 mario@marioiseli.com
☕ [Buy me a coffee](https://buymeacoffee.com/marioiseli) if this makes your Spark go brrrrrr! 🚀
