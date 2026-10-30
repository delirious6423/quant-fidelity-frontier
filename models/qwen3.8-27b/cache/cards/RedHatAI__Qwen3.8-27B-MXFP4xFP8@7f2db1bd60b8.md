---
tags:
- mxfp4
- fp4
- fp8
- vllm
- llm-compressor
- compressed-tensors
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-MXFP4xFP8

## Model Overview
- **Model Architecture:** Qwen3_5ForConditionalGeneration
  - **Input:** Text / Image / Video
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** Mixed (MXFP4/FP8)
  - **Activation quantization:** FP8
- **Release Date:** 2026-10-28
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

This is a mixed-precision quantization where the MLP layers use MXFP4 weights and the attention layers use FP8 weights, with FP8 activations throughout.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) to mixed MXFP4 and FP8 precision, ready for inference with vLLM.

Typical MXFP4 quantization pairs MXFP4 weights with MXFP4 activations, a combination that is most performant on Blackwell GPUs. In this model, the MLP projections, which hold most of the parameters, instead use MXFP4 weights with FP8 activations quantized dynamically on a per-group basis, making it well suited for inference on Hopper GPUs. The full-attention and Gated DeltaNet (linear-attention) projections are kept at FP8 precision with block-wise weight scaling and FP8 dynamic activations. The built-in multi-token prediction (MTP) head is quantized the same way, so it remains available for speculative decoding.

This optimization reduces the number of bits per parameter from 16 to 4 for the MLP weights and to 8 for the attention weights, reducing the disk size and GPU memory requirements by approximately 59% (from ~55.6 GB to ~22.7 GB). The vision encoder, embeddings, and `lm_head` are kept in their original BF16 precision.

Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

This model requires vLLM 0.32.0 or later on an NVIDIA GPU.

```
vllm serve RedHatAI/Qwen3.8-27B-MXFP4xFP8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

To enable speculative decoding with the built-in MTP head, add:

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Additional model-specific options (long-context scaling, text-only serving, etc.) are documented in the [vLLM recipe for Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a calibration-free mixed-precision recipe: MXFP4 weights with per-group FP8 dynamic activations for the MLP layers, and FP8 block-wise quantization for the attention layers. The quantized model was exported in the [compressed-tensors](https://github.com/vllm-project/compressed-tensors) format, as presented in the code snippet below.

<details>

```python
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, MXFP4xFP8

from llmcompressor import model_free_ptq

MODEL_ID = "Qwen/Qwen3.8-27B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4xFP8"

# Qwen3.8-27B is a dense hybrid model: every layer has a gated MLP, and
# attention alternates between full attention and linear (gated delta) attention.
# Apply MXFP4xFP8 to the MLP projections, which hold most of the weights.
# Use FP8_BLOCK for the full-attention and linear-attention projections.
# The targets also match the MTP head's layers, so it is quantized the same way.
# Keep the embeddings, lm_head and the vision tower unquantized.
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    config=QuantizationConfig(
        config_groups={
            "attention": QuantizationScheme(
                **FP8_BLOCK,
                targets=[
                    r"re:.*self_attn\.(q|k|v|o)_proj$",
                    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                ],
            ),
            "mlp": QuantizationScheme(
                **MXFP4xFP8,
                targets=[r"re:.*mlp\.(gate|up|down)_proj$"],
            ),
        },
        ignore=[
            "lm_head",
            r"re:.*embed_tokens$",
            r"re:model\.visual.*",
        ],
    ),
    device="cuda:0",
)
```
</details>

## Evaluation

This model was evaluated on GPQA Diamond, AIME 2026, MMLU-Pro, and Terminal-Bench 2.1 using [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) ([inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals)), served with vLLM 0.32.0 (OpenAI-compatible API). The quantized and unquantized models were evaluated with an identical setup, so only the checkpoint differs between the two columns.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>Qwen/Qwen3.8-27B</th>
      <th>RedHatAI/Qwen3.8-27B-MXFP4xFP8</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>Reasoning</b></td>
      <td>GPQA Diamond (0-shot, pass@1)</td>
      <td>89.9%</td>
      <td>88.9%</td>
      <td>98.9%</td>
    </tr>
    <tr>
      <td>AIME 2026 (0-shot, pass@1)</td>
      <td>98.9%</td>
      <td>98.9%</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (0-shot, pass@1)</td>
      <td>86.0%</td>
      <td>85.5%</td>
      <td>99.4%</td>
    </tr>
    <tr>
      <td rowspan="1"><b>Agentic Coding</b></td>
      <td>Terminal-Bench 2.1 (pass@1)</td>
      <td>60.7%</td>
      <td>62.9%</td>
      <td>103.7%</td>
    </tr>
  </tbody>
</table>

### Reproduction

All tasks were evaluated 0-shot with thinking enabled (`reasoning_effort=xhigh`). Each task was run for 3 epochs and the reported pass@1 is the mean across epochs. Generation used the sampling parameters from the model's `generation_config.json` (temperature 1.0, top_p 0.95, top_k 20) and up to 131,072 output tokens. The model was served with vLLM 0.32.0 using four data-parallel replicas of two H200s each and a BF16 KV cache. All evaluations were run with the setup in [neuralmagic/evals-refresh](https://github.com/neuralmagic/evals-refresh). The results were obtained using the following commands:

<details>

#### vLLM server

```
vllm serve RedHatAI/Qwen3.8-27B-MXFP4xFP8 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --max-model-len 262144 \
  --kv-cache-dtype bfloat16 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --port 8000
```

#### Client environment

```
pip install "inspect-ai==0.3.263" "inspect-evals==0.20.0" "openai==3.14.0"
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=local
```

#### GPQA Diamond, AIME 2026, MMLU-Pro (Inspect AI, 3 epochs)

```
inspect eval inspect_evals/gpqa_diamond \
  --model vllm/RedHatAI/Qwen3.8-27B-MXFP4xFP8 \
  --epochs 3 \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens 131072 \
  --max-connections 128 \
  -M client_timeout=7200

inspect eval inspect_evals/aime2026 \
  --model vllm/RedHatAI/Qwen3.8-27B-MXFP4xFP8 \
  --epochs 3 \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens 131072 \
  --max-connections 128 \
  -M client_timeout=7200

inspect eval inspect_evals/mmlu_pro \
  --model vllm/RedHatAI/Qwen3.8-27B-MXFP4xFP8 \
  --epochs 3 \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens 131072 \
  --max-connections 256 \
  -M client_timeout=7200
```

#### Terminal-Bench 2.1 (Inspect AI, 3 epochs)

The Terminal-Bench 2.1 configuration and reproduction steps are available in [neuralmagic/evals-refresh](https://github.com/neuralmagic/evals-refresh).

</details>
