---
language: en
tags:
- quantization
- 8bit
- mtp
- multi-token-prediction
- text-generation
library_name: mlx-lm
pipeline_tag: text-generation
license: apache-2.0
model_name: Qwen3.8-27B-MTP-8bit
base_model: Qwen/Qwen3.8-27B
---
# Model Card for Qwen3.8-27B-MTP-8bit

8-bit quantized version of Qwen3.8-27B with multi-token prediction (MTP) capabilities preserved. Quantization performed using mlx-lm pull request https://github.com/ml-explore/mlx-lm/pull/990 to preserve MTP weights. Original model: https://huggingface.co/Qwen/Qwen3.8-27B

## Key Features

- **Multi-Token Prediction (MTP)**: This quantized model preserves the MTP capabilities from the original Qwen3.8-27B, enabling faster inference through speculative decoding.
- **YaRN Support**: The repository includes `config_1M.json` for hot-swapping to enable YaRN (Yet another RoPE extensioN) for ultra-long context processing up to 1M tokens, and `config_default.json` to switch back. The main `config.json` ships with the default 256K context configuration. Per Qwen's guidance, static YaRN can impact performance on shorter texts, so only swap to `config_1M.json` when long-context processing is required. For details on using YaRN with this model, see https://huggingface.co/Qwen/Qwen3.8-27B
- **Thinking Preservation (on by default)**: Unlike Qwen3.6, Qwen3.8 retains thinking blocks from all historical messages by default, maintaining a complete reasoning trace across the conversation. This is especially beneficial for agent scenarios where decision consistency and reduced redundant reasoning are critical, and it improves KV cache utilization. It can be disabled per request with `preserve_thinking: False`.

## Performance

8-bit is the highest-quality MTP quantization of Qwen3.8-27B, making it the recommended quantized option on hardware where the full bf16 weights saturate memory bandwidth (e.g. Apple M4 Max) — in that regime, enabling MTP on the full weights actually decreases TPS, while 8-bit with MTP restores and exceeds it. The full weights remain the best choice for maximum quality wherever they fit, while 8-bit with MTP provides the best speed/quality trade-off for large output tasks. For the optimal balance of speed and quality, you can re-use your KV cache and hot toggle between the full weights and MTP-8bit within the same conversation:
- **Use full weights for:** Prefill, thinking, planning, and design tasks
- **Use MTP-8bit for:** Large output tasks such as code editing or generation

You can give your agent a `toggle_mtp` tool with these instructions so it can dynamically switch between the appropriate weights for maximum performance for both prefill and decode.

## Model Details

- **Base Model:** Qwen/Qwen3.8-27B
- **Library:** mlx-lm
- **Quantization:** 8-bit (group size 64, affine)
- **Scope:** Language model only (vision encoder weights not included)
- **Context Length:** 262,144 natively; extensible up to 1,000,000 tokens via `config_1M.json` (YaRN)
- **License:** apache-2.0
- **Pipeline Tag:** text-generation

## Installation

MTP support for this model currently requires installing mlx-lm from [PR #990](https://github.com/ml-explore/mlx-lm/pull/990) (not yet in main/release):

```bash
# Option 1: pip install
pip install https://github.com/ml-explore/mlx-lm/archive/refs/pull/990/head.zip

# Option 2: uv run (no virtual env needed)
uv run --with https://github.com/ml-explore/mlx-lm/archive/refs/pull/990/head.zip python example.py
```

## Usage

```python
import mlx_lm
from mlx_lm.sample_utils import make_sampler

model_path = "petergilani/Qwen3.8-27B-MTP-8bit"
model, tokenizer = mlx_lm.load(model_path)

sampler = make_sampler(temp=1.0, top_p=0.95, top_k=20)

prompt = "Explain multi-token prediction in language models."
response = mlx_lm.generate(
    model,
    tokenizer,
    prompt=prompt,
    sampler=sampler,
    max_tokens=512,
    mtp=True  # Enable Multi-Token Prediction for faster generation
)
print(response)
```

## Related

- 4-bit release: https://huggingface.co/petergilani/Qwen3.8-27B-MTP-4bit