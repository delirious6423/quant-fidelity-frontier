---
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: mlx
tags:
- mlx
- vision
pipeline_tag: image-text-to-text
---

# ToPo-ToPo/Ornith-1.5-35B-A3B-mlx-4bit

MLX **4bit** conversion of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for Apple Silicon (mlx-vlm), **with the vision tower kept**.

Since 2026-10-31 this repo uses a mixed-precision recipe: the routed experts are 4bit and every other linear layer is 8bit.

## Provenance (self-converted from official weights)
- Source: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (license: mit)
- Tool: `mlx-vlm 0.7.10` (mlx 0.32.4), `mlx_vlm.convert(..., quantize=True, q_bits=4, q_group_size=64, quant_predicate=predicate)` with the predicate below
- Quantization (affine, group size 64):
  - Routed experts (`mlp.switch_mlp.{gate,up,down}_proj`): **4bit**
  - All other linear layers: **8bit**. This covers `lm_head`, `embed_tokens`, full attention, Gated DeltaNet projections, the shared expert and the routers.
  - Vision tower: not quantized (bf16)
- Effective: **4.925 bits/weight** (weights about 21.6 GB; the previous uniform 4bit was about 20.4 GB)
- Vision: the source checkpoint is `Qwen3_5MoeForConditionalGeneration` and contains the vision tower (333 tensors) and the image/video processor configs. This conversion keeps them, so the model accepts image input. The official [`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit) is a text-only conversion without the vision tower.
- The MTP (multi-token prediction) weights of the source are not included.

```python
from mlx_vlm.utils import skip_multimodal_module

def predicate(path, module):
    if skip_multimodal_module(path):
        return False
    if ".switch_mlp." in path:
        return {"group_size": 64, "bits": 4}
    return {"group_size": 64, "bits": 8}
```

## Why mixed precision
The previous revision (`88f27a0`, 2026-09-18) used the default uniform 4bit conversion, with only the routers kept at 8bit. That is the same recipe as the official MLX 4bit.
Only 8 of the 256 experts run for each token, while the non-expert layers run for every token. In the uniform 4bit model, Japanese answers sometimes mixed in Simplified Chinese, for example a tool-call preamble written in Chinese. Keeping the non-expert layers at 8bit removed this in our tests. It costs about 6% more weight bits.

Small in-house evaluation, thinking off, 5 samples per case: 34 short Japanese/English probes and 20 tool-calling scenarios, each in Japanese and English. All models were served through local-llm-server (mlx-vlm backend) on the same Apple Silicon Mac.

| Metric | bf16 (source) | previous uniform 4bit | this revision |
|---|---|---|---|
| Responses with Simplified Chinese / Hangul (probes) | 0.6% | 5.9% | 0% |
| Same, Japanese tool calling | 0% | 4% | 0% |
| Probe accuracy | 96.5% | 93.0% | 95.7% |
| Japanese tool-calling accuracy | 100% | 100% | 99% |
| English tool-calling accuracy | 100% | 100% | 100% |
| Median generation speed | 64 tok/s | 102 tok/s | 91 tok/s |

The differences between bf16 and this revision are within one sample. The suite is small and does not cover long agentic tasks.

## Validation
- Image input tested through local-llm-server (mlx-vlm backend). The model correctly named the colors, shapes and positions in a synthetic test image.
- Tokenization and the chat template give the same token IDs as the source checkpoint on Japanese, English, code, tool-call markup and combining characters.

## Usage
```python
from mlx_vlm import load, generate
model, processor = load("ToPo-ToPo/Ornith-1.5-35B-A3B-mlx-4bit")
```

To use the previous uniform 4bit files, pin the revision `88f27a0ff513bc11fe6b62a5068c3e660639a595`.
