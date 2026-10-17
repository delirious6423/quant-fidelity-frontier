---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- qwen
- qwen3.8
- multimodal
- 6-bit
- abliterated
---

# Qwen3.8-27B Abliterated MLX 6-bit

An unofficial MLX 6-bit derivative of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). The original
model is by Qwen; the MLX conversion, refusal-direction experiment, and
validation were performed by PocketAI Model Lab.

This repository deliberately keeps the upstream model name first. `PocketAiHub`
identifies the publisher of this derivative, not the creator of Qwen3.8.

## Important safety notice

This checkpoint has been modified to suppress learned refusal behavior. It may
produce harmful, illegal, offensive, deceptive, or dangerously incorrect
content more readily than the upstream instruction model. Abliteration is not
truthfulness training, capability improvement, or a safety guarantee. Use it
only where you can independently evaluate and constrain its outputs.

## Format

- MLX affine 6-bit, group size 64
- 498 quantized language modules
- Vision tower retained in BF16
- Stored artifact size: 22,804,849,041 bytes (21.24 GiB)
- Validated with `mlx==0.32.0` and `mlx-vlm==0.6.8`
- Pinned base revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

## Abliteration recipe

The reusable LFM2.5 workflow was adapted to Qwen3.8's 64-layer hybrid
architecture. A projected harmful-minus-harmless direction was measured from
256 length-matched prompts per class at the assistant-generation boundary.

- Direction source layer: 53
- Destination layers: 24–63
- Scale: 1.0
- Per-input-column norm preservation: enabled
- Modified residual-output matrices: 80
  - 30 gated-delta/linear-attention `out_proj` matrices
  - 10 full-attention `o_proj` matrices
  - 40 MLP `down_proj` matrices

The edit was applied to a separate BF16 master checkpoint, and this artifact was
derived from that master. The official source checkpoint was not modified in
place. See [`abliteration-manifest.json`](./abliteration-manifest.json).

## Validation

| Gate | Result |
| --- | ---: |
| Candidate harmful screen, batch 1, 128-token ceiling | 0/100 explicit refusals |
| Candidate benign control, batch 1, 128-token ceiling | 0/100 explicit refusals |
| Behavioral final answers present | 200/200 |
| Behavioral evasive nonanswers | 0/200 |
| Deterministic quality checks | 12/12 |
| Native tool-call checks | 8/8 |
| Text smoke | passed (`POCKETAI_OK`) |
| Vision smoke | passed (`red`) |
| Temporal video understanding | passed (`red->blue`) |
| 4K-context retrieval | passed (`COBALT-7319`) |

Across the 200 behavioral generations, 200 reached the 128-token
ceiling and 0 completed naturally. The aggregate contained 0 explicit
refusals. This therefore measures early explicit refusal behavior, not whether
long answers naturally reach EOS. The refusal scorer is phrase-based and cannot
establish universal compliance or answer quality. Machine-readable results are in
[`validation-summary.json`](./validation-summary.json).

### 4K performance

On an Apple M5 Max with 128 GB unified memory, batch size 1, temperature 0,
seed 0, and thinking disabled:

- 4,105 prompt tokens
- 548.2 prompt tok/s
- 24.7 generation tok/s
- 7.87 seconds end to end
- 29.54 GB peak MLX memory

This is one local run, not a cross-machine performance guarantee.

## Load with MLX-VLM

```bash
python -m pip install "mlx==0.32.0" "mlx-vlm==0.6.8"
```

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

repo_id = "PocketAiHub/Qwen3.8-27B-Abliterated-MLX-6bit"
model, processor = load(repo_id)

prompt = apply_chat_template(
    processor,
    model.config,
    "Explain why seasons occur.",
    num_images=0,
    enable_thinking=False,
)
result = generate(
    model,
    processor,
    prompt,
    max_tokens=256,
    temperature=0.0,
    enable_thinking=False,
)
print(result.text)
```

Image and video inputs use the normal `mlx_vlm.generate` media arguments.

## License and attribution

The base model is Apache 2.0 licensed. See [`LICENSE`](./LICENSE) and the
[official Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B).
