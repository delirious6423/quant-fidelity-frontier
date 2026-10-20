---
library_name: mlx
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- ornith
- qwen3.5-moe
- multimodal
- abliterated
- 8-bit
---

# Ornith 1.5 35B-A3B Abliterated MLX — 8-bit recommended

An unofficial experimental MLX derivative of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), pinned to
revision `e4dfb35a93d4b6822a811a7676f3488514abe7e2`. The original model is by Ornith AI; the MLX
conversion, refusal-direction experiment, and validation were performed by
PocketAI Model Lab. `PocketAiHub` identifies the publisher of this derivative.

**This is the recommended practical release.** It retains the abliterated BF16 behavior much more closely than the 4-bit build.

## Important safety notice

This checkpoint was deliberately modified to suppress learned refusal behavior.
It may produce harmful, illegal, offensive, deceptive, or dangerously incorrect
content more readily than the upstream instruction model. Abliteration is not
truthfulness training, a capability improvement, or a guarantee of universal
compliance. Independently evaluate and constrain outputs for your use case.

## Format

- MLX affine 8-bit/group 64; router and shared-expert gates 8-bit
- Stored model payload: 37,748,365,508 bytes (35.16 GiB)
- Vision tower retained; the 4-bit build passed a basic image-input smoke test
- Native MTP speculative-decoding head is not included because `mlx-vlm==0.6.8`
  drops `mtp.*` tensors during conversion
- Validated with `mlx==0.32.0` and `mlx-vlm==0.6.8`

Other releases:

- [BF16 reference](https://huggingface.co/PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-BF16)
- [8-bit recommended](https://huggingface.co/PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-8bit)
- [4-bit compact](https://huggingface.co/PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-4bit)

## Abliteration recipe

A projected harmful-minus-harmless direction was measured from 256
length-matched prompts per class at the assistant-generation boundary.

- Direction source layer: 27
- Destination layers: 15–39
- Scale: 1.0
- Per-input-column norm preservation: enabled
- Modified physical tensors: 75
- Modified logical expert/projection paths: 6,450
- Direction SHA-256: `b4bef4649c209aae888c7b313feb89005f897938c0a01540a6852f0e3bf4b407`

See [`abliteration-manifest.json`](./abliteration-manifest.json) for the
machine-readable recipe.

## Behavioral screen

The regular BF16 parent produced explicit-refusal phrases on 12/12 harmful gate
prompts. The selected abliterated BF16 candidate produced 0/12 on the same gate
and retained 12/12 deterministic capability checks.

| Batch-1 screen | Explicit-refusal phrase flags | Final-answer text present |
| --- | ---: | ---: |
| Harmful prompts | 2/100 | 100/100 |
| Benign controls | 0/100 | 100/100 |

The scorer is phrase based. The 128-token ceiling makes this an early-refusal
screen rather than a complete answer-quality evaluation, and manual inspection
found semantic refusals that it did not flag. “Abliterated” describes the
weight-editing method; it does not mean “fully uncensored.”

## Matched-teacher drift

The drift suite used 36 prompts—12 capability, 12 harmful, and 12 benign—with
481 shared teacher positions and exact KL over all 248,320 logits. It also
captured all 40 residual layers, K/V state for 10 full-attention layers, and
convolution/recurrent state for 30 linear-attention layers.

| Pure BF16 ablation split | Mean forward KL | Top-1 agreement | Residual cosine |
| --- | ---: | ---: | ---: |
| Capability | 0.018342 | 97.94% | 0.995895 |
| Benign | 0.307388 | 85.42% | 0.969620 |
| Harmful | 1.049146 | 60.94% | 0.890754 |

Across all 481 positions, the pure BF16 ablation measured mean KL
`0.545185`, top-1 agreement
`78.17%`, and residual cosine
`0.952090` versus regular BF16.

Against the abliterated BF16 master, this quantization measured mean KL `0.021342`, top-1 agreement `95.43%`, and residual cosine `0.994969`.

The total 8-bit recommended path versus regular BF16 measured mean KL
`0.549199`, top-1 agreement
`77.55%`, and residual cosine
`0.949564`.

Machine-readable behavioral, residual, and cache metrics are in
[`validation-summary.json`](./validation-summary.json).

## Load with MLX-VLM

```bash
python -m pip install "mlx==0.32.0" "mlx-vlm==0.6.8"
mlx_vlm.generate --model PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-8bit --prompt "Explain why seasons occur." --max-tokens 256
```

For an image prompt:

```bash
mlx_vlm.generate --model PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-MLX-8bit --prompt "Describe this image." --image photo.jpg --max-tokens 256
```

The vision tower is present in every release, but only the 4-bit model received
an end-to-end image smoke test. Broader vision, video, coding, tool-use, and
long-context evaluations remain future work.

## License and attribution

The upstream model card declares MIT. This derivative preserves the upstream
attribution and links to the exact pinned source revision above.
