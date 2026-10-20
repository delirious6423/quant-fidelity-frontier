---
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/fbb995a79eedd569a5edc5f2af9644c0fa1124fc/LICENSE
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- mtplx
- qwen3.5
- moe
- vision
- speculative-decoding
- apple-silicon
---

# ornith-1.5-35b-a3b-affine4-router8-bf16recurrence-vision-mtplx

Affine 4-bit language body with 8-bit MoE router gates and recurrent-state inputs retained in BF16, plus BF16 vision and native BF16 MTP. Built from [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) revision [`fbb995a`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/fbb995a79eedd569a5edc5f2af9644c0fa1124fc) for Apple Silicon.

## Format

- Logical size: 24.29 GB
- Qwen3.5 MoE multimodal: 35B total parameters, 3B active
- Language body: affine 4-bit/group-32 with 80 MoE router gates at affine 8-bit/group-64; 60 recurrent input modules retained in BF16
- Vision: 333 same-parent BF16 tensors
- MTP: 785 same-parent BF16 tensors in `mtp/weights.safetensors`
- Context metadata: 262,144 tokens
- Tokenizer, chat template, and image/video processor metadata preserved

The fail-closed precision recipe and conversion assertions are included in `BUILD_RECIPE.json` and `conversion_receipt.json`.

The matched AR/D1/D2/D3 sweep selected **D1** at 118.50 decode tok/s on the qualification host.

## Usage

```bash
mtplx serve \
  --model Shiftedx/ornith-1.5-35b-a3b-affine4-router8-bf16recurrence-vision-mtplx \
  --profile turbo \
  --generation-mode mtp \
  --load-mtp \
  --depth 1
```

```bash
python -m mlx_vlm.generate \
  --model Shiftedx/ornith-1.5-35b-a3b-affine4-router8-bf16recurrence-vision-mtplx \
  --image image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

## Qualification and limitations

- Structural loading, deterministic text, vision, and MTPLX tensor gates passed locally.
- This control quant is experimental; behavior may differ from the BF16 parent and the higher-precision hybrid.
- Full BF16 parent parity was not run on the 64 GiB qualification host.
- Shiftedx Bench results below are tied to the immutable evaluated weight revision.
- Review the [upstream model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for intended use, training details, license, and safety considerations.

<!-- shiftedx-bench:start -->
## Shiftedx Bench post-publication qualification

This table was generated from the frozen lightweight quant gate after the model weights were published. Categories remain separate; the benchmark does not produce a composite intelligence score.

| Lane | Passed | Accuracy | Mean wall time | Mean decode | Peak active memory |
|---|---:|---:|---:|---:|---:|
| Quality | 6/10 | 60.0% | 7.60 s | 126.10 tok/s | 25.39 GiB |
| Long context | 11/15 | 73.3% | 37.04 s | 105.99 tok/s | 36.42 GiB |
| Tool calling | 6/6 | 100.0% | 1.54 s | 96.90 tok/s | 26.62 GiB |
| Agentic | 1/2 | 50.0% | 5.37 s | — tok/s | — |
| Vision | 2/4 | 50.0% | 1.35 s | 123.72 tok/s | 28.54 GiB |

- Tested model revision: [`bbaff3c9946d1c2fcd30f88138cf8db2970629f1`](https://huggingface.co/Shiftedx/ornith-1.5-35b-a3b-affine4-router8-bf16recurrence-vision-mtplx/tree/bbaff3c9946d1c2fcd30f88138cf8db2970629f1)
- Benchmark: [Shiftedx Bench v0.3.0](https://huggingface.co/datasets/Shiftedx/shiftedx-bench/tree/3bbb0bfa01e33503163cb34ef52b4d507e456265)
- Context lengths represented: 4,096, 16,384, 65,536, 131,072 prompt tokens; effective tested context: 131,072 tokens
- Runtime contract: MTPLX 2.7.1; MLX 0.32.0; mlx-lm 0.31.3; D1; turbo; thinking on; reasoning effort medium; KV cache `off`; MTP depth `1`
- Host: Apple M4 Max, 64 GiB unified memory
- Total measured request wall time: 656.97 seconds
- 260,096-token status: not run; it is outside the lightweight quant gate.

Scores are specific to the linked model revision, benchmark revision, runtime contract, and host. Changing weight precision, KV-cache precision, reasoning mode, template, or speculative depth creates a different benchmark candidate.
<!-- shiftedx-bench:end -->
