---
license: apache-2.0
library_name: mlx
base_model:
- Qwen/Qwen3.8-27B
- harness-hub/Qwen3.8-27B-pi-adapted
base_model_relation: quantized
tags:
- mlx
- mtplx
- vision
---

# Qwen3.8-27B pi-adapted · MLX Attention8 + BF16 recurrence

The [harness-hub pi adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted/tree/e786f8491742268f9b2bc729697b13bfac26d4ad) was merged into the [Qwen BF16 base](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0) before quantization. This release retains the adapter's tokenizer and chat template.

The language model uses 234 affine 8-bit modules (group size 64), 168 affine 4-bit modules (group size 32), and 96 BF16 recurrent-input modules. The package includes the Qwen vision tower and the original native BF16 MTP sidecar.

This unedited quant passed strict MLX-LM text loading. MTPLX still marks the runtime contract as unverified.

## Verified benchmark results

<!-- shiftedx-bench:start -->
[Published weights `7ebb7a2`](https://huggingface.co/Shiftedx/Qwen3.8-27B-pi-adapted-MLX-Attention8-BF16-Recurrence/tree/7ebb7a2d750b863a3c31704fa46fc8ed2f5aee48) · [Shiftedx Bench v0.5.1 `9a43027`](https://huggingface.co/datasets/Shiftedx/shiftedx-bench/tree/9a43027a84dc4a4a283220f58314b5768245ec58)

| Lane | Passed |
|---|---:|
| Quality | 8/10 |
| Tool calling | 6/6 |
| Agentic | 2/2 |
| Vision | 4/4 |
| Long context | 3/3 at 4,096 tokens |

The context gate is partial. Twelve of 15 cases remain unscored after host memory limits interrupted longer requests.

Apple M4 Max, 64 GiB · MTPLX 2.12.2 Turbo · medium reasoning, greedy decoding, native template and tools · q8 KV configured · target-only text, MTP depth 2 for vision.
<!-- shiftedx-bench:end -->

```bash
mtplx run --model Shiftedx/Qwen3.8-27B-pi-adapted-MLX-Attention8-BF16-Recurrence --no-mtp --reasoning off --prompt "Reply with OK."
```

See `SHA256SUMS` for file integrity.
