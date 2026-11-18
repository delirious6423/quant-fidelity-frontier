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

This is the unedited quant. It passed strict MLX-LM text loading and six local utility checks. The packaged model passed a short target-only text smoke test in MTPLX 2.12.2. Vision inference, long-context performance, and speculative decoding parity are pending qualification. MTPLX reports the runtime contract as unverified.

```bash
mtplx run --model Shiftedx/Qwen3.8-27B-pi-adapted-MLX-Attention8-BF16-Recurrence --no-mtp --reasoning off --prompt "Reply with OK."
```

See `SHA256SUMS` for file integrity.
