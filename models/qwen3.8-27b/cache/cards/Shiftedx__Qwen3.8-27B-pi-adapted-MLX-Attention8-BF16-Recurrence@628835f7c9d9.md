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

# Qwen3.8-27B pi-adapted · MLX

[Pi LoRA](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted/tree/e786f8491742268f9b2bc729697b13bfac26d4ad) merged into [Qwen3.8-27B BF16](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0). Attention8 + BF16 recurrence, with vision and the native BF16 MTP sidecar. About 22 GiB for the target.

## Run

Kapitsa + DFlash2 is the tested fast path for this quant. Tested on an M4 Max with 64 GiB using Kapitsa 0.2.31.

```bash
hf download z-lab/Qwen3.8-27B-DFlash2 --revision \
  50307d4c4cde6860d4eee73e2547cd786fe8e8a4 --local-dir ./dflash2

kapitsa serve --model Shiftedx/Qwen3.8-27B-pi-adapted-MLX-Attention8-BF16-Recurrence@7ebb7a2d750b863a3c31704fa46fc8ed2f5aee48 \
  --dflash ./dflash2 --context 133120 --slots 1 --disk-cache 0 \
  --host 127.0.0.1 --port 8000
```

Point your OpenAI-compatible client at `http://127.0.0.1:8000/v1`. For the validated thinking-on settings, send `reasoning_effort: "medium"` and use greedy sampling (`temperature: 0`, `top_p: 1`, `top_k: 0`). For target-only decoding, replace `--dflash ./dflash2` with `--dflash none --spec 0`.

## Verified

With thinking on, both modes passed all 15 [Shiftedx Bench context cases](https://huggingface.co/datasets/Shiftedx/shiftedx-bench/tree/9a43027a84dc4a4a283220f58314b5768245ec58) and matched on 15/15 captured responses. Thinking-on DFlash2 tool, agentic, and vision parity remains unverified.
Both modes passed the 65,536- and 131,072-token sentinels with exact prompt counts.

86.3 tok/s median decode with DFlash2 across the context lane (19.1 target-only). Cold 131K requests took about 12 minutes in both modes.
