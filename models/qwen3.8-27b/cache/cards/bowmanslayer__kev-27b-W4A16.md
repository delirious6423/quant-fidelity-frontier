---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- jaredpalmer/kev-27b
base_model_relation: quantized
pipeline_tag: text-classification
tags:
- kev
- auto-round
- int4
- experimental
---
# Kev-27B W4A16 — experimental community quantization

Community conversion of [jaredpalmer/kev-27b](https://huggingface.co/jaredpalmer/kev-27b). Kev selects and scores supplied options using a pointer head; it is not a text-generation chat model. This repository contains the full merged quantized backbone, tokenizer and original decision head. Do not apply the upstream LoRA again.

## Status and limitations

Source revisions, tensor packing, shard/index consistency, file hashes and unchanged head have been checked. **Inference, decision quality, latency and deployment compatibility have not been validated in this release workflow.** No upstream benchmark score should be interpreted as a score for this quantization.



## Format and provenance

AutoRound 0.12.3, W4A16, symmetric group size 128, `auto_round:auto_gptq` packing. Eligible projections are INT4; small hybrid projections remain 16-bit. Activations and retained weights are BF16; the original pointer head remains FP32, with its original temperature. Calibration used 256 rows (192 public Kev calibration rows and 64 independent Chinese synthetic rows), sequence length 512, 200 iterations, batch size 4. The quantization-time attention-mask replay correction is not a runtime hook.

- Adapter: `jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195`
- Base: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Kev code: `jaredpalmer/kev@0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91`
- The backbone uses the Qwen3.5 text architecture, including the 27B source named Qwen3.8.
- See `release.json` and `SHA256SUMS` for public provenance and integrity.

## Loading

Use Linux, a CUDA GPU and a compatible AutoRound Triton environment. The conversion environment used PyTorch 2.10.0+cu128, Transformers 5.17.0, AutoRound 0.12.3, PEFT 0.21.1 and Accelerate 1.15.0. This is an observed environment, not a validated support matrix: pinned upstream Kev declares PyTorch <2.9. The included loader is adapted from the 4B runtime; 9B/27B loader execution remains unvalidated. 27B retains BF16 and requires BF16-capable hardware. Allow GPU memory beyond the weight-file size for runtime buffers and inputs.

Clone the pinned Kev source and place it on `PYTHONPATH` in your prepared environment. Download this repository with `huggingface_hub.snapshot_download`, then run from the downloaded directory (so `load_kev.py` is importable):

```python
import torch
from load_kev import load
from kev.api import SystemOneRequest, to_record

tokenizer, model, info = load(".", device="cuda:0")
request = SystemOneRequest.model_validate({
    "state": "The parcel has been delivered.",
    "questions": {"status": {"type": "choice", "criteria": {
        "delivered": "Delivery is complete", "pending": "Waiting for delivery"
    }}}
})
record, keys = to_record(request)
encoded = model.encode(tokenizer, record, max_state=2048, max_branch=3072, strict=True)
with torch.inference_mode():
    probabilities = model.probs(encoded)
print(dict(zip(keys[0]["keys"], probabilities[0].tolist())))
```

The loader uses `auto_round:tritonv2_zp`; the zero-point-aware backend matters for this packing. It reuses upstream option encoding and decision semantics, and does not rely on `trust_remote_code` or generic text-generation pipelines. The `.pt` head is the original upstream file and is read with `weights_only=True`.

## License and attribution

Apache-2.0. Original Kev adapter, head and code by Jared Palmer; base model by Qwen. See `LICENSE` and `NOTICE`. This is an unofficial community conversion; upstream authors do not endorse its quality. No calibration dataset, private deployment configuration or evaluation service is included.
