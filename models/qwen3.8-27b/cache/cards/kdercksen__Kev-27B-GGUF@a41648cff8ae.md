---
license: apache-2.0
pipeline_tag: text-classification
base_model:
- jaredpalmer/kev-27b
- Qwen/Qwen3.8-27B
tags:
- gguf
- quantized
- decision-model
- systemone
---

# Kev-27B GGUF

GGUF conversion of [jaredpalmer/kev-27b](https://huggingface.co/jaredpalmer/kev-27b) (v2, the released full bf16 weights with the pointer head), a Kev decision model on the Qwen3.8-27B base. Apache 2.0, like the source weights.

This is a decision model: it answers typed questions (`noul`, `choice`, `score`) over a state text and returns probabilities. It is served through the `/v1/systemone` API of llama.cpp (≥ 0.6.0) and LM Studio, not through chat or completion endpoints.

```bash
llama-server -m Kev-27B-Q4_K_M.gguf -np 4
curl http://localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "The package arrived two weeks late and was damaged.",
  "questions": {
    "upset":  {"type": "noul", "instructions": "Is the customer upset?"},
    "action": {"type": "choice", "instructions": "Best resolution?",
               "criteria": {"refund": "Refund the order", "replacement": "Send a replacement"}}
  }}'
```

## Files

| File | Bits/weight | Size | Notes |
|---|---|---|---|
| `Kev-27B-Q4_K_M.gguf` | 4.94 | 15.8 GB | fits a 24 GB GPU or a 32 GB Mac |
| `Kev-27B-Q6_K.gguf` | ~6.6 | 21 GB | fits an RTX 5090 (32 GB) with room for parallel slots |

The pointer head (`cls.output.weight`) is kept at Q8_0 in both files. The calibrated temperature (T = 1.3195) from `head.pt` is stored in the GGUF metadata and applied by the server.

## Conversion

Converted with llama.cpp `convert_hf_to_gguf.py` plus a small change to `conversion/lev.py` so that a Kev release shipped as merged weights (rather than a LoRA adapter) is recognised: detection via `head.pt` (`weights == "full"`), no base-model download or adapter merge. Quantized with `llama-quantize --tensor-type cls.output.weight=q8_0`.

Source commit: jaredpalmer/kev-27b `main` as of 2026-10-12 (head.pt sha256 `7968f17b…`, see the source model card).

## Caveats

- Kev is trained on English classification data; performance on other languages is transfer and should be measured on your own labelled data before thresholds are set.
- Quantization effects on calibration have not been measured for this size; the source card recommends refitting the temperature on your own data (`python -m kev.calibrate`).
- No vision input.
