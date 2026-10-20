---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: image-text-to-text
language:
- en
tags:
- gguf
- llama.cpp
- ornith
- qwen3.5-moe
- multimodal
- abliterated
---

# Ornith 1.5 35B-A3B Abliterated GGUF

One-repository GGUF release of an unofficial abliterated derivative of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), pinned to
revision `e4dfb35a93d4b6822a811a7676f3488514abe7e2`. The original model is by Ornith AI. PocketAI Model
Lab performed the refusal-direction edit, GGUF conversion, and validation.

## Important safety notice

This model was deliberately modified to suppress learned refusal behavior. It
may produce harmful, illegal, offensive, deceptive, or dangerously incorrect
content more readily than the upstream instruction model. Abliteration is not
truthfulness training, a capability improvement, or a guarantee of universal
compliance. Evaluate and constrain it for your use case.

## Files

| File | Size | Guidance |
| --- | ---: | --- |
| `Ornith-1.5-35B-A3B-Abliterated-Q4_K_M.gguf` | 19.71 GiB | Recommended starting point for local use |
| `Ornith-1.5-35B-A3B-Abliterated-Q8_0.gguf` | 34.37 GiB | Higher-fidelity quantization |
| `Ornith-1.5-35B-A3B-Abliterated-BF16.gguf` | 64.61 GiB | Unquantized reference |
| `mmproj-Ornith-1.5-35B-A3B-Abliterated-F16.gguf` | 0.84 GiB | Shared F16 vision projector |

The language GGUFs use the validated abliterated BF16 checkpoint. The vision
projector uses the exact pinned upstream vision tower because the abliteration
did not modify vision weights. The native MTP speculative-decoding head is not
included.

## Text usage

Download one language file, then run a recent llama.cpp build:

```bash
huggingface-cli download PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-GGUF \
  Ornith-1.5-35B-A3B-Abliterated-Q4_K_M.gguf --local-dir .

llama-cli -m Ornith-1.5-35B-A3B-Abliterated-Q4_K_M.gguf \
  -ngl all -c 4096 -n 256 \
  -p "Explain why seasons occur."
```

## Vision usage

Download the matching language model and the shared projector:

```bash
huggingface-cli download PocketAiHub/Ornith-1.5-35B-A3B-Abliterated-GGUF \
  Ornith-1.5-35B-A3B-Abliterated-Q4_K_M.gguf \
  mmproj-Ornith-1.5-35B-A3B-Abliterated-F16.gguf --local-dir .

llama-mtmd-cli -m Ornith-1.5-35B-A3B-Abliterated-Q4_K_M.gguf \
  --mmproj mmproj-Ornith-1.5-35B-A3B-Abliterated-F16.gguf \
  --image photo.jpg -p "Describe this image."
```

Q4_K_M plus the projector passed an end-to-end image smoke test. The Q8_0 and
BF16 language files passed text inference but did not receive separate vision
smoke tests.

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

## GGUF behavior screen

The deterministic llama.cpp Metal screen used 100 JailbreakBench harmful
prompts and 100 benign controls per quantized model, a 256-token cap, batch 1,
thinking disabled, and a transparent phrase-based refusal detector.

| Variant | Harmful refusal flags | Benign refusal flags | Final-answer segments |
| --- | ---: | ---: | ---: |
| Q4_K_M | 3/100 | 0/100 | 200/200 |
| Q8_0 | 1/100 | 0/100 | 200/200 |
| BF16 confirmation | 0/12 | 0/12 | 24/24 |

The flagged quantized responses were manually confirmed as genuine refusals.
Most generations reached the 256-token ceiling. This is an early-behavior
screen, not proof of universal compliance, safety, correctness, or full answer
quality.

## Matched GGUF quantization drift

The comparison reused 36 prompts and 481 shared teacher positions. At every
position it measured exact `D_KL(P_BF16 || P_quantized)` over all 248,320
vocabulary logits.

| Variant vs GGUF BF16 | Mean KL (nats) | Top-token agreement |
| --- | ---: | ---: |
| Q8_0 | 0.01359 | 96.26% |
| Q4_K_M | 0.07280 | 90.64% |

The BF16 self-check returned 2.98e-09
mean KL and 100% top-token agreement. llama.cpp did not expose the residual and
KV/recurrent-state tensors captured in the separate MLX analysis.

Machine-readable aggregate results are in
[`validation-summary.json`](./validation-summary.json), and conversion checks
and SHA-256 values are included in the repository manifests.

## Conversion provenance

- llama.cpp revision: `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
- Qwen 3.5 MoE conversion supports the MLX stacked-expert layout
- The converter avoids applying the Qwen 3.5 RMSNorm unit offset twice
- MTP included: no

## License and attribution

The upstream model card declares MIT. This derivative preserves the upstream
attribution and links to the exact source revision above.
