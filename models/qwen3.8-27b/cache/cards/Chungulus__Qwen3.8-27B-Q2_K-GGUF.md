---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
license: apache-2.0
tags:
  - qwen3.8
  - quantized
  - gguf
  - vision-language
  - mtp
---

# Qwen3.8-27B Q2_K GGUF

This repository contains one independently downloadable `Q2_K` GGUF of
`Qwen/Qwen3.8-27B`. It is an unmodified vanilla quantization, not a fine-tune,
merge, ablation, alignment change, or chat-template modification.

The source weights are pinned to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The official checkpoint's
internal `Qwen3_5ForConditionalGeneration` / `qwen3_5` architecture identifier
does **not** mean this is a Qwen3.5 model.

## Files and conversion

- Main model: `Qwen3.8-27B-Q2_K.gguf` (10.865 GB decimal)
- Vision projector: `mmproj-Qwen3.8-27B-F16.gguf`
- Complete repository size: 11.805 GB decimal
- Quantization: `Q2_K` using llama.cpp K/IQ quantization
- llama.cpp revision: `5f754ea0e2fd21e1213db7ebebfd65d938d9d69c`
- Calibration: none for K-quants; representative local prompts where required by IQ conversion
- Source inventory: 1,199 tensors, including 333 vision tensors and 15 source MTP tensors

## Validated component status

- Text generation: passed
- Tool calling: passed all five native-format cases
- Vision and video: passed all three deterministic local-image cases
- Thinking controls and chat template: preserved and checked against the locked source
- Hybrid Gated DeltaNet/full-attention structure: retained
- MTP tensors and metadata: retained; speculative acceleration is **not** advertised
- Per-file runtime smoke: passed at 18.920043894501834 generated tokens/s on the validation host
- Weight SHA-256: `185bc2aa1f15a34ef9d2b14f51f9edda69f4e43b5ee9e39e3c2ae8c439ddd1a5`
- Validation lineage: the same bytes passed in the combined release at
  [`f519a212d6c15cd3292b6ca835dd8ebf235642c0`](https://huggingface.co/Chungulus/Qwen3.8-27B-GGUF/tree/f519a212d6c15cd3292b6ca835dd8ebf235642c0)

`validation_result.json` records the exact per-file smoke result and its inherited
aggregate functional gates. These split repositories do not claim a new benchmark
run; they preserve byte-identical, already-validated artifacts.

## Inference

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 5f754ea0e2fd21e1213db7ebebfd65d938d9d69c
cmake -S . -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

hf download Chungulus/Qwen3.8-27B-Q2_K-GGUF --local-dir ../qwen38-q2_k
cd ../qwen38-q2_k
../llama.cpp/build/bin/llama-mtmd-cli \
  -m ./Qwen3.8-27B-Q2_K.gguf \
  --mmproj ./mmproj-Qwen3.8-27B-F16.gguf \
  -p 'Describe the image.' --image ./image.png
```

Allow at least approximately 15 GB of available memory for the model,
vision projector, and modest runtime overhead; KV-cache needs grow with context.
This is planning guidance, not a measured peak-memory guarantee.

Use the source chat-template controls for `enable_thinking`, `reasoning_effort`,
and `preserve_thinking`, together with Qwen's native tool-call format.

## Limitations

Quantization can reduce quality, particularly at lower bit widths. The longest
recorded validation prompt was 73 tokens; do not infer that the architectural
maximum context was tested. A runtime must support the Qwen3.8 hybrid language
graph, vision tower/projector, tokenizer, and retained MTP metadata rather than
loading only a single language tensor.

## License and attribution

The parent model and this vanilla quantization use the source model's Apache-2.0
license. See [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
