---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
license: apache-2.0
tags:
  - qwen3.8
  - quantized
  - vision-language
  - mtp
---

# Qwen3.8-27B GGUF

This is a vanilla quantization of `Qwen/Qwen3.8-27B`. It is not a fine-tune,
merge, ablation, alignment change, or chat-template modification. The source
weights are pinned to commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The official checkpoint uses `Qwen3_5ForConditionalGeneration` / `qwen3_5` as
its internal architecture identifier. That string does **not** mean these
weights came from a Qwen3.5 model.

## Conversion

```json
{
  "algorithm": "llama.cpp stable K-quants and IQ quants from one F16 GGUF",
  "bit_width": [
    2,
    3,
    4,
    5,
    6,
    8
  ],
  "group_size": "format_defined",
  "calibration_source": "none for K-quants; local representative prompts for IQ variants if required"
}
```

- Source tensor inventory: 1199 tensors,
  including 333 vision tensors and
  15 source MTP tensors.
- Conversion tool/runtime requirement: `llama.cpp` / `5f754ea0e2fd21e1213db7ebebfd65d938d9d69c`.
- Artifact size: 206.295 GB (decimal).
- Expected hardware: Apple Silicon Metal or CPU; 120 GB temporary headroom.

Calibration source: none for K-quants; local representative prompts for IQ variants if required.

## Component status

- Text: passed release tests.
- Vision/video: passed deterministic local image tests.
- Tool calling: passed all native XML tool tests.
- MTP: source MTP tensors were retained by the structural gate; this repository does not claim speculative acceleration.
- Chat template, tokenizer, processor, generation config, and special-token IDs:
  checked against the locked source by the structural gate.
- Quality comparison: passed against the
  locked BF16 source using the exact same functional cases. Semantic similarity
  uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at
  `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`
  as a measured proxy, not as ground-truth accuracy.
- Longest recorded validation prompt: 73 prompt tokens. This is a measured
  test boundary, not a claim that the architectural maximum was exercised.

## Validation results

```json
{
  "release_gate": "PASS",
  "text": [
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true
  ],
  "tools": [
    true,
    true,
    true,
    true,
    true
  ],
  "vision": [
    true,
    true,
    true
  ],
  "mtp": {
    "passed": true,
    "acceleration_claimed": false,
    "retention_gate": "GGUF tensor and nextn metadata inspection",
    "advertise_acceleration": false
  },
  "bf16_source_comparison": {
    "passed": true,
    "mean_semantic_similarity": 0.9082151889801026,
    "exact_matches": 5,
    "measurements": {
      "average_generation_tps": 8.600119274947904,
      "peak_memory_gb": null,
      "artifact_bytes": 206294771716,
      "maximum_prompt_tokens_tested": 73,
      "loop_rate": 0.0
    },
    "evaluator": {
      "repo_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
      "pooling": "attention-mask mean pooling followed by L2 normalization",
      "maximum_tokens": 256
    }
  },
  "bf16_fixed_logit_comparison": "not applicable to the original 30-slot matrix"
}
```

No acceleration is advertised unless the MTP report contains a measured
throughput improvement. Exact measurements are artifact-, prompt-, context-,
and hardware-specific.

## Inference

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 5f754ea0e2fd21e1213db7ebebfd65d938d9d69c
cmake -S . -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
hf download Chungulus/Qwen3.8-27B-GGUF --local-dir ../qwen38-gguf
cd ../qwen38-gguf
../llama.cpp/build/bin/llama-mtmd-cli -m ./Qwen3.8-27B-Q4_K_M.gguf --mmproj ./mmproj-Qwen3.8-27B-F16.gguf -p 'Describe the image.' --image ./image.png
```

Use the exact source chat-template controls for thinking (`enable_thinking`,
`reasoning_effort`, and `preserve_thinking`) and the native Qwen tool format.

## Limitations

Quantization can reduce quality, especially at very low bit widths. Runtime
support for the hybrid Gated DeltaNet/full-attention graph, vision tower,
projector, processor, and MTP component is format-specific. A loader that reads
only a language tensor is not sufficient. Tested context length and resource
measurements are recorded in `validation_result.json`; untested context lengths
must not be inferred from the architectural maximum.

## License and attribution

The parent model and this unmodified quantization are distributed under the
source model's Apache-2.0 license. See the
[official Qwen3.8-27B repository](https://huggingface.co/Qwen/Qwen3.8-27B) for
the upstream model card and attribution.
