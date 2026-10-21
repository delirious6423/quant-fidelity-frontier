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

# Qwen3.8-27B JANG_4S

This is a vanilla quantization of `Qwen/Qwen3.8-27B`. It is not a fine-tune,
merge, ablation, alignment change, or chat-template modification. The source
weights are pinned to commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The official checkpoint uses `Qwen3_5ForConditionalGeneration` / `qwen3_5` as
its internal architecture identifier. That string does **not** mean these
weights came from a Qwen3.5 model.

## Conversion

```json
{
  "algorithm": "official JANG_4S adaptive mixed-precision MSE quantization",
  "bit_width": "approximately_4.1_average",
  "group_size": "converter_auto",
  "calibration_source": "none; official JANG MSE weight-only conversion"
}
```

- Source tensor inventory: 1199 tensors,
  including 333 vision tensors and
  15 source MTP tensors.
- Conversion tool/runtime requirement: `JANG/MLX` / `2.5.46`.
- Artifact size: 17.081 GB (decimal).
- Expected hardware: Apple Silicon.

Calibration source: none; official JANG MSE weight-only conversion.

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
    "preserved": true,
    "tensor_count": 31,
    "metadata_mode": "preserved_enabled",
    "runtime_supported": false,
    "measurement_required": false,
    "acceptance_rate": null,
    "baseline_tps": null,
    "mtp_tps": null,
    "speedup": null,
    "measured_improvement": false,
    "advertise_acceleration": false,
    "limitation": "JANG 2.5.46 preserves the Qwen3.8 MTP tensors, but its public mlx-vlm loader filters them because the model class has no native MTP module; no acceptance-rate or speed A/B can be measured."
  },
  "bf16_source_comparison": {
    "passed": true,
    "mean_semantic_similarity": 0.9505505561828613,
    "exact_matches": 4,
    "measurements": {
      "average_generation_tps": 17.21587225265724,
      "peak_memory_gb": 19.511131463,
      "artifact_bytes": 17080508919,
      "maximum_prompt_tokens_tested": 73,
      "loop_rate": 0.0
    },
    "evaluator": {
      "repo_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
      "pooling": "attention-mask mean pooling followed by L2 normalization",
      "maximum_tokens": 256
    }
  }
}
```

No acceleration is advertised unless the MTP report contains a measured
throughput improvement. Exact measurements are artifact-, prompt-, context-,
and hardware-specific.

## Inference

```bash
python -m pip install vllm huggingface-hub
vllm serve Chungulus/Qwen3.8-27B-JANG_4S
```

Use `JANG/MLX` at or above `2.5.46`; the exact validated container digest is recorded in `quantization_manifest.json`.

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
