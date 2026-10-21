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

# Qwen3.8-27B MLX-6bit

This is a vanilla quantization of `Qwen/Qwen3.8-27B`. It is not a fine-tune,
merge, ablation, alignment change, or chat-template modification. The source
weights are pinned to commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The official checkpoint uses `Qwen3_5ForConditionalGeneration` / `qwen3_5` as
its internal architecture identifier. That string does **not** mean these
weights came from a Qwen3.5 model.

## Conversion

```json
{
  "algorithm": "MLX affine weight quantization with FP16 vision and MTP drafter companion",
  "bit_width": 6,
  "group_size": 64,
  "calibration_source": "none"
}
```

- Source tensor inventory: 1199 tensors,
  including 333 vision tensors and
  15 source MTP tensors.
- Conversion tool/runtime requirement: `mlx-vlm plus mlx-mtp compatible drafter` / `mlx-vlm 0.6.1; mlx 0.31.2; mlx-lm 0.31.3`.
- Artifact size: 23.670 GB (decimal).
- Expected hardware: Apple Silicon with 64 GB unified memory.

Calibration source: none.

## Component status

- Text: passed release tests.
- Vision/video: passed deterministic local image tests.
- Tool calling: passed all native XML tool tests.
- MTP: loaded and passed a temperature-zero equivalence and throughput A/B.
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
    "drafter_kind": "mtp",
    "output_equivalent_temperature_zero": true,
    "accepted_drafts": 32,
    "drafted_tokens": 34,
    "acceptance_rate": 0.9411764705882353,
    "baseline_tps": 9.478878586754751,
    "mtp_tps": 11.958225851342917,
    "speedup": 1.261565462823067,
    "measured_improvement": true,
    "baseline_wall_seconds": 5.657700166921131,
    "mtp_wall_seconds": 4.457088207942434,
    "advertise_acceleration": true
  },
  "bf16_source_comparison": {
    "passed": true,
    "mean_semantic_similarity": 0.988053274154663,
    "exact_matches": 7,
    "measurements": {
      "average_generation_tps": 9.953450085041483,
      "peak_memory_gb": 25.573573157,
      "artifact_bytes": 23669991658,
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
python -m pip install 'mlx==0.31.2' 'mlx-lm==0.31.3' 'mlx-vlm==0.6.1' 'huggingface-hub[cli]'
hf download Chungulus/Qwen3.8-27B-MLX-6bit --local-dir ./qwen38-quant
python -m mlx_vlm.generate --model ./qwen38-quant --draft-model ./qwen38-quant/mtp-drafter --draft-kind mtp --draft-block-size 3 --prompt 'Describe this image.' --image ./image.png --max-tokens 256 --no-verbose
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
