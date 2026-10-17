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

# Qwen3.8-27B MLX-4bit-Group32

This is a vanilla quantization of `Qwen/Qwen3.8-27B`. It is not a fine-tune,
merge, ablation, alignment change, or chat-template modification. The source
weights are pinned to commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The official checkpoint uses `Qwen3_5ForConditionalGeneration` / `qwen3_5` as
its internal architecture identifier. That string does **not** mean these
weights came from a Qwen3.5 model.

## Conversion

```json
{
  "algorithm": "MLX affine 4-bit quantization with group size 32",
  "bit_width": 4,
  "group_size": 32,
  "calibration_source": "none"
}
```

- Source tensor inventory: 1199 tensors,
  including 333 vision tensors and
  15 source MTP tensors.
- Conversion tool/runtime requirement: `mlx-vlm` / `0.6.1`.
- Artifact size: 18.628 GB (decimal).
- Expected hardware: Apple Silicon with at least 32 GB unified memory.

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
    "accepted_drafts": 84,
    "drafted_tokens": 87,
    "acceptance_rate": 0.9655172413793104,
    "baseline_tps": 14.271337491869996,
    "mtp_tps": 15.976497489790209,
    "speedup": 1.119481443059671,
    "measured_improvement": true,
    "baseline_wall_seconds": 9.261349542066455,
    "mtp_wall_seconds": 8.232959791086614,
    "advertise_acceleration": true
  },
  "bf16_source_comparison": {
    "passed": true,
    "mean_semantic_similarity": 0.9419277489185334,
    "exact_matches": 4,
    "measurements": {
      "average_generation_tps": 15.275914766074624,
      "peak_memory_gb": 19.774269782,
      "artifact_bytes": 18627492882,
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
  "bf16_fixed_logit_comparison": {
    "positions": 106,
    "inputs_sha256": "f1a0af6b9580739ebc9efa9375ae31aa6940dafe7ff6446097ed6cf8d9ab37de",
    "mean_kl_divergence": 0.01875024640335227,
    "reference_perplexity": 9.84727010437735,
    "candidate_perplexity": 10.045437946476026,
    "perplexity_delta": 0.19816784209867677,
    "top1_token_agreement": 0.9433962264150944,
    "selection": "selected",
    "warnings": [
      "KL is measured on fixed original text, not a public benchmark.",
      "BF16 log-probabilities are stored in float16 after float32 log-softmax; reported KL therefore has finite-storage approximation error."
    ]
  }
}
```

No acceleration is advertised unless the MTP report contains a measured
throughput improvement. Exact measurements are artifact-, prompt-, context-,
and hardware-specific.

## Inference

```bash
python -m pip install 'mlx==0.31.2' 'mlx-lm==0.31.3' 'mlx-vlm==0.6.1' 'huggingface-hub[cli]'
hf download Chungulus/Qwen3.8-27B-MLX-4bit-Group32 --local-dir ./qwen38-quant
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
