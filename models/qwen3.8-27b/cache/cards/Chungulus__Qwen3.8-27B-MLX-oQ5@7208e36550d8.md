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

# Qwen3.8-27B MLX-oQ5

This is a vanilla quantization of `Qwen/Qwen3.8-27B`. It is not a fine-tune,
merge, ablation, alignment change, or chat-template modification. The source
weights are pinned to commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The official checkpoint uses `Qwen3_5ForConditionalGeneration` / `qwen3_5` as
its internal architecture identifier. That string does **not** mean these
weights came from a Qwen3.5 model.

## Conversion

```json
{
  "algorithm": "oMLX oQ5 sensitivity-driven mixed-precision quantization",
  "bit_width": "mixed around 5.5 bpw",
  "group_size": "64 with mode-specific overrides",
  "calibration_source": "local fixed representative prompts; no benchmark answers"
}
```

- Source tensor inventory: 1199 tensors,
  including 333 vision tensors and
  15 source MTP tensors.
- Conversion tool/runtime requirement: `oMLX and standard MLX loaders` / `71b9d52039c3058041c5029fdb3d3e833d13d624`.
- Artifact size: 20.314 GB (decimal).
- Expected hardware: Apple Silicon with at least 64 GB unified memory.

Calibration source: local fixed representative prompts; no benchmark answers.

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
    "backend": "oMLX Lightning MTP (integrated source MTP head)",
    "output_equivalent_temperature_zero": true,
    "baseline": {
      "text": "To list the first ten square numbers, we calculate the square of each integer from 1 to 10 ($n^2$).\n\n1.  $1^2 = 1$\n2.  $2^2 = 4$\n3.  $3^2 = 9$\n4.  $4^2 = 16$\n5.  $5^2 = 25$\n6.  $6^2 = 36$\n7.  $7^2 = 49$\n8.  $8^2 = 64",
      "finish_reason": "length",
      "usage": {
        "prompt_tokens": 21,
        "completion_tokens": 128,
        "total_tokens": 149,
        "input_tokens": 21,
        "output_tokens": 128,
        "prompt_tokens_details": {
          "cached_tokens": 0
        },
        "total_time": 10.37
      },
      "wall_seconds": 10.375257540959865,
      "generation_tps": 12.343297974927676
    },
    "mtp_measurement": {
      "text": "To list the first ten square numbers, we calculate the square of each integer from 1 to 10 ($n^2$).\n\n1.  $1^2 = 1$\n2.  $2^2 = 4$\n3.  $3^2 = 9$\n4.  $4^2 = 16$\n5.  $5^2 = 25$\n6.  $6^2 = 36$\n7.  $7^2 = 49$\n8.  $8^2 = 64",
      "finish_reason": "length",
      "usage": {
        "prompt_tokens": 21,
        "completion_tokens": 128,
        "total_tokens": 149,
        "input_tokens": 21,
        "output_tokens": 128,
        "prompt_tokens_details": {
          "cached_tokens": 0
        },
        "total_time": 4.23
      },
      "wall_seconds": 4.238896624883637,
      "generation_tps": 30.260047281323875
    },
    "baseline_tps": 12.343297974927676,
    "mtp_tps": 30.260047281323875,
    "speedup": 2.4515366430260044,
    "measured_improvement": true,
    "advertise_acceleration": true,
    "native_stats": {
      "finish_reason": "length",
      "tokens": 128,
      "cycles": 36,
      "tokens_per_cycle": 3.56,
      "accepted_drafts": 93,
      "drafted_tokens": 96,
      "acceptance_rate": 0.96875
    },
    "failure": null
  },
  "bf16_source_comparison": {
    "passed": true,
    "mean_semantic_similarity": 0.8852966785430908,
    "exact_matches": 2,
    "measurements": {
      "average_generation_tps": 19.016428184965864,
      "peak_memory_gb": 21.0919424,
      "artifact_bytes": 20314066494,
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
    "mean_kl_divergence": 0.00603105514398161,
    "reference_perplexity": 9.84727010437735,
    "candidate_perplexity": 9.75069257534557,
    "perplexity_delta": -0.09657752903178007,
    "top1_token_agreement": 0.9811320754716981,
    "selection": "selected",
    "warnings": [
      "KL is measured on fixed original text, not a public benchmark.",
      "BF16 log-probabilities are stored in float16 after float32 log-softmax; reported KL therefore has finite-storage approximation error.",
      "The stock MLX-VLM logit scorer ignored 29 strict-loader extras, all proven to be under language_model.mtp; native oMLX validation separately loaded and tested MTP."
    ]
  }
}
```

No acceleration is advertised unless the MTP report contains a measured
throughput improvement. Exact measurements are artifact-, prompt-, context-,
and hardware-specific.

## Inference

```bash
python -m pip install "omlx @ git+https://github.com/jundot/omlx.git@71b9d52039c3058041c5029fdb3d3e833d13d624"
hf download Chungulus/Qwen3.8-27B-MLX-oQ5 --local-dir ./models/Qwen3.8-27B-MLX-oQ5
mkdir -p ./omlx-state
python - <<'PY'
import json
from pathlib import Path
model_id = 'Qwen3.8-27B-MLX-oQ5'
Path('omlx-state/model_settings.json').write_text(json.dumps({
    'version': 1, 'models': {model_id: {
        'mtp_enabled': True, 'mtp_num_draft_tokens': 3
    }}
}, indent=2) + '\n')
PY
omlx serve --model-dir ./models --base-path ./omlx-state --port 8000
```

Then send OpenAI-compatible multimodal chat requests to `http://127.0.0.1:8000/v1/chat/completions`.

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
