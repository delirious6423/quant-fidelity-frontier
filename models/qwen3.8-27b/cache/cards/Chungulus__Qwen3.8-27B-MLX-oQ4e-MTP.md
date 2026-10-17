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

# Qwen3.8-27B MLX-oQ4e-MTP

This is a vanilla quantization of `Qwen/Qwen3.8-27B`. It is not a fine-tune,
merge, ablation, alignment change, or chat-template modification. The source
weights are pinned to commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The official checkpoint uses `Qwen3_5ForConditionalGeneration` / `qwen3_5` as
its internal architecture identifier. That string does **not** mean these
weights came from a Qwen3.5 model.

## Conversion

```json
{
  "algorithm": "oMLX oQ4e language quantization with FP16 vision and native MTP",
  "bit_width": "effective_4",
  "group_size": "format_defined",
  "calibration_source": "pinned oMLX oQe packaged multilingual/code/tool/reasoning corpus; SHA-256 04d98e2d7367f7a53e3efa759acded87717d8b70562df142e1e02963af571aa0"
}
```

- Source tensor inventory: 1199 tensors,
  including 333 vision tensors and
  15 source MTP tensors.
- Conversion tool/runtime requirement: `oMLX / vllm-mlx native MTP` / `71b9d52039c3058041c5029fdb3d3e833d13d624`.
- Artifact size: 16.995 GB (decimal).
- Expected hardware: Apple Silicon with 64 GB unified memory.

Calibration source: pinned oMLX oQe packaged multilingual/code/tool/reasoning corpus; SHA-256 04d98e2d7367f7a53e3efa759acded87717d8b70562df142e1e02963af571aa0.

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
      "text": "The first ten square numbers are the squares of the integers from 1 to 10. Here is the step-by-step calculation:\n\n1. $1^2 = 1$\n2. $2^2 = 4$\n3. $3^2 = 9$\n4. $4^2 = 16$\n5. $5^2 = 25$\n6. $6^2 = 36$\n7. $7^2 = 49$\n8. $8^2 = 64$\n9. $9^2 =",
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
        "total_time": 8.53
      },
      "wall_seconds": 8.531850000028498,
      "generation_tps": 15.00586166471278
    },
    "mtp_measurement": {
      "text": "The first ten square numbers are the squares of the integers from 1 to 10. Here is the step-by-step calculation:\n\n1. $1^2 = 1$\n2. $2^2 = 4$\n3. $3^2 = 9$\n4. $4^2 = 16$\n5. $5^2 = 25$\n6. $6^2 = 36$\n7. $7^2 = 49$\n8. $8^2 = 64$\n9. $9^2 =",
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
        "total_time": 3.62
      },
      "wall_seconds": 3.624543041922152,
      "generation_tps": 35.35911602209945
    },
    "baseline_tps": 15.00586166471278,
    "mtp_tps": 35.35911602209945,
    "speedup": 2.3563535911602207,
    "measured_improvement": true,
    "advertise_acceleration": true,
    "native_stats": {
      "finish_reason": "length",
      "tokens": 128,
      "cycles": 36,
      "tokens_per_cycle": 3.56,
      "accepted_drafts": 92,
      "drafted_tokens": 95,
      "acceptance_rate": 0.968421052631579
    },
    "failure": null
  },
  "bf16_source_comparison": {
    "passed": true,
    "mean_semantic_similarity": 0.8991085827350617,
    "exact_matches": 4,
    "measurements": {
      "average_generation_tps": 21.81981842061714,
      "peak_memory_gb": 17.785847808,
      "artifact_bytes": 16994902573,
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
python -m pip install "omlx @ git+https://github.com/jundot/omlx.git@71b9d52039c3058041c5029fdb3d3e833d13d624"
hf download Chungulus/Qwen3.8-27B-MLX-oQ4e-MTP --local-dir ./models/Qwen3.8-27B-MLX-oQ4e-MTP
mkdir -p ./omlx-state
python - <<'PY'
import json
from pathlib import Path
model_id = 'Qwen3.8-27B-MLX-oQ4e-MTP'
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
