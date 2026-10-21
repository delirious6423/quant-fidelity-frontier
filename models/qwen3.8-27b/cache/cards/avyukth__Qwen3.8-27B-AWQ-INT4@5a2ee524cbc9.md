---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - qwen3.8
  - qwen3_8
  - qwen
  - qwen3_5
  - awq
  - int4
  - w4a16
  - compressed-tensors
  - vllm
  - quantized
---

# Qwen3.8-27B — AWQ INT4 (compressed-tensors)

4-bit weight quantisation of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
built so the model fits a single **24 GB** consumer GPU.

> Note on the `qwen3_5` tag: Qwen3.8 is implemented with the Qwen3.5
> architecture (`model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`), so
> the Hub derives that family tag automatically — the upstream
> `Qwen/Qwen3.8-27B` repo carries it too. This is Qwen **3.8**.

**55.6 GB → 18 GB.** Verified serving on an RTX 3090 (sm_86) under vLLM.

| | |
|---|---|
| Language model | INT4, group size 128, symmetric (`pack-quantized`) |
| Vision tower | **bf16, not quantised** (see *Why the vision tower is bf16*) |
| `lm_head`, MTP head | bf16 |
| Quantised `Linear` modules | 496 |
| Size on disk | 18 GB |
| Weights in VRAM | 17.68 GiB |

## Usage

### vLLM (recommended)

```bash
vllm serve avyukth/Qwen3.8-27B-AWQ-INT4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.93 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Text-only serving reclaims the vision encoder cache (~5k extra KV tokens on a
24 GB card):

```bash
  --limit-mm-per-prompt '{"image": 0}'
```

### transformers

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "avyukth/Qwen3.8-27B-AWQ-INT4", device_map="auto")
processor = AutoProcessor.from_pretrained("avyukth/Qwen3.8-27B-AWQ-INT4")
```

## Measured results

Measured on an RTX 3090 (24 GB, sm_86), vLLM, `max_model_len=8192`.

| Benchmark | Result |
|---|---|
| GSM8K (exact match, **n = 40**, temperature 0) | **39/40 = 97.5%** |
| Vision: read rendered text | exact |
| Vision: identify shape / colour / position | correct |

**The GSM8K figure is on 40 problems, not the full 1319-item test set.** It is
enough to show reasoning survived quantisation; it is not a leaderboard number,
and no bf16 side-by-side was run, so the exact delta from the base model is
unmeasured.

### Throughput (RTX 3090, 256 tokens/request, `ignore_eos`)

| Concurrency | Aggregate tok/s | Per request | TTFT p50 |
|---|---|---|---|
| 1 | 41.1 | 41.1 | 0.38 s |
| 4 | 136.4 | 34.1 | 1.28 s |
| **8** | **221.5** | 27.7 | 2.21 s |
| 16 | 216.9 | 13.6 | 3.14 s |
| 32 | 211.8 | 6.6 | 13.72 s |

Throughput saturates at concurrency 8; beyond that, aggregate is flat and only
latency grows.

## Limitations

* **48 of 64 layers received INT4 without activation-aware scaling.**
  Qwen3.5/3.8 interleaves 16 full-attention layers with 48 `linear_attn`
  (`Qwen3_5GatedDeltaNet`) layers. AWQ's scaling search must replay a parent
  module to collect reference outputs, and `Qwen3_5GatedDeltaNet.forward` has
  the runtime signature `(self, *args, **kwargs)` — a decorator drops the real
  one — so llm-compressor's captured arguments collapse into a single nested
  `kwargs` key and replay fails. Those layers are therefore quantised by
  round-to-nearest rather than AWQ. Their MLPs and all full-attention layers
  *do* get proper AWQ scaling. No measurable reasoning cost was found
  (GSM8K above), but this is not the same as a full AWQ model.
* **Vision tower is bf16.** Its `intermediate_size` is 4304, which is not
  divisible by `group_size=128` (4304/128 = 33.625; the only divisor under 256
  is 16), so those layers cannot be group-quantised at all. This costs 0.92 GB.
  Per-channel quantisation would be the workaround if that matters.
* **Marlin thread-tile padding.** vLLM warns that some GDN projection shapes
  need padding, so those layers pad/slice on every forward. Correctness is
  unaffected; some throughput is lost.
* **Symmetric W4A16**, chosen for Marlin support on Ampere. Asymmetric
  (`W4A16_ASYM`) tracks AWQ's usual formulation slightly more closely.
* Calibration: 128 samples × 512 tokens from `HuggingFaceH4/ultrachat_200k`.
  Calibration data is general chat, not domain-specific.

## Build recipe

Produced with `llm-compressor` 0.13.0. Two details that are easy to get wrong
on this architecture:

```python
from llmcompressor.modifiers.transform.awq import AWQModifier   # NOT modifiers.awq
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq.dynamic_mappings import (
    build_hybrid_attention_mappings)

# The dynamic builder emits layer-index-scoped rules per attention kind.
# Drop only the linear-attention rule; the rest keep proper AWQ scaling.
maps = [m for m in build_hybrid_attention_mappings(model)
        if not any("linear_attn" in b for b in m.balance_layers)]

recipe = [
    AWQModifier(mappings=maps, duo_scaling="both"),
    QuantizationModifier(targets=["Linear"], scheme="W4A16",
                         ignore=["lm_head", "re:.*visual.*", "re:.*mtp.*"]),
]
```

Load the model with the class its config declares
(`Qwen3_5ForConditionalGeneration`), **not** `AutoModelForCausalLM` — the
latter resolves to the text-only class, which silently drops the vision tower
and writes a config that no longer matches the weights.

Save the tokenizer from a **fresh** instance. HF fast tokenizers persist
truncation state, so saving the tokenizer used for calibration bakes
`"truncation": {"max_length": 512}` into `tokenizer.json`, which silently
clips every prompt beyond the calibration length.

## License

Apache 2.0, inherited from the base model. All credit for the model itself
goes to the Qwen team; this repository contains only a quantisation of their
weights.
