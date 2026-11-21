---
base_model:
  - Qwen/Qwen3.8-27B
  - harness-hub/Qwen3.8-27B-pi-adapted
base_model_relation: quantized
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - kapitsa
  - apple-silicon
  - quantized
  - oq6
  - mtp
  - speculative-decoding
  - dflash2
  - pi
  - agents
---

# Qwen3.8-27B pi-adapted — oQ6, bf16 scales, MTP sidecar, DFlash2 (for Kapitsa)

[harness-hub's pi adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted),
a LoRA trained with RL to drive the [pi](https://github.com/earendil-works/pi)
coding agent, **merged into
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** and quantized to
**oQ6** so that [Kapitsa](https://github.com/nace-ai/kapitsa) can serve it on
Apple Silicon: no LoRA support needed at serving time, with the stock MTP head
in a sidecar and a DFlash2 drafter bundled.

All of the model's behaviour is harness-hub's work. This repo only merges,
quantizes and packages it. The adapter's own evaluation is in
[`SOURCE_MODEL_CARD.md`](SOURCE_MODEL_CARD.md) (SWE-bench Pro hard-51
39.2% → 64.7%, Terminal-Bench 2.1 39.5% → 55.8%, under pi, thinking off).
**Those numbers were measured on BF16 weights with vLLM and were not
re-measured on this 6-bit build.**

## Differences from the adapter as published

- **Merged and quantized.** The adapter's card warns that lower-bit bases
  drift further from the trained behaviour. oQ6 keeps 50 sensitive modules
  at 8-bit and everything else at 6-bit group 64, well above the Q4 GGUF bases
  the card mentions, but it is not BF16.
- **Thinking is on by default**, as in the base model's template (effort
  `xhigh` unless `reasoning_effort` is set). The adapter was *trained* with
  thinking off, and its card reports thinking on as level on Terminal-Bench 2.1
  (56.6% both ways) and ahead on SWE-bench Pro (74.5% vs 64.7%, not
  significant on 51 tasks) at about 4.5× the tokens. Send
  `"chat_template_kwargs": {"enable_thinking": false}` for the trained mode.
  With thinking on, allow up to 32k output tokens per call: the card found a
  16k cap cut reasoning short.
- Full list in [`CHANGES.md`](CHANGES.md); provenance and settings in
  [`conversion_metadata.json`](conversion_metadata.json).

## Requirements

- **Kapitsa 0.2.25 or newer** (6-bit MTP sidecar). Tested on 0.2.37.
- Apple Silicon with about 28 GB free: 22.5 GB of weights, 1.86 GB resident
  drafter, plus KV.

## Download and run

Kapitsa's catalog is compiled into the binary, so a third-party model is
loaded by **folder path**, not by name:

```bash
hf download jongstra/Qwen3.8-27B-pi-adapted-oQ6-bf16-mtp-dflash2-kapitsa \
    --local-dir ~/.cache/kapitsa/models/qwen3.8-27b-pi-adapted-oq6

kapitsa serve --model ~/.cache/kapitsa/models/qwen3.8-27b-pi-adapted-oq6
```

Putting it under `~/.cache/kapitsa/models/` makes `kapitsa list` show it
(`.kapitsa/manifest.json` is included). Expected log lines:

```
MTP head loaded in 191 ms (dense, trunk hidden pre-norm, drafts over the full vocabulary)
loaded …: 64 layers, dense, 22.5 GB mapped; weights 6-bit group 64 87%, 8-bit group 64 13%
DFlash2 drafter …/dflash2: 5 layers, hidden 5120, 4-bit projections, 1.86 GB resident
engine ready: qwen3.8-27b-pi-adapted-oq6, context 262144, KFlash speculative decoding (DFlash2 lattice, tree verify), images on
```

### With pi

Point pi at the server with the **`openai-completions`** API. The adapter was
not trained against the Responses API:

```json
{
  "providers": {
    "kapitsa": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "models": [{ "id": "qwen3.8-27b-pi-adapted-oq6", "maxTokens": 32768 }]
    }
  }
}
```

The adapter's evaluations sampled at temperature 0.8, top-p 1.0.

## Speed

Mac Studio M2 Ultra (60-core GPU, 64 GB), Kapitsa 0.2.37, `--slots 1`, 700-token
generations, temperature 0.3, thinking off, median of three. The Swift 1.5 oQ6
Kapitsa build was run with the same script in the same session for comparison:

| build | prose | code |
|---|---:|---:|
| **this model** | **55.9 tok/s** | **94.2 tok/s** |
| Swift 1.5 Qwen3.8-27B oQ6 (Kapitsa) | 52.4 tok/s | 96.8 tok/s |

Runs overlap (this model 53.9–56.8 / 93.4–102.1; Swift 51.1–54.1 /
94.5–102.4), so the two are tied: identical bit allocation, same drafter.
Time to first token was 0.08 s on short prompts.

## Licence

Apache-2.0, as are the base model, the adapter and the bundled drafter.
[`LICENSE`](LICENSE) is the base model's copy.
