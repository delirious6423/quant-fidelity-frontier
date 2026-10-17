---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: llama.cpp
pipeline_tag: image-text-to-text
tags:
  - gguf
  - qwen
  - qwen3.8
  - llama.cpp
  - multimodal
  - vision-language
  - text-generation
  - quantized
language:
  - en
  - zh
---

# Qwen3.8-27B GGUF

Source-faithful and architecture-aware GGUF conversions of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), including its
native vision projector and one-layer MTP draft model.

This release was independently built from source revision
[`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
with llama.cpp revision
[`1692f9e50bb20fd96b963af38a282daf78feea64`](https://github.com/ggml-org/llama.cpp/commit/1692f9e50bb20fd96b963af38a282daf78feea64).
All 18 source `safetensors` shards were verified against their Hub LFS SHA-256
digests before conversion.

## Files

### Main model

| File | Size | Purpose |
|---|---:|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 18.97 GB | **Recommended practical model.** Conservative 5.64 BPW mixed quantization. |
| `Qwen3.8-27B-Q8_0.gguf` | 28.60 GB | Near-lossless high-quality quantization. |
| `Qwen3.8-27B-BF16.gguf` | 53.81 GB | Maximum-fidelity BF16 conversion. |

The recommended Q4 file is deliberately not a stock Q4_K_M quant. Its FFN and
embedding matrices use Q4_K, the final output and attention-output projections
use Q6_K, and all other attention plus Gated DeltaNet/SSM matrices use Q8_0.
This protects the model's architecture-sensitive paths while keeping the file
under 19 GB.

### Vision projector

Download one projector to use images or video. It is not needed for text-only
inference.

| File | Size | Purpose |
|---|---:|---|
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.63 GB | Recommended practical projector. |
| `mmproj-Qwen3.8-27B-BF16.gguf` | 0.93 GB | Maximum-fidelity projector. |

### MTP speculative-decoding sidecar

These files are optional. They enable Qwen3.8's native one-layer
multi-token-prediction draft model; they do not change the target model's final
sampling distribution.

| File | Size | Purpose |
|---|---:|---|
| `mtp-Qwen3.8-27B-Q4_0.gguf` | 1.68 GB | Smallest and usually best practical draft sidecar. |
| `mtp-Qwen3.8-27B-Q8_0.gguf` | 3.16 GB | Higher-fidelity draft sidecar. |
| `mtp-Qwen3.8-27B-BF16.gguf` | 5.95 GB | Maximum-fidelity draft sidecar. |

## Download

Recommended text + vision pair:

```bash
hf download Mike0021/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-Q4_K_M.gguf \
  mmproj-Qwen3.8-27B-Q8_0.gguf \
  --local-dir ./Qwen3.8-27B-GGUF
```

Add `mtp-Qwen3.8-27B-Q4_0.gguf` to that command if you want speculative
decoding.

## llama.cpp usage

This new architecture requires llama.cpp revision `1692f9e` or a tested newer
revision. Other GGUF runtimes and older GUIs may not support Qwen3.8 yet.

Text chat with thinking disabled:

```bash
llama-cli \
  -m Qwen3.8-27B-Q4_K_M.gguf \
  -ngl all -c 32768 --jinja --conversation \
  --chat-template-kwargs '{"enable_thinking":false}'
```

Image understanding:

```bash
llama-cli \
  -m Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf \
  --image image.jpg \
  --image-min-tokens 1024 \
  -p "Describe this image precisely." \
  -ngl all -c 32768 -n 512 --jinja
```

MTP speculative decoding:

```bash
llama-cli \
  -m Qwen3.8-27B-Q4_K_M.gguf \
  -md mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl all -ngld all -c 32768 --jinja --conversation \
  --chat-template-kwargs '{"enable_thinking":false}'
```

The model's native context window is 262,144 tokens. Start with a smaller
runtime context such as 32K unless you need the full window, because KV and
recurrent-state memory grow with the configured context and concurrency.

## Thinking and sampling

The exact source Jinja template is embedded in every text and MTP GGUF. Thinking
defaults to `xhigh`. Select another supported effort explicitly:

```bash
--chat-template-kwargs \
  '{"enable_thinking":true,"reasoning_effort":"low","preserve_thinking":true}'
```

Supported effort values are `low`, `medium`, and `xhigh`. Use
`{"enable_thinking":false}` to disable thinking. At the pinned llama.cpp
revision, pass the effort through `chat_template_kwargs`; top-level OpenAI API
`reasoning_effort` forwarding is still being completed in
[llama.cpp PR #26941](https://github.com/ggml-org/llama.cpp/pull/26941).

Qwen's source model card recommends these starting points:

| Mode | Temperature | Top-p | Top-k | Presence penalty |
|---|---:|---:|---:|---:|
| Thinking | 1.0 | 0.95 | 20 | 0.0 |
| Non-thinking | 0.7 | 0.8 | 20 | 1.5 |

Disabling thinking does not automatically change the sampler settings.

## Validation and provenance

The release passed the following checks before upload:

- GGUF v3 metadata: `qwen35`, 64 target layers, 262,144 context, 248,320-token vocabulary.
- Exact source chat-template comparison.
- Expected tensor counts: 851 text, 334 projector, and 18 MTP tensors.
- Expected type distributions for every BF16, Q8, and Q4 artifact.
- Full SHA-256 checksums (see `SHA256SUMS`).
- CUDA tensor checks and deterministic text-generation smoke tests.
- Native image/projector and MTP speculative-decoding smoke tests.

The exact conversion commands, tool revisions, tensor policy, and validation
details are in [`CONVERSION.md`](./CONVERSION.md). Raw conversion and
quantization logs are included under [`logs/`](./logs/).

## Why no importance-matrix Q4?

llama.cpp supports importance-matrix quantization, but the architecture was new
at conversion time and no representative Qwen3.8-specific calibration plus
held-out evaluation was available. An unvalidated calibration corpus could bias
the remaining Q4 FFNs. This release therefore follows ggml-org's conservative,
architecture-aware reference recipe, which already keeps the attention and
DeltaNet/SSM paths at Q8. Any future imatrix build should be published as a
separate variant with its calibration provenance and held-out KLD/task results.

## License

The original model and these converted weights are distributed under the
[Apache License 2.0](./LICENSE). See the
[`Qwen/Qwen3.8-27B` model card](https://huggingface.co/Qwen/Qwen3.8-27B) for
the upstream model's documentation and intended-use guidance.
