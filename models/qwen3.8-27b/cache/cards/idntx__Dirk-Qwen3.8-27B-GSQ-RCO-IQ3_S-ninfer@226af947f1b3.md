---
library_name: ninfer
pipeline_tag: text-generation
inference: false
license: apache-2.0
base_model:
- peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF
- Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- ninfer
- qwen3.8
- qwen3_5
- gguf-converted
- gsq-rco
- iq3_s
- mtp
- speculative-decoding
- cuda
---

# Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S (NInfer)

This repository contains a **NInfer v3 `.ninfer` engine artifact** converted from the GGUF model
[`peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF`](https://huggingface.co/peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF).

It is **not** a GGUF, SafeTensors, or Transformers checkpoint. It is intended to be loaded by the
NInfer engine on NVIDIA CUDA devices. The artifact contains the text model and the MTP
speculative-decoding head. The separate vision projector from the source GGUF release is **not**
included in this conversion.

## Files

| File | Size (bytes) | LFS SHA-256 | Xet hash |
| --- | ---: | --- | --- |
| `Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.ninfer` | 12,116,808,448 | `a1e5fce8f01bd975634c02abba7f8a61ed103bf13135a10b57ffa0062da8df80` | `bb538ac93d4c0bfca5ffbad1a6a93379bf9983c89df98bfe4fc471d5f10036f0` |
| `Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.ninfer.conversion.json` | 4,954 | — | — |

## Conversion provenance

| Field | Value |
| --- | --- |
| Source model | [`peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF`](https://huggingface.co/peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF) |
| Source file | `Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` |
| Source size | 12,120,037,696 bytes |
| Source LFS SHA-256 | `108cb89de873d36ad9a47356085d2d960eeadaf9df1040cedce3749dc11fddad` |
| Source Xet hash | `7f355a756c1120e3f5862214a8a1a410ca61e46ed88b573dd4c2437c75401e59` |
| Converter | NInfer converter `1.0.0` |
| Engine revision | `51cdf17108d28d7b772dfcc56a80dcf5de517291` |
| Conversion time | 76.719 seconds |
| Output artifact | `Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.ninfer` |

The conversion preserves the GGUF tensor blocks without re-quantization. NInfer stores the GGUF
block payloads and performs quantized matrix multiplication in place at runtime.

## Artifact contents

The `.ninfer` artifact contains:

- a `text` component for `Qwen3_5ForCausalLM`
- an `mtp` component for `Qwen3_5MTP`, used for speculative decoding
- embedded resources:
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `chat_template.jinja`
  - `generation_config.json`

Reported artifact summary fields:

| Field | Value |
| --- | ---: |
| Artifact version | 3 |
| Artifact ID | `514b60e21c2345179c732cc1ddbad120` |
| Objects | 846 |
| Tensors | 842 |
| Resources | 4 |
| Bindings | 979 |
| Uses | 619 |

Storage layout:

| Layout | Objects |
| --- | ---: |
| `contiguous_le_v1` | 457 |
| `gguf_blocks_v1` | 385 |

Stored tensor format counts:

| Format | Count |
| --- | ---: |
| `bf16` | 360 |
| `fp32` | 96 |
| `gguf_iq1_m` | 1 |
| `gguf_iq2_s` | 17 |
| `gguf_iq2_xs` | 9 |
| `gguf_iq2_xxs` | 6 |
| `gguf_iq3_s` | 131 |
| `gguf_iq3_xxs` | 68 |
| `gguf_iq4_xs` | 89 |
| `gguf_q2_k` | 19 |
| `gguf_q4_k` | 40 |
| `gguf_q6_k` | 5 |
| `int32` | 1 |

## Text model configuration

| Field | Value |
| --- | ---: |
| Architecture | `Qwen3_5ForCausalLM` |
| Model type | `qwen3_5_text` |
| Hidden size | 5120 |
| Layers | 64 |
| Vocabulary size | 248320 |
| Max position embeddings | 262144 |
| Attention heads | 24 |
| KV heads | 4 |
| Head dimension | 256 |
| Linear attention key heads | 16 |
| Linear attention key head dimension | 128 |
| Linear attention value heads | 48 |
| Linear attention value head dimension | 128 |
| Linear conv kernel dimension | 4 |
| Intermediate size | 17408 |
| RoPE theta | 10,000,000 |
| Partial rotary factor | 0.25 |
| MRoPE sections | `[11, 11, 10]` |
| `rms_norm_eps` | `9.999999974752427e-07` |

The 64 layers use a mixed attention layout: every fourth layer uses full attention, while the
remaining layers use linear attention.

## Runtime notes

- NInfer loads the stored GGUF quantized blocks directly.
- Activations are quantized to `q8_1` BF16 for the quantized matmul path.
- Accumulation is performed in FP32.
- The GGUF blocks are not re-quantized during conversion.
- The artifact is meant for the NInfer engine, not for `llama.cpp`, Transformers, or other GGUF loaders.

## Launch example

The example below uses a `ninfer-serve.exe` build from
[`qwased/ninfer-16g-5070ti-qwen3.8-27b-gsq-rco-precision-tail`](https://github.com/qwased/ninfer-16g-5070ti-qwen3.8-27b-gsq-rco-precision-tail/).
Adjust the model path before running.

Pre-built binaries, runtimes, and a configured quickstart batch file to Hugging Face:
👉 https://huggingface.co/idntx/ninfer-qwen3.8-27b-win-5070ti-precision-tail

```bat
ninfer-serve.exe c:\ai\Dirk-Qwen3.8-27B-GSQ-RCO-IQ3_S.ninfer ^
  --host 0.0.0.0 --port 8081 ^
  --model-id ninfer ^
  --max-context 65000 ^
  --kv-capacity auto ^
  --kv-dtype rk4v4-e8 ^
  --kv-headroom-mib 300 ^
  --spec mtp --draft-tokens 3 ^
  --max-concurrency 1 ^
  --kv-tail-tokens 1024 ^
  --kv-tail-type f16
```

## Observed session statistics

The numbers below combine the provided log fragments from one real interactive OpenAI-compatible
chat session on a single NVIDIA GeForce RTX 5070 Ti 16 GB system running Windows 10, using the
launch flags above. This is not a synthetic benchmark. The workload includes tool calls, cached
conversation prefixes, cold prefix rebuilds, MTP speculation, and ngram speculation.

| Metric | Observed |
| --- | ---: |
| Completed requests | 25 |
| Rejected requests | 2 (`context length exceeded`) |
| Stop reasons | 7 stop tokens, 17 tool calls, 1 context capacity |
| Tool calls issued | 24 |
| Tool markup recovery warnings | 1 |
| Prompt tokens | 1,018,133 |
| Output tokens | 51,971 |
| Cached prompt tokens | 837,446 |
| Uncached prompt tokens | 180,687 |
| Token-weighted prompt cache hit rate | 82.3% |
| TTFT | mean 4.43 s, median 0.595 s, range 0.152–33.1 s |
| Total request latency | mean 22.29 s, median 10.9 s, range 0.531–79.2 s |
| Queue latency | mean 21.5 ms, range 13.3–33.3 ms |
| Uncached prefill | 1.65k tok/s token-weighted, 92–1.93k tok/s per request |
| Effective decode | 116 tok/s token-weighted, 79–291 tok/s per request |
| MTP/ngram mixed speculation acceptance | 35,567/59,386 (59.9%) |
| Ngram acceptance | 7,282/12,694 (57.4%) |
| Ngram rounds | 848 |

Startup observed in the same session:

| Metric | Observed |
| --- | ---: |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| OS | Windows 10 |
| Engine ready | 6.6 s |
| Weight load | 11.3 GiB in 4.0 s, 2.85 GiB/s |
| Host state pinned | 1.15 GiB |
| Host KV pinned | 8.00 GiB |
| CUDA graphs ready | 1.7 s |
| Warmup | 431 ms |
| KV capacity | 65,024 tokens, `rk4v4-e8`, 1,016/1,016 pages |
| Runtime footprint | 2.98 GiB, 854.4 MiB free after readiness |

## Limitations

- This conversion includes the text model and MTP head only; the vision projector is not bundled.
- Quality is bounded by the source GGUF quantization.
- This card does not claim independent NInfer-specific benchmark results for this exact artifact.

## License

This model is provided under the **Apache 2.0** license.