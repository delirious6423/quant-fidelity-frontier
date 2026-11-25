---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exl3-int8
  - rocm
  - quantized
---

# Qwen3.8-27B - EXL3-INT8 3.05 bpw

EXL3-INT8 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), targeting 3.05 bits per weight for the language-model decoder. The LM head and vision tower use 6 bits and MTP uses 4 bits. Embeddings, norms and recurrent controls are unquantized. Calibration used 250 rows of 2,048 tokens. The safetensors total about 12.69 GiB.

These are not standard Transformers or vLLM weights. Upstream ExLlamaV3 cannot decode the altered codebook correctly. The measurements below use text only; vision and MTP behavior were not evaluated here.

## What EXL3-INT8 is

The EXL3 trellis uses the mul24 codebook: every linear's `.mul1` marker contains `0x00DCD12D`. Decoded weights lie on a fixed INT8 grid. This format requires quantization with that codebook; changing a standard EXL3 checkpoint's metadata does not convert it. `config.json` retains the inherited `codebook: mul1` label; `quantization_config.json` identifies the format as `exl3-int8`.

## Contents

Keep the safetensors, index, configuration, tokenizer and preprocessor files together. `SOURCE_MODEL_CARD.md` contains the upstream card; its claims describe the base model, not this quantization.

## Evaluation

Measured on 2026-11-20 with one Radeon AI PRO R9700 (32 GiB, gfx1201), ROCm 10.2, and shelf. Generation was greedy, single request, without speculative decoding. Japanese and English prompts each produced 64 tokens of coherent text.

### BF16 reference quality

Eight cases, 2,632 positions, valid vocabulary 248,077; KL(reference BF16 || candidate), temperature 1, FP64 calculation, nats. The saved reference uses llama.cpp BF16 weights with FP16 KV and chunk 64; shelf uses BF16 KV and up to 256 prefill tokens. The measurements therefore include runtime and KV differences as well as quantization. Shelf `quality/bf16-kld` revision `96fe6a89`, B path, two runs:

| Model | Mean KLD | Top-1 agreement |
|---|---:|---:|
| This 3.05 bpw model, run 1 | 0.059129 | 91.3374% |
| This 3.05 bpw model, run 2 | 0.059129 | 91.3374% |
| 4 bpw EXL3-INT8, earlier shelf B measurement | 0.020745 | 94.8708% |

The two runs produced identical logits dump hashes for all eight cases. This 3.05 bpw model has lower quality than the 4 bpw model on this comparison. The case-level top-1 agreement was 82.17% on each of the English and Japanese prose cases, so the aggregate score does not describe every prompt equally.

### Single-request speed

Shelf revision `9f5bde32`, BF16 KV, page size 16, `prefill-chunk 4096`, one warmup followed by three measurements, GPU event timing. Decode used no speculative draft.

| Workload | Median | Range |
|---|---:|---:|
| Prefill, 2,048 tokens | 1,147.38 tok/s | 1,144.89–1,148.73 |
| Decode, context 2,048, 128 tokens | 29.09 tok/s | 29.07–29.11 |

These are observations on this system, not speed guarantees.

### Context allocation and 16 GiB estimate

One request, prefix cache disabled, a 21-token input and one generated token, `shelf-compute`, page size 16; no speculative decoding. Peak GPU used bytes were sampled from the R9700 sysfs counter every 50 ms. Shelf reserves an arena, so the peak includes the entire chosen arena and runtime overhead; it is not the minimum memory required. The tests below measure allocation for the configured maximum length, not generation across a fully populated long context.

| KV type | Max sequence | Arena | Peak GPU use | 16 GiB minus peak | Result |
|---|---:|---:|---:|---:|---|
| BF16 | 8,192 | 14.5 GiB | 14.82 GiB | +1.18 GiB | allocated |
| BF16 | 32,768 | 18 GiB | 18.32 GiB | −2.32 GiB | allocated; 15 GiB arena failed |
| BF16 | 65,536 | 24 GiB | 24.32 GiB | −8.32 GiB | allocated; 15 GiB arena failed |
| BF16 | 131,072 | auto, 31.29 GiB arena | 31.60 GiB | −15.60 GiB | allocation failed |
| octave:c5v5 | 8,192 | 14.5 GiB | 14.82 GiB | +1.18 GiB | allocated |
| octave:c5v5 | 32,768 | 14.5 GiB | 14.82 GiB | +1.18 GiB | allocated |
| octave:c5v5 | 65,536 | 14.5 GiB | 14.82 GiB | +1.18 GiB | allocated |
| octave:c5v5 | 131,072 | 20 GiB | 20.32 GiB | −4.32 GiB | allocated; 15 GiB arena failed |

A peak of about 15 GiB or less is an approximate screening criterion for a 16 GiB card. These measurements were made on a 32 GiB R9700, not on a 16 GiB RX 9070 XT. Actual capacity depends on runtime allocations, prompt size, concurrency, other GPU usage and system overhead. The larger-arena successes do not establish the minimum arena size; failures at 15 GiB do not establish the exact maximum context on a 16 GiB GPU.

### Later shelf measurements on one R9700

On cases-v1 (8 cases, 2,632 teacher-forced positions), the default prefill path A gave mean KL(BF16 reference ‖ shelf) **0.059471** and top-1 agreement **91.1474%**. Path B gave 0.059129 and 91.3374%. The llama.cpp reference used FP16 KV, and shelf used BF16 KV. Runtime and KV precision both contribute to this comparison. [BF16 reference and prefill paths](https://github.com/kusuriya-labs/shelf/blob/1fb31fd963cc4c71210aa0c9c461679af150be71/docs/perf/exl3-int8-prefill-quality.md)

With ROCm 10.2, BF16 KV and one request, ordinary prefill at 2,048 tokens reached **2,350.8 tok/s**, and ordinary decode at context 2,048 reached **35.18 tok/s**. With the EXL3-INT8 4 bpw DFlash2 drafter, decode averaged **104.66 tok/s** across six English and Japanese code-editing tasks at 2k, 8k and 16k input lengths. The November 23 comparison also reports BF16-reference KLD 0.0597 and 91.11% top-1 agreement under its BF16-KV configuration. The speculative run used `9c7be540`, before the streaming-stall fix. The suite excluded one stalled trial. [R9700 benchmark report](https://github.com/kusuriya-labs/shelf/blob/1fb31fd963cc4c71210aa0c9c461679af150be71/docs/perf/engine-compare-qwen38-27b-r9700.en.md)

## Runtime

The [public shelf runtime](https://github.com/kusuriya-labs/shelf) supports this checkpoint. The measured build targets `gfx1201`. Example, with no speculative decoding:

```bash
shelf-server --model-dir /path/to/Qwen3.8-27B-EXL3-INT8-3.05bpw \
  --max-seq-len 8192 --kv-cache-dtype bf16 --max-concurrent-requests 1 \
  --prefix-cache 0 --port 8123
```

The example uses the runtime's default arena allocation. Adjust the arena and context for the available GPU memory. The measurements above do not establish compatibility with other GPUs, NixOS or other environments.

## License

Apache-2.0, as the base model. See `LICENSE`.
