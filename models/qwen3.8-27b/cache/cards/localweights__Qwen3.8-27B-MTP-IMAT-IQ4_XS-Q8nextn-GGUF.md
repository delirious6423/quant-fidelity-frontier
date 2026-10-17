---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - qwen
  - qwen3.8
  - 27b
  - mtp
  - nextn
  - speculative-decoding
  - gguf
  - quantized
  - imatrix
pipeline_tag: text-generation
---

# Qwen3.8-27B-MTP-IMAT-IQ4_XS-Q8nextn (GGUF)

Dense 27B Qwen3.8 trunk + embedded NextN-MTP head, quantized for single-GPU inference. Same recipe and per-tensor layout as [localweights/Qwen3.6-27B-MTP-IMAT-IQ4_XS-Q8nextn-GGUF](https://huggingface.co/localweights/Qwen3.6-27B-MTP-IMAT-IQ4_XS-Q8nextn-GGUF).

- **Trunk:** IQ4_XS (imatrix-calibrated); attention `qkv`/`v` at Q5_K, output at Q6_K
- **MTP head:** Q8_0 (NextN, `kv_only_nextn=true`)
- **File size:** 14,616 MiB (~15.3 GB, 4.49 BPW)

The MTP head is **embedded inside the GGUF** — no separate drafter file. Recent `llama.cpp` builds activate it via `--spec-type draft-mtp`.

## Memory

Qwen3.8-27B is a hybrid: of 64 layers, only the 16 full-attention layers keep a KV cache that grows with context (4 KV heads × 256 head dim). The 48 Gated-DeltaNet layers hold a fixed-size recurrent state per sequence.

| component | size |
|---|---|
| weights | 14.27 GiB |
| KV cache, Q8/fp8 | ~32 KiB per token (~1 GiB per 32K tokens) |
| recurrent state | ~150 MB per sequence, independent of context |

Rough VRAM for one sequence with Q8 KV, plus ~1.5–2 GiB runtime overhead: ~17 GB at 8K, ~18 GB at 32K, ~21 GB at 128K, ~24 GB at the native 262K.

Measured: on a 24 GB RTX 3090, vLLM (GGUF plugin, fp8 KV, `gpu_memory_utilization=0.95`) fits the weights plus a 190,814-token KV pool with a 139,264-token max context per request.

## Quick start (no spec decode)

```bash
llama-server -m Qwen3.8-27B-MTP-IMAT-IQ4_XS-Q8nextn.gguf \
  -ngl 999 -fa on -c 32768 --parallel 1 \
  -ctk q8_0 -ctv q8_0 --kv-unified \
  --host 0.0.0.0 --port 8080 --jinja
```

## With speculative decoding

Requires `llama.cpp` built from master after [ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673):

```bash
llama-server -m Qwen3.8-27B-MTP-IMAT-IQ4_XS-Q8nextn.gguf \
  -ngl 999 -fa on -c 32768 --parallel 1 \
  -ctk q8_0 -ctv q8_0 --kv-unified \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.75 \
  --host 0.0.0.0 --port 8080 --jinja
```

The 3.6 settings are a good starting point (`p_min 0.75` was essential there). MTP accept rate on 3.8 has **not** been benchmarked yet.

## Benchmarks (vLLM, RTX 3090, no MTP)

vLLM 0.25.0 + GGUF plugin (with large-batch dequant+cuBLAS dispatch), fp8 KV, 4096-token prefill chunks, measured in-pod.

| fresh prefill | tok/s |
|---|---|
| 5K tokens | 1,057 |
| 21K tokens | 1,045 |
| 42K tokens | 967 |
| 85K tokens | 836 |

| decode @ 42K context | aggregate tok/s |
|---|---|
| 1 stream | 36.7 |
| 2 streams | 63.4 |
| 8 streams | 69.8 |

## Quant details

- Converted from `Qwen/Qwen3.8-27B` BF16 with `convert_hf_to_gguf.py` (arch `qwen35`, MTP layer kept as `blk.64` NextN tensors).
- imatrix: [bartowski's Qwen3.8-27B imatrix](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) (calibration v6). Thanks!
- Per-tensor types forced with `llama-quantize --tensor-type-file` to exactly match our Qwen3.6-27B quant (866 tensors, 0 type differences). A uniform layout matters on vLLM's GGUF path: a mixed per-tensor layout (F32 `ssm_alpha/beta`, Q4_0/Q4_K/Q6_K/Q8_0 shards) prefilled ~6× slower there.
- Text-only `config.json` (`Qwen3_5ForCausalLM`) and the Qwen3.8 tokenizer + chat template are included for vLLM `--hf-config-path`. Note the 3.8 chat template defaults `reasoning_effort` to `xhigh` and `preserve_thinking` to true.

## Sister models

- Qwen3.6-27B: [localweights/Qwen3.6-27B-MTP-IMAT-IQ4_XS-Q8nextn-GGUF](https://huggingface.co/localweights/Qwen3.6-27B-MTP-IMAT-IQ4_XS-Q8nextn-GGUF)
- Qwen3.6-35B-A3B MoE: [localweights/Qwen3.6-35B-A3B-MTP-IMAT-IQ4_XS-Q8nextn-GGUF](https://huggingface.co/localweights/Qwen3.6-35B-A3B-MTP-IMAT-IQ4_XS-Q8nextn-GGUF)

## Reference

- Base model: <https://huggingface.co/Qwen/Qwen3.8-27B>
- MTP architecture: NextN-style embedded prediction layer, `kv_only_nextn=true`.

Apache 2.0 license.
