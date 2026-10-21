---
license: other
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - deepseek-v4
  - vllm
  - quantized
  - int4
  - int8
---

# DeepSeek-V4-Flash-0731 INT4/INT8 for RTX 30xx and DGX Spark

DeepSeek-V4-Flash-0731 converted for the
[`AppMana/forks-vllm-consumer-nvidia-platforms`](https://github.com/AppMana/forks-vllm-consumer-nvidia-platforms)
runtime:

- routed experts: INT4 W4A16 Marlin;
- dense, shared-expert and attention linears: INT8 W8A16 AllSpark;
- sparse-MLA KV cache and indexer: packed INT8 using `int8_ds_mla`;
- three DSpark MTP draft stages.

Stock vLLM cannot load `quant_method=dsv4_int` or the `int8_ds_mla` cache.
The revision to serve is `3b25318eca9ccebc48130251d2c8b825341ad0ea`. The
multimodal sibling,
[`appmana/deepseek-v4-flash-vision-exp-int4-int8`](https://huggingface.co/appmana/deepseek-v4-flash-vision-exp-int4-int8),
carries the same language quantization plus an INT8 vision tower; both load
on the fork at `1d8b0eae46` (upstream vLLM `729ebac498` merged 2026-09-18),
whose startup validation resolves every kernel named in this checkpoint's
`vllm` block.

## Platform support

| Platform |
| --- |
| RTX 30xx |
| DGX Spark GB10 |

Both platforms use the same weights and the same kernel configuration. The
runtime image must contain the native extension code for the actual GPU.

## Kernel map

This table describes what executes for this INT4/INT8 checkpoint. It does not
describe the separate NVFP4/FP8 SparkInfer checkpoint.

| Function | RTX 30xx | DGX Spark GB10 | Source |
| --- | --- | --- | --- |
| Routed-expert MoE GEMM | Marlin W4A16 | Marlin W4A16 | community ([Marlin](https://github.com/IST-DASLab/marlin)) |
| Dense, shared-expert and attention linear GEMMs | AllSpark W8A16 | AllSpark W8A16 | vLLM |
| Sparse-MLA decode | `flash_mla.sparse_mla_decode_int8` | `flash_mla.sparse_mla_decode_int8` | AppMana ([forks-flash-mla-int](https://github.com/AppMana/forks-flash-mla-int)) |
| Sparse-MLA prefill | `flash_mla.sparse_mla_prefill_int8` | `flash_mla.sparse_mla_prefill_int8` | AppMana ([forks-flash-mla-int](https://github.com/AppMana/forks-flash-mla-int)) |
| Indexer K cache write | `indexer_k_quant_and_cache_int8` | `indexer_k_quant_and_cache_int8` | vLLM |
| Indexer Q RoPE/quantization | fused Triton | fused Triton | vLLM |
| Indexer logits | Triton IMMA | Triton IMMA | vLLM |
| Indexer top-k | native CUDA row/persistent top-k | native CUDA row/persistent top-k | vLLM |
| KV cache | 528-byte `int8_ds_mla` slots | 528-byte `int8_ds_mla` slots | vLLM |
| mHC mixing and HCHead | torch/Triton | torch/Triton | vLLM |
| Speculative decoding | three-stage DSpark | three-stage DSpark | vLLM |

### Kernel configuration

We added a `vllm` block to our checkpoint configs so each checkpoint can state
which cache and kernels it needs:

```json
"vllm": {
  "kernels": [
    "flash_mla.sparse_mla_decode_int8",
    "flash_mla.sparse_mla_prefill_int8",
    "vllm._custom_ops.indexer_k_quant_and_cache_int8",
    "vllm.models.deepseek_v4.common.ops.fused_indexer_q.fused_indexer_q_rope_quant_int8",
    "vllm.model_executor.layers.quantization.utils.marlin_utils.marlin_act_int8_process_scales",
    "vllm.model_executor.layers.sparse_attn_indexer.streaming_prefill_topk"
  ],
  "cache_type": "int8_ds_mla"
}
```

vLLM reads this block at startup. `--hf-overrides` can replace it for testing.
Streaming top-k is enabled by default to keep prefill memory bounded at long
context.

## Deployment

See the public Kubernetes manifests in the
[`examples/deployment`](https://github.com/AppMana/forks-vllm-consumer-nvidia-platforms/tree/appmana/vllm-consumer-nvidia-platforms/examples/deployment)
directory.

## How the checkpoint was built

Converted from
`deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
with `tools/ampere/dsv4_requant_checkpoint.py` in the AppMana vLLM fork. The
published `config.json` supplies the kernel names and `int8_ds_mla` selection
shown above.

The three DSpark MTP stages are native to 0731 and are converted in place.
0731 declares `num_nextn_predict_layers: 1` while shipping `mtp.0`, `mtp.1`
and `mtp.2`; the published config states the three stages actually present,
since vLLM sizes the draft model from that field.

## Quantization

The source routed experts use packed MXFP4 E2M1 values and E8M0 group scales.
`tools/ampere/dsv4_requant_checkpoint.py` in the vLLM fork:

1. unpacks the E2M1 values;
2. applies the E8M0 scale to recover FP32 group values;
3. groups values by 32 along the final dimension;
4. searches BF16-rounded candidate scales;
5. requantizes with `round(group / scale).clamp(-8, 7)`;
6. selects the lowest round-trip squared-error scale (`scale_mode: mse`,
   recorded in `quantization_config`), worth +4.2 to +8.7 dB reconstruction
   SNR over dividing by the group max;
7. stores two signed INT4 values per byte as uint4b8 (`code + 8`).

Stored expert weights therefore use signed INT4 nibbles, BF16 per-group
scales and group size 32. Dense/shared/attention weights use biased UINT8 plus
channel scales for AllSpark W8A16, except `attn.wo_a`, which is dequantized
straight to BF16 with no scale companion: it is consumed by an inverse-RoPE
BF16 einsum rather than a GEMM, so quantizing it costs accuracy and buys
nothing.

Weights stay in the per-expert, separate-projection checkpoint layout and are
fused while loading. Do not pre-fuse the checkpoint.

## Benchmarks

Per stream is the median request of a batch; aggregate is the whole batch
over its wall time. Verbatim: each request hides a unique 1,000-token
passage in its prompt, at a depth set by its index in the batch, and asks
for it back; the column counts how many of the C simultaneous requests
returned it exactly.

### RTX 3090

11 nodes, each a Ryzen 9 7950X with 64 GB of RAM and one RTX 3090 power
limited to 250 W, linked by Thunderbolt. Pipeline parallel over the 11 GPUs
(PP=11, TP=1) with NCCL over Thunderbolt networking, served by
[AppMana/forks-vllm-consumer-nvidia-platforms](https://github.com/AppMana/forks-vllm-consumer-nvidia-platforms)
from revision `3b25318eca9ccebc48130251d2c8b825341ad0ea` of this checkpoint,
1,000,000-token window, 2 GiB of KV cache per rank, chunked prefill of
1,024 tokens, thinking off. Measured 2026-09-04 on the runtime that
preceded the 2026-09-18 upstream merge. Tables are the output of
`tools/ampere/dsv4_needle_matrix_report.py --markdown` in the fork. The
merged runtime (`1d8b0eae46`) has not been
re-measured on this checkpoint yet; its 8,000-token rows on the sibling
vision checkpoint, whose language path is the same code and quantization,
are on that card (2026-09-19: 55 output tok/s per stream at C=1, 316
aggregate at C=8 warm, every row verbatim).

```shell
vllm serve appmana/deepseek-v4-int4-int8 \
  --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
  --pipeline-parallel-size 11 --distributed-executor-backend ray \
  --max-model-len 1000000 --kv-cache-memory-bytes 2147483648 \
  --enable-chunked-prefill --max-num-batched-tokens 1024 \
  --long-prefill-token-threshold 4096 --max-num-seqs 8 \
  --stream-interval 4 --async-scheduling --no-enable-prefix-caching \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}'

python tools/ampere/dsv4_needle_bench.py --base-url http://HOST:8080 \
  --model appmana/deepseek-v4-int4-int8 --input-tokens 64000 \
  --needle-tokens 1000 --concurrency 4 --run-seed 98502 --output-json row.json
```

Speculative decoding off, `--max-num-seqs 8`:

| Input tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate | Output tok/s per stream | Output tok/s aggregate | Verbatim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 | 1 | 2.7 | 2,985 | 2,985 | 37.6 | 34.1 | 1/1 |
| 8,000 | 2 | 2.3 | 3,602 | 6,206 | 36.7 | 67.6 | 2/2 |
| 8,000 | 4 | 3.1 | 2,601 | 6,954 | 21.8 | 81.4 | 4/4 |
| 8,000 | 8 | 5.2 | 1,545 | 7,765 | 12.5 | 93.8 | 8/8 |
| 16,000 | 1 | 2.9 | 5,587 | 5,587 | 37.4 | 33.8 | 1/1 |
| 16,000 | 2 | 3.6 | 4,593 | 7,260 | 36.1 | 63.5 | 2/2 |
| 16,000 | 4 | 5.5 | 3,012 | 7,834 | 21.7 | 73.4 | 3/4 |
| 16,000 | 8 | 9.2 | 1,765 | 8,203 | 12.0 | 84.4 | 7/8 |
| 64,000 | 1 | 10.5 | 6,113 | 6,113 | 36.9 | 26.6 | 1/1 |
| 64,000 | 2 | 12.3 | 5,714 | 8,004 | 32.7 | 46.0 | 2/2 |
| 64,000 | 4 | 20.0 | 3,317 | 8,061 | 18.3 | 51.6 | 3/4 |
| 64,000 | 8 | 35.9 | 1,805 | 8,023 | 10.8 | 61.4 | 7/8 |

Speculative decoding on: DSpark with 7 draft tokens and probabilistic draft
sampling, the same server otherwise with `--max-num-seqs 32`:

```shell
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

| Input tokens | C | TTFT s | Prefill tok/s per stream | Prefill tok/s aggregate | Output tok/s per stream | Output tok/s aggregate | Verbatim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,000 | 1 | 2.9 | 2,795 | 2,795 | 62.3 | 52.7 | 1/1 |
| 8,000 | 2 | 2.4 | 3,553 | 5,391 | 59.8 | 103.8 | 2/2 |
| 8,000 | 4 | 3.4 | 2,444 | 6,657 | 56.1 | 180.6 | 4/4 |
| 8,000 | 8 | 5.6 | 1,454 | 6,917 | 39.0 | 252.7 | 8/8 |
| 16,000 | 1 | 3.0 | 5,306 | 5,306 | 60.9 | 51.5 | 1/1 |
| 16,000 | 2 | 4.0 | 4,212 | 6,478 | 58.3 | 92.0 | 2/2 |
| 16,000 | 4 | 6.1 | 2,704 | 6,984 | 50.6 | 145.7 | 3/4 |
| 16,000 | 8 | 10.2 | 1,588 | 7,146 | 34.9 | 193.8 | 7/8 |
| 64,000 | 1 | 10.8 | 5,923 | 5,923 | 58.5 | 35.9 | 1/1 |
| 64,000 | 2 | 13.8 | 5,175 | 7,068 | 48.6 | 56.6 | 2/2 |
| 64,000 | 4 | 22.4 | 2,970 | 7,229 | 34.0 | 76.3 | 2/4 |
| 64,000 | 8 | 40.3 | 1,608 | 7,195 | 19.2 | 86.4 | 5/8 |
| 512,000 | 1 | 111.3 | 4,600 | 4,600 | 45.9 | 7.5 | 1/1 |
| 512,000 | 2 | 159.9 | 3,599 | 4,805 | 27.7 | 9.1 | 1/2 |
| 512,000 | 4 | 266.0 | 2,005 | 4,786 | 6.1 | 9.3 | 1/4 |
| 512,000 | 8 | 484.5 | 1,071 | 4,637 | 3.6 | 9.6 | 2/8 |
| 950,000 | 1 | 266.0 | 3,571 | 3,571 | 52.0 | 3.9 | 0/1 |
| 950,000 | 2 | 393.5 | 2,722 | 3,613 | 30.6 | 4.1 | 0/2 |
| 950,000 | 4 | 657.9 | 1,505 | 3,565 | 3.1 | 4.0 | 0/4 |

Output tok/s per stream counts only requests that generated their full
passage.

Tool calling (Berkeley Function Calling Leaderboard, DSpark on, 2026-09-04):
non-live AST 87.3%, live 77.9%, multi-turn 0.0%, overall 21.1%.

### DGX Spark GB10

Two DGX Spark GB10 systems, tensor parallel across the two (TP=2), DSpark
with 5 speculative tokens, 500,000-token admission limit, 8 GiB of KV cache
per rank. 96 output tokens per request, no verbatim check.

C=1:

| Input tokens | TTFT s | Prefill tok/s per stream | Output tok/s per stream | ITL ms p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1,104 | 1.2 | 928.9 | 70.56 | 85.2 |
| 8,823 | 10.5 | 841.0 | 63.68 | 88.3 |
| 68,924 | 77.7 | 886.7 | 49.71 | 121.2 |
| 137,847 | 182.2 | 756.5 | 44.98 | 127.6 |
| 215,386 | 279.6 | 770.4 | 65.31 | 91.8 |
| 430,771 | 581.8 | 740.4 | 64.45 | 98.7 |

C=2:

| Input tokens | TTFT s p50 | Prefill tok/s aggregate | Output tok/s per stream |
| ---: | ---: | ---: | ---: |
| 1,108 | 2.0 | 501.4 | 43.89 |
| 8,827 | 8.2 | 1,423.1 | 35.68 |
| 68,928 | 66.7 | 1,560.7 | 33.14 |
| 137,851 | 140.1 | 1,468.1 | 33.51 |

C=1, 256 output tokens:

| Input tokens | TTFT s | Prefill tok/s per stream | Output tok/s per stream | ITL ms p50 | Accepted tokens per step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,841 | 5.7 | 1,557.5 | 69.54 | 85.7 | 6.0 |
| 68,942 | 43.8 | 1,575.3 | 69.11 | 86.9 | 6.0 |

The admission limit trades against decode latency: the decode CUDA graphs
carry logits scratch sized to the limit, and a 1,000,000-token limit on the
same hardware decodes at 13 to 19 tok/s per stream over the same ladder.

Tool calling: 20 of 20.
