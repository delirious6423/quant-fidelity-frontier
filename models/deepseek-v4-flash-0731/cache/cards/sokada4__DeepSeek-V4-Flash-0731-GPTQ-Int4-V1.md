---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
library_name: transformers
tags:
- gptq
- gptqmodel
- 4bit
- w4a16
- int4
- quantized
- vllm
---

# DeepSeek-V4-Flash-0731-GPTQ-Int4-V1

4-bit GPTQ (W4A16) quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), produced with [GPTQModel](https://github.com/modelcloud/gptqmodel).

This is the **V1** release (first calibration run). A V2 release, requantized with a
different calibration dataset, is planned as a separate repository
(`DeepSeek-V4-Flash-0731-GPTQ-Int4-V2`) once available.

## Quantization details

| | |
|---|---|
| Method | GPTQ (GPTQModel 7.3.4) |
| Bits | 4 |
| Group size | 128 |
| Symmetric | yes |
| `desc_act` | false |
| `lm_head` quantized | no |
| Pack dtype | int32 |

### Calibration data

128 samples (32 each of zh / en / ja / code), `seq_len=2048`:

- `allenai/c4` — zh, en, ja splits (mangled/mis-decoded documents filtered out)
- `codeparrot/codeparrot-clean-valid` — Python source

zh/en/ja/code were weighted evenly. `seq_len=2048` was chosen above this model's
sparse-indexer `index_topk` (512) so the top-k pruning path is actually exercised
during calibration, while staying within the indexer's dense-attention memory budget
on the quantization rig.

## Serving with vLLM

This model requires a vLLM build with DeepSeek-V4 GPTQ support on the Ampere
(`SM86`, e.g. RTX 3090/4090-class, non-Hopper) backend, which is not yet present in
either upstream vLLM or the unmodified `vllm-dsv4-ampere` community fork. The
required patch is available at:

https://github.com/okdshin/vllm-dsv4-ampere/tree/gptq-w4a16-support

Apply the `main-port/vllm` overlay from that branch on top of a vanilla vLLM
checkout, then serve as usual, e.g.:

```bash
vllm serve sokada4/DeepSeek-V4-Flash-0731-GPTQ-Int4-V1 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --enforce-eager \
  --max-num-seqs 8 \
  --max-model-len 4096
```

(`--cpu-offload-gb` may be needed depending on available VRAM; the full checkpoint
is ~153GB.)

## Benchmark

Measured on a 4x GPU Ampere rig (`--cpu-offload-gb 20`, `vllm bench serve`),
compared against the unquantized fp8 checkpoint:

| concurrency | metric | fp8 (unquantized) | GPTQ 4bit | delta |
|---|---|---|---|---|
| 1 | TTFT mean | 2345.56 ms | 1184.35 ms | -49% |
| 1 | TPOT mean | 335.33 ms | 317.98 ms | -5% |
| 1 | output tok/s | 2.73 | 3.02 | +11% |
| 8 | TTFT mean | 8434.68 ms | 4486.18 ms | -47% |
| 8 | TPOT mean | 735.13 ms | 432.72 ms | -41% |
| 8 | output tok/s | 9.34 | 16.11 | +72% |

At concurrency=1, decode is bound by PCIe transfer of CPU-offloaded weights rather
than GPU compute/bandwidth, so the TPOT improvement from quantization is modest;
at concurrency=8, batching amortizes that fixed cost and the quantization benefit
is much larger.

## License

MIT, inherited from the base model.
