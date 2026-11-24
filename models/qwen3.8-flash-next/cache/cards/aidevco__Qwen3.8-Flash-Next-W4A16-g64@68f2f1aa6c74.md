---
library_name: transformers
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - compressed-tensors
  - int4
  - w4a16
  - rocm
  - mi210
---

# Qwen3.8-Flash-Next — W4A16 (int4 experts, group 64)

A data-free, symmetric int4 weight-only quantization of the routed experts of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in the compressed-tensors `pack-quantized` format.
It was built to serve the model on two AMD Instinct MI210 (gfx90a, 64 GB each) at tensor-parallel size 2. The published
symmetric W4A16 builds use group 128, and TP=2 splits each expert's 640-wide intermediate into 320, which 128 does not divide;
group 64 does.

Everything about the model itself (architecture, training, intended use, benchmarks, limitations) is in the
[original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). This card covers only what the quantization changes.

## What is quantized

| Part                                                                            | Precision                 | Why                                                                  |
| ------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------- |
| Routed experts `gate_proj` / `up_proj` / `down_proj` (48 layers × 512 experts)  | int4, group 64, symmetric | 241.6 GB of the 360 GB bf16 source; fused MoE WNA16 kernel on gfx90a |
| n-gram PLE embedding tables (~102 GB)                                           | bf16                      | served from pinned host memory; a lookup touches only a few rows     |
| Gated DeltaNet, gated attention, QSA sparse indexer, hyper-connections, routers | bf16                      | recurrent, state and routing layers stay full precision              |
| Shared experts, norms, embeddings, `lm_head`, MTP head, vision tower            | bf16                      | standard                                                             |

- 182.6 GB in 131 shards. About 76 GB goes on the GPUs (~38 GB per card); the PLE tables stay in host memory.
- The source stores the experts fused (`experts.gate_up_proj`, `experts.down_proj`). The quantizer splits them into per-expert
  `gate_proj` / `up_proj` / `down_proj`, the layout vLLM's `qwen4_exp` loader expects (gate is the first half of
  `gate_up_proj`).

## How it was made

- Tool: llm-compressor 0.14.0 `model_free_ptq` with compressed-tensors 0.19.0, data-free (no calibration set), on two MI210s
  with compressed-tensors' eager PyTorch kernels (`CT_ENFORCE_EAGER=1`: its Triton observer kernel does not compile on ROCm).
- Scheme: `num_bits=4`, `symmetric=true`, `strategy=group`, `group_size=64`, observer `memoryless-mse`; no activation
  quantization.
- Targets (both the checkpoint's and vLLM's module names):
  `re:^(?:model\.language_model|language_model\.model)\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$`
- Source revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Verified after the build: every scheme field, the targets in both name forms, every shard present. Then served on
  2 × MI210 (vLLM v0.33.0, ROCm 10.0): `CompressedTensorsWNA16MoEMethod`, 35.62 GiB of GPU weights per card, KV cache
  534,118 tokens, and four checks at temperature 0 passed (17 × 60 = 1,020; a 155-minute train trip with thinking off
  and on; a `get_weather` tool call).

## Serving (vLLM, ROCm)

Quantization is detected from the config; do not pass `--quantization`.

```bash
vllm serve <this repo> \
  --tensor-parallel-size=2 --engram-config='{"cpu_offload": true}' \
  --max-model-len=32768 --gpu-memory-utilization=0.92 --kv-cache-dtype=auto \
  --max-num-batched-tokens=2048 --language-model-only \
  --limit-mm-per-prompt='{"image":0,"video":0}' \
  --reasoning-parser=qwen3 --tool-call-parser=qwen3_coder
```

- `--engram-config={"cpu_offload": true}` keeps the PLE tables in pinned host memory. Without it the AMD PLE layer puts them
  on the GPUs, which do not have room next to the experts. Give the server roughly 200 GB of host memory.
- vLLM v0.31.0's `qwen4_exp` PLE loader refuses any compressed-tensors model, even with the tables left in bf16. The serving
  stack that built this model ships a small `sitecustomize.py` patch that makes the loader treat the tables as unquantized.
- Pipeline parallelism is not supported for this architecture; use tensor parallelism.

## Tuned fused-MoE config (optional)

`moe-configs/E=512,N=320,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json` is a Triton fused-MoE kernel config tuned for this build on
the MI210 at tensor-parallel size 2 (`N` is the per-card expert width, 640 / 2). vLLM does not read it from the model repository: download the
folder and point `VLLM_TUNED_CONFIG_FOLDER` at it. The startup log then names the file (`Using configuration from ...`) instead of
`Using default MoE config`.

```bash
hf download aidevco/Qwen3.8-Flash-Next-W4A16-g64 --include 'moe-configs/*' --local-dir .
export VLLM_TUNED_CONFIG_FOLDER=$PWD/moe-configs
```

- Tuned with vLLM v0.33.0 (`vllm/vllm:v0.33.0-rocm`, ROCm 10, Triton 3.8.0), all 12 batch sizes from 1 to 2048, on both MI210s, with the
  `BLOCK_SIZE_K` tiles capped at the int4 group size (64); the finalists were re-timed 10 times on one card.
- Serving A/B on 2 × MI210 (default, tuned, tuned, default; 1,536 prompt / 128 output tokens): output tokens per second +0.1% to +2.9%,
  inter-token latency 1.6% to 5.0% lower, time to first token and energy per token unchanged within the noise.
- Only valid for this exact shape, card and dtype. ROCm 7.2 images name the card `Instinct_MI210`: copy the file under that name too. A newer
  vLLM or Triton can change the kernel; re-check before relying on it.

## Results

Measured on 2 × MI210 at TP=2, vLLM v0.31.0, on a CPU build of the same recipe and source revision:

| Check                                                      | Result                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| GPU weights per card                                       | 35.6 GiB                                                                  |
| KV cache capacity (`--gpu-memory-utilization=0.92`)        | 522,103 tokens; 15.9x concurrency at 32,768 tokens per request            |
| Output tokens per second, 1,536 prompt / 128 output tokens | 43.2 at concurrency 1, 89.7 at 4; time to first token 274 ms              |
| lm-evaluation-harness, 50 samples per task, thinking on    | gsm8k 0.96, minerva algebra 0.74, humaneval (chat) 0.90, mbpp (chat) 0.90 |

The task scores are a 50-sample sanity check that the quantization kept the model's quality, not a full benchmark. Every
missed code task was an empty answer: the model was still thinking at the 8,192-token budget.

## License and attribution

This is a derivative of Qwen3.8-Flash-Next by the Qwen team, distributed under the same license as the original, the Qwen
Community License 1.0, whose full text (including its copyright and permission notice) is in this repository's `LICENSE`
file. Its conditions apply to this build unchanged, among them: a commercial product or service with more than 100 million
monthly active users or US$20 million monthly revenue must display the model name prominently, and a licensee that runs a
Model-as-a-Service or AI Work Assistant business needs a separate license from Qwen for commercial use. The tokenizer, chat
template and configuration files are copied unchanged from the original repository; `config.json` adds only the
quantization config.

The quantization changes the numerical precision of the weights listed above and nothing else. It is not an official Qwen
release, and the Qwen team has not reviewed or endorsed it.
