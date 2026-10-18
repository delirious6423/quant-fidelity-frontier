---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
library_name: transformers
tags:
- deepseek-v4
- nvfp4
- blackwell
- moe
license: mit
---

# DeepSeek V4 Flash 0731-NVFP4

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V4" />
</div>

## Community NVIDIA-style NVFP4 version

> [!IMPORTANT]
> This is a community conversion of the official DeepSeek V4 Flash 0731
> checkpoint. DeepSeek's architecture, tokenizer, DSpark/MTP configuration,
> attention, shared experts, router, head, and other excluded tensors are
> preserved. This repository is not affiliated with or endorsed by DeepSeek or
> NVIDIA.

<p align="center">
  <a href="https://arxiv.org/abs/2606.19348"><b>Technical Report</b> 👁️</a>
</p>

## What was converted

- Calibrated routed-expert inputs across all 43 main transformer layers.
- Used exactly 500,000 calibration tokens across a diverse mixture of SWE and
  coding trajectories, agentic/tool-use traces, GLM/Kimi traces, CUDA/Nsight
  material, math/Triton material, and science material.
- Generated routed-expert `w1`, `w2`, and `w3` input scales from the calibration
  maxima.
- Converted the routed-expert MXFP4 packed weights losslessly to the NVIDIA-style
  NVFP4 representation.
- Converted source GS32 block scales to target GS16 block scales by splitting
  each 32-value source block into two 16-value target blocks with the same exact
  scale. The packed four-bit weight values are unchanged.

The conversion covers 33,024 routed expert projections. Attention, shared
experts, router/head, embeddings, and MTP tensors remain in their official 0731
source representation.

## Validation

- 33,024 packed routed weights: byte-identical to the source.
- 33,024 weight-scale tensors: exact GS32→GS16 conversion.
- 66,048 generated global/input scale tensors: validated against the frozen
  calibration result.
- 6,269 passthrough tensors: byte-identical to the source.
- 48 safetensor shards and 138,365 indexed output tensors validated.

The complete conversion receipt is included as `conversion-receipt.json`.

## Runtime status

The checkpoint was verified to load and generate through vLLM on two NVIDIA DGX
Spark systems using tensor parallelism across both machines. An
OpenAI-compatible chat smoke test completed successfully. The following is the
tested configuration; it favors compatibility and deterministic startup over
maximum throughput (`--enforce-eager` disables CUDA graphs and compilation
optimizations).

- vLLM: `0.26.1rc1.dev191+gaef9bd1e9.d20260731`
- Tensor parallel size: `2` across `2` nodes
- Maximum model length: `250000` tokens
- Maximum sequences: `8`
- Maximum batched tokens / chunked prefill: `4096`
- GPU memory utilization: `0.83`
- KV cache: FP8, block size `256`
- Load format: `instanttensor`

Start the rank-1 worker first:

```bash
vllm serve /path/to/DeepSeek-V4-Flash-0731-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 2 --nnodes 2 --node-rank 1 \
  --master-addr <rank-0-address> --master-port 29513 \
  --headless --distributed-executor-backend mp \
  --load-format instanttensor \
  --max-model-len 250000 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.83 --kv-cache-dtype fp8 \
  --block-size 256 --enforce-eager
```

Then start rank 0 with the same command, changing `--node-rank 1` to
`--node-rank 0` and removing `--headless`.

This runtime check confirms loading and basic generation compatibility. It is
not a claim of production performance certification; the tested eager-mode
configuration produced low decode throughput.

## Introduction

**DeepSeek-V4-Flash-0731** is the official release of **DeepSeek-V4-Flash**, superseding the preview version, with substantially enhanced agentic capabilities. It has the same model structure as DeepSeek-V4-Flash-DSpark, i.e. it comes with a speculative decoding module attached.

DeepSeek-V4-Flash-0731 outperforms DeepSeek-V4-Pro (Preview) on benchmarks listed below despite its far smaller activated parameter count, and is broadly competitive with the strongest proprietary models available.

<div align="center">

| Benchmark | DeepSeek-V4-Flash-0731 | DeepSeek-V4-Flash (Preview) | DeepSeek-V4-Pro (Preview) | GLM-5.2 | Opus-4.8 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Terminal Bench 2.1 | 82.7 | 61.8 | 72.1 | 81.0 | 85.0 |
| NL2Repo | 54.2 | 39.4 | 38.5 | 48.9 | 69.7 |
| Cybergym | 76.7 | 38.7 | 52.7 | - | 83.1 |
| DeepSWE | 54.4 | 7.3 | 12.8 | 46.2 | 58.0 |
| Toolathlon-Verified | 70.3 | 49.7 | 55.9 | 59.9 | 76.2 |
| Agents' Last Exam | 25.2 | 15.8 | 16.5 | 23.8 | 25.7 |
| AutomationBench Public | 25.1 | 10.8 | 12.8 | 12.9 | 27.2 |
| DSBench-FullStack † | 68.7 | 37.0 | 41.8 | 61.8 | 71.6 |
| DSBench-Hard † | 59.6 | 25.8 | 31.1 | 54.5 | 71.7 |

</div>

Notes:

1. For the Code Agent tasks among the public benchmarks above, DeepSeek-V4-Flash-0731 is evaluated with the minimal mode of DeepSeek Harness (to be released) as the agent framework, using the `max` reasoning effort level with `temperature = 1.0, top_p = 0.95`.
2. † DSBench-FullStack is an internal full-stack development test set; DSBench-Hard is an internal test set of difficult coding-agent problems.

## Chat Template

This release does not include a Jinja-format chat template. Instead, we provide a dedicated `encoding` folder with Python scripts and test cases demonstrating how to encode messages in OpenAI-compatible format into input strings for the model, and how to parse the model's text output. Please refer to the [`encoding`](encoding/README.md) folder for full documentation.

The `reasoning_effort` parameter now supports three levels — `low`, `high`, and `max` — which control how much deliberation the model spends before answering.

A brief example:

```python
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

messages = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hello! I am DeepSeek.", "reasoning_content": "thinking..."},
    {"role": "user", "content": "1+1=?"}
]

# messages -> string
prompt = encode_messages(messages, thinking_mode="thinking", reasoning_effort="max")

# string -> tokens
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("MJPansa/DeepSeek-V4-Flash-0731-NVFP4")
tokens = tokenizer.encode(prompt)
```

## How to Run with vLLM

For the upstream DeepSeek deployment recipe, DSpark speculative decoding is
enabled with a single flag — add `--speculative-config` with method `dspark` to
your vLLM launch command. For this NVFP4 conversion, use the tested
two-node command in the Runtime status section above.

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --data-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention-config '{"use_fp4_indexer_cache": true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

For example, the command above serves the upstream model on a single 4×GB300
node. See the vLLM recipe for detailed instructions and other hardware
configurations.

## How to Run Locally

Please refer to the [inference](inference/README.md) folder for detailed instructions on running DeepSeek-V4 locally, including model weight conversion and interactive chat demos.

For local deployment, we recommend setting the sampling parameters to `temperature = 1.0`, with `top_p = 0.95` for agentic scenarios and `top_p = 1.0` otherwise. For the `high` and `max` reasoning effort levels, we recommend a maximum output length of **384K** tokens.

## License

This repository and the model weights are licensed under the [MIT License](LICENSE).

## Citation

```text
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```

## Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](mailto:service@deepseek.com).
