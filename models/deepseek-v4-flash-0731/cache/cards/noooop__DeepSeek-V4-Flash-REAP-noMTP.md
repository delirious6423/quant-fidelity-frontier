---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
library_name: transformers
pipeline_tag: text-generation
tags:
  - moe
  - pruning
  - reap
---

# DeepSeek-V4-Flash-REAP-noMTP

[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) with its routed experts
pruned from **256 to 128 per layer** (50% sparsity) by REAP
(Router-weighted Expert Activation Pruning), leaving **82.4 GB** across
43 layers.

The point of the exercise is to fit a single 128 GB unified-memory host with
room left over to serve, which the original does not.

## What was changed

REAP scores each expert by how much the router actually leans on it over a
calibration set, then drops the lowest scorers. **It does not modify the weights
of the experts that survive**, so this checkpoint keeps the original
quantization exactly: FP4 for routed experts, FP8 elsewhere. Nothing was
requantized, retrained, or distilled.

Three things are worth knowing before you use it:

* **The first 3 layers route by a frozen token-id table, not by a
  learned score.** `n_routed_experts` is one value for the whole model, so
  pruning any layer forces those to the same count. Their table was rewritten to
  send each dropped expert's token ids to the surviving expert whose router row
  points in the most similar direction, balanced so no survivor absorbs a
  disproportionate share. That is a merge, not a pure prune.
* **The MTP (multi-token prediction) blocks were dropped.** Speculative
  decoding through them is not available on this checkpoint. A later build keeps
  them — see [Builds that keep the draft head](#builds-that-keep-the-draft-head)
  below.
* **Hash-layer handling: `prune-remap`.** See `reap_pruning.json` for the exact
  surviving expert ids per layer.

## Builds that keep the draft head

This checkpoint has no MTP blocks, so it cannot speculate. A separate build of
the same base model does:

**[noooop/DeepSeek-V4-Flash-REAP128-FP4](https://huggingface.co/noooop/DeepSeek-V4-Flash-REAP128-FP4)**
— the same 128-of-256 target sparsity, plus all three DSpark MTP blocks carried
across and pruned to 128 experts of their own by a separate live calibration of
the draft head. On a DGX Spark it decodes at 23.5 tok/s with speculation against
16.5 without, and it ships the calibration records, the vLLM overlays that
speculation needs on SM121, and the full build and evaluation scripts. A
152-expert variant of it is being published separately.

It is not strictly better: it is a different pipeline (dequantize → reference
REAP → requantize, rather than file-level surgery), and on the shared benchmarks
the two land within noise of each other except on multilingual, where this
checkpoint is ahead by about 2 points. Its card carries that comparison.

## Serving

Sized for a 128 GB unified-memory host: 82.4 GB resident leaves room to
serve, which the original does not. Measured on a DGX Spark (GB10, 121 GiB
usable). Load takes about five minutes.

The image is stock vLLM with one package moved forward. Save this as
`Dockerfile`:

```dockerfile
FROM vllm/vllm-openai:v0.25.1
RUN pip install --no-cache-dir flashinfer-python==0.6.14
ENV FLASHINFER_DISABLE_VERSION_CHECK=1
```

`v0.25.1` was `latest` when this was built (same image id), and is pinned here
so the recipe keeps meaning what it said. A newer vLLM may well carry a
FlashInfer new enough to drop the last two lines.

```bash
docker build -t vllm-dsv4 .

docker run -d --name dsv4-reap --gpus all --ipc=host -p 8000:8000 \
    -v /path/to/DeepSeek-V4-Flash-REAP-noMTP:/model:ro \
    --entrypoint vllm vllm-dsv4 serve /model \
    --served-model-name dsv4-reap \
    --gpu-memory-utilization 0.75 \
    --max-model-len 65536 --max-num-seqs 16 \
    --kv-cache-dtype fp8
```

Four things there are not optional, and each cost something to find out:

* **`--kv-cache-dtype fp8`** — DeepSeek-V4's sparse-MLA kernel rejects anything
  else.
* **`--gpu-memory-utilization 0.75`**, not vLLM's default 0.9. On unified memory
  that fraction comes out of system RAM rather than a separate VRAM pool: 0.9 on
  a 121 GiB machine reserves ~109 GiB and leaves the OS about 8 GiB. Long context
  then collapses — measured at 32K with 16 concurrent requests, generation fell
  to 0.1-2.2 tokens/s with only 4-6% of the KV cache in use, so the pool was
  never the constraint. Pointing a second client at it in that state hung the
  host hard enough to need a power cycle.
* **The FlashInfer bump, and `FLASHINFER_DISABLE_VERSION_CHECK=1` with it** (the
  Dockerfile sets both). vLLM 0.25.1 pins flashinfer 0.6.13 while its own code
  passes arguments that only exist in 0.6.14, so the stock image crashes on
  load; flashinfer-cubin never shipped 0.6.14, which is why the version check
  then has to be off. Later vLLM releases may not need any of this.
* **`--entrypoint vllm`** spelled out. The Dockerfile above inherits vLLM's own
  `ENTRYPOINT ["vllm", "serve"]` and does not need it, but an image built the
  quick way — `docker commit` of a container started with `--entrypoint bash` —
  keeps bash as its entrypoint, and `docker run IMAGE serve /model` then
  silently runs nothing. Spelling it out works either way.

Long context is exercised up to 64K in the table below. 128K is untested on this
checkpoint.

## Evaluation

Served through vLLM on a DGX Spark: `--kv-cache-dtype fp8`, 64K context,
`--gpu-memory-utilization 0.75`.

| Benchmark | Result | Baseline |
|---|---|---|
| Japanese JCommonsenseQA (full 1,119, 3-shot) | 0.9088 ± 0.0086 | 0.20 random |
| English MMLU (570-item diagnostic) | 0.6526 ± 0.0194 | 0.25 random |
| Chinese global_mmlu_zh (full 400) | 0.4975 ± 0.0250 | 0.25 random |
| Japanese global_mmlu_ja (full 400) | 0.5075 ± 0.0249 | 0.25 random |
| English global_mmlu_en (full 400) | 0.6425 ± 0.0238 | 0.25 random |
| Perplexity (262,016 held-out tokens) | 6.5891 | — |
| RULER 4K / 16K / 32K / 64K (12-task mean) | 98.68 / 96.10 / 96.93 / 94.51 | 85.6 effective-length threshold |

Long context survives the prune: RULER stays well above the 85.6 effective-length
threshold at every length measured. Retrieval in particular is untouched — the
eight needle-in-a-haystack tasks score 1.000 at every length up to 32K. What
erodes with length is reading comprehension over the retrieved span:
`ruler_qa_squad` falls from 0.842 at 4K to 0.542 at 64K, and it is the worst task
at all four lengths.

**There is no unpruned control.** The original does not fit the evaluation host,
so these are absolute health checks, not a measured degradation against the
model this came from.

## Calibration

Which experts survive is decided entirely by the calibration mix. This one was
weighted toward Japanese, English and code; a checkpoint aimed at other
languages or domains needs its own scoring pass rather than this file.

## Reproducing

Pipeline, scripts and the reasoning behind each choice:
<https://github.com/g667300/deepseek-v4-flash-reap>

---

# Original model card: deepseek-ai/DeepSeek-V4-Flash-0731

# DeepSeek-V4-Flash-0731

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V4" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V4-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2606.19348"><b>Technical Report</b>👁️</a>
</p>

## Introduction

**DeepSeek-V4-Flash-0731** is the official release of **DeepSeek-V4-Flash**, superseding the preview version, with substantially enhanced agentic capabilities. It has the same model structure as [DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark), i.e. it comes with a speculative decoding module attached.

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
tokenizer = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Flash-0731")
tokens = tokenizer.encode(prompt)
```

## How to Run with vLLM

DSpark speculative decoding is enabled with a single flag — add --speculative-config with method: dspark to your vLLM launch command:

`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'`

For example, the command below serves the model with vLLM on a single 4×GB300 node. 
See the [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash?hardware=b300&features=tool_calling,reasoning) for detailed instructions and other hardware configurations.

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --data-parallel-size 4 --enable-expert-parallel \
  --moe-backend deep_gemm_mega_moe \
  --attention-config '{"use_fp4_indexer_cache": true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
```

## How to Run with SGLang

Enable DSpark with `--speculative-algorithm DSPARK` and do not set a separate `--speculative-draft-model-path` as the target and draft weights therefore come from the same checkpoint.
See the [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4#hw=gb300&variant=flash-official&quant=fp4&strategy=low-latency&nodes=single) for detailed instructions, benchmarks and other hardwares configurations.

```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tp 4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --swa-full-tokens-ratio 0.1 \
```

## How to Run Locally

Please refer to the [inference](inference/README.md) folder for detailed instructions on running DeepSeek-V4 locally, including model weight conversion and interactive chat demos.

For local deployment, we recommend setting the sampling parameters to `temperature = 1.0`, with `top_p = 0.95` for agentic scenarios and `top_p = 1.0` otherwise. For the `high` and `max` reasoning effort levels, we recommend a maximum output length of **384K** tokens.

## License

This repository and the model weights are licensed under the [MIT License](LICENSE).

## Citation

```
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```

## Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).
