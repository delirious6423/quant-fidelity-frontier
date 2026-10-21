---
license: mit
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
quant_method: dsv4_mxfp4_int8
tags:
  - deepseek-v4-flash
  - deepseek-v4
  - mxfp4
  - int8
  - int8-kv
  - ampere
  - sm86
  - sm_86
  - rtx-3090
  - consumer-gpu
  - vllm
  - vllm-fork
  - appmana
  - moe
  - mixture-of-experts
  - mla
  - multi-latent-attention
  - long-context
  - quantization
  - flash-mla
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731 MXFP4-INT8

Ampere (RTX 3090, sm86) has no native FP8 or MXFP4 tensor cores.
The released 0731 checkpoint (FP8+MXFP4) using mainline vLLM has no
officially supported loading path.

Using AppMana's vLLM fork at its current public tip plus the two patches
shipped in this repository's `serving/` folder, this quant enables serving
on Ampere.

Tested on 8x RTX 3090 (24GB, sm86). See "What to expect" below.

## What this is

DeepSeek-V4-Flash-0731 converted to a format that runs on Ampere GPUs:

- **Experts**: MXFP4 (E2M1) — lossless. On sm86, E2M1 is reinterpreted as
  signed INT4 and run through Ampere INT4 tensor cores via Marlin. E8M0
  power-of-2 scales applied in epilogue. No quality loss vs original.
- **Dense linears**: FP8 E4M3 → INT8 (channelwise). Lossy but minimal:
  measured 40 dB signal-to-noise ratio (~1% mean relative error) on the
  original checkpoint's FP8 weights after round-trip. For comparison, a
  typical W4A16 quant measures ~20-30 dB.
- **KV cache**: int8_ds_mla. Doubles context vs fp8 (437K vs ~220K on 8x24GB).
- **DSpark**: 3 MTP draft stages preserved from 0731 source. Observed 0%
  draft acceptance with int8 KV on sm86. Disable for production.

## How it was created

Converted with `tools/ampere/dsv4_requant_checkpoint.py` from the
AppMana `vllm-consumer-nvidia-platforms` fork:

```bash
python tools/ampere/dsv4_requant_checkpoint.py \
  --src deepseek-ai/DeepSeek-V4-Flash-0731 \
  --dst ./output \
  --expert-format mxfp4 \
  --dense-int8-strategy channel \
  --device cuda:0 \
  --overwrite
```

MXFP4 path is 95% passthrough — expert weights copied byte-for-byte, only
~8 FP8 linears per shard requanted to INT8. Conversion takes ~15-30 min
on a single GPU. See `serving/convert.sh`.
Regenerate with `serving/convert.sh`; compare shard hashes against
`serving/checksums/` (sha256 of the published files).


Source: `deepseek-ai/DeepSeek-V4-Flash-0731`
Fork: [AppMana/forks-vllm-consumer-nvidia-platforms](https://github.com/AppMana/forks-vllm-consumer-nvidia-platforms) branch `appmana/upstream-merge`, tip `760ced86b` + the two serving/patches (== commit `623739503`)
Similar quants: `appmana/deepseek-v4-mxfp4-int8` (pre-0731, no DSpark),
`appmana/deepseek-v4-int4-int8` (0731, INT4 experts, faster but less headroom)

## What to expect

Tested on 8x RTX 3090 (24GB, PCIe, no NVLink), CUDA 13.1, torch 2.13.0+cu130,
fork commit 623739503 (= public tip 760ced86b + the two serving/patches).
One bench harness, Sep 2026:

| Profile | KV pool | Single | 4-way agg | Notes |
|---------|---------|--------|-----------|-------|
| TP4xPP2, async, 512K window (recommended default) | 1,008,842 tok = 1.92x @512K | 49 tok/s | 135 tok/s | 492K-token ingest validated E2E |
| TP8, async, 262K | 413,887 tok = 1.58x @262K | 58 tok/s | 139 tok/s | max single-stream |
| TP8, sync, 262K + 20 GiB CPU offload | 461,011 tok + CPU spill | 49 tok/s | 120 tok/s | clean teardown |

- Context auto-fit: 524,288-token window with a 1.0M-token pool
- Single-stream is below the pre-merge engine's 62 tok/s — the merged engine
  trades ~10-15% single-stream decode for 2.3x context and a fixed
  long-context prefill path (details in serving/profiles.md)
- Correct math, coherent long-form generation, working tool calls
- No quality evals yet — evaluate before production use
- GMU=0.93 is the ceiling at the 512K window (0.95 always OOMs on 24GB)
- `--max-num-batched-tokens 128` gives 100% Triton JIT cache hit

No attempt was made to make DSpark work. We prioritized serving context.

No claims are made for suitability for any purpose.

The only intent is to provide inspiration for owners of 8x3090 rigs.
## How to serve

Requires the AppMana fork plus the two patches shipped in `serving/patches/`,
and `flash-mla==2.0.0+8ec3de6`.

The flash-mla wheel index is `https://appmana.github.io/forks-flash-mla-int/`,
which redirects to
[GitHub releases](https://github.com/AppMana/forks-flash-mla-ampere-dsv4/releases).
If it is unavailable, the sm86 attention path has no fallback. Consider
mirroring the wheel if you depend on this setup.

```bash
# 1. Quant (also provides serving/patches/ and serving/serve.sh)
Q="$(hf download Jon-Nielsen/__REPO__)"

# 2. Fork at the pinned commit (current public tip of the branch)
git clone --branch appmana/upstream-merge \
  https://github.com/AppMana/forks-vllm-consumer-nvidia-platforms.git
cd forks-vllm-consumer-nvidia-platforms
git checkout 760ced86b

# 3. Patches: upstream engine sync, then the consumer-NVIDIA fixes
git apply "$Q/serving/patches/0001-upstream-sync-20260906.patch"
git apply "$Q/serving/patches/0002-consumer-nvidia-fixes-20260906.patch"

# 4. Dependencies
python -m venv .venv && source .venv/bin/activate
pip install torch==2.13.0+cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130
pip install flash-mla==2.0.0+8ec3de6 \
  --extra-index-url https://appmana.github.io/forks-flash-mla-int/

# 5. Build and install (sm86 only — saves 3-5x build time)
MAX_JOBS=12 TORCH_CUDA_ARCH_LIST="8.6" pip install -e . --no-build-isolation

# 6. Serve — defaults are the recommended measured profile
# (TP4 x PP2, async scheduling, 512K context, ~1.0M-token KV pool)
VENV="$PWD/.venv" MODEL="$Q" bash "$Q/serving/serve.sh"
```

On a box with mixed PCIe widths, order ranks so the first PP stage gets the
x16 cards, e.g. x16 = 0,1,2,7: `CUDA_VISIBLE_DEVICES=0,1,2,7,3,4,5,6`.

All engine fixes (prefill workspace reservation, allocator settings,
graph-memory profiling opt-out, dense capture sizes) are baked into
`serving/serve.sh` — see `serving/README.md` and `serving/profiles.md` for
what each knob does and the alternative profiles.
## Patches

Two anonymous diffs against the fork's public tip `760ced86b` (branch
`appmana/upstream-merge`), applied in order:

### 0001-upstream-sync-20260906.patch

Brings the engine to the post-merge state used for validation: 57
upstream/main commits (graph memory profiling default, Fast Start, MRV2 DBO,
DSv4 fixes) plus conflict resolutions.

### 0002-consumer-nvidia-fixes-20260906.patch

The consumer-NVIDIA delta on top:

- int8_ds_mla bounds-guarded cache insert kernels (PR-A port)
- KV-offload IMA-safe host memory + block-id range checks (PR-B port)
- mxfp4 w13 loader + conversion tool kernel block (PR-C port)
- worst-case prefill K-gather workspace reservation + env-overridable
  `VLLM_DSV4_PREFILL_CHUNK_SIZE` (long-context `workspace locked` fix)
- `_vllm_fa3_C` optional (Ampere builds produce no FA3 `.so`)

`760ced86b` + both patches equals commit `623739503` of the maintainer's line
(verified in a throwaway worktree; modulo serving documentation). The previous
recipe's three patches (expert weight loader, conversion kernel block, int8
prefill safety) are superseded — all three are contained in 0002.
## Serving directory

```
serving/
  serve.sh                                          — launcher; defaults = recommended profile
  profiles.md                                       — measured profile matrix + envelope rules
  README.md                                         — recipe, knob explanations, known-good facts
  convert.sh                                        — re-run the conversion (tool in the patched fork)
  tests/                                          — bench harness behind profiles.md + card numbers
  checksums/                                      — sha256 of published shards (base/ablit)
  patches/0001-upstream-sync-20260906.patch         — upstream engine sync (745 KB)
  patches/0002-consumer-nvidia-fixes-20260906.patch — consumer-NVIDIA fixes (43 KB)
```
---

Below is the original README from `deepseek-ai/DeepSeek-V4-Flash-0731`,
included for reference only. The `encoding/` and `inference/` folders it
mentions are not part of this repository.

# Original DeepSeek-V4-Flash-0731 README

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
See the [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash?hardware=b300&features=tool_call,reasoning) for detailed instructions and other hardware configurations.

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
