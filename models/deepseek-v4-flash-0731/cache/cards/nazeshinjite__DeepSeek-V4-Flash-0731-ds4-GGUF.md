---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- dwarfstar
- ds4
- deepseek
- deepseek-v4
- moe
- quantized
- imatrix
---

<div align="center">
  <img src="https://github.com/antirez/ds4/blob/main/logo.svg?raw=true" width="220" alt="DwarfStar" />
</div>

<div align="center">

# DeepSeek-V4-Flash-0731 — DwarfStar (ds4) GGUF

**Quantizations of the final DeepSeek-V4-Flash release, built for the DwarfStar inference engine — mixed Q2/Q4 (the headline build), flat Q2, and flat Q4.**

[![Engine](https://img.shields.io/badge/engine-DwarfStar%20(ds4)-c0392b)](https://github.com/antirez/ds4)
[![Docs](https://img.shields.io/badge/docs-dwarfstar.sh-2c3e50)](https://dwarfstar.sh/docs/quickstart/)
[![Base](https://img.shields.io/badge/base-DeepSeek--V4--Flash--0731-536af5)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
[![License](https://img.shields.io/badge/License-MIT-f5de53)](https://opensource.org/licenses/MIT)

</div>

---

## ⚠️ These GGUFs only run on DwarfStar ⚠️

**This is not a llama.cpp GGUF.** DwarfStar (`ds4`) is a purpose-built C inference engine for DeepSeek V4 and GLM 5.2 with its own custom GGUFs. These files will not load in llama.cpp, Ollama, or LM Studio, and llama.cpp GGUFs don't load in DwarfStar.

- **Engine:** <https://github.com/antirez/ds4>
- **Docs / quickstart:** <https://dwarfstar.sh/docs/quickstart/>

DwarfStar targets high-memory Macs (96 GiB+ unified), CUDA, and ROCm. It ships an OpenAI- and Anthropic-compatible HTTP API, a CLI, a coding agent, and disk-backed KV cache.

### Why bother with another engine? Speed.

On my testbed — **Apple M5 Max, 128 GiB** — DwarfStar decodes DeepSeek V4 Flash **2.0–2.7× faster than llama.cpp**:

| Engine | Quant | Decode t/s (2K → 32K ctx) |
|---|---|---|
| **DwarfStar (ds4)** | antirez mixed Q2/Q4, 90.9 GiB | **36.8 → 27.1** |
| llama.cpp | Unsloth `UD-IQ3_XXS`, 95.9 GiB | 13.7 → 12.9 |

Since the two engines cannot load each other's files, I matched the quants as closely as formats allowed (90.9 GiB vs 95.9 GiB,  ~2.5–3 bpw). These are **preview-checkpoint** measurements — 0731 numbers coming soon.

### On-disk KV cache for agentic workflows

DwarfStar is built on the premise that fast local SSDs mean that **KV cache can become a first-class disk citizen**. With DeepSeek's compressed MLA prompt cache, **256k context costs under 3.5 GiB**, so extremely long contexts (up to 1M tokens) are viable on local hardware with constrained RAM. 

**Why this dominates for agentic work:** agents re-send a conversation history that grows continually. Every tool call, every reasoning trace, each turn shares an identical prefix. With disk KV, that prefix is a cache hit and you pay prefill only for what's new. Unlike RAM, DS4's disk cache is persistent, so restarting doesn't cost you that first expensive prefill all over again if you resume your session. For multi-turn or agent workloads this is an even bigger win than the 2× decode advantage.

Enable with `--kv-disk-dir` and `--kv-disk-space-mb`:

```
./ds4-server --ctx 262144-0pp --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192
```
DwarfStar also ships expert SSD streaming with hotlist routing, which allows you to run the model at usable speeds even if you can't fit it all in RAM. 

---

## What's in this repo

| File | Size | What it is |
|---|---|---|
| `DeepSeek-V4-Flash-0731-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | 90.9 GiB | **Mixed Q2/Q4 main model — the headline build** |
| `DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | 80.8 GiB | Flat Q2 main model |
| `DeepSeek-V4-Flash-0731-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf` | 153.3 GiB | Flat Q4 main model |
| `DeepSeek-V4-Flash-0731-DSpark-support-Q2K-w2Q4K.gguf` | 7.8 GiB | DSpark speculative-decoding draft head, hybrid `Q2_K` with `Q4_K` w2 (optional, see below) |
| `DeepSeek-V4-Flash-0731-DSpark-support-Q4KExperts.gguf` | 10.6 GiB | DSpark speculative-decoding draft head, `Q4_K` experts throughout (optional, see below) |

The mixed Q2/Q4 reproduces **antirez's original mixed recipe**, rebuilt against the **final 0731 checkpoint**:

- **Routed experts, layers 0–36:** `IQ2_XXS` gate/up, `Q2_K` down
- **Routed experts, layers 37–42:** `Q4_K` (last six layers, where 2-bit error hurts most)
- **Attention projections, shared experts, output:** `Q8_0`

The insight behind the recipe is that routed experts dominate the file size (277B of the model's ~284B parameters), so 2-bit precision there buys aggressive compression. Quality is retained by keeping original precision for the last six layers (4-bit) and the non-routed paths (8-bit). The result is around 90.9 GiB instead of ~153 GiB, so it fits in 128 GiB with context to spare.

The flat Q2 (`IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8`) and flat Q4 (`Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out`) are antirez's formats, rebuilt on 0731 tensors.

---

## Download and run

Downloading into `gguf/` using HF CLI for resumable downloads:

```
hf download nazeshinjite/DeepSeek-V4-Flash-0731-ds4-GGUF \
  --include "*Layers37-42*" --local-dir gguf/
```

This downloads just the Q2-Q4 mixed checkpoint. Add `--include "*DSpark*"` if you also want the draft head.

### Interactive chat

Running `ds4` with no `-p` drops you into the interactive CLI prompt:

```
./ds4 -m gguf/DeepSeek-V4-Flash-0731-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf -c 262144
```

Inside the session, `/help` lists the commands; `/think`, `/think-max`, and `/nothink` switch reasoning modes, and `/ctx N` restarts with a different context size.

### API server

OpenAI **and** Anthropic compatible HTTP server:

```
./ds4-server \
  -m gguf/DeepSeek-V4-Flash-0731-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -c 262144 \
  --kv-disk-dir ~/.ds4/server-kv \
  --kv-disk-space-mb 8192
```
Point any OpenAI client at the base URL
`http://127.0.0.1:8000/v1` (just like with llama.cpp), or set `ANTHROPIC_BASE_URL=http://127.0.0.1:8000`
for Anthropic clients like Claude Code.

### Reasoning Effort

Thinking defaults **on at high effort**, in both CLI and server. There is no `--reasoning-effort` flag — on the server, effort is a per-request field.

| Effort | CLI flag | Server request field |
|---|---|---|
| **High** (default) | — | — |
| Max | `--think-max` | `"reasoning_effort": "max"` |
| Off | `--nothink` | `"thinking": {"type": "disabled"}` |

In an interactive session, `/think`, `/think-max`, and `/nothink` switch mid-conversation. **Max requires `-c 393216` or higher.** Below that it silently falls back to high.

**Sampling is ignored while thinking**, matching the official DeepSeek API. `temperature` and `top_p` apply only in non-thinking mode, where DeepSeek recommends `1.0` / `0.95` for agentic use and `1.0` / `1.0` otherwise.

### Turn on the disk KV cache!

It is **off unless you pass `--kv-disk-dir`** to enable it; `--kv-disk-space-mb` caps how much disk it uses. 

### Doesn't fit in RAM?

DwarfStar can stream routed experts from SSD rather than requiring full residency:

```bash
./ds4 -m gguf/DeepSeek-V4-Flash-0731-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  -c 32768 --ssd-streaming
```

Read the cache report it prints at startup, then raise or lower `--ssd-streaming-cache-experts` to suit your memory needs.

---

## DSpark MTP head

The 0731 release ships a substantially larger DSpark speculative-decoding module than the preview — **4,705 draft tensors against the preview's 1,575**, across three draft stages, each with its own 256-expert MoE and attention. DwarfStar's converter reads all of it cleanly (`unknown_dspark_tensors=0`), so a draft model GGUF is included in this repo.

Enable it with `--mtp <dspark-filename>.gguf --dspark`, pointed at the sidecar (recommended: Q2-Q4 mixed) alongside the main model.

**I am not promising a speedup.** Speculative decoding pays off when decode is *memory-bandwidth-bound*: you verify several drafted tokens in roughly the time it takes to stream the weights once, so the extra compute rides along for free. DeepSeek V4 Flash at these quants is **compute-bound on Metal** — dequantization dominates the decode step — so verification work is *not* free, and its cost can exceed the savings. In my own testing on the preview checkpoint, MTP was a **net loss of 3–11%**.

The more-sophisticated 0731 draft model may perform better, particularly on CUDA/ROCm, which generally see better MTP speedups. **If you benchmark it, please report back!**

---

## Status: benchmarks forthcoming

Numbers are not published yet, but will be added here as they land:

- **KLD and perplexity** against the official weights, per quant
- **Prefill and decode tok/s** at several context depths on my testbed: **Apple M5 Max, 128 GiB**
- Quality-eval pass rates versus the preview-checkpoint quant

**Note:** The importance matrix used here is antirez's, collected on the preview checkpoint rather than on 0731. 

I measured routed-expert drift, unpacking the FP4 nibbles and applying the E8M0 block scales, sampled across ten layers spanning the full depth:

| Layer depth | Weight correlation | Relative RMS drift | 4-bit values identical |
|-------------|--------------------|--------------------|------------------------|
| 0–5 (early) | 0.9936             | 11.4%              | 80%                    |
| 20 (middle) | 0.9926             | 12.2%              | 79%                    |
| 42 (final)  | 0.9903             | 13.9%              | 75%                    |

Correlation holds above 0.99 at every depth, and over three-quarters of individual 4-bit weights are bit-identical. This is a continued-training refresh, not a retrain from scratch — expert indices still mean the same thing, so the per-expert importance vectors carry over rather than degrading into noise.

The drift is small but increasing with depth — the final layers moved about 20% more than the first. That happens to work in this recipe's favor: layers 37–42, which drifted most, get Q4_K treatment, and 4-bit is far more forgiving of an imperfect imatrix than 2-bit. The aggressive IQ2_XXS lives in layers 0–36, which are the most stable.

It is still not a matched imatrix. A 0731-native one is planned, and I'll publish the delta with a v2, if it turns out to matter.

## Please send feedback

If you run these, I would genuinely like to hear about it in the **Community** tab:

- Hardware (GPU, total memory) and DwarfStar backend (Metal/CUDA/ROCm)
- Prefill and decode tok/s, and at what context length
- DSpark / MTP results, positive or negative
- Quality impressions versus llama.cpp or the preview quant — especially failure modes
- Evals with the built-in `ds4-eval` and `ds4-bench` features
- Whether you ran with `--ssd-streaming`, and your expert-cache settings

I'll add reported numbers to this README (with credit). Negative results are as useful as positive ones — particularly on non-Metal backends, which I cannot test.

---

## Credit

**All of this exists because of [Salvatore Sanfilippo (antirez)](https://github.com/antirez).** He created the DwarfStar engine, the standalone C quantizer, the mixed Q2/Q4 recipe these files reproduce, and the importance matrix they were built with — and published every piece of it. This repo is a reprise of his work on newer weights, nothing more. If you find these useful, the credit belongs upstream:

- **Engine and tooling:** <https://github.com/antirez/ds4>
- **Original GGUFs and imatrix:** <https://huggingface.co/antirez/deepseek-v4-gguf>

Thanks also to **DeepSeek-AI** for the weights under MIT, and to the **llama.cpp / GGML** project, whose quantization formats and hard-won engineering DwarfStar builds on.


---
---

# Upstream model card

*The following is reproduced verbatim from the [official DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). All credit to DeepSeek-AI.*

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
