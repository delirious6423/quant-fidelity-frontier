---
license: other
license_name: nvidia-open-model-agreement
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
- qwen3.8
- qwen3.8-flash-next
- nvfp4
- modelopt
- mixture-of-experts
- vllm
- b12x
- speculative-decoding
- mtp
- dgx-spark
- gb10
- blackwell
---

# Qwen3.8-Flash-Next FP4-Local for one DGX Spark

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (125B total / 6B active parameters, vision, 51B
n-gram embedding tables) served on **a single DGX Spark** and tuned for agentic coding. The weights are NVIDIA's
[Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) quantization, with an MTP
draft head fine-tuned for agentic workloads. The serving recipe in `recipe/` is a pinned public vLLM + b12x image with
patched vLLM / b12x files and custom CUDA kernels for the Spark's GB10.

## Highlights

- **The same quality as stock vLLM**: GSM8K 97.5 % and HumanEval 97.0 %, against 97.0 % and 97.6 %; KL divergence to
  the original BF16 model 0.0465.
- **1.34× faster decode and 1.45× faster prefill** than stock vLLM on the same Spark: 65 tok/s on agentic coding and
  3,236 tok/s on a 33K-token prompt.
- **Agent follow-up turns start in under a second**: 0.90 s to first token against 3.70 s (4.1×).
- **Four agent sessions at once** finish 4.0× sooner than in the 2026-11-03 release, which used vLLM's stock scheduler,
  and each waits about 1 s for its first token.
- OpenAI-compatible API with a 131K context, text, image and video input, tool calls and reasoning.

## Agentic coding speed

One request at a time, against stock vLLM on the same DGX Spark:

| | Stock vLLM | This repo | |
|---|---:|---:|---:|
| Decode, agentic coding | 48.8 tok/s | **65.3 tok/s** | **1.34×** |
| Prefill, 33K-token prompt | 2,226 tok/s | **3,236 tok/s** | **1.45×** |
| Time to first token, 112K-token prompt | 51.3 s | 35.5 s | 1.44× |
| Time to first token, agent follow-up turn (median) | 3.70 s | **0.90 s** | **4.1×** |
| Agentic coding benchmark, wall clock (92 turns) | 1,161 s | 693 s | **1.68×** |

Stock vLLM is NVIDIA's documented DGX Spark command for this checkpoint, on the same image build. This table was
measured on the 2026-11-03 release.

Later releases added cache-aware scheduling for several agent sessions, leaner prefix-cache checkpoints and faster
prefill kernels. Current release against the 2026-11-03 release, on a replay of multi-turn coding-agent sessions with tool
definitions and tool calls (75 requests, prompts up to 131K tokens):

| | 2026-11-03 release | Current release | |
|---|---:|---:|---:|
| Wall clock, one agent session at a time | 523.5 s | 469.6 s | 1.11× |
| Wall clock, four agent sessions at once | 1,554.9 s | **390.6 s** | **3.98×** |
| Geometric mean of the two | | | **2.11×** |
| Time to first token, four sessions at once (median) | 84 s | **1.0 s** | |

Cold prefill is about 15 % faster than in the 2026-11-03 release (3,597 against 3,099–3,118 tok/s on a 25K-token prompt);
decode speed is unchanged.

## Quality

| | Stock vLLM | This repo |
|---|---:|---:|
| GSM8K (first 200) | 97.0 % | 97.5 % |
| HumanEval | 97.6 % | 97.0 % |
| Needle at 29K / 96K / 125K tokens, tool calls | pass | pass |
| KL divergence to the BF16 model | | 0.0465 (top-1 agreement 92.1 %) |

GSM8K and HumanEval are greedy with thinking off, measured on the 2026-11-03 release; both move by about 1.5 points between
boots. The changes since then are scheduling, prefix-cache checkpoints and bit-identical kernels, so the forward pass
is the same (`recipe/RELEASE_NOTES.md` has the quality checks of each release). The KL divergence is the mean over 24
chunks of 4,096 tokens of AtomicChat's held-out neutral set
([reference](https://huggingface.co/datasets/AtomicChat/Qwen3.8-Flash-Next-Uncensored-GGUF-metrics)), full vocabulary,
in llama.cpp `--kl-divergence` semantics.

## How it works

- **A draft head fine-tuned for agentic workloads.** Only the 29 dense tensors of the MTP layer (91M parameters) were
  trained. Speculative decoding checks every drafted token against the full model, so the head changes speed, not
  outputs: 3.48 instead of 3.26 tokens per step on agentic coding.
- **Custom decode and prefill kernels for GB10 (sm_121a)**, including a persistent prefill MoE kernel, fused
  HyperConnection and norm kernels and 8192-token prefill chunks.
- **Fast follow-ups.** The prefix-cache checkpoint of each prompt sits just before its end, so a follow-up turn
  recomputes only the new message.
- **Cache-aware scheduling.** With several agent sessions sharing the KV cache, the request with the longest cached
  prefix starts first, so sessions stop evicting each other's prefixes.
- **bf16 activations in every MoE call.** Stock uses FP4 activations for prefill-size MoE calls; this recipe keeps bf16
  there too, which stays closer to the BF16 model.

Every change has an off switch in `recipe/serve/serve.sh`. `recipe/README.md` describes each one with its measured
effect, and `recipe/RELEASE_NOTES.md` has the release history.

## Quick start

```bash
hf download diffbot/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark --local-dir ~/models/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark
cd ~/models/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark/recipe
API_KEY=change-me bash serve/serve.sh     # OpenAI-compatible API on :8000, model qwen3.8-flash-next
```

The first start compiles and autotunes the kernels into `~/.cache/qwen38-flash-next-spark`, which takes about 25
minutes; later starts take about 5 minutes. Text, image and video input are on; tool calls use vLLM's
`qwen3_xml` parser and reasoning the `qwen3` parser.

Check the running server:

```bash
cd ~/models/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark/recipe/bench && bash fetch-evaldata.sh
export API_KEY=change-me SERVED_MODEL=qwen3.8-flash-next EVALDATA=$PWD/evaldata
python3 test-toolcall.py && python3 needle-test.py 1800 && python3 quality-probe.py
```

## Hardware requirements

- One DGX Spark, or another GB10 system with 128 GB of unified memory. Tested with NVIDIA driver 580.126.09 (CUDA 13.0).
- Docker with the NVIDIA container toolkit.
- About 110 GB of free disk on fast NVMe: the 102 GB download plus about 5 GB of kernel caches. The n-gram tables are
  read from disk at every step.
- vLLM gets 73% of the unified memory and needs 88.8 GiB free to start; the server peaks at about 93 GiB during a
  112K-token prefill.

To leave room for another memory-heavy service, this setting frees about 7 GiB, keeps the 131K context and turns off
image and video input; prefill is about 8% slower and decode is unchanged:

```bash
MAX_NUM_BATCHED_TOKENS=4096 GPU_MEMORY_UTILIZATION=0.68 \
EXTRA_ARGS="--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0" bash serve/serve.sh
```

## Configuration

`recipe/serve/serve.sh` reads these environment variables; `recipe/RELEASE_NOTES.md` lists all of them.

| Variable | Default | Effect |
|---|---|---|
| `API_KEY` | none | bearer token for the API; unset serves without one |
| `PORT`, `NAME` | `8000`, `qwen38-fp4` | port and container name |
| `MAX_MODEL_LEN` | `131072` | context window |
| `MAX_NUM_SEQS` | `4` | concurrent requests |
| `GPU_MEMORY_UTILIZATION` | `0.73` | memory fraction for vLLM |
| `MTP_TOKENS` | `4` | draft tokens per step; `0` turns speculative decoding off (about 30 tok/s) |
| `DRAFT_VOCAB` | `en` | `cjk` adds every CJK, kana and Hangul token to the draft vocabulary for Chinese, Japanese and Korean answers |
| `CA_SCHED` | `1` | cache-aware scheduling for several agent sessions at once; `0` is the stock scheduler, which is faster for unrelated parallel requests that share no prefix |

## Limitations

- Tested on one DGX Spark with the pinned image digest. The patches replace files of that build, so newer
  `eugr/spark-vllm-b12x` tags are not drop-in replacements, and the custom kernels are built for sm_121a (GB10) only.
- The context window is 131K tokens; stock vLLM fits 262K in the same memory.
- Up to 4 concurrent requests. The default scheduler favors multi-turn sessions: unrelated parallel requests that share
  no prefix are served nearly one at a time, with about one third less aggregate throughput than the stock scheduler
  (`CA_SCHED=0` restores it).
- The fine-tuned draft head helps most on agentic coding, and the default draft vocabulary favors English and code.
- b12x's MoE sums expert outputs with atomics, so greedy outputs can differ between runs.
- The quality evidence is GSM8K, HumanEval, KL divergence, needle and tool-call tests, not a full benchmark suite.

## How we measured

One DGX Spark (GB10, 128 GB), no other clients, every reading on a fresh server boot. Stock vLLM is NVIDIA's DGX Spark
command from `NVIDIA_MODEL_CARD.md` on the image build this recipe pins, at the same memory fraction;
`recipe/bench/stock-vllm.sh` starts it as measured. The agentic workloads are multi-turn coding-agent sessions with
tool definitions and tool calls, prompts up to 131K tokens, replayed in order at temperature 0; the sessions are not
part of this repo. Prefill and decode speeds come from fixed-prompt benches (`recipe/bench/c1bench.py` for the stock
comparison), quality from `recipe/bench/quality-probe.py`. The full method and every table of the earlier releases are in
`recipe/RELEASE_NOTES.md`; raw outputs are in `recipe/results/`.

## What's in this repo

| Files | Contents |
|---|---|
| `tensors/`, `config.json`, tokenizer, processor and generation files, `chat_template.jinja` | NVIDIA's `nvidia/Qwen3.8-Flash-Next-FP4-Local`; `tensors/model-00015-of-00021.safetensors` carries the fine-tuned MTP head (only its 29 dense MTP tensors differ from NVIDIA's file), every other file is unmodified |
| `NVIDIA_MODEL_CARD.md` | NVIDIA's model card for the checkpoint |
| `recipe/` | launcher, patches, CUDA kernels (source, build lines and built libraries), draft vocabularies, benchmarks, KL client, raw results |
| `LICENSE`, `LICENSE.Qwen`, `NOTICE` | NVIDIA Open Model Agreement, Qwen Community License 1.0, attribution and the list of modified tensors |

## Credits and license

Built on the work of the Qwen team (Qwen3.8-Flash-Next), NVIDIA (the FP4-Local checkpoint and Model Optimizer),
local-inference-lab and the FlashInfer team (b12x), eugr (the DGX Spark vLLM + b12x image), the vLLM team, and AtomicChat
(the KL reference). This repo is not affiliated with or endorsed by NVIDIA or Qwen.

- **Weights.** Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement (`LICENSE`). The fine-tuned MTP head
  is a derivative of NVIDIA's checkpoint under the same agreement; `NOTICE` lists the tensors that differ.
- **Base model.** Qwen3.8-Flash-Next is under the Qwen Community License 1.0 (`LICENSE.Qwen`). It requires a separate
  license from Qwen before commercial "Model as a Service" or "AI Work Assistant" use, and a visible model name in
  products above 100M monthly active users or US$20M monthly revenue.
- **Code.** The recipe's own code is MIT; the modified and new vLLM / b12x files in `recipe/patches/` are Apache-2.0
  (`recipe/THIRD_PARTY_NOTICES.md`).
