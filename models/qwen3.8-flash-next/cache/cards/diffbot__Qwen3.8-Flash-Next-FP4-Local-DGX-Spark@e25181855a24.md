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

NVIDIA's [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) checkpoint
(unmodified), with a serving recipe for **a single DGX Spark** (GB10, 128 GB unified memory) on vLLM and the b12x
kernels. The recipe is a pinned public image plus 22 patched or new vLLM / b12x files, all in `recipe/`.

- **Agentic coding sessions: 63–64 tok/s decode** (54 ms per speculative step, 3.4–3.6 tokens accepted per step) and
  0.76 s median time to first token on follow-up turns. The public 40-prompt category harness: 72.8–73.2 tok/s median.
- **Prefill 2,130–2,160 tok/s** from 16K to 64K tokens (2,207–2,247 with the opt-in `A4_PREFILL=1`, KLD 0.0516).
- **KL divergence 0.0492 as served** against the BF16 model on AtomicChat's neutral set (top-1 91.7%, PPL +0.8%).
- GSM8K (first 200) 96.5 %, HumanEval 95.7 % (greedy, thinking off); a needle is found at 29K, 96K and 125K tokens.

Measured on one DGX Spark at a 0.70 GPU-memory fraction (131K context, FP8 KV cache, 4 MTP draft tokens, CUDA
graphs). Changes over the stock launcher: 71.2 → 54.2 ms per decode step, 46.9 → 64 tok/s on the same sessions.

## Fidelity

Full-vocabulary KL(BF16 ‖ served model) in llama.cpp `--kl-divergence` semantics over AtomicChat's held-out neutral
set: the original BF16 model's logits, 87 chunks of 4096 tokens, 178,089 scored positions, BF16 PPL 4.045
([reference](https://huggingface.co/datasets/AtomicChat/Qwen3.8-Flash-Next-Uncensored-GGUF-metrics)). Every row is
measured against that same reference.

| Model | Size | Mean KLD | Top-1 | PPL ratio |
|---|---:|---:|---:|---:|
| **This repo, as served** (W4A16 experts) | 102 GB | **0.0492** | **91.7 %** | 1.008 |
| Same checkpoint, vLLM default FP4 activations (W4A4) | 102 GB | 0.0665 | 90.4 % | |
| This repo with `A4_PREFILL=1` (two-plane FP4 activations for prefill-size MoE calls) | 102 GB | 0.0516 | 91.6 % | 1.012 |
| Same, one FP4 plane (b12x's `A4_PREFILL` default) | 102 GB | 0.0665 | 90.3 % | 1.012 |
| Vardaan97 Splash Quality-PLE8 GGUF ([card](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF)) | 129–157 GB | 0.0126 | | 1.000 |
| unsloth UD-Q4_K_XL + Q8 n-gram table (measured in the Splash card) | 141 GB | 0.0210 | 94.8 % | 1.005 |
| AtomicChat AD-4.27bpw-Q4_K_M ([card](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)) | 95 GB | 0.0842 | 89.5 % | 1.026 |

"As served" is the server's own forward pass through `recipe/patches/qwen4_exp/nvidia/logit_dump.py`, scored with
`recipe/kl/atomic_kld.py`. NVIDIA's FP4 checkpoint is further from BF16 than the best GGUF builds; what it buys on the
Spark is vLLM serving speed. Running the routed experts with bf16 activations (`MOE_A16`, the setting NVIDIA's own
card recommends) is both closer to BF16 and faster than the FP4-activation default.

Quality probes with thinking off and greedy decoding (`recipe/bench/quality-probe.py`):

| | GSM8K-200 | HumanEval |
|---|---:|---:|
| This repo, as served | 96.5 % | 95.7 % |

A needle planted in prompts of 28,842, 96,042 and 124,842 tokens is found at every length
(`recipe/bench/needle-test.py`), and tool calls with `tool_choice=auto` parse (`recipe/bench/test-toolcall.py`).

## Speed

Test setup: one DGX Spark (GB10), `recipe/serve/serve.sh` with its defaults, no other clients, Spark idle otherwise.
Throughput is in tokens per second.

**Agentic coding sessions** (private session set, not published: multi-turn coding agents with tool definitions and
tool calls in the history; 1,362 decode steps per run):

| | Decode (weighted) | Decode (median per turn) | Time to first token (median) | Tokens per step |
|---|---:|---:|---:|---:|
| This release | 63.5–64.0 | 63.7–63.9 | 0.74–0.76 s | 3.40–3.56 |
| Stock launcher (image + NVIDIA's flags) | 46.9 | | | |

**Public 40-prompt category harness** (tonyd2wild's `bench_categories.py`, thinking off, temperature 0): 72.8–73.2 tok/s
median decode, 0.28 s median time to first token, 53.5 ms per step.

**Prefill** (cold, single request):

| Prompt | Prefill |
|---:|---:|
| 2K tokens | 1,696 tok/s |
| 16K tokens | 2,134–2,169 tok/s |
| 64K tokens | 2,131 tok/s |

`A4_PREFILL=1` runs prefill-size MoE calls on the FP4 tensor cores with two FP4 activation planes (value + residual,
`B12X_W4A16_A4_PREFILL_TERMS=2`): 1,869 / 2,247 / 2,207 tok/s at 2K / 16K / 64K tokens (+3.5–4%), decode unchanged, KLD
0.0516 instead of 0.0492. One plane, as NVIDIA's card configures it, is +6–8% but KLD 0.0665, the W4A4 level. Both are off by
default.

**Synthetic prompts** (`recipe/bench/bench-sbs.py`: a random-word document plus a question, non-streaming; decode
speed is the difference between a 400-token and a 16-token reply on a warm prefix cache). Random-word text is hard
for the drafter, so decode here is below the agentic numbers above:

| Benchmark | Prefill | Time to first token | Decode |
|---|---:|---:|---:|
| 2K prompt, 1 stream | 1,850 | 1.1 s | 51.7 |
| 2K prompts, 4 streams | 2,230 aggregate | 4.1 s | 107.0 aggregate (26.7 per stream) |
| 46K prompt, 1 stream | 2,259 | 20.3 s | 45.8 |
| 46K prompts, 4 streams | ~2,190 aggregate (184K tokens in 84 s) | 84 s | 37.4 aggregate (9.4 per stream) |

Four 46K-token contexts (184K tokens) exceed the KV cache, so vLLM preempts and recomputes; the configuration is
sized for one long agentic session at a time.

**Chinese, Japanese and Korean answers.** The MTP drafter scores a vocabulary subset. The default subset (`DRAFT_VOCAB=en`,
65,536 tokens) was ranked on English agentic coding and holds almost no CJK tokens, so CJK answers get few accepted
drafts. `DRAFT_VOCAB=cjk` adds every CJK, kana and Hangul token (130,402 in all). 12 prompts, 400-token greedy answers,
thinking off (`tok/s` includes the short prefill):

| Answers in | `DRAFT_VOCAB=en` (default) | `DRAFT_VOCAB=cjk` | Draft acceptance |
|---|---:|---:|---:|
| Chinese | 37.3 | **54.3** (+46%) | 24.6% → 51.8% |
| Japanese | 30.1 | **48.8** (+62%) | 14.4% → 42.3% |
| Korean | 34.7 | **45.0** (+30%) | 19.7% → 36.4% |
| English (40-prompt harness, median) | 72.8–73.2 | 69.9 (−4%) | unchanged |

The larger subset costs about 3 ms per step on every answer (the draft head reads twice the rows), so pick `cjk` when most
answers are in those languages.

## How it fits in one Spark

The FP4-Local checkpoint is 102 GB, including 51B parameters of n-gram (PLE) embedding tables. The launcher serves
those tables from NVMe instead of memory (b12x reads them with io_uring each step, overlapped with the first decoder
layer and kept warm with a keepalive read), so about 68 GiB of weights stay resident. vLLM gets 70% of the unified
memory; the rest stays free for the OS and other services. The KV cache holds about 175–240K tokens depending on the
boot, enough for one 131K-token request plus a smaller one.

The launcher also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. On the Spark's unified memory, the
default PyTorch caching allocator keeps freed load-time buffers as cached GPU segments, which are system RAM. During
weight post-processing that peaked 35–40 GB above the resident weights and could exhaust the 128 GB, so boots failed
or were OOM-killed at random. With expandable segments the boot never drops below about 29 GB free, and serving
speed is unchanged (measured 2026-10-30).

## What's in this repo

| Files | Contents |
|---|---|
| `tensors/`, `config.json`, tokenizer, processor and generation files, `chat_template.jinja` | NVIDIA's `nvidia/Qwen3.8-Flash-Next-FP4-Local`, unmodified. |
| `NVIDIA_MODEL_CARD.md` | NVIDIA's model card for those files. |
| `recipe/` | Launcher, patches, draft vocabulary, benchmarks, KL client, raw results (`recipe/README.md`). |
| `LICENSE`, `LICENSE.Qwen`, `NOTICE` | NVIDIA Open Model Agreement, Qwen Community License 1.0, attribution. |

## Quick start

You need one DGX Spark (or another GB10 system with 128 GB), Docker with the NVIDIA container toolkit, and about
110 GB of free disk on fast NVMe.

```bash
hf download diffbot/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark --local-dir ~/models/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark
cd ~/models/Qwen3.8-Flash-Next-FP4-Local-DGX-Spark/recipe
API_KEY=change-me bash serve/serve.sh     # OpenAI-compatible API on :8000, model qwen3.8-flash-next
```

The first start compiles and autotunes the b12x kernels into `~/.cache/qwen38-flash-next-spark` and took 15.5 minutes
from an empty cache on our Spark; later starts take about 5 minutes. Tool calling uses vLLM's `qwen3_xml` parser and reasoning the `qwen3` parser.

## Limitations

- Tested only on one DGX Spark with the pinned image; other GB10 systems should work but are untested.
- b12x's MoE sums expert outputs with atomics, so greedy outputs can differ between runs. `MOE_DETERMINISTIC=1`
  fixes this only for FP4-activation MoE.
- `TAIL_CKPT` places the prompt checkpoint just before the prompt end; upstream vLLM's `--prefix-match-unit` does not
  replace it on this image.
- Up to 4 concurrent requests (`MAX_NUM_SEQS`); the configuration is tuned for single-user agentic work.
- The default draft vocabulary is tuned for English and code; use `DRAFT_VOCAB=cjk` for Chinese, Japanese or Korean.
- The quality evidence is the KL table, GSM8K-200, HumanEval and needle tests, not a full benchmark suite.

## Credits and license

Built on the work of NVIDIA (the FP4-Local checkpoint and Model Optimizer), the Qwen team (Qwen3.8-Flash-Next),
local-inference-lab and the FlashInfer team (b12x), eugr (the Spark vLLM + b12x image), the vLLM team, AtomicChat
(the KL reference) and Vardaan97 (the Splash measurements on the same reference).

The weights are NVIDIA's, under the NVIDIA Open Model Agreement (`LICENSE`): "Licensed by NVIDIA Corporation under the
NVIDIA Open Model Agreement." The base model is under the Qwen Community License 1.0 (`LICENSE.Qwen`), which requires a
separate license from Qwen for commercial "Model as a Service" offerings. The recipe's own code is MIT; the files in
`recipe/patches/` are Apache-2.0 (`recipe/THIRD_PARTY_NOTICES.md`).
