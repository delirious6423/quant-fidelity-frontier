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
kernels. The recipe is a pinned public image plus patched or new vLLM / b12x files and one CUDA kernel, all in
`recipe/`.

- **Prefill 3,150 tok/s at 33K tokens and 3,070 at 112K** (2,815–2,850 at 8.6K), +39–48% over the 2026-10-21 release
  on the same bench, from the 2026-11-01 prefill update (below).
- **Agentic coding sessions: 63–64 tok/s decode** (54 ms per speculative step, 3.4–3.6 tokens accepted per step) and
  0.76 s median time to first token on follow-up turns. The public 40-prompt category harness: 72.8–73.2 tok/s median.
  The prefill update leaves decode unchanged.
- **KL divergence 0.0492** against the BF16 model on AtomicChat's neutral set (top-1 91.7%, PPL +0.8%; 2026-10-21
  configuration, all 87 chunks). The 2026-11-01 changes are bit-identical or within noise on 6- and 24-chunk subsets,
  and the HyperConnection fusion lowers it by 3.3% (24 chunks, paired).
- GSM8K (first 200) 97.5 %, HumanEval 96.3–97.0 % (greedy, thinking off; 96.5 / 95.7 % at the 2026-10-21 release,
  within run-to-run noise); a needle is found at 29K, 96K and 125K tokens.

Measured on one DGX Spark at a 0.73 GPU-memory fraction with 8192-token prefill chunks (131K context, FP8 KV cache,
4 MTP draft tokens, CUDA graphs). Changes over the stock launcher: 71.2 → 54.2 ms per decode step, 46.9 → 64 tok/s
on the same sessions.

## Fidelity

Full-vocabulary KL(BF16 ‖ served model) in llama.cpp `--kl-divergence` semantics over AtomicChat's held-out neutral
set: the original BF16 model's logits, 87 chunks of 4096 tokens, 178,089 scored positions, BF16 PPL 4.045
([reference](https://huggingface.co/datasets/AtomicChat/Qwen3.8-Flash-Next-Uncensored-GGUF-metrics)). Every row is
measured against that same reference.

| Model | Size | Mean KLD | Top-1 | PPL ratio |
|---|---:|---:|---:|---:|
| **This repo, as served on 2026-10-21** (W4A16 experts) | 102 GB | **0.0492** | **91.7 %** | 1.008 |
| Same checkpoint, vLLM default FP4 activations (W4A4) | 102 GB | 0.0665 | 90.4 % | |
| 2026-10-21 configuration with `A4_PREFILL=1` (two-plane FP4 activations for prefill-size MoE calls; unmeasured on the 2026-11-01 configuration) | 102 GB | 0.0516 | 91.6 % | 1.012 |
| Same, one FP4 plane (b12x's `A4_PREFILL` default) | 102 GB | 0.0665 | 90.3 % | 1.012 |
| Vardaan97 Splash Quality-PLE8 GGUF ([card](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF)) | 129–157 GB | 0.0126 | | 1.000 |
| unsloth UD-Q4_K_XL + Q8 n-gram table (measured in the Splash card) | 141 GB | 0.0210 | 94.8 % | 1.005 |
| AtomicChat AD-4.27bpw-Q4_K_M ([card](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF)) | 95 GB | 0.0842 | 89.5 % | 1.026 |

"As served" is the server's own forward pass through `recipe/patches/qwen4_exp/nvidia/logit_dump.py`, scored with
`recipe/kl/atomic_kld.py`. NVIDIA's FP4 checkpoint is further from BF16 than the best GGUF builds; what it buys on the
Spark is vLLM serving speed. Running the routed experts with bf16 activations (`MOE_A16`, the setting NVIDIA's own
card recommends) is both closer to BF16 and faster than the FP4-activation default.

The 2026-11-01 prefill update was checked on subsets of the same reference (`recipe/results/kl-neutral-24ch-*`).
These figures are not comparable with the 87-chunk table above.

| 2026-11-01 change | Chunks | Before | After |
|---|---:|---:|---:|
| PLE lookahead, prefill bundle, qpmoe r17 → r18 | 6 | identical logits | identical logits |
| qpmoe prefill MoE kernel (fixed-order expert sum instead of b12x's bf16 atomics) | 6 | 0.0386 / 0.0391 (two runs) | 0.0375 |
| HyperConnection prefill fusion (as served) | 24 | 0.04811 ± 0.00057, top-1 92.03 % | **0.04653 ± 0.00056**, top-1 92.06 % |

The HC fusion row is a paired per-position comparison (−3.3%, t = −4.7; `recipe/results/kl-neutral-24ch-paired.txt`).

Quality probes with thinking off and greedy decoding (`recipe/bench/quality-probe.py`):

| | GSM8K-200 | HumanEval |
|---|---:|---:|
| This repo, as served (2026-11-01, two boots) | 97.5 / 97.5 % | 96.3 / 97.0 % |
| This repo, as served on 2026-10-21 | 96.5 % | 95.7 % |

b12x's decode MoE sums expert outputs with bf16 atomics, so these scores move by about 1.5 points between boots.

A needle planted in prompts of 28,842, 96,042 and 124,842 tokens is found at every length
(`recipe/bench/needle-test.py`), and tool calls with `tool_choice=auto` parse (`recipe/bench/test-toolcall.py`).

## Speed

Test setup: one DGX Spark (GB10), `recipe/serve/serve.sh` with its defaults, no other clients, Spark idle otherwise.
Throughput is in tokens per second.

**Agentic coding sessions** (private session set, not published: multi-turn coding agents with tool definitions and
tool calls in the history; 1,362 decode steps per run):

| | Decode (weighted) | Decode (median per turn) | Time to first token (median) | Tokens per step |
|---|---:|---:|---:|---:|
| 2026-10-21 release | 63.5–64.0 | 63.7–63.9 | 0.74–0.76 s | 3.40–3.56 |
| Stock launcher (image + NVIDIA's flags) | 46.9 | | | |

With the 2026-11-01 prefill update, the same session set runs at 53.7–53.8 ms per step with a 0.72 s median time
to first token, so decode is unchanged.

**Public 40-prompt category harness** (tonyd2wild's `bench_categories.py`, thinking off, temperature 0): 72.8–73.2 tok/s
median decode, 0.28 s median time to first token, 53.5 ms per step (2026-10-21 configuration).

**Prefill** (cold, single request, `recipe/bench/c1bench.py`; two runs each):

| Prompt | 2026-10-21 release | 2026-11-01 | |
|---:|---:|---:|---:|
| 8.6K tokens | 2,034–2,053 tok/s | 2,815–2,850 tok/s | +39% |
| 33K tokens | 2,105–2,144 tok/s | **3,151–3,159 tok/s** | +48% |
| 112K tokens | 2,129–2,139 tok/s (TTFT 52.6–52.9 s) | **3,063–3,070 tok/s** (TTFT 36.7 s) | +44% |

The release column was measured on 2026-10-30 with the release's launcher settings (4096-token chunks); the
2026-10-21 card listed 1,696 / 2,134–2,169 / 2,131 tok/s at 2K / 16K / 64K on an earlier bench. What changed is
listed in `recipe/README.md` ("Prefill update"), with A/B results per step in `recipe/results/abba-1101.txt`.

`A4_PREFILL=1` (off by default) runs prefill-size MoE calls on b12x's FP4 tensor-core path with two FP4 activation
planes (value + residual, `B12X_W4A16_A4_PREFILL_TERMS=2`) instead of the qpmoe kernel. Its speed and KLD are
unmeasured on the 2026-11-01 configuration; the 2026-10-21 figures (KLD 0.0516, one plane 0.0665) are in the
fidelity table above.

**Synthetic prompts** (`recipe/bench/bench-sbs.py`: a random-word document plus a question, non-streaming; decode
speed is the difference between a 400-token and a 16-token reply on a warm prefix cache). Random-word text is hard
for the drafter, so decode here is below the agentic numbers above (2026-10-21 configuration, not re-run):

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
layer and kept warm with a keepalive read), so about 68 GiB of weights stay resident. vLLM gets 73% of the unified
memory (70% before 2026-11-01; the larger 8192-token prefill chunks need the extra room). The KV cache holds about
160–190K tokens depending on the boot, enough for one 131K-token request plus a smaller one.

The server needs about 93 GiB of the Spark's 121.7 GiB at the peak of a 112K-token prefill (GPU allocations plus
~6 GB of process memory), and vLLM refuses to start unless 88.8 GiB (0.73 × 121.7) is free. That leaves about
26 GiB available on an otherwise idle Spark. If another memory-heavy service has to run next to it, use the lean
setting:

```bash
MAX_NUM_BATCHED_TOKENS=4096 GPU_MEMORY_UTILIZATION=0.68 \
EXTRA_ARGS="--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0" bash serve/serve.sh
```

It frees about 7 GiB, keeps the full 131K context (KV 140K tokens) and turns off image and video input. Prefill
is about 8% lower (2,856 vs 3,096 tok/s at 112K); decode is the same. Keeping 8192-token chunks with image and
video off at `GPU_MEMORY_UTILIZATION=0.71` frees 2.4 GiB at full speed (`recipe/results/memory-1101.txt`).

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

The first start compiles and autotunes the b12x kernels into `~/.cache/qwen38-flash-next-spark`. That took 15.5
minutes from an empty cache on our Spark before the 2026-11-01 update, whose kernels add compile time to the first
start; later starts take about 5 minutes. Tool calling uses vLLM's `qwen3_xml` parser and reasoning the `qwen3` parser.

## Limitations

- Tested only on one DGX Spark with the pinned image; other GB10 systems should work but are untested.
- The patches replace files of the pinned image digest and are validated only against it. Newer
  `eugr/spark-vllm-b12x` tags are not drop-in replacements (the 2026-11-01 nightly carries an older FlashInfer b12x
  port without the 2026-10-15 MoE race fixes).
- The qpmoe prefill MoE library is built for sm_121a (GB10) only.
- The first boot compiles and autotunes kernels and takes much longer than later boots (see Quick start); the
  2026-11-01 numbers were measured on warm caches.
- b12x's MoE sums expert outputs with atomics, so greedy outputs can differ between runs. `MOE_DETERMINISTIC=1`
  fixes this only for FP4-activation MoE.
- `TAIL_CKPT` places the prompt checkpoint just before the prompt end; upstream vLLM's `--prefix-match-unit` does not
  replace it on this image.
- Up to 4 concurrent requests (`MAX_NUM_SEQS`); the configuration is tuned for single-user agentic work.
- The default draft vocabulary is tuned for English and code; use `DRAFT_VOCAB=cjk` for Chinese, Japanese or Korean.
- The quality evidence is the KL table, GSM8K-200, HumanEval and needle tests, not a full benchmark suite.

## Changelog

- **2026-11-01: prefill update.** +39–48% c=1 prefill (3,150 tok/s at 33K), decode unchanged. New defaults:
  8192-token prefill chunks at a 0.73 memory fraction, and four file sets under `recipe/patches/`: a prefill set
  (MTP feedback on cuBLAS, GDN identity path, fused HC up-gate, dense-GEMM swizzle), the qpmoe prefill MoE kernel
  (CUDA source, build script and library), a PLE read lookahead, and a prefill bundle (GDN strided conv, pipelined
  QSA gathers, MXFP8 tiles, HC fusion). KLD on a 24-chunk subset −3.3%. Rollback knobs are listed in
  `recipe/README.md` ("Rollback"); the previous release is revision `e2518185`.
- 2026-10-30: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fixes random boot OOMs on unified memory).
- 2026-10-21: CJK draft vocabulary (`DRAFT_VOCAB=cjk`); `A4_PREFILL` uses two FP4 planes.
- 2026-10-20: first release.

## Credits and license

Built on the work of NVIDIA (the FP4-Local checkpoint and Model Optimizer), the Qwen team (Qwen3.8-Flash-Next),
local-inference-lab and the FlashInfer team (b12x), eugr (the Spark vLLM + b12x image), the vLLM team, AtomicChat
(the KL reference) and Vardaan97 (the Splash measurements on the same reference).

The weights are NVIDIA's, under the NVIDIA Open Model Agreement (`LICENSE`): "Licensed by NVIDIA Corporation under the
NVIDIA Open Model Agreement." The base model is under the Qwen Community License 1.0 (`LICENSE.Qwen`), which requires a
separate license from Qwen for commercial "Model as a Service" offerings. The recipe's own code is MIT; the files in
`recipe/patches/` are Apache-2.0 (`recipe/THIRD_PARTY_NOTICES.md`).
