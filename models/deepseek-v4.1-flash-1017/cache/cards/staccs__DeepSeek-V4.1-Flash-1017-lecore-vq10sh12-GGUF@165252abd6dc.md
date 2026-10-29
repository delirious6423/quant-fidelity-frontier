---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
  - gguf
  - quantized
  - vector-quantization
  - deepseek
  - moe
pipeline_tag: text-generation
library_name: gguf
language:
- en
base_model_relation: quantized
---

# DeepSeek-V4.1-Flash-1017 — lecore vq10sh12 GGUF (99.1 GB)

**A DeepSeek V4.1 Flash built to run at home**: the 763B-parameter model in one 99 GB file, made to fit a
single 128 GB machine.

**TL;DR — side by side**

| | original (DeepSeek) | [Baekpica VQ GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF) | **this build** |
| --- | ---: | ---: | ---: |
| model weights | 307.2 GB | 113.6 GB | **99.1 GB** |
| routed experts | FP4/FP8 | VQ 13-bit (layers 0–13) / 12-bit (14–39) | VQ 12-bit (0–13) / 10-bit (14–39) |
| Engram n-gram tables | 203.1 GB | 203.1 GB (original shards) | 203.1 GB (original shards) |
| total on disk | 510.3 GB | ~318 GB | 302.2 GB |
| WikiText-2 perplexity | **1.838** | not published | **3.615** |
| same next word as the original, WikiText-2 | 100% (reference) | 78.5%\* | **74.5%** |
| mean KL vs the original, WikiText-2 | 0 (reference) | 0.694\* | **0.797** |
| runs on | a multi-GPU server | one 128 GB DGX Spark | one GPU with ≥104 GB (H200 tested); native-Linux Strix Halo untested |
| decode speed, one stream | — | 54–66 tok/s (DGX Spark, Baekpica's card) | **46–54 tok/s** (1× H200 NVL, measured) |

\* Baekpica's quality figures are YoungAi's measurement on the bare base, quoted on Baekpica's card. They were
taken on V4.1 Flash (the release before 1017) with YoungAi's own text window, so they are close to but not exactly
the same test as ours. Our two columns (original and this build) are the same test: the first 2,048 tokens of the
WikiText-2 test set, the same official forward pass, with Engram.

### Pros and cons

**Pros**
- Fits one 128 GB machine with room for context: 99.1 GB of weights. The other two need 113.6 GB or a server.
- The full model: 763B parameters, all 384 experts per layer, Engram and the MTP drafter all included. Nothing is pruned or distilled.
- Engram tables are used unchanged from DeepSeek's own files.
- 15 GB smaller than the Baekpica build, for about 4 points less next-word agreement.

**Cons**
- Clearly weaker than the original: perplexity 3.6 against 1.8, and the next-word choice differs about 1 time in 4.
- A little weaker than the 113.6 GB build, with no calibration and no correction sidecar yet.
- Needs a custom runtime (ds4-dfm-rs with a small patch, below). llama.cpp, vLLM and Ollama cannot load it.
- Still a ~300 GB download in total, because of the Engram tables.
- Task accuracy is not measured yet: so far, next-word fidelity plus a few spot-check prompts.

### Verified running (2026-10-22)

`ds4-server` on 1× H200 NVL (CUDA 13, sm_90) with this GGUF and the Engram shards, greedy, thinking off:

- boot: weights copied to the GPU in 5.2 s, server ready in 10 s
- "What is the capital of Australia, and why is it not Sydney?" → *"Canberra is the capital of Australia because it
  was chosen as a compromise between Sydney and Melbourne, the two rival cities that both wanted to be the
  national capital."*
- "Write a Python function that returns the n-th prime, with two doctests" → correct trial-division `nth_prime`
  with correct doctests (`nth_prime(1) == 2`, `nth_prime(10) == 29`)
- time to first token 0.24–0.31 s; steady decode 46–54 tok/s on 400-token answers

**In plain words:** we had the original model and this build read the same article and guess every next word.
They picked the same word **74 times out of 100**. Where they disagree, this build is usually less sure rather
than wildly off, but some words go clearly wrong. So expect a noticeably weaker model than the full DeepSeek:
more factual slips and shakier long reasoning. The trade is size — the full model needs a 500 GB-class server,
this one fits a single 128 GB machine. For reference, YoungAi's notes put their ~113 GB, all-12-bit build
(14% bigger, with gain correction) at about 79 out of 100 on a similar test, so this is roughly what this size buys.
Real task scores (coding, Q&A) have not been measured yet.

A 99.1 GB build of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(763B parameters), sized to run on a single 128 GB unified-memory machine
(tested target: Ryzen AI Max+ 395 / Radeon 8060S with 96 GB Variable Graphics Memory).

| file | bytes | sha256 |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-lecore-vq10sh12-q4k-mtpnative.gguf` | 99,116,576,416 | `d60d8dfae87d0e7480b24c45093ae8b0058f5407cbf14fbe851f179960da63d4` |

## Recipe

Quantized from the official checkpoint at revision `1bae0e1088993b9b82de08f59de5be5a34e42170` with
[fodelf/YoungAi](https://github.com/fodelf/YoungAi) `v41_quantize` (commit `7fb26cc`), zero calibration:

| part | format |
| --- | --- |
| routed experts, layers 0–13 | 8-dim VQ, 4096 codewords (12-bit indices), shared per-layer E4M3 codebook |
| routed experts, layers 14–39 | 8-dim VQ, 1024 codewords (10-bit indices), shared per-layer E4M3 codebook |
| MTP drafter towers | 8-dim VQ, 1024 codewords (10-bit) |
| everything else (attention, shared experts, embeddings, head) | q4_K |
| Engram n-gram tables (2 × 101.5 GB) | **not in this file** — read from the official shards at runtime |

Command: `v41_quantize --skel q4k --mtp-vq --vq-nc 1024 --vq-nc-layers 0:14=4096 --vq-shared-cb --vq-cb-fp8`.
YoungAi's converter and the runtime only accepted 13-bit v3 indices; this build needed a small patch to both
for 10/11/12-bit indices (index width `min(nbit, 12)`, persist-kernel mask fix).

Weight-space fidelity reported by the quantizer: 12-bit layers cos ≈ 0.911, 10-bit layers cos ≈ 0.873.
Excluding the Engram tables (served from the original files), the remaining ~566B parameters average about 1.4 bits per weight.

## Benchmark: original vs this build

Same official forward pass for both (YoungAi `v41_teacher.py`, DeepSeek's `inference/model.py` with weights
streamed per layer, dequantized to full-precision matmuls), same tokens: the first 2,048 tokens of the
WikiText-2 raw test set, tokenized with the model's own tokenizer. Scored with YoungAi `anchor_metrics`.

| metric (WikiText-2 test, 2,048 tokens) | original (FP8/FP4 checkpoint) | this build |
| --- | ---: | ---: |
| perplexity | 1.838 | 3.615 (ratio 1.967) |
| mean NLL | 0.6089 | 1.2851 (+0.6763) |
| mean KL divergence vs original | — | 0.797 (median 0.195, p95 3.788) |
| probability mass recovered, Σ min(p, q) | — | 0.692 (median 0.815, p5 0.054) |
| same top-1 token as original | — | 74.46% |
| RMS Δp over the original's top-32 | — | 6.94% |
| Δp on the original's top token | — | 0.269 (p95 0.923) |

For scale: the original predicts the next WikiText token correctly 85.4% of the time, so this text is easy for it.
This is a real quality loss — about twice the perplexity, and a different top token on roughly one token in four.
YoungAi's own target for the mass-recovered metric is ≥ 0.90; this zero-calibration 10/12-bit build is at 0.69.
It is one 2,048-token text; downstream task accuracy has not been measured.

Hardware for the benchmark: 1× H100 NVL 94 GB, 314 GB RAM. The original checkpoint (476 GiB) does not fit in
GPU + RAM, so its pass streams the weights from disk layer by layer; the quantized pass (this build's
weights, 15,744 VQ experts read back from the quantized files) took 156 s of forward time.

DeepSeek's model card reports agentic benchmarks (HLE, Terminal-Bench 4.0, DeepSWE, …). Those were **not** run on
this build; no claims are made about them here.

**Exploratory, not a score — BabyVision (no tools).** A fixed, stratified 100-item sample of
[UnipatAI/BabyVision](https://huggingface.co/datasets/UnipatAI/BabyVision) (99 scored), thinking on, greedy,
8,192-token budget, choice items by exact option and blank items by normalized exact match: 11/99 (11.1%) raw.
58 of the 99 answers hit the token budget before giving an answer; on the 41 that finished, 10 were right (24.4%).
This says more about the budget than the model. It is not comparable to DeepSeek's 92.5 (*with tools*) or to the
paper's no-tool leaderboard, and a proper run needs a much larger budget.

## Running it

This GGUF targets the V4.1 VQ runtime [ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs) (`ds4-server`), built with the
10/11/12-bit v3 index patch described above (the patch is not yet published upstream; a pull request is being
prepared, with CPU-oracle and CUDA fixture tests). It needs:

- the two Engram shards from the official repo: `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors`
  (revision above), passed with `--engram-dir`;
- optionally the vision tower sidecar `DeepSeek-V4.1-Flash.vision.gguf` (`--vision-gguf`).

```bash
ds4-server -m DeepSeek-V4.1-Flash-1017-lecore-vq10sh12-q4k-mtpnative.gguf \
  --engram-dir <dir with the two engram shards> --vision-gguf DeepSeek-V4.1-Flash.vision.gguf \
  --mtp-mode auto -c 32768 --host 0.0.0.0 --port 8008
```

`--v41-no-engram` runs without the Engram tables, but that removes part of the model and does not represent its quality.

On a GPU with its own memory (anything but a unified-memory box like the DGX Spark), set `DS4_CUDA_COPY_MODEL=1`.
Without it the weights stay host-mapped and every token reads them over the bus: on the H200 that was 7–9 tok/s
instead of 46–54. A GPU needs about 104 GB free for this file (weights plus the runtime's 4 GB floor), so an 80/94 GB
card does not fit; an H200 (141 GB) does.

The runtime patch has two parts: 10/11/12-bit index support in the decode kernels, and routing 10/11-bit layers'
prompt processing to the fused kernels (the tensor-core prefill only reads 12/13-bit layers).

### AMD (ROCm, gfx1151 / Strix Halo)

A HIP port of the runtime for the Radeon 8060S is in progress. Status as of 2026-10-22:

- the ds41 GPU fixture tests pass on the Radeon 8060S under ROCm 7.2 (WSL2): VQ decode at 10/11/12/13 bits
  bit-exact, MoE at 10 and 12 bits, FP8, Engram rows, sampling;
- the full 99 GB model has **not** yet been run on it end to end.

Memory: **WSL2 cannot run this model.** With Variable Graphics Memory at 96 GB, HIP under WSL2 reports
111.7 GiB and lets you *allocate* 108 GiB, but writing to it stalls after about 32 GiB (the WSL GPU bridge
stops making memory resident), and the server hangs while loading the weights. Native Linux with ROCm is
the expected route on Strix Halo; it has not been tested with this file yet.

## Credits

DeepSeek for the model (MIT). fodelf/YoungAi for the VQ quantizer and the reference-forward judge.
Baekpica for the ds4-dfm-rs V4.1 runtime and the original VQ build this recipe follows.

## Runtime

Needs the patched **ds4 / ds4-server** runtime — llama.cpp, vLLM and Ollama cannot load the VQ
blobs. Source, Windows/HIP build scripts, the launcher and the supervisor:
[staccs/ds4-winhip](https://huggingface.co/staccs/ds4-winhip).

## Files in this repo

| file | size | what it is |
|---|---:|---|
| `DeepSeek-V4.1-Flash-1017-lecore-vq10sh12-q4k-mtpnative.gguf` | 99.12 GB | model weights |
| `DeepSeek-V4.1-Flash-1017-lecore-vq10sh12-q4k-mtpnative.gguf.sha256` | 141 B | support file |

One 99.12 GB (92.3 GiB) file of weights. A `DeepSeek-V4.1-Flash-1017-lecore-vq10sh12-q4k-mtpnative.gguf.sha256` sidecar is included.

## Sidecars and Engram tables

The 103 GB file is the model, not the whole runtime. Three companion files plus DeepSeek's Engram
shards are needed to serve it:

| what | file | size | where |
|---|---|---:|---|
| Vision tower (image input) | `DeepSeek-V4.1-Flash.vision.gguf` | 0.90 GiB | [staccs/ds41-sidecars](https://huggingface.co/staccs/ds41-sidecars) |
| DSpark speculative drafter | `DeepSeek-V4.1-Flash-1017-abliterated-pqhaz-lecore-vq11cal.dspark.llama.gguf` | 5.05 MiB | [staccs/ds41-sidecars](https://huggingface.co/staccs/ds41-sidecars) |
| Dense-skeleton GGUF (llama.cpp layout, no experts) | `DeepSeek-V4.1-Flash-1017-abliterated-pqhaz-lecore-vq11cal.llama.gguf` | 605.8 MiB | [staccs/ds41-sidecars](https://huggingface.co/staccs/ds41-sidecars) |
| Engram n-gram tables (used unchanged) | `model-00047-of-00048.safetensors`, `model-00048-of-00048.safetensors` | 101.5 GB each | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |

Drop the vision file next to the big GGUF (the launcher points `--vision-gguf` at that directory) and
pass `--engram-dir <dir>` with the two shards in it. The drafter and skeleton files are the pqhaz
build's; other abliterations can use the vision tower and Engram tables but should expect a lower
DSpark accept rate with a drafter trained on different weights.
