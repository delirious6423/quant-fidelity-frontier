---
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
base_model:
- autotrust/GLM-5.3-SLIM-E192
- zai-org/GLM-5.3
language:
- en
- zh
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- dgx-spark
- moe
- expert-pruning
- iq2_xxs
- glm
- glm-dsa
---

# GLM-5.3-GGUF-DGX-Spark

[![AutoTrust/GuruSearch recommendation & search live demo: news.guru.so](https://huggingface.co/autotrust/JEV-27B/resolve/main/assets/autotrust_gurusearch_demo.png)](https://news.guru.so/ui)

## 🔴 Live: AutoTrust/GuruSearch recommendation & search demo → [news.guru.so](https://news.guru.so/ui)

> [!IMPORTANT]
> **Try AutoTrust/GuruSearch live (new, 11 October 2026).** A recommendation and search demo in which every ranking is a
> calibrated System 1 decision, made in about half a second:
> * **Recommendation:** the latest headlines from 6 news feeds, ranked by importance.
> * **Search:** results from Google News, Bing News and Yahoo News re-ranked by relevance, side by side with the
>   search engines' own order, plus a short answer with citations.

**GLM 5.3 at frontier quality, 24 % smaller — Sparse-Squared compression with AutoTrust SLIM-Q, packaged for DGX Spark owners.**

This repository is the llama.cpp GGUF build of **autotrust/GLM-5.3-SLIM-E192**, the SLIM-Q-compressed checkpoint that powers **Guru Turbo 2.0**, served in NVFP4 through **ScienceGuru** and the **Guru Apps**. Here it ships as one 149.7 GiB GGUF (glm-dsa architecture, runs on upstream llama.cpp) sized to fit a pair of DGX Sparks, a single 180 GB GPU, or a Mac Studio.

---

## SLIM-Q: AutoTrust's Sparse-Squared compression recipe

**SLIM-Q** — **S**elective expert pruning + **L**ow-bit quantization for **I**nference of **M**oE — is AutoTrust's two-stage pipeline for post-training compression of frontier open-source MoE models.

The idea in one line is **Sparse-Squared (Sparse²) compression**. A MoE is already sparse at inference: each token activates only the top-8 of 256 routed experts. SLIM-Q adds a second, structural sparsity axis — permanently removing the experts the router rarely uses — and then drives the surviving experts to low precision. Sparsity in activation × sparsity in the expert pool, at low bit-width:

**Stage 1 — Selective expert pruning (SLIM).** Expert usage in a MoE is highly skewed and task-dependent. AutoTrust profiles routing frequencies across representative workloads and *structurally removes* the 64 least-used of 256 routed experts per layer (25 %) from Z.AI's 744B-parameter GLM 5.3. Unlike dynamic expert-skipping methods, which save latency but keep every expert in memory, structural pruning permanently shrinks the weight footprint — the axis that determines what hardware can serve the model at all.

**Stage 2 — Low-bit quantization (Q).** The pruned checkpoint is then quantized aggressively on the routed experts while attention, routers, and shared experts stay at high precision — preserving the routing behavior that MoE quality depends on. This build uses the IQ2_XXS recipe (2.06 bits/weight on routed experts, Q8_0 elsewhere); the Turbo 2.0 production build uses NVFP4.

Compressing a trillion-scale MoE this way is rare and technically demanding — published expert-pruning-plus-quantization research tops out around 50B-parameter models, more than an order of magnitude below GLM 5.3. The payoff is a structural cost advantage quantization alone cannot deliver:

- **−47 GiB (−24 %) footprint** versus the full GLM 5.3 Q2 at the identical quantization recipe — the difference between "needs a 200 GB-class deployment" and "fits two DGX Sparks fully resident."
- **Zero added per-token cost.** Routing still selects the top-8 of the remaining 192 experts, so compute and memory traffic per token are unchanged; only the footprint drops.
- **Quality retained where it counts.** On the FP8 pruned base versus the original (author's A/B under vLLM): HumanEval 95.1 → 95.1, CyberMetric 88.0 → 87.7, BFCL live 69.6 → 69.4, BFCL multi-turn 73.0 → 72.5, AIME −2.5 pt. SLIM-Q's pruning profile is deliberately workload-targeted: the trade-off lands on knowledge-heavy Chinese-exam benchmarks (C-Eval 92.0 → 84.7; GPQA-Diamond 86.9 → 83.3), while coding, agentic, and cybersecurity performance — the design target — is preserved.

---

## Deployment options

| Where | How | What to expect |
|---|---|---|
| **Two DGX Sparks** (ConnectX-7 link, NVIDIA's dual-Spark setup) | llama.cpp RPC: model split across the two 128 GB memories (~75 GiB each), fully resident | The intended Spark configuration for this file; decode is memory-bandwidth bound (~25 GB of weights per token) |
| **One DGX Spark** (128 GB) | llama.cpp mmap: partial residency, remainder paged from NVMe per token | Works, but slow (low single-digit t/s); use [GLM-5.3-Flash-GGUF-DGX-Spark](https://huggingface.co/autotrust/GLM-5.3-Flash-GGUF-DGX-Spark) (79 GiB) for a single Spark |
| **One 180 GB GPU** (B200 / GB200) | Resident | 42 t/s single stream, 177 t/s aggregate at 32 parallel requests (measured) |
| **Mac Studio 256 GB** (192 GB with small context) | Resident, Metal | Same class as the full GLM 5.3 Q2 on the same Mac |

---

## Download

```bash
hf download autotrust/GLM-5.3-GGUF-DGX-Spark --local-dir ./GLM-5.3-GGUF-DGX-Spark
```


The model is stored as four GGUF shards (Hugging Face's 50 GB per-file limit); llama.cpp loads them together when pointed at the first one: `GLM-5.3-Q2-DGX-Spark-00001-of-00004.gguf` … `-00004-of-00004.gguf`, 149.7 GiB total (GLM-5.3-SLIM-E192 in IQ2_XXS). Checksums in `GLM-5.3-Q2-DGX-Spark.sha256`.

---

## DGX Spark quick start

`glm-dsa` is supported by upstream llama.cpp. Build on each Spark:

```bash
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build build --config Release -j
```

**Two Sparks (resident):** start the RPC worker on the second machine, then the server on the first. Both machines read the weights they own, so put the file on both NVMe drives (or on shared storage).

```bash
# Spark B (worker)
./build/bin/rpc-server -H 0.0.0.0 -p 50052 -c

# Spark A (server): local GPU + Spark B over the ConnectX link
./build/bin/llama-server -m GLM-5.3-Q2-DGX-Spark-00001-of-00004.gguf -ngl 99 -fa on \
    --rpc <spark-b-ip>:50052 --tensor-split 1,1 \
    -c 32768 -np 2 --cont-batching --host 0.0.0.0 --port 8080
```

`--tensor-split 1,1` places about half of the 78 layers on each Spark (~75 GiB of weights each, leaving ~40 GiB per machine for context and buffers). Only activations cross the link per token, so the 200 GbE ConnectX connection is not the bottleneck. This configuration is llama.cpp's standard RPC path; the author has not run it on Spark hardware — please report numbers.

**One Spark (paged):** the same `llama-server` command without `--rpc`. llama.cpp maps the file and the GB10 pages weights in from NVMe as experts are needed; expect low single-digit tokens/s. For a single Spark the resident choice is [GLM-5.3-Flash-GGUF-DGX-Spark](https://huggingface.co/autotrust/GLM-5.3-Flash-GGUF-DGX-Spark).

**Everywhere:**
- Thinking is on by default (the template opens `<think>`); `--reasoning-budget 0` disables it, `--chat-template-kwargs '{"reasoning_effort":"low"}'` selects the template's effort level.
- The server returns reasoning separately as `reasoning_content` and tool calls as OpenAI `tool_calls`.
- Sample with `--temp 0.7 --min-p 0.05` (or a reasoning budget) rather than pure greedy: at 2 bits the model occasionally loops in very long greedy chains of thought.
- **Memory:** 149.7 GiB weights + ~1.2 GiB per 16 K tokens of context (compressed MLA/DSA cache) + compute buffers. On a 180 GB GPU `-np 8 -c 131072` fits; on a 192 GB Mac keep the context small.

---

## Speed reference

llama.cpp, CUDA, one B200, model resident (`llama-bench` / `llama-batched-bench`, 256-token prompts, 128 generated tokens, flash attention):

| | tokens/s |
|---|---|
| Prompt processing (pp512 / pp2048) | 730–760 |
| Generation, 1 sequence | 42 |
| Generation, 4 / 8 / 16 / 32 parallel sequences (aggregate) | 96 / 120 / 154 / 177 |

Per-token work equals the full GLM 5.3 Q2 (8 routed experts per layer either way).

---

## Quality

**SLIM stage (pruned FP8 base vs. original GLM-5.3, author's A/B under vLLM):** HumanEval 95.1 → 95.1 · CyberMetric 88.0 → 87.7 · BFCL live 69.6 → 69.4 · BFCL multi-turn 73.0 → 72.5 · AIME −2.5 pt · GPQA-Diamond 86.9 → 83.3 (−3.6) · C-Eval 92.0 → 84.7 (−7.3, the deliberate trade-off). Held-out perplexity vs. the original: code +1.0 %, English chat +4.7 %, Chinese +7.3 %.

**Q stage (this 2-bit file)** — DwarfStar's ds4-eval harness on the same weights (thinking on, 16 000-token budget, greedy), one B200:

| Set | Passed | Wrong | Out of budget |
|---|---|---|---|
| GPQA Diamond (25) | 12 | 0 | 13 |
| SuperGPQA (25) | 17 | 4 | 4 |
| AIME 2025 (25) | 14 | 1 | 10 |
| COMPSEC (17) | 15 | 2 | 0 |
| **Core total (92)** | **58** | **7** | **27** |

Read it as: when it answers, it is almost always right (7 wrong in 92), but at 2 bits long reasoning chains often do not close within the budget — which is why non-greedy sampling or a reasoning budget is recommended above.

**Versus Flash:** on the first 40 of these cases, the smaller GLM-5.3-Flash-GGUF-DGX-Spark scored 35/40 (3 wrong, 2 out of budget) against 30/40 here (1 wrong, 9 out of budget). Where this model keeps a clear edge over Flash is language modelling of agent/SWE trajectories and code (held-out PPL 6.2 vs 12.4 on agent traces, 3.1 vs 4.6 on SWE traces, 2.96 vs 3.17 on code); Flash is better on general and Chinese text.

**Companion builds:** GLM-5.3-SLIM-E160 (128 GiB, held-out PPL +3.6 % vs this file). A per-layer non-uniform pruning study at equal budget found no gain over uniform expert counts.

---

## What is in the file

| Role | Tensors | Type | Bytes |
|---|---|---|---|
| Routed experts gate/up/down (75 layers × 192 experts) | 225 | IQ2_XXS | 140.1 GB |
| Attention (MLA q_a/q_b/kv_a/k_b/v_b/o, DSA indexer) | 78 layers | Q8_0 | 14.5 GB |
| Shared experts gate/up/down | 75 layers | Q8_0 | 3.0 GB |
| Dense FFN (layers 0–2) | 3 layers | Q8_0 | 0.7 GB |
| Token embedding, output head | 2 | Q8_0 | 2.0 GB |
| Norms, routers, bias, indexer weights_proj | — | F32 | 0.4 GB |

78 layers (3 dense + 75 MoE) · 64 MLA heads · DSA indexer (top-2048) · top-8 of 192 routed experts + 1 shared · 1,048,576-position metadata · GGUF v3 · 1,782 tensors.

Routed experts in IQ2_XXS (2.06 bits/weight), everything else in Q8_0 — the recipe of the full GLM 5.3 Q2, so per-token compute and memory traffic are unchanged; only the footprint drops. Standard llama.cpp glm-dsa layout (same tensor names, MLA `attn_k_b`/`attn_v_b` split and metadata as llama.cpp's own GLM 5.2/5.3 conversions). No MTP head.

### Shards

| File | Bytes |
|---|---|
| GLM-5.3-Q2-DGX-Spark-00001-of-00004.gguf | 44,477,268,576 |
| GLM-5.3-Q2-DGX-Spark-00002-of-00004.gguf | 44,708,574,752 |
| GLM-5.3-Q2-DGX-Spark-00003-of-00004.gguf | 44,708,574,752 |
| GLM-5.3-Q2-DGX-Spark-00004-of-00004.gguf | 26,865,884,128 |
| GLM-5.3-Q2-DGX-Spark.sha256 | checksums of the four shards |

Total 160,760,302,208 bytes (149.7 GiB); build name `GLM-5.3-SLIM-E192-IQ2_XXS`. Split with `llama-gguf-split --split-max-size 45G`; merge back with `llama-gguf-split --merge` if a single file is wanted. On a two-Spark RPC setup every machine needs all four shards on local storage.

---

## Limitations

- **2-bit routed experts.** A measurable drop versus the FP8 checkpoint on knowledge-heavy and Chinese-exam tasks; coding, agent, and cybersecurity use are the intended workloads.
- Long greedy reasoning can fail to converge; use sampling or `--reasoning-budget`.
- One 128 GB machine cannot hold it resident — two Sparks (RPC), a 180 GB GPU, or a 192–256 GB Mac. For a single Spark use GLM-5.3-Flash-GGUF-DGX-Spark.
- No MTP head, no imatrix.
- `general.source.revision` in the metadata carries the quantizer's default (the official GLM-5.3 revision); the SLIM checkpoint revision used is `e45b62eb3f5a22232f1e4980da255266ab933f31`.

---

## License and credits

- **Weights:** GLM-5.3 License (Z.AI), including the Model-as-a-Service clause; derivative of `zai-org/GLM-5.3` via `autotrust/GLM-5.3-SLIM-E192`.
- **SLIM-Q pipeline and expert pruning:** AutoTrust AI (`GLM-5.3-SLIM-E192`) — the compression recipe behind Guru Turbo 2.0.
- **IQ2_XXS quantization recipe and quantizer:** DwarfStar (antirez/ds4), on llama.cpp / GGML.
- **Tooling and this build:** yuhai-china — https://github.com/yuhai-china/ds4-glm-slim
