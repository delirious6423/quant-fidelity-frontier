---
language:
  - es
  - en
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
datasets:
  - HuggingFaceH4/ultrachat_200k
tags:
  - sysadmin
  - opencode
  - linux
  - vllm
  - qwen3.8
  - w4a16
  - int4
  - gptq
  - compressed-tensors
  - mtp
  - speculative-decoding
  - long-context
  - rtx3090
  - ampere
---

# Qwen3.8-27B — GPTQ W4A16 + INT4 embeddings, for 24 GB GPUs (MTP included)

> **Install from this card, not from generic snippets.** This checkpoint keeps its token embeddings quantized, which stock vLLM cannot load without a small patch. Follow [Requirements & install](#requirements--install-vllm-with-a-small-patch) and [Quick start](#quick-start-single-rtx-3090-24-gb); generic vLLM, SGLang or Transformers commands will fail.

A 4-bit quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) built to run very long contexts on
**a single 24 GB Ampere GPU (RTX 3090)** with vLLM, including the official **MTP head for speculative decoding**.
The same recipe was also applied to two fine-tunes of this model,
[JRamirez-UAB/Qwopus3.8-27B-Flash-V2-GPTQ-W4A16-embed-int4-24GB](https://huggingface.co/JRamirez-UAB/Qwopus3.8-27B-Flash-V2-GPTQ-W4A16-embed-int4-24GB)
and [JRamirez-UAB/Swift-1.5-Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB](https://huggingface.co/JRamirez-UAB/Swift-1.5-Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB);
where noted, the numbers below were measured on this checkpoint only. It follows the layout popularised by Marcin116: besides the linear layers,
the **token embeddings and `lm_head` are also INT4**, which frees ≈3.5 GiB of VRAM for KV cache. It is the fastest member of the Qwen3.8 family
below on one GPU; for higher fidelity to the BF16 model see the EXL3 sibling.

> **Tested setup:** RTX 3090 (24 GB, Ampere, power limit 350 W), vLLM 0.31.0 + the 4-line embedding patch in `patches/`, weights downloaded from this repo, the Quick start configuration (2026-10-14). Only this setup has been tested.

**Highlights (one RTX 3090 24 GB, measured on 2026-10-14; details in *Family comparison*):**
- **KLD 0.0400 against BF16** on our evaluation set (0.0312 on turboderp's public trace).
- **222,000 (text) · 176,128 with vision** on one 24 GB GPU with MTP (2.95 tokens per step); needle 8/8 at 204,551.
- **Tool calls with the real opencode request 56/60**, opencode agent tasks **28/30** (10 sysadmin tasks × 3 runs), images **15/15**.
- **Reasoning 47/48 at `xhigh`** with 32,190 reasoning tokens in total.
- Decode 104.0 / 182.4 / 192.4 tok/s with 1 / 2 / 8 requests; 282 MiB of VRAM really free at peak.

> **Our focus.** We quantize and evaluate models for one use case: a **local Linux system-administration agent** (opencode with tool calling)
> for a university IT department, **used in Spanish**. Our calibration traces include real sysadmin/agent conversations in Spanish, and our quality
> checks (tool calls with the real opencode request, agent tasks with real command execution, long-context retrieval, long answers in Spanish) reflect
> that use. KL divergence is also reported on turboderp's public trace, so it can be compared and reproduced. Numbers for other domains or languages may differ.

## What is in this repository

### Which one to use

All Qwen3.8-27B variants we publish, measured on 2026-10-14 under the same conditions (see *Family comparison* for every number):

| Variant | Weights | Context on one RTX 3090 | KLD vs BF16 | Agent tasks | Decode, 1 request | Pick it for |
|---|---|---|---|---|---|---|
| **[GPTQ INT4](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB/tree/mtp-int8) (this repo)** | 14.21 GiB | 222,000 (text) · 176,128 with vision | 0.0400 | 28/30 | 104.0 tok/s | the longest context on stock vLLM (+ a 4-line patch): 222,000 text-only, or 176,128 with vision |
| [GPTQ INT8 g64](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB) | 15.75 GiB | 176,000 (text) · 131,072 with vision | 0.0175 | 29/30 | 101.4 tok/s | a more faithful GPTQ (KLD 0.0175) on stock vLLM (+ the same patch), with less context |
| [EXL3 3.85](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB) | 13.62 GiB | 215,000 with vision | 0.0073 | 29/30 | 87.9 tok/s | the closest to BF16 (KLD 0.0073) with vision at 215,000; needs exllamav3 + our plugin |
| [EXL3 3.70](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB/tree/3.70bpw) | 13.19 GiB | 226,000 with vision | 0.0089 | 29/30 | 79.3 tok/s | more context (226,000) and 4 sessions with vision, a little less faithful than 3.85 |

### Variants / revisions

| Revision | MTP head | Size | Notes |
|---|---|---|---|
| `main` | BF16 (0.79 GiB) | 14.6 GiB | reference; use without speculation for the longest context |
| `mtp-int8` | INT8 g128 (0.40 GiB) | 14.21 GiB | **recommended with MTP**: frees ≈0.4 GiB for KV cache |

Sibling repositories of the same family: EXL3 3.85 bpw (+ branch `3.70bpw`), see the table above.

### Files and reproducibility

- `scripts/quantize.py`: exact quantization script (llm-compressor); `recipe.yaml`: recipe as serialized by llm-compressor.
- `scripts/mtp_quant.py`: script that builds the `mtp-int8` revision from `main`.
- `patches/`: vLLM patch and its Apache-2.0 license. `SHA256SUMS`: checksums of the model files.

## Results
### Family comparison (Qwen3.8-27B, same conditions)

Measured on 2026-10-14 for every variant in the same way: weights downloaded from Hugging Face, one RTX 3090 24 GB (power limit 350 W, no other use), vLLM 0.31.0 (GPTQ: + the embedding patch / EXL3: + exllamav3 1.5.1 + plugins), each variant with its own Quick start configuration. KLD and perplexity come from exllamav3's `qbench` (see *How to read it*).

| Variant | Weights | KLD mean / median (internal · public) | Perplexity (BF16 1.5860 · pub. 1.3332) | Needle (8 facts) near max. context | Tools: real opencode request | Agent tasks (10 × 3) | Reasoning `xhigh`: correct · tokens | Vision |
|---|---|---|---|---|---|---|---|---|
| **[GPTQ W4A16 + INT4 embeddings](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB/tree/mtp-int8) (`mtp-int8`, this repo)** | **14.21 GiB** | **0.0400 / 0.0127 · pub. 0.0312 / 0.0017** | **1.6631 · pub. 1.3779** | **8/8 at 204,551** | **56/60** | **28/30** | **47/48 · 32,190** | **15/15** |
| [GPTQ W4A16 g64 + INT8 embeddings](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB) (`main`) | 15.75 GiB | 0.0175 / 0.0049 · pub. 0.0171 / 0.0009 | 1.6203 · pub. 1.3568 | 8/8 at 162,208 | 59/60 | 29/30 | 48/48 · 20,414 | 15/15 |
| [EXL3 3.85 bpw](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB) (`main`) | 13.62 GiB | 0.0073 / 0.0018 · pub. 0.0067 / 0.0003 | 1.5976 · pub. 1.3430 | 8/8 at 197,991 | 60/60 | 29/30 | 48/48 · 22,474 | 15/15 |
| [EXL3 3.70 bpw](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB/tree/3.70bpw) (`3.70bpw`) | 13.19 GiB | 0.0089 / 0.0023 · pub. 0.0085 / 0.0004 | 1.5999 · pub. 1.3466 | 8/8 at 208,158 | 60/60 | 29/30 | 48/48 · 20,873 | 15/15 |

| Variant | Context on one RTX 3090 | Sessions | Prefill near max. context | Decode, 1 / 2 / 8 requests (tok/s) | MTP tokens per step | VRAM really free at peak |
|---|---|---|---|---|---|---|
| GPTQ INT4 | 222,000 (text) · 176,128 with vision | 2 | 605 tok/s | 104.0 / 182.4 / 192.4 | 2.95 | 282 MiB · vision config: 930 MiB |
| GPTQ INT8 g64 | 176,000 (text) · 131,072 with vision | 2 | 680 tok/s | 101.4 / 164.6 / 170.2 | 3.07 | 196 MiB · vision config: 840 MiB |
| EXL3 3.85 | 215,000 with vision | 2 | 556 tok/s | 87.9 / 137.4 / 149.5 | 3.03 | 300 MiB |
| EXL3 3.70 | 226,000 with vision | 4 | 545 tok/s | 79.3 / 142.0 / 264.4 | 3.03 | 148 MiB |

How to read it:
- **KLD / perplexity**: next-token KL divergence against the BF16 model with exllamav3's `qbench` (lower is better). *Internal*: our sysadmin/agent evaluation set (not published, separate from the calibration data); *public*: turboderp's `qbench_prompts_gen.json`, reproducible by anyone. GPTQ checkpoints were evaluated as a BF16 copy holding the dequantized weights. These two columns come from `qbench`, not from this campaign.
- **Needle**: 8 facts spread through one prompt at about 92 % of the served context; all 8 must come back. **Tools**: the real request of the opencode agent (system prompt + 11 tools), 3 cases × 20 tries, at the model's default sampling (`generation_config.json`); a try counts when it calls a valid tool with valid arguments and the expected command. **Agent**: 10 sysadmin tasks run with `opencode run` and real command execution, 3 runs, at the recommended temperature; with sampling on, the same checkpoint varies by about ±2 of 30 between runs, so other measurements in this card may differ by that much. **Reasoning**: 6 checkable problems × 8 at `xhigh`; total reasoning tokens over the 48. **Vision**: our 4-image test.
- **Prefill / decode**: prefill from the needle prompt; decode is aggregate throughput with 1 / 2 / 8 simultaneous requests (temperature 0; with more requests than *Sessions*, the extra ones wait). **MTP tokens per step**: vLLM's speculative-decoding metrics over the whole run.
- **VRAM really free**: lowest `memory.free` while serving (nvidia-smi every 0.5 s, from the end of start-up to the end of the run); we require ≥100 MiB.
- Older sections below keep their own date and setup; use this table to compare variants.

**How many sessions fit** with each Quick start configuration (KV size in tokens as reported by vLLM at start-up; a session uses the KV of its own context, and vLLM never runs more than *max sessions* at once):

| Variant | KV pool (tokens) | Max sessions | 32k each | 64k each | 128k each | Full context each |
|---|---|---|---|---|---|---|
| GPTQ INT4 | 223,441 | 2 | 2 | 2 | 1 | 1 × 222,000 |
| GPTQ INT4 (vision config) | 178,923 | 2 | 2 | 2 | 1 | 1 × 176,128 |
| GPTQ INT8 g64 | 178,816 | 2 | 2 | 2 | 1 | 1 × 176,000 |
| GPTQ INT8 g64 (vision config) | 132,423 | 2 | 2 | 2 | 1 | 1 × 131,072 |
| EXL3 3.85 | 216,433 | 2 | 2 | 2 | 1 | 1 × 215,000 |
| EXL3 3.70 | 231,757 | 4 | 4 | 3 | 1 | 1 × 226,000 |

**Few users or many users.** The Quick start configuration is tuned for 1–4 simultaneous users (fastest single request). For more users, the *many users* option below (measured the same day, same GPU) gives more total throughput; each request gets slower when many run at once. Same model, same quality: only the serving settings change.

| Variant | Configuration | Decode by simultaneous requests (aggregate tok/s) | N sessions at once, needle 8/8 | Tools (5 × 3) | VRAM really free |
|---|---|---|---|---|---|
| GPTQ INT4 | Quick start (2 sessions, MTP 3) | 1: 104 · 2: 182 · 8: 192 | — | 56/60 | 282 MiB |
| GPTQ INT4 | many users (8 sessions of up to 222,000 tokens, MTP 3) | 1: 105 · 2: 185 · 4: 341 · 8: 475 | 8/8 of 22k | 15/15 | 176 MiB |
| GPTQ INT8 g64 | Quick start (2 sessions, MTP 3) | 1: 101 · 2: 165 · 8: 170 | — | 59/60 | 196 MiB |
| GPTQ INT8 g64 | many users (8 sessions of up to 165,000 tokens, MTP 3) | 1: 94 · 2: 177 · 4: 316 · 8: 470 | 8/8 of 22k | 14/15 | 386 MiB |
| EXL3 3.85 | Quick start (2 sessions, MTP 3) | 1: 88 · 2: 137 · 8: 150 | — | 60/60 | 300 MiB |
| EXL3 3.85 | many users (8 sessions of up to 215,000 tokens, MTP 3) | 1: 84 · 2: 140 · 4: 252 · 8: 296 | 8/8 of 22k | 15/15 | 138 MiB |
| EXL3 3.70 | Quick start (4 sessions, MTP 3) | 1: 79 · 2: 142 · 8: 264 | — | 60/60 | 148 MiB |
| EXL3 3.70 | many users (8 sessions of up to 220,000 tokens, MTP 3) | 1: 82 · 2: 141 · 4: 253 · 8: 298 | 8/8 of 22k | 15/15 | 338 MiB |

<!-- end of the controlled comparison -->

### Benchmarks and validated configurations (single RTX 3090 24 GB, this checkpoint)

_Earlier measurements, before the controlled comparison of 2026-10-14: other dates, vLLM versions or settings, as stated in each table. Use *Family comparison* above to compare variants._

#### Speed and retrieval

Setup: vLLM 0.29.0 + patch, CUDA 13.0 wheels, driver 595 (CUDA 13.2), fp8 (e4m3) KV cache without calibrated scales, FlashInfer,
`--language-model-only`, `mtp-int8` revision, MTP 3 tokens, 2 sequences, prefix caching, 200k context (209,096-token KV pool).
Decode is measured with 800 forced output tokens (`ignore_eos`).

| Metric | Result |
|---|---|
| Decode, 1 request (t=0 / t=0.6) | **97.6 / 90.5 tok/s** |
| Decode, 2 concurrent (aggregate) | 172.6 tok/s |
| MTP accepted tokens per step (official head) | 2.80–3.05 (mean ≈2.93); per position ≈80-85 % / 59-68 % / 40-52 % |
| 8 facts at 3–97 % depth, 57,874-token prompt | **8/8**, 60.8 s (951 tok/s prefill) |
| Tool calling, 10 sysadmin tasks (`qwen3_coder`) | **10/10** called, right tool and valid JSON |
| Tool calling with the **real opencode agent request** (system prompt + 11 tools; fresh, after a long story, and a full multi-turn session; 5 tries each) | **15/15** (2026-10-02, 20 tries each: 60/60 at T=1.0, 60/60 at T=0.3) |
| Peak VRAM | 23,658 MiB of 24,576 (918 MiB free) |

**Quality sanity checks** (small in-house battery): arithmetic, a multi-step word problem, a Python function with asserts, multi-turn memory
and a tool call were all correct. A free-form historical summary (Spanish) was largely accurate; a factual question in Catalan about the
location of a local university campus was answered incorrectly.

#### Validated serving configurations (run as systemd services, 24 GB RTX 3090)

Each configuration below was started as a real service and checked with a near-maximum-length prompt. The VRAM peak barely moves with prompt length
(prefill runs in 2048-token chunks): with 222k, idle 24,050 MiB vs. 24,052 MiB at peak.

| Configuration | Context (KV pool) | Checks | Peak VRAM (free) |
|---|---|---|---|
| **Text, MTP 3 (INT8 head), prefix caching, 2 seqs, `qwen3_coder`** (used daily from opencode) | **222,000** (223,441) | 8/8 facts at 205,792 tokens (348 s, 591 tok/s prefill); decode 94.4 / 88.5 tok/s (t=0 / 0.6), 187 tok/s with 2 concurrent; MTP ≈2.9 accepted tokens per step; real opencode request 5/5 tool calls | 24,052 MiB (524 MiB) |
| Vision + MTP 3 (INT8 head), prefix caching, 2 seqs, `qwen3_coder` | 176,128 (178,923) | real opencode request 5/5 tool calls and a real `ls` executed from opencode; sysadmin tool calling 10/10; 3/3 test images (terminal screenshot, chart, dense nft ruleset); prefix caching with an image (2.1 s → 1.5 s); ≈1 MP image + 127k-token prompt correct | 23,710 MiB (866 MiB) |
| Text, MTP 3 (INT8 head), prefix caching, 2 seqs | 204,800 (209,096) | 8/8 facts at 58k; decode 97.6 tok/s | 23,658 MiB (918 MiB) |

(The 222k row is the configuration before the two memory settings; with them the peak is 23,760 MiB, 363 MiB really free, see *Important notes*.)

Note: with MTP enabled, KV capacity grows in block-sized steps. With 8.33–8.35 GB of KV cache vLLM accepted at most 212,800 tokens,
so that is the maximum `--max-model-len` for the text configuration (not 212,992).

Images with the new 4-image battery (2026-10-02, vision configuration: 176,128 context, KV 7.14e9 B, MTP 3, `--limit-mm-per-prompt '{"image":1,"video":0}'`):
**4/4 tasks, 18/18 checks** (failed-service screenshot 3/3, chart 3/3, dense firewall ruleset 3/3, port held by another process with a dead PID as a trap 6/6),
follow-up turn with prefix cache OK, 957 MiB really free at the end of the run (not a long-prompt peak), no engine errors.

**Sizing tips.** With MTP enabled vLLM needs the KV pool to exceed `--max-model-len` by ≈1.44k tokens (MTP look-ahead plus GatedDeltaNet
state), and capacity grows in block-sized steps, so after changing `--kv-cache-memory-bytes` always check the `GPU KV cache size` log line.
The KV pool is shared by all running requests and by the prefix cache: `--max-model-len` caps a single request, it does not reserve memory per session.
With agent clients that compact the history at `context − max_output` (e.g. opencode), set the output limit accordingly
(222,000 − 22,000 → compaction at ≈200k tokens of input).

#### Agent turns and 8 sessions (2026-10-02, 350 W, text)

Agent turn = the real opencode request (≈13.5k → 14.5k tokens over 10 turns) with prefix caching. 8 sessions = `--max-num-seqs 8`, the KV pool
shared by all (it does not multiply the context).

| Configuration | Turn: time to first token / total (median) | First turn (no cache) | 8 sessions: needle each, aggregate prefill | Decode with 8 sessions (t=0 / 0.6) | Really free (min) |
|---|---|---|---|---|---|
| 150,000, KV 6.25e9 | 2.7 / 3.7 s | 11.6 s | 8 × 16.5k, 8/8 in all, 1,129 tok/s | 459.9 / 466.9 tok/s | 2,569 MiB |
| 200,000, KV 8.15e9 | 2.7 / 3.6 s | 11.6 s | 8 × 22k, 8/8 in all, 1,106 tok/s | 472.2 / 463.9 tok/s | 751 MiB |

#### Quality: KL divergence against BF16 (2026-10-02)

Measured with exllamav3's `qbench` (see *Family comparison* for the method). Internal evaluation set: KLD 0.0400 (median 0.0127), perplexity
1.6631 vs 1.5860 for BF16. Public trace: 0.0312 (median 0.0018), perplexity 1.3779 vs 1.3332. Most of the gap is attributed to the INT4 RTN
embeddings / `lm_head` (on the Swift sibling an INT8-embedding variant lowered the KLD by ≈12–15 %).

#### Reasoning length

Problems with checkable answers (arithmetic, word problem, counting, logic, subnetting, systemd) × 8 tries, temperature 0.6, top_p 0.95,
reasoning tokens counted with `/tokenize`, totals over 48 requests (2026-09-30):

| Level | low | medium | xhigh |
|---|---|---|---|
| Reasoning tokens (mean / p90 / max at xhigh: 552 / 1,291 / 2,615) | 17,957 | 21,317 | 26,481 |
| Correct | 48/48 | 48/48 | 48/48 |

Repeat on 2026-10-02: 20,004 / 21,714 / 27,761, 48/48 at every level. Almost all of the `xhigh` excess comes from one problem (a systemd question:
mean 1,450 and max 2,615 reasoning tokens on 2026-09-30; on turboderp's 3.00 bpw EXL3 the same problem stayed below 1,000). Agent tasks with opencode (4 sysadmin tasks × 2, `xhigh`, temperature 0.3, our setting until 2026-10-13): 8/8, 4,817 reasoning tokens, 245 s.
Comparison with the fine-tunes: see the [Swift 1.5](https://huggingface.co/JRamirez-UAB/Swift-1.5-Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB) card.

#### Alternative: DFlash 2 draft (tested, not recommended on one 24 GB GPU)

Instead of the MTP head, this checkpoint also works with the DFlash 2 draft
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), quantized for 24 GB GPUs in
[JRamirez-UAB/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-DFlash2-W4A16) (INT4 in `main`, INT8 in `w8a16`).
Measured on one RTX 3090 (vLLM 0.29.0, 7 speculative tokens, fp8 KV cache, 2 sequences):

| Speculation | Context (KV pool) | Accepted tokens per step (decode · reasoning · tools) | Decode, 1 request (t=0 / 0.6) | 2 requests |
|---|---|---|---|---|
| MTP 3 (INT8 head), this card's text configuration | 222,000 (223,441) | ≈2.9 | 94 / 89 tok/s | 187 tok/s |
| DFlash 2 W4A16 draft, 6.3 GB KV cache | 69,842 | 3.79 · 4.62 · 4.70 | 115 / 109 tok/s | 173 / 175 tok/s |
| DFlash 2 W8A16 draft, 5.5 GB KV cache | 60,854 | 4.00 · 4.36 · 3.90 | 113 / 137 tok/s | 170 / 177 tok/s |

The draft accepts more tokens per step, but on one GPU it gives only +10–25 % with one request and nothing with two, and its KV cache
(≈90 KB per token against ≈35 KB with MTP in vLLM 0.29) limits the context to ≈60–70k tokens. The draft is meant for a two-GPU setup, where there is room for a long context;
we do not have two GPUs, so that setup has not been tested. Details and commands on the draft's card.

## Sampling and repetition loops

Measured on 2026-10-14 with the same software and GPU as *Family comparison*, on our EXL3 3.85 build of Qwen3.8-27B (loops come from the model and the sampling, not from the quantization format): 5 long Spanish sysadmin prompts × 6 runs per temperature, up to 16,000 tokens, top_p 0.95 and top_k 20; a loop = more than 30 % repeated 8-grams. Serving: the Quick start with a 32k context and 8 sessions.

| Temperature | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| Loops / 30 | 12 | 12 | 7 | 8 | 1 | 1 | 1 | 0 |
| Answers that finished before 16,000 tokens / 30 | 6 | 4 | 5 | 5 | 4 | 4 | 2 | 0 |

From temperature **0.7** up, loops are rare (at most 1 of 30). This repo's default (`generation_config.json`) is **1.0**.

Other models, same test and setup (loops / 30):

| Model (build) | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-pi · EXL3 3.85 | 9 | 4 | 9 | 10 | 4 | 2 | 1 | 0 |
| Qwopus3.8-27B-Flash-V2 · EXL3 3.85 | 3 | 1 | 2 | 1 | 2 | 0 | 0 | 0 |
| Swift 1.5 (Qwen3.8-27B) · EXL3 3.85 | 7 | 5 | 3 | 4 | 3 | 1 | 1 | 0 |
| Ornith-1.5-9B · EXL3 6.00 (16 GB) | 6 | 2 | 3 | 1 | 1 | 0 | 0 | 0 |
| NeoHorse-1-9B · EXL3 6.00 (16 GB) | 6 | 3 | 3 | 1 | 0 | 0 | 0 | 0 |
| Qwen3.6-35B-A3B · EXL3 4.00 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| Ornith-1.5-35B-A3B · EXL3 4.00 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwopus3.6-35B-A3B-Coder · EXL3 4.00 | 9 | 13 | 12 | 6 | 11 | 6 | 5 | 4 |
| Swift 1.5 Qwen3.8-Flash-Next (125B-A6B) · EXL3 2.25 · vLLM | 24 | 24 | 21 | 21 | 17 | 8 | 0 | 0 |
| Qwen3.8-Flash-Next (125B-A6B) · EXL3 2.25 · vLLM | — | — | 13 | 7 | 5 | 2 | 0 | 0 |

## Requirements & install (vLLM with a small patch)

Stock vLLM (checked on 0.29.0 and 0.30.0) does not pass `quant_config` to the embedding layer of `qwen3_5` models, so a checkpoint with
quantized embeddings fails to load (`no parameter named 'embed_tokens.weight_packed'`). Apply the 4-line patch in
[`patches/qwen3_5-embed-quant.patch`](patches/qwen3_5-embed-quant.patch) (from [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090), Apache-2.0):

```bash
uv venv --python 3.12 venv && uv pip install --python venv/bin/python "vllm==0.31.0" --torch-backend=auto        # torch 2.13.0
VLLM_DIR=$(venv/bin/python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
patch -p1 -d "$VLLM_DIR" < patches/qwen3_5-embed-quant.patch
```

## Quick start (single RTX 3090, 24 GB)

**With MTP (`mtp-int8` revision), text only, 222k context, 2 concurrent sequences** — validated as a service on a GPU used only for the LLM:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB --revision mtp-int8 \
  --language-model-only \
  --max-model-len 222000 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-memory-bytes 8640000000 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

With the two memory settings this leaves 363 MiB of VRAM really free at peak (see *Important notes*). If the GPU also drives a display or runs anything else, use the 200k variant below:

**Same, 200k context (more VRAM headroom):**

```bash
vllm serve JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB --revision mtp-int8 \
  --language-model-only \
  --max-model-len 204800 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-memory-bytes 8160437862 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**With MTP and vision (`mtp-int8` revision), 172k context, 2 concurrent sequences, 1 image per request (≤1 MP):**

```bash
vllm serve JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB --revision mtp-int8 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1003520}' \
  --max-model-len 176128 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-memory-bytes 7140000000 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**8 concurrent sessions (text, shared 200k KV pool):** the 222k text command with `--max-model-len 200000 --kv-cache-memory-bytes 8150000000 --max-num-seqs 8`
(validated, 751 MiB really free at peak; see *Benchmarks*). With 8 sequences vLLM 0.29 runs out of memory while capturing CUDA graphs at ≥222k.

**Without MTP**, the same layout fits the full 262k context with `--max-model-len 262144 --kv-cache-memory-bytes 9101639680 --max-num-seqs 4`
(measured on the fine-tune quantized with the same recipe; **not** re-measured on this checkpoint).

**For many users** (measured on 2026-10-14, same GPU): the same command with `--max-num-seqs 8` — 8 sessions of up to 222,000 tokens, MTP 3. Aggregate decode 475 tok/s with 8 requests (Quick start: 192 with 8); a single request runs at 105 tok/s (104 with the Quick start); 8 of 8 simultaneous sessions returned all 8 needle facts; 176 MiB of VRAM really free. Rule for CUDA graphs: they must cover sessions × (1 + MTP) tokens per step.

## Using it with opencode (how we tested)

This checkpoint is meant to be used as a **coding / sysadmin agent from [opencode](https://opencode.ai)** (tested with opencode 1.18.33).
What was measured with opencode:
- **Tool calling with the real opencode request:** the actual request that opencode sends (its long system prompt and its 11 tools),
  captured and replayed against vLLM, in three cases (fresh request, after a long story, full multi-turn session), several tries each.
- **End-to-end runs with `opencode run`:** multi-step tasks (list files, count lines, read a file, write a summary) with real tool execution
  and streaming.

Context, speed and memory figures were measured directly against the vLLM API, not through opencode.

Start vLLM with `--served-model-name qwen3.8-27b` (the model key in opencode must match it), then add a provider to `~/.config/opencode/config.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local vLLM",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8-27B (local)",
          "tool_call": true,
          "temperature": true,
          "limit": { "context": 222000, "output": 22000 },
          "variants": {
            "none":   { "chat_template_kwargs": { "enable_thinking": false } },
            "low":    { "chat_template_kwargs": { "reasoning_effort": "low" } },
            "medium": { "chat_template_kwargs": { "reasoning_effort": "medium" } },
            "xhigh":  { "chat_template_kwargs": { "reasoning_effort": "xhigh" } }
          }
        }
      }
    }
  },
  "agent": {
    "build": { "model": "local-vllm/qwen3.8-27b", "variant": "xhigh" },
    "plan":  { "model": "local-vllm/qwen3.8-27b", "variant": "xhigh" }
  }
}
```

Parameters that matter:
- **`limit.context` = vLLM's `--max-model-len`** (222,000 in the text configuration above). opencode compacts the history at about
  `context − output`, so `output: 22000` makes it compact at ≈200k tokens of input and never exceed the server's limit.
- **Declare `"temperature": true` in the model.** Without it opencode does not send any temperature to a custom model, even if the agent sets one
  (checked by capturing the requests).
- **Reasoning levels:** the variants pass `chat_template_kwargs`; `reasoning_effort` accepts only `xhigh`, `medium` and `low` (`high` returns
  HTTP 400). They are prompt instructions and do not limit the amount of reasoning (see the notes above).
- **Tool parser:** `--tool-call-parser qwen3_coder` (Qwen3.8's XML tool-call format). In our tests we did not set a temperature for this
  model (vLLM then uses the model's `generation_config.json`, temperature 1.0) and tool calling was 15/15; in the 2026-10-02 test with 20 tries
  per case it was 60/60 both at T=1.0 and at T=0.3.
- **Start opencode inside a project directory, not in your home directory:** before each step opencode takes a git snapshot of the directory,
  and on a large home directory that takes minutes and the request is never sent.
- **For `opencode run` from a script, close stdin** (`opencode run "..." </dev/null`); otherwise it may wait for input and hang at start-up.

## Important notes

- **Real VRAM headroom (corrected 2026-09-30):** the driver reserves 453 MiB of the 24,576 MiB that vLLM cannot use, and the "MiB free"
  figures in this card were computed as 24,576 − used, so they over-estimate the headroom by 453 MiB. Check the real value with
  `nvidia-smi --query-gpu=memory.used,memory.free --format=csv`. The original 222k command left only ≈50–70 MiB really free, and the same
  configuration crashed once with out-of-memory on the Swift 1.5 checkpoint (same recipe and size) during an agent request. The 222k command
  above now adds `--max-num-batched-tokens 1024` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: measured on this checkpoint the peak
  drops to 23,760 MiB (**363 MiB really free**) with the same prefill (586 tok/s at 205,792 tokens, 8/8 facts) and decode speed, no OOM.
  (The "really free" values in the 8-session rows of *Benchmarks* were read directly with `nvidia-smi`.)
- **Size the KV cache with `--kv-cache-memory-bytes`, not `--gpu-memory-utilization`.** With this hybrid (GatedDeltaNet) model, vLLM 0.29's
  memory profiling under-estimates the prefill activations (it measured 0.4–0.7 GiB; the real peak is ≈2 GiB). With a percentage it
  over-allocates the KV cache and runs out of memory on the first long prompt. The values above leave ≈0.9 GB of VRAM free.
- Use `--tool-call-parser qwen3_coder` (Qwen3.8's XML tool-call format).
- On Ampere (sm86), fp8 KV cache only works with the **FlashInfer** attention backend (the default).
- `reasoning_effort` (via `chat_template_kwargs`) accepts only `xhigh` (default), `medium` and `low`; `enable_thinking: false` disables
  reasoning. The levels are prompt instructions, not token budgets, and in our measurements they do not regulate the amount of reasoning
  (6 checkable problems × 3 tries, temperature 0.6: 7,304 / 8,592 / 8,605 reasoning tokens in total for low / medium / xhigh, 17–18/18 correct).
  For a hard cap, send `thinking_token_budget` in the request: with `--reasoning-parser qwen3`, vLLM 0.29 enforces it (also with MTP)
  by forcing `</think>` when the budget is reached (e.g. 569 → 199 reasoning tokens with a budget of 200, answer still correct).
- Two fields of the `config.json` generated by llm-compressor are adjusted for vLLM (already applied): the `lm_head` target is
  `re:.*lm_head$`, and the MTP ignore rule is `re:^mtp\.(fc|layers\..*)$`, so that the MTP shares the quantized embeddings and `lm_head`.

## Quantization recipe

| Component | Treatment |
|---|---|
| Linear layers of the language model (attention, MLP, GatedDeltaNet `in_proj_qkv` / `in_proj_z` / `out_proj`) | **GPTQ W4A16**, symmetric INT4, group size 128, `actorder=static`, dampening 0.01 |
| `embed_tokens`, `lm_head` | INT4 symmetric, group size 128, RTN (min-max) |
| GatedDeltaNet `in_proj_a`, `in_proj_b`, norms, `A_log`, `dt_bias`, `conv1d` | kept in BF16 |
| MTP head (`mtp.*`) | BF16 in `main`; INT8 g128 RTN in `mtp-int8` (0.8 % mean relative error; shares the INT4 embeddings / `lm_head`) |
| Vision tower | kept in BF16 (validated with MTP and prefix caching, see below; use `--language-model-only` for text-only serving) |

- Tool: [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.14.0, compressed-tensors 0.19.0, transformers 5.17.0, torch 2.14.0+cu130.
- Calibration: 512 samples × 2048 tokens from `HuggingFaceH4/ultrachat_200k` (the first 512 of the `train_sft` split, shuffled with seed 42; chat template applied). Only the GPTQ linear layers use it: `embed_tokens` / `lm_head` (INT4) and the INT8 MTP head of the `mtp-int8` revision are round-to-nearest, with no data.
- Base revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; quantization took 26.9 min on a single RTX PRO 6000 Blackwell (96 GB).
- Tokenizer and chat template are byte-identical to the base model. Format: `compressed-tensors` `pack-quantized` (Marlin kernels).

## Integrity and validation

What was checked for this checkpoint, how, and where the numbers are (measured on 2026-10-14; the method and every figure are in *Family comparison*).

**Artifact integrity**
- Every weight file is checked against our local copy with SHA-256 before publication (`SHA256SUMS` in this repo).
- The measurements used the weights downloaded from this repo at commit `eb9615d0bd77`; later commits change only metadata, documentation or the plugin, never the `.safetensors` files.
- vLLM loads every tensor listed in `model.safetensors.index.json` on start-up (a missing tensor or a wrong shape would stop it): checked from a fresh download.

**Quantization**
- Quantization recipe and calibration in `recipe.yaml` and `scripts/` (llm-compressor GPTQ, `compressed-tensors` format).
- Fidelity: KL divergence against the BF16 Qwen3.8-27B with exllamav3's `qbench` on two traces (ours and turboderp's public one).

**Chat template**
- The published template renders correctly in 40 edge cases (empty and very long system prompt, no tools, two tool calls in one message, nested JSON and multi-line arguments, images with tools, two images, earlier reasoning, multi-turn with several tool results) × the reasoning settings (`enable_thinking: false`, `reasoning_effort` low / xhigh): all pass (`prueba_plantillas.py`). With Transformers directly, pass tool-call arguments as dicts (vLLM converts JSON strings itself).

**Runtime**
- Load, long-context needle, vision, tool calls with the real opencode request, 10 agent tasks × 3 runs, concurrency and VRAM: see *Family comparison*; engine errors during the run: 0.
- On a desktop RTX 3090 that also drives the display (300 W): Quick start at 222,000 tokens: 299 MiB really free; vision config at 176,128 tokens: 921 MiB really free.

**Scope**: results apply to the revision, software stack (vLLM 0.31.0 + the embedding patch), hardware and settings stated above; other setups are not tested.


## Limitations

- **Review the commands before they run.** As an agent, the model proposes shell commands and opencode can execute them: review them before running anything as root or with `sudo`, and give the agent a non-privileged account and a test machine where you can.
- **Quality has not been evaluated with standard benchmarks**: only the checks above (KL divergence against BF16, retrieval, tool calling,
  reasoning problems, agent tasks) were run. Calibration used general chat data (ultrachat).
- Among our Qwen3.8 quantizations this is the least faithful to BF16 (KLD 0.040 vs 0.007 for EXL3 3.85), mainly because of the INT4 embeddings / `lm_head`.
- Not measured on this checkpoint: 262k context without MTP and prefix-caching turn latencies in long text sessions. Vision, MTP and tool calling were validated (see above).
- The fp8 KV cache uses the default scale 1.0 (no calibrated KV scales).
- All numbers come from one machine and one software stack.

## Feedback welcome

I would be grateful for reports of installations of this checkpoint, **both working and failed ones**. Please open a discussion in the
[Community](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB/discussions) tab with your GPU(s), driver, vLLM version, the full command line and the result (speed, context, or the
error message). Failed attempts are just as useful as successful ones: they help to document which setups work.

## Credits & license

### Credits & acknowledgements

This work is directly inspired by **[Marcin116](https://huggingface.co/Marcin116)** and their checkpoint
[Qwen3.8-27B-W4A16-AutoRound-fast-embed-int4](https://huggingface.co/Marcin116/Qwen3.8-27B-W4A16-AutoRound-fast-embed-int4),
which showed that Qwen3.8-27B can serve very long contexts on RTX 3090-class GPUs by quantizing the token embeddings to INT4.
We served and measured that checkpoint on our RTX 3090 before building our own.

What we took from that line of work: the **INT4 `embed_tokens` / `lm_head` layout** (group 128, `compressed-tensors` pack-quantized),
keeping the **GatedDeltaNet `in_proj_a` / `in_proj_b` projections in BF16**, and the **vLLM patch** for quantized embeddings in `qwen3_5`.
Marcin116's checkpoint builds on [born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast](https://huggingface.co/born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast)
and [syvai/qwen3.8-27b-3090-fast-variant](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant)
(patch: [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)), developed with the Club3090 community.
Many thanks to all of them. Any mistakes in this checkpoint are ours.

What is different here: GPTQ (llm-compressor) instead of AutoRound for the linear layers, and the MTP head kept in BF16 or INT8 instead of INT4.

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Qwen team).
- INT4-embedding layout: [Marcin116](https://huggingface.co/Marcin116) / [born2bewild](https://huggingface.co/born2bewild); vLLM patch: [syv-ai](https://github.com/syv-ai/qwen38-27b-rtx3090).
- Tools: [llm-compressor](https://github.com/vllm-project/llm-compressor), [compressed-tensors](https://github.com/neuralmagic/compressed-tensors), [vLLM](https://github.com/vllm-project/vllm).
- KL-divergence evaluation: `qbench` from [exllamav3](https://github.com/turboderp-org/exllamav3) (turboderp) and its public evaluation trace;
  reference rows in the family table: [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3).

### License

Apache-2.0, same as the base model. The included patch is Apache-2.0 (syv-ai). This is a community quantization, not an official Qwen release.
