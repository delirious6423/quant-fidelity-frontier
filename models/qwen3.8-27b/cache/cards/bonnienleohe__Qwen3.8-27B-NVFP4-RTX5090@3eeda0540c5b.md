---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pretty_name: Qwen3.8-27B-NVFP4-RTX5090
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - modelopt
  - nvidia
  - rtx-5090
  - blackwell
  - quantization
  - qwen3.8
  - vllm
---

# Qwen3.8-27B-NVFP4-RTX5090

**GeForce RTX 5090–specific** NVFP4 checkpoint of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

Built so a **32 GB RTX 5090** can serve the **full native 262,144-token context** and decode **faster** than [Unsloth’s NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) on the same GPU, **with accuracy almost maintained**.

> ### 🚀 Now 2.04× faster with the matching DSpark drafter
> A speculative drafter trained **and** NVFP4-quantized specifically against this checkpoint now ships:
> [**`Qwen3.8-27B-DSpark-NVFP4`**](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4) — **88.5 → 180.3 tok/s**, 1.41 GB, outputs unchanged.
> It beats the built-in MTP head by 13.8% while using a quarter of its memory. [See the numbers](#speculative-decoding-191-faster-on-the-same-gpu).

<p align="center">
  <img src="assets/rtx5090-overview.png" alt="One RTX 5090: decode 42.4 (Unsloth) vs 88.5 (this checkpoint) vs 180.3 tok/s (+DSpark v2); context 262K served vs 77K cap; accuracy parity 75% vs 75%" width="100%"/>
</p>

Blackwell tensor cores only. Hopper can load the files but cannot run NVFP4.

| Item | Value |
| --- | --- |
| Base | `Qwen/Qwen3.8-27B` BF16 (~53 GB) |
| Quant | NVIDIA ModelOpt NVFP4 W4A4, group size 16, FP8 KV |
| Export | ~18.8 GB (3 shards), **~17.1 GB** in VRAM |
| Target | **GeForce RTX 5090 32 GB** |
| Context | **262,144 tokens native — fits on 5090** (vLLM util 0.97) |
| Decode | 88.5 tok/s alone · **180.3 tok/s with [DSpark-NVFP4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4)** |
| Accuracy | **Almost held vs Unsloth NVFP4** (20-item smoke: **75% vs 75%** overall) |
| License | Apache 2.0 |

---

## Why this checkpoint on RTX 5090

| | **This checkpoint** (ModelOpt) | Unsloth NVFP4 |
| --- | --- | --- |
| Weights in VRAM | **17.1 GB** | 22.7 GB |
| FP8 KV on 32 GB | **275,941 tokens** | 77,184 tokens |
| Full **256k** context | **yes** (1.05× at 262,144) | **no** (~77k cap) |
| Decode conc 1 | **80.6 tok/s** | 42.4 tok/s |
| Decode + own drafter | **180.3 tok/s** | not available |
| Decode @ ~61k | **74.3 tok/s** | 40.0 tok/s |
| TTFT @ ~62k | **8.52 s** | 11.08 s |
| Longest completed prompt | **242,686** | ~62k |
| Tool calls | **5/5** | 2/5 |
| Accuracy (n=20 / task) | **75% overall** | 75% overall |

Same GPU (RTX 5090 32 GB). Speed: `temperature=0`, thinking off. Accuracy: `temperature=1.0`, thinking on, same 20 items per task. This checkpoint: vLLM **0.27.1**, util **0.97**. Unsloth speed numbers: SGLang 0.5.18 (NEXTN/MTP does not fit usefully on 32 GB). Unsloth accuracy: same vLLM 0.27.1 stack.

Unsloth is a different recipe (`compressed-tensors` mixed NVFP4/FP8, larger weights, MTP draft). On 32 GB that extra size eats the KV pool, so it cannot hold native 256k and decode is about **half** of this ModelOpt export.

---

## Accuracy vs Unsloth NVFP4

Quality is **almost maintained** on a matched 20-question smoke of GPQA Diamond, AIME 2025, and MMLU-Pro. **Overall 45/60 (75%) for both checkpoints.** MMLU-Pro tied; GPQA and AIME split by one item each.

| Task | This checkpoint | Unsloth NVFP4 | Δ |
| --- | --- | --- | --- |
| GPQA Diamond | **13/20 (65%)** | 14/20 (70%) | −1 |
| AIME 2025 | **15/20 (75%)** | 14/20 (70%) | +1 |
| MMLU-Pro | **17/20 (85%)** | 17/20 (85%) | 0 |
| **Overall** | **45/60 (75%)** | **45/60 (75%)** | **0** |

**Method (same for both).** RTX 5090, vLLM **0.27.1**, thinking **on**, `temperature=1.0`, Qwen sampling (`top_p=0.95`, `top_k=20`), `--max-model-len 65536`, concurrency 4. Same 20 random items per task (`sample-seed=20260815`, AIME 1 sample each). Datasets: OpenAI simple-evals GPQA Diamond CSV (HF GPQA is gated), `MathArena/aime_2025`, `TIGER-Lab/MMLU-Pro` 5-shot same-category.

**Caveats.** This is a **20-item smoke**, not a full-test ranking. Several misses on both sides were **length truncations** (thinking filled the 24k/32k generation cap; extract `None`), not clean wrong answers. Do not treat these as published GPQA / AIME / MMLU-Pro scores.

**Next weights.** Calibration and export are still being tightened. The next drop of this repo is intended to be **more accurate and faster** on the same RTX 5090 32 GB envelope.

---

## Serve on RTX 5090

```bash
vllm serve gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.97 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

vLLM **0.27.x**. First boot JITs FlashInfer SM120 FP4 GEMM (`nvcc` + CUDA 13 headers). Limit parallel JIT on smaller host RAM (`MAX_JOBS=2`).

Util **0.97** is required for native 256k on 32 GB (0.90 only holds ~205k KV). Qwen3.8 is a thinking model; for short answers pass `"chat_template_kwargs": {"enable_thinking": false}` — see [Chat template](#chat-template) for the full set of knobs.

---

## Chat template

Qwen3.8's own template, with agentic fixes. It renders **byte-identical to upstream `Qwen/Qwen3.8-27B` on every non-tool path**, so ordinary chat, thinking and vision behave exactly as the base model does.

### Reasoning effort

`xhigh` (default), `medium`, `low` — invalid values raise:

```json
{"chat_template_kwargs": {"reasoning_effort": "medium"}}
```

On this checkpoint **`xhigh` is the cheapest setting at equal accuracy.** It reasons more directly rather than simply longer:

| Effort | Correct | Avg output tokens |
| --- | --- | --- |
| **xhigh** (default) | 12/12 | **245** |
| medium | 12/12 | 584 |
| low | 12/12 | 579 |

12 verifiable problems, `temperature=0`, 3k budget, no truncations. A harder 10-problem set held the same ordering (562 vs 939 tokens). Keep the default unless you have a specific reason not to.

### Tool calling

XML by default, matching the `qwen3_xml` (vLLM) and `qwen3_coder` (SGLang) parsers:

```text
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

Assistant `tool_calls` replayed from an OpenAI-style response now render whether `function.arguments` is a **dict or a JSON string**. The string form previously raised `TypeError: Can only get item pairs from a mapping` on the second turn of an agentic loop, which broke direct `apply_chat_template`, llama.cpp and LM Studio users. Server-side tool calling was never affected — SGLang and vLLM normalize arguments to a dict before rendering.

### Template kwargs

| Kwarg | Default | Effect |
| --- | --- | --- |
| `reasoning_effort` | `xhigh` | `xhigh` / `medium` / `low` |
| `enable_thinking` | `true` | `false` emits a closed, empty think block |
| `preserve_thinking` | `true` | keeps reasoning in history; `false` is prefix-cache safe |
| `tool_call_format` | `xml` | `json` switches to `{"name": ..., "arguments": {...}}` |
| `continue_final_message` | `false` | prefills the final assistant turn; takes precedence over `add_generation_prompt` |
| `auto_disable_thinking_with_tools` | `false` | turns thinking off when tools are present |
| `max_tool_arg_chars` / `max_tool_response_chars` | `0` | opt-in truncation, off by default |

`system` and `developer` roles are both accepted, and consecutive tool responses are grouped into a single turn.

**Validation on this checkpoint.** Agent correctness 8/8 (tool selection, argument accuracy, parallel calls, no-tool restraint, synthesis, multi-step, failure recovery, tools with thinking off); 240 held-out conversations render clean across four configurations; multi-turn prompts are token-level prefix-stable, so KV cache reuse holds across turns; and a paired A/B against the previous template left DSpark acceptance unchanged to slightly better.

Template contributions from [@TheChola](https://huggingface.co/TheChola).

---

## RTX 5090 bench detail

vLLM 0.27.1 (V1), FlashInfer SM120 NVFP4 GEMM, torch 2.13.0+cu130, CUDA 13.0, driver 580.173.02. Streaming `/v1/chat/completions`. Decode: `ignore_eos`, OSL 256, `--max-num-seqs 16`.

### Decode

| Concurrency | Per-request | Aggregate | TTFT p50 |
| --- | --- | --- | --- |
| **1** | **80.6 tok/s** | 78 tok/s | 126 ms |
| 2 | 73.1 tok/s | 137 tok/s | 220 ms |
| 4 | 71.0 tok/s | 267 tok/s | 227 ms |
| 8 | 73.5 tok/s | 552 tok/s | 233 ms |
| **16** | 68.2 tok/s | **1,029 tok/s** | 231 ms |

At ~61k input / 128 output: **74.3 tok/s** (TTFT 8.36 s).

### Prefill (full 256k window)

| Input tokens | TTFT | Result |
| --- | --- | --- |
| 508 | 106 ms | OK |
| 1,967 | 134 ms | OK |
| 61,775 | **8.52 s** | OK |
| 121,625 | 26.3 s | OK |
| 182,425 | 53.8 s | OK |
| 237,525 | 87.1 s | OK |
| **242,686** | 90.3 s | OK · longest prompt |

Four concurrent ~62k prompts: **4/4 OK** (~247k tokens vs a 276k KV pool).

### Tool calling

| Test | Result | Latency |
| --- | --- | --- |
| `tool_choice=auto` get_weather | `{"city": "Paris"}` | 0.42 s |
| `tool_choice=required` | get_weather (New York) | 1.24 s |
| Parallel two cities | Tokyo + London | 0.77 s |
| Named `add_numbers` | `{"a": 19, "b": 23}` | 0.58 s |
| Tool-result follow-up | Berlin 18C cloudy | 0.30 s |

---

## Speculative decoding: 2.04× faster on the same GPU

A DSpark drafter **trained and quantized specifically against this NVFP4 checkpoint** now ships alongside it:

- 🚀 [**`Qwen3.8-27B-DSpark-NVFP4`**](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4) — recommended, 1.41 GB
- [`Qwen3.8-27B-NVFP4-RTX5090-DSpark`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-DSpark) — BF16 source, 2.72 GB

**88.5 → 180.3 tok/s on a single RTX 5090, with byte-identical outputs.**

<img src="assets/rtx5090-speculation.png" alt="Speculative decoding: 88.5 to 180.3 tok/s across six profiles; acceptance per workload, agentic tool calling +17.9%" width="100%"/>

### Measured on 240 held-out prompts, identical across every profile

| Profile | Decode | Accept length | Drafter | vs no-spec |
| --- | --- | --- | --- | --- |
| No speculation | 88.45 tok/s | — | — | 1.00× |
| Built-in MTP head | 136.90 tok/s | 2.758 | 5.53 GB | 1.68× |
| Stock [RadixArk DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) (FP8-trained) | 139.35 tok/s | 2.421 | 2.72 GB | 1.71× |
| Ours — DSpark BF16 | 141.99 tok/s | 2.717 | 2.72 GB | 1.74× |
| Ours — DSpark NVFP4 (v1) | 150.74 tok/s | 2.761 | 1.41 GB | 1.85× |
| **Ours — DSpark NVFP4 (v2)** | **180.30 tok/s** | **2.904** | **1.41 GB** | **2.04×** |

**+31.7% throughput over the built-in MTP head while using a quarter of its memory**, and +29.3% over the stock DSpark drafter it was adapted from.

Two changes compound here: the v2 drafter (retrained on serving-format data) and an **NVFP4 `lm_head`** on the target. The no-speculation and v2 rows are measured on the current build; the MTP, stock-DSpark, BF16 and v1 rows come from earlier sessions on the previous build, where `lm_head` was BF16.

The v2 drafter is retrained on a corpus rendered exactly as the model is served — XML tool calls with a real tools array, think blocks and the reasoning-effort preamble present — closing a train/serve mismatch that had cost the most on agentic traffic. Same size, same quantization recipe. The two current rows were measured cold in one session; older rows come from earlier sessions, where run-to-run variance on this harness is roughly 2%.

### Per domain

Acceptance below is measured per request with the prefix cache flushed between requests, from cumulative verify counters rather than a sampled gauge.

| Domain | v1 drafter | **current** | Δ |
| --- | --- | --- | --- |
| Math | 3.810 | **4.496** | +18.0% |
| Coding | 3.729 | **3.864** | +3.6% |
| JSON / structured output | 3.328 | **3.543** | +6.5% |
| Chat | 2.305 | **2.363** | +2.5% |
| Long-context | 2.484 | 2.474 | −0.4% |
| Instruction | 2.195 | **2.329** | +6.1% |
| **Overall** | **2.761** | **2.904** | **+5.2%** |

Structured output sustains longer accepted blocks than open-ended prose: predictable token streams are easier to draft. Long-context is the one regression — the training corpus caps sequences at 2,048 tokens, so it was never well represented.

**Agentic tool calling.** The "JSON / structured output" row is schema-constrained generation — those held-out prompts declare no tools and contain no `<tool_call>` blocks, so they do not measure agentic tool use. Measured separately on 60 purpose-built agentic scenarios across 10 tool schemas, this is where the v2 retrain paid off most:

| Agentic phase | v1 drafter | **current** | Δ |
| --- | --- | --- | --- |
| Parallel calls | 4.169 | **5.199** | +24.7% |
| Initial call | 3.683 | **4.692** | +27.4% |
| After a tool result | 3.524 | **3.910** | +11.0% |
| Second call in a loop | 3.155 | **3.787** | +20.0% |
| Error recovery | 3.028 | **3.630** | +19.9% |
| Two-step chain | 2.605 | **2.749** | +5.5% |
| **Overall** | **3.299** | **3.891** | **+17.9%** |

Every phase improved. Tool-call syntax is predictable, so emitting several calls at once accepts longest; multi-step chains accept least, since each new tool boundary is a fresh decision point. Tool-call emission rate is unchanged (38/60 for both), so this is an acceptance gain, not a change in behaviour.

### Why `lm_head` is NVFP4 here

At concurrency 1 this model is **weight-bandwidth bound, not compute bound**. The previous build streamed 18.80 GiB of weights per token at 81.6 tok/s — a **1.65 TB/s** read rate against the RTX 5090's 1.79 TB/s spec, about **92% of peak**. Decode reads the whole model once per token, so bytes removed convert almost linearly into tokens per second.

`lm_head` is a full-vocabulary (248,320 × 5,120) GEMM evaluated on **every** token — 2.54 GB of that read in BF16, 0.72 GB in NVFP4. It was the only large BF16 block left on the per-token critical path (embeddings are the same size but are a *gather*, ~10 KB/token, so quantizing them saves capacity, not speed).

| | previous build | **current** |
| --- | --- | --- |
| Checkpoint | 20.59 GB | **18.77 GB** |
| Weights in VRAM | 18.80 GB | **17.10 GB** |
| Decode, no speculation | 81.6 tok/s | **88.45 tok/s** |
| Decode + DSpark v2 | 155.8 tok/s | **180.3 tok/s** |

The bandwidth model predicted 89.7 tok/s; measured 88.45 — within 1.4%.

**Accuracy is unchanged.** On a 60-item smoke (same items, thinking on, `temperature=1.0`) the two builds scored 38/60 and 42/60. At n=20 per task with 6–9 truncations each, that 4-item gap is inside noise: the supported claim is *no degradation*, not an improvement. DSpark acceptance also held (2.886 → 2.904 held-out, 3.766 → 3.891 agentic), which matters because the drafter was trained against the previous build's logits.

The previous build is preserved on the [`pre-lmhead4`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090/tree/pre-lmhead4) branch.

### Why it is faster *and* more accurate than BF16

The NVFP4 drafter is not a compromise. Mixed precision — MLP and the attention output projection at 4 bits, Q/K/V and all norms/heads in BF16 — makes each drafted block cheaper without hurting the prediction, so more of the target's time converts into accepted tokens:

| Drafter | tok/s | Accept | VRAM |
| --- | --- | --- | --- |
| BF16 | 141.99 | 2.717 | 2.72 GB |
| **NVFP4 (MLP + O)** | **147.87** | **2.792** | **1.41 GB** |
| NVFP4 (+ QKV) | 150.73 | 2.546 | 1.17 GB |

Quantizing Q/K/V as well is faster still, but acceptance collapses — which is why the shipped recipe leaves them alone.

All three rows come from one session on the v1 drafter, so they are internally comparable; they are not directly comparable to the v2 figures above, which were measured later.

### Choosing a profile

| | **With the drafter** | **This checkpoint alone** |
| --- | --- | --- |
| Max context | ~116K | **262K** (verified: 252,849-token prompt served) |
| Decode | **180.3 tok/s** | 88.5 tok/s |
| Concurrency | 1 request | 1 request |

On one 32 GB card you can have the full 262K window **or** 2.04× decode. The drafter's weights plus a draft KV pool sized to the full target context claim roughly what the target itself needs for KV at 262K. Speculation also accelerates **decode only, never prefill** — cold TTFT at 250K is ~121 s either way.

**Output quality is unchanged.** The target verifies every drafted token under strict acceptance, so speculation changes speed, not outputs.

---

## Recipe (NVIDIA ModelOpt)

| Item | Value |
| --- | --- |
| Tool | [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) git `c4129b6` (`quant_method: modelopt`) |
| Weights / activations | NVFP4 W4A4, group size 16 |
| KV cache | FP8 (`fp8_cast` at PTQ; serve with `--kv-cache-dtype fp8`) |
| Calibration | 128 image-text samples (`--calib_with_images`) |

Left in **BF16** on purpose: vision tower, embeddings, MTP, and Gated-DeltaNet `conv1d` / `in_proj_a` / `in_proj_b`. MLP, `lm_head` and remaining Linear layers are NVFP4.

---

## License

Apache 2.0, same as the Qwen3.8-27B base model.
