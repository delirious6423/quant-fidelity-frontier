---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - litert
  - litert-lm
  - litertlm
  - on-device
  - edge
  - hybrid
  - gated-deltanet
  - qwen3_5
pipeline_tag: text-generation
library_name: litert-lm
base_model_relation: quantized
---

# Qwen3.8-27B — LiteRT-LM

**Which file:**
- ⭐ **`Qwen3.8-27B_wi4b32_32k.litertlm`** *(recommended for LiteRT-LM ≥ 0.20.0)* — latest full-composite export (added 2026-11-20) with blockwise-32 INT4 weights (`dynamic_wi4b32_afp32`), full GPU composites, 32K dynamic context window, externalized embedder and weights, and both thinking and tool-calling enabled. Smaller (15.17 GB) and significantly faster on GPU than earlier files, and also verified on CPU with `--cache disk`. Specifically, it requires LiteRT-LM >= 0.20.0 (`litert-lm-nightly 0.20.0.dev*`, a source build from `main`, or the upcoming `0.20.0` stable release).
- **`Qwen3.8-27B_mixed_int4.litertlm`** — compatibility file for LiteRT-LM releases `0.18.1` to `0.19.1` (and CPU); it also runs on LiteRT-LM `main`. 4K context, thinking off only.
- **`Qwen3.8-27B_mixed_int4_gpu_composite_channels_last.litertlm`** — earlier 2026-11-07 GPU-composite preview (4K context, thinking off only), superseded by `Qwen3.8-27B_wi4b32_32k.litertlm` on LiteRT-LM ≥ 0.20.0.

This repository is a conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) to the LiteRT-LM (`.litertlm`) format for Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime. All files hold the **text decoder only** for desktop-class machines (26.9B parameters; image/video input and the multi-token-prediction head are not exported).

Qwen introduces Qwen3.8 as "the most capable generation in the Qwen open-model family to date", built "on the architectural foundation of Qwen3.5", and Qwen3.8-27B as "a compact, deployment-friendly dense model: a native vision-language model that understands images and videos, with flexible thinking control". Qwen's card reports GPQA Diamond 89.2 and IFBench 79.5 for it (vendor-reported, not run here) and recommends sampling with temperature 1.0, top_p 0.95 and top_k 20 in the thinking mode and temperature 0.7, top_p 0.8, top_k 20 and presence_penalty 1.5 in the non-thinking mode.

The architecture is Qwen3.5's hybrid at 27B: 64 layers in 16 groups of three GatedDeltaNet linear-attention layers and one gated full-attention layer, each with a dense FFN; hidden size 5120, FFN size 17,408, 24 attention heads with 4 KV heads of 256 (rotary position on 64 of the 256 dimensions), linear attention with 16 query/key heads and 48 value heads of 128, a 248,320-token vocabulary with untied embeddings. Only the 16 attention layers keep a growing KV cache (up to 32,768 tokens dynamically in `Qwen3.8-27B_wi4b32_32k.litertlm`, or 4,096 tokens in the older files); the 48 linear-attention layers carry a constant-size conv and recurrent state.

| File | Recipe | Size | sha256 |
|---|---|---|---|
| ⭐ `Qwen3.8-27B_wi4b32_32k.litertlm` | **Block-32 INT4, full GPU composites, 32K dynamic context** *(LiteRT-LM ≥ 0.20.0)*: symmetric int4 blockwise-32 weights (`dynamic_wi4b32_afp32`), full GPU composites, externalized embedder & weights, 32,768 dynamic KV budget, thinking + tool calling enabled | 15.17 GB (15,169,077,248 bytes) | `94867fc29ccf3e1fe422ddede11f6c858ec063af181ce59251fc602c3e99ded6` |
| `Qwen3.8-27B_mixed_int4.litertlm` | **Mixed INT4** *(LiteRT-LM 0.18.1–0.19.1 & CPU)*: int4 blockwise-128 (min-max) on the linear layers, int8 on the lm_head and the embedding, the 96 GatedDeltaNet decay/gate projections (`in_proj_a` / `in_proj_b`) float; fp32 GPU activations declared; 4K context; thinking off only | 15.38 GB (15,379,595,264 bytes) | `c3a33a2c17b57d198849adda94298728375e4d30d5fa522b03e8a5ccb620032e` |
| `Qwen3.8-27B_mixed_int4_gpu_composite_channels_last.litertlm` | **Mixed INT4, GPU composites** *(2026-11-07 preview; superseded by `Qwen3.8-27B_wi4b32_32k.litertlm`)*: int4 blockwise-128 (min-max) on the linear layers, int8 on the lm_head, the embedding and the 48 fused GatedDeltaNet z/b/a projections (`in_proj_zba`); fp32 GPU activations declared; 4K context; thinking off only | 15.97 GB (15,974,703,760 bytes) | `62720411c566caf79b5e5fcc422d7f178418463a0bf3a33fbb9a96bcc3f23763` |

### Release notes

- **2026-11-20 (`Qwen3.8-27B_wi4b32_32k.litertlm`):** Added the full-composite 32K export built with `litert-torch`'s Converter V2 (`dynamic_wi4b32_afp32`, full GPU composites, externalized embedder and weights, `prefill_128` + `prefill_1024` + `decode`, 32K dynamic KV cache, and canonical `qwen3_5` metadata with thinking + tool calling enabled and `min_runtime_version: "0.20.0"`). On Apple M5 Max GPU (1024p/256d), it reaches **~940–970 tok/s prefill** and **~33–34 tok/s decode** on Metal (and **~293 / 29.7 tok/s** on WebGPU) from 1K up to 32K context. Verified on Apple Silicon GPU (single-turn, multi-turn, thinking, and 2.7K-token needle retrieval) and on Linux CPU with `litert-lm-nightly` (`0.20.0.dev20261112`, `--cache disk`).
- **2026-11-07 (`Qwen3.8-27B_mixed_int4_gpu_composite_channels_last.litertlm`):** Added an earlier GPU-composite export of the same checkpoint with `litert-torch` main at [d600176b](https://github.com/google-ai-edge/litert-torch/commit/d600176b2e29bcc900a7e193f054ab3d6c3c747e) (4K KV budget, thinking off, channels-last `odml.short_conv_step`). Answered the 8 questions under Correctness 8/8 on Mac M4 Max GPU with LiteRT-LM main `b1dcfbd0` and `litert-lm-nightly 0.20.0.dev20261106`.
- **2026-11-03 (`Qwen3.8-27B_mixed_int4.litertlm`):** Updated LLM metadata to carry LiteRT-LM v0.19.0's [`models/qwen3_5` tool-call configuration](https://github.com/google-ai-edge/LiteRT-LM/blob/v0.19.0/models/qwen3_5/LlmMetadataProto.pbtext) (`tool_call_config`, `supports_function_calling: true`). Checked with `litert-lm 0.19.0` on Mac CPU (3/3 tool prompts returned structured tool calls).

Unless explicitly labeled otherwise, the older community notes below ("this file") describe `Qwen3.8-27B_mixed_int4.litertlm`.

Time for a 1024-token prefill plus 256 decoded tokens at context 2048 for `Qwen3.8-27B_mixed_int4.litertlm`, computed from the rates in the Performance table below: **19.87 s on the Mac M4 Max GPU**, 92.74 s on the Mac M4 Max CPU (with the CPU weight cache, see Usage). On GSM8K (thinking off, 2,048 output tokens, table in Accuracy) it scored 97 on the Mac GPU against 96 for the bf16 model in transformers.

## Correctness

### `Qwen3.8-27B_wi4b32_32k.litertlm` (LiteRT-LM ≥ 0.20.0)

- **Apple Silicon GPU (Metal):** Passed single-turn, multi-turn, and thinking conversation checks, short factual QA with thinking enabled, and multi-chunk 2.7K-token needle retrieval (`--max-num-tokens 8192`) with zero runtime errors.
- **Linux CPU (`litert-lm-nightly 0.20.0.dev20261112`):** Loaded with dynamic context resizing (`32768 → 4096`) and `--cache disk`, answering `"What is 17 + 25? Answer briefly."` (`--thinking false --temperature 0`) with `42`.

### `Qwen3.8-27B_mixed_int4.litertlm` (LiteRT-LM 0.18.1–0.19.1)

Qwen3.8 thinks by default. `Qwen3.8-27B_mixed_int4.litertlm` (and the 2026-11-07 preview) runs with **thinking off only** (`supports_thinking` is false) because thinking on broke multi-turn conversations on LiteRT-LM 0.18.1's default settings; use `Qwen3.8-27B_wi4b32_32k.litertlm` on LiteRT-LM ≥ 0.20.0 if you want thinking mode enabled.

Unless a line names the file published on 2026-10-23, the checks and the benchmarks below ran on an earlier build of `Qwen3.8-27B_mixed_int4.litertlm` that still carried a thinking switch in its metadata; that file differs from it only in the LlmMetadata section (the template, the channel and supports_thinking), with the TFLite, tokenizer and executor sections byte for byte the same, and the published template renders the same thinking-off prompts as the earlier one, byte for byte, in nine kinds of conversation (216 of 216 renders). On that file itself, on the Mac GPU: the 8 questions below 8/8 with the same answers, and the three-turn conversation with thinking off ("Give me one word that means 'very big'." / "Use it in a short sentence." / "Now make that sentence a question.") answered "Enormous" / "The elephant was enormous." / "Was the elephant enormous?" with no chat marker in the text.

8-question check on the Mac M4 Max ([the questions](https://github.com/john-rocky/hf-to-litertlm/blob/main/scripts/verify_quality.py) with "Answer briefly." appended, `--temperature 0`, one `litert-lm run` process each, thinking unset): 8/8 on the CPU and 8/8 on the GPU, no degenerate answer (none empty, none a run of one repeated word or character) (42 / Tokyo / Cold / 7 / Merci / 56 / 0.9 / blue). With the runtime's default fp16 GPU activations instead of the declared fp32 ones, the same weights answered the first question with a run of `!` to the token cap (one run, 2026-10-22; the check stops at its first failure), which is why `Qwen3.8-27B_mixed_int4.litertlm` declares fp32 activations.

Step-by-step decode comparison against the bf16 model (transformers 5.14.1 `Qwen3_5ForCausalLM`, bf16 on the CPU; teacher-forced over the first 24 positions of a GSM8K question in the chat format, both models fed the same prompt tokens one at a time; this file's TFLite section through LiteRT's CompiledModel API on the CPU): the top token is the same at 23 of 24 positions and within the top 5 at all 24; the one miss is a near tie in the reference (its top two logits 0.25 apart). Mean KL divergence 0.0356, mean Pearson correlation of the logits 0.973 (lowest 0.935), largest logit difference 4.81.

A multi-digit multiplication check on the CPU (six questions, three fresh processes each, thinking off, greedy, 400-token cap) ran on the block-32 int4 file of the same export, which is not published here: 18 of 18 right, every run ending by itself with no repeated line. On this file the CPU checks are the 8 questions above and the step-by-step decode comparison.

The three-turn conversation with thinking off (Correctness, first paragraph) also passed on the earlier file: on the Mac GPU with the same three replies, which are the bf16 model's in transformers word for word, and on the Mac CPU (with the weight cache) with "Huge" / "The house was huge." / "Was the house huge?".

**q / k norm weights checked (2026-11-10).** In the bytes of `Qwen3.8-27B_mixed_int4_gpu_composite_channels_last.litertlm`, the q / k norm weights of the 16 attention layers equal 1 + those of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (the scale its RMSNorm applies), with the GPU composites' partial-rotary permutation applied once, as the composites expect: 32 of 32 in every one of its 7 signatures.

## Accuracy

GSM8K (measured on `Qwen3.8-27B_mixed_int4.litertlm`): test questions 0–99 with a step-by-step instruction and a final `#### <number>` line, the file's template (thinking off), greedy, at most 2,048 output tokens, a 4096-token KV budget, a fresh process per question with at most 3 at once (litert-lm 0.18.1 Python API, Mac GPU, on the earlier file with the same thinking-off prompts). The bf16 reference is transformers 5.14.1 on MPS with Qwen's template at `enable_thinking=False`, the same prompts, greedy, 2,048 new tokens. The answer is the number after the last `####`, else the reader script's next rules (`\boxed{}`, "answer is", the last number). IFBench and GPQA were not run; Qwen's own table reports them above.

| Configuration | GSM8K | wrong questions |
|---|---|---|
| bf16 reference (transformers, MPS) | 96 / 100 | 12, 62, 85, 93 |
| Mixed INT4, litert-lm Mac GPU | 97 / 100 | 62, 85, 93 |

Question by question the two give the same answer on 98 of 100. Question 12 is counted right only in this file, and its reply is not a clean one: it reached the 2,048-token cap without a final line after writing "answer is 13" and "answer is 12" twice each, and the reader script read the last of these, 13. Question 62 is wrong in both, with different answers. 15 of this file's replies ended without the final line and were read by the next rules, all of them counted right; question 12 is the only one of its 100 replies that reached the cap, and none of the 100 processes logged a WebGPU validation error. Three bf16 replies (45, 62 and 93) reached the cap.

## Usage

### Recommended: `Qwen3.8-27B_wi4b32_32k.litertlm` (LiteRT-LM ≥ 0.20.0)

Specifically, it requires LiteRT-LM >= 0.20.0 (`litert-lm-nightly 0.20.0.dev*`, a source build from `main`, or the upcoming `0.20.0` stable release).

```bash
pip install --pre litert-lm-nightly

# GPU (thinking enabled by default; supports up to 32K context)
litert-lm run ./Qwen3.8-27B_wi4b32_32k.litertlm \
  --prompt "What is the tallest building in the world? Answer in two sentences." \
  --backend gpu

# GPU with thinking disabled
litert-lm run ./Qwen3.8-27B_wi4b32_32k.litertlm \
  --prompt "What is 17 + 25? Answer briefly." \
  --backend gpu --thinking false --temperature 0

# CPU (always pass --cache disk on CPU to avoid high weight-packing RAM)
litert-lm run ./Qwen3.8-27B_wi4b32_32k.litertlm \
  --prompt "What is 17 + 25? Answer briefly." \
  --backend cpu --cache disk --max-num-tokens 4096 --thinking false --temperature 0

# Benchmark (1024 prefill + 256 decode at 32K context)
litert-lm benchmark ./Qwen3.8-27B_wi4b32_32k.litertlm \
  -p 1024 -d 256 --max-num-tokens 32768 --runs 3 --backend gpu
```

### Legacy: `Qwen3.8-27B_mixed_int4.litertlm` (LiteRT-LM 0.18.1–0.19.1)

```bash
litert-lm run ./Qwen3.8-27B_mixed_int4.litertlm --prompt "What is 17 + 25? Answer briefly." --backend gpu --cache no
litert-lm run ./Qwen3.8-27B_mixed_int4.litertlm --prompt "What is 17 + 25? Answer briefly." --backend cpu --cache disk
litert-lm benchmark ./Qwen3.8-27B_mixed_int4.litertlm -p 1024 -d 256 --max-num-tokens 2048 --runs 3 --cache no --backend gpu
```

Run as printed on the Mac M4 Max with the file under this name (litert-lm 0.18.1, 2026-10-23), the two `run` commands printed `42` and exited 0, and the benchmark exited 0 with prefill 176.99 and decode 18.42 tok/s.

- **The GPU engine takes 3½ to 4 minutes to build** on the Mac M4 Max for `Qwen3.8-27B_mixed_int4.litertlm`: the first command above took 217 s from start to answer, and a benchmark process spends 214–247 s per engine (three processes, from their wall times); the CLI prints about twice that as its init time because it adds up overlapping init phases ([LiteRT-LM issue 4057](https://github.com/google-ai-edge/LiteRT-LM/issues/4057)). Build it once and keep it for the conversation. (With GPU composites on LiteRT-LM ≥ 0.20.0, GPU engine build is dramatically faster.)
- **The CPU needs the weight cache (`--cache disk`).** Without it (`--cache no`) the CPU engine packs the int4 weights once per signature: on the block-32 file of the original export the memory footprint grew from 5.8 GB to 91 GB in 48 s during the engine build on a 128 GiB Mac, with swap growing, and the run was stopped. With `--cache disk` the first CPU run writes an XNNPACK weight cache next to the model and later runs read it at much lower peak memory.
- **One Engine per conversation (LiteRT-LM ≤ 0.19.1).** [LiteRT-LM issue 3165](https://github.com/google-ai-edge/LiteRT-LM/issues/3165): a new Conversation on an Engine that already ran another conversation could continue from the earlier conversation's recurrent state. A fix landed on LiteRT-LM `main` on 2026-11-03 ([e1f86b05](https://github.com/google-ai-edge/LiteRT-LM/commit/e1f86b05bf8a731331622a2622440e9f93d9a9f5)), included in LiteRT-LM ≥ 0.20.0 (`litert-lm-nightly`).
- **One Engine per process on the Mac GPU (0.18.1).** The benchmark builds 4 engines one after another in one process. Across 11 such processes on `Qwen3.8-27B_mixed_int4.litertlm` and variants of it, WebGPU validation errors appeared in 2 of the 44 engines, each time in the third engine of its process; none appeared in 115 single-engine processes.
- Without `--backend`, `run` uses the file's default backend, the CPU. Pass `--backend gpu` for GPU acceleration. For `Qwen3.8-27B_mixed_int4.litertlm`, the file sets no sampler; pass `--top-k 20 --top-p 0.8 --temperature 0.7` for Qwen's non-thinking recommendation. `Qwen3.8-27B_wi4b32_32k.litertlm` embeds default sampler params (`top_k=20`, `top_p=0.95`, `temperature=0.6`).
- Tool calling: both `Qwen3.8-27B_wi4b32_32k.litertlm` and `Qwen3.8-27B_mixed_int4.litertlm` (since 2026-11-03) declare `supports_function_calling: true` and the regex tool-call configuration of LiteRT-LM's `models/qwen3_5` ([the fix for issue 3927](https://github.com/google-ai-edge/LiteRT-LM/commit/ce42d882bde880a5ef28afd68544457e5140027a)), so LiteRT-LM ≥ 0.19.0 returns Qwen XML `<tool_call>` blocks as structured tool calls.
- **Replies that take over 10 minutes.** On `litert-lm 0.18.x` and `0.19.0`, a blocking `send_message` that is still generating after 10 minutes fails with `DEADLINE_EXCEEDED`. For long replies on a slow backend (long thinking in particular), lower `max_output_tokens` or use streaming `send_message_async`.

**Mac app:** [litertlm-chat-mac](https://github.com/john-rocky/LiteRT-Models/tree/main/litertlm-chat-mac) is a SwiftUI chat app that opens `Qwen3.8-27B_mixed_int4.litertlm` and answers on the Mac's GPU through LiteRT-LM's Swift package (0.18.1), one engine process per conversation.

## Performance

### `Qwen3.8-27B_wi4b32_32k.litertlm` — Apple M5 Max (40-core GPU, 48 GB)

Workload: **1024-token prefill + 256-token decode**, batch size 1, LiteRT-LM `main` (≥ 0.20.0).

| Backend | Context | Prefill 1024 (tok/s) | Decode 256 (tok/s) | TTFT (s) |
|---|---|---:|---:|---:|
| **GPU (Metal)** | 1K–4K (`--max-num-tokens 1024..4096`) | **954–956** | **34.0–34.1** | **1.10** |
| **GPU (Metal)** | 32K (`--max-num-tokens 32768`) | **940–960** | **33.2–34.1** | **1.10–1.26** |
| **GPU (WebGPU)** | 1K–4K | **293** | **29.7** | **3.53** |
| **GPU (WebGPU)** | 32K | **293** | **29.5** | **3.53** |

*(Metal measurements reflect native Metal GPU execution on macOS with LiteRT-LM `main`; WebGPU reflects the cross-platform WebGPU backend on the same host.)*

### Legacy files — Apple M4 Max (128 GiB)

Mac Studio M4 Max (128 GiB), `litert-lm 0.18.1` (and `litert-lm-nightly 0.20.0.dev20261106` for the 2026-11-07 file): 1024-token prefill, 256-token decode, context 2048, one warmup, then the mean of 3 iterations across two benchmark processes. GPU is WebGPU on Metal with `--cache no`; CPU is `--cache disk` with the weight cache already written.

| Mac M4 Max | Prefill 1024 (tok/s) | Decode 256 (tok/s) | TTFT (s) | Engine build (s) | Peak memory (GB) |
|---|---|---|---|---|---|
| Mixed INT4 GPU (`0.18.1`) | 175.50 | 18.24 | 5.89 | 246.8 | 24.9 |
| Mixed INT4 CPU (weight cache, `0.18.1`) | 21.82 | 5.59 | 47.13 | 123.8 | 4.1 |
| Mixed INT4, GPU composites (2026-11-07 file), WebGPU | 233.71 | 23.85 | 4.42 | 3.7 | 18.6 |

- Not run on a phone or on iOS: the 27B model (~15.2 GB) does not fit in typical mobile RAM; it is intended for desktop-class hardware.
- Not run on an NPU: LiteRT has no NPU accelerator on macOS.

## Conversion notes

### `Qwen3.8-27B_wi4b32_32k.litertlm`

Exported from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) using `litert-torch`'s Converter V2 streaming pipeline (now on `google-ai-edge/litert-torch` `main`):

```bash
litert-torch export_hf \
  --model=Qwen/Qwen3.8-27B \
  --output_dir=./qwen3_8_27b_wi4b32_32k \
  --quantization_recipe=dynamic_wi4b32_afp32 \
  --apply_gpu_composites=True \
  --externalize_weights=True \
  --externalize_embedder=True \
  --bundle_litert_lm=True \
  --prefill_lengths='[128,1024]' \
  --cache_length=32768 \
  --llm_metadata_max_num_tokens_override=32768 \
  --litert_lm_llm_metadata_override=qwen3_5 \
  --use_v2=True \
  --delete_in_memory_params=True
```

- **Quantization:** `dynamic_wi4b32_afp32` (symmetric INT4, block size 32, fp16 scales, fp32 activations).
- **Graph & container:** `prefill_128`, `prefill_1024`, and `decode` signatures with full GPU composites, 32K dynamic KV cache, externalized embedder (`tf_lite_embedder`) and weights (`TFLiteWeights` sections), and canonical `qwen3_5` metadata (`supports_thinking: true`, `supports_function_calling: true`, `pad_token: "<|endoftext|>"`, `stop_tokens: "<|im_end|>"`, `min_runtime_version: "0.20.0"`).

### `Qwen3.8-27B_mixed_int4.litertlm`

- **Source and tools:** Qwen/Qwen3.8-27B at revision `1d4bf0f2`; `litert-torch` at commit `115a1360` with the Qwen3.5 hybrid patch of [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm), transformers 5.14.1, torch 2.12.1, ai-edge-quantizer 0.8.0; packed and checked with litert-lm 0.18.1 (metadata updated with litert-lm 0.19.0 on 2026-11-03).
- **Export & quantization:** 6 prefill signatures (1024, 256, 64, 16, 4, 1) plus decode over a 4096-position KV budget; int4 blockwise-128 min-max on the 400 linear layers, int8 channelwise on lm_head and embedding, 96 decay/gate projections kept float; fp32 GPU activations declared; thinking-off template; repo `tokenizer.json` packaged verbatim.
- **Not in any file:** the vision tower and the multi-token-prediction head (see [LiteRT-LM issue 3439](https://github.com/google-ai-edge/LiteRT-LM/issues/3439)).

## Known runtime issues

Measurements of `Qwen3.8-27B_mixed_int4.litertlm` on this page made before 2026-11-11 used earlier files, not the current one.

**Android GPU with LiteRT-LM v0.18.0 and v0.18.1.** `Qwen3.8-27B_mixed_int4.litertlm` keeps some of its weights outside the model graph (LiteRT's external weights), and on a Pixel 8a the GPU backend of the v0.18.0 and v0.18.1 Android builds crashes (SIGSEGV) on a Qwen3-0.6B file with external weights ([LiteRT-LM#3986](https://github.com/google-ai-edge/LiteRT-LM/issues/3986)).

From a conversation preface (the system message, initial messages, tools and extra context a conversation is created with) that has tools and no messages, LiteRT-LM 0.18.1 renders `Qwen3.8-27B_mixed_int4.litertlm`'s chat template with an empty message list, and the template reads `messages[0]` without checking that the list is non-empty, so the render fails with `Failed to apply template: undefined value` when `prefillPrefaceOnInit` is enabled or `renderPrefaceIntoString()` is called. (The canonical `qwen3_5` template in `Qwen3.8-27B_wi4b32_32k.litertlm` guards `messages and messages[0]['role'] == 'system'`.)

On LiteRT-LM 0.18.1 and 0.19.1, once a message fails because it does not fit in the context window, every later message in that conversation fails as well (the next one with `Task failed with state: 7`). Start a new conversation after such an error.

## License and changes

Apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (the repo's metadata and its LICENSE file, "Copyright 2026 Alibaba Cloud"); this repository carries a copy of that LICENSE. **Changes from the original work:** the text decoder of the multimodal checkpoint exported (the vision tower and the multi-token-prediction head not loaded), weights converted from safetensors bf16 to LiteRT flatbuffers and quantized as described above; tokenizer and chat template packaged in LiteRT-LM format. This repository is a community conversion and is not affiliated with Qwen or Alibaba Cloud.
