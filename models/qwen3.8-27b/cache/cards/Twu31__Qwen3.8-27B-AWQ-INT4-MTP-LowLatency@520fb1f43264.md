---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
  - zh
tags:
  - qwen3.8
  - qwen3.5
  - awq
  - int4
  - w4a16
  - compressed-tensors
  - marlin
  - mtp
  - dflash2
  - speculative-decoding
  - low-latency
  - vllm
  - quantized
  - conversational
  - autoround
  - gptq
  - llmcompressor
---

# Qwen3.8-27B-AWQ-INT4-MTP-LowLatency

**A latency-optimized INT4 (W4A16) build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for interactive, thinking-off serving on vLLM: MTP speculative head kept intact and measured, `lm_head` in INT8, DFlash2 drafting verified, plus the serving flags that make this hybrid GDN+attention model's prefix cache actually hit.**

**Why this build.** Most 4-bit checkpoints of this model are made and judged for throughput or benchmark accuracy. This one targets the other regime: real-time conversation (voice agents, embodied assistants, chat UIs) where a single GPU must turn a \~1.5 k-token prompt into a 20 to 60-token reply with very low time-to-first-token and per-turn latency, thinking off, one or a few streams at a time. The calibration data, the preserved MTP draft head, the INT8 output layer and the serving flags below were chosen for that use, and the figures on this page are measured end-to-end turn latencies rather than peak throughput.

> **TL;DR**: 18.2 GB (BF16 base 55.6 GB) · INT4 symmetric g128, `compressed-tensors`, vLLM Marlin kernel on **SM 8.0+** GPUs (Machete on Hopper) · vision tower + MTP draft head BF16, `lm_head` INT8 · zero-config in vLLM 0.24 through 0.33 · **MTP verified working** (64 to 73 % draft acceptance on chat traffic) · **DFlash2 verified working** with the official z-lab draft (10.0 ms per token on one RTX 6000 Ada) · with vLLM 0.30, MTP and `--prefix-match-unit 16`, an 18-token turn after a 1.5 k-token prompt takes **237 ms with TTFT 59 ms** on one RTX 6000 Ada, down from 415 ms with the 0.24-era recipe; vLLM 0.31 gives 56 ms / 248 ms, vLLM 0.32 57 ms / 236 ms and vLLM 0.33 57 ms / 235 ms on the same recipe, and on 0.33 a new question after a cached system prompt gets 133 ms TTFT instead of 285 ms · **vLLM 0.33 cuts DFlash2's TTFT from about 240 to 84 ms** (249 ms per turn). Single stream, same harness for every number on this page.

> **Naming note**: "AWQ" is the checkpoint class people search for (4-bit W4A16, same format and kernel path as AWQ-INT4 checkpoints). We built the same model with three algorithms on the same calibration data, **AutoRound, AWQ and GPTQ**, and measured them side by side; `main` ships AutoRound + INT8 `lm_head`, the AWQ and GPTQ builds are the `awq-llmcompressor` / `gptq-llmcompressor` branches (see [Which algorithm](#which-algorithm)). Serving speed is identical across the three by construction.

## Quick start (vLLM 0.29 or newer; older versions below)

**Recipe A: shortest TTFT, MTP off**
```bash
vllm serve Twu31/Qwen3.8-27B-AWQ-INT4-MTP-LowLatency \
  --served-model-name qwen38 --max-model-len 12288 --max-num-seqs 8 \
  --enable-prefix-caching --prefix-match-unit 16 --mamba-ssm-cache-dtype bfloat16 \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
```
One RTX 6000 Ada: **TTFT 51 ms, 396 ms per 18-token turn**. No prompt padding needed any more; see [Prefix caching](#prefix-caching-on-a-hybrid-gdn-model).

**Recipe B: shortest turn, MTP on (recommended for chat)**
```bash
vllm serve Twu31/Qwen3.8-27B-AWQ-INT4-MTP-LowLatency \
  --served-model-name qwen38 --max-model-len 12288 --max-num-seqs 8 \
  --enable-prefix-caching --prefix-match-unit 16 --mamba-ssm-cache-dtype bfloat16 \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```
One RTX 6000 Ada, vLLM 0.30.0: **TTFT 67 ms, 273 ms per turn**, 11.0 ms per token; after one prefix-warming call (below) **59 ms / 237 ms**; later turns of a conversation land at 80 to 90 ms TTFT with 95 to 98 % cache hit. On 0.29.0 the same recipe gives 141 ms / 388 ms (89 / 276 warmed); 0.30 added the Mamba-state fixes for speculative resume ([vllm#53945](https://github.com/vllm-project/vllm/pull/53945), [vllm#55178](https://github.com/vllm-project/vllm/pull/55178)). vLLM 0.31.0: TTFT 56 ms (71 ms for a 0.30.0 run on the same day), 247 ms per warm turn, 11.0 ms per token. vLLM 0.32.0: TTFT 58 ms (58 ms for a 0.31.0 run right before it), 237 ms per warm turn, 11.0 ms per token. vLLM 0.33.0: TTFT 57 ms, 235 ms per warm turn, 10.9 ms per token, level with a 0.32.0 run right before it, and the first request with a new question after the cached system prompt and history now takes 133 ms TTFT instead of 285 ms ([vllm#61589](https://github.com/vllm-project/vllm/pull/61589), see [Prefix caching](#prefix-caching-on-a-hybrid-gdn-model)). On 0.32 the experimental `"disable_eagle_block_drop": true` in the speculative config gets that first request to 136 ms too, with a small cost on fully cached turns (same section).

**Recipe C: fastest decode, DFlash2 draft (long replies, single stream)**
```bash
vllm serve Twu31/Qwen3.8-27B-AWQ-INT4-MTP-LowLatency \
  --served-model-name qwen38 --max-model-len 12288 --max-num-seqs 8 \
  --enable-prefix-caching --mamba-ssm-cache-dtype bfloat16 \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
```
The official draft ([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), 5 layers, 3.85 GB) loads against this checkpoint's INT8 `lm_head` unchanged and gives **10.0 ms per token** (K=7, 2.7 tokens accepted per step on chat). Its cost is TTFT: the draft's sliding-window layers switch off vLLM's fine-grained prefix hits for the whole model ([vllm#53786](https://github.com/vllm-project/vllm/issues/53786), [vllm#57032](https://github.com/vllm-project/vllm/issues/57032), both still open). Up to 0.32 vLLM also dropped the last cached block on every hit, so a 1.5 k-token prompt paid about 240 ms TTFT and about 400 ms per 18-token turn, and Recipe C won only on replies past roughly 150 tokens. **vLLM 0.33 keeps that block** ([vllm#54163](https://github.com/vllm-project/vllm/pull/54163)): TTFT 84 ms and 249 ms per warm turn (Recipe B: 235 ms), 79 / 229 ms with the INT4 draft, decoding at 9.5 against 10.9 ms per token on these replies. Hits are still whole 448-token blocks, though: in a running conversation TTFT climbs from 79 ms on the first turn to 130 ms on the fourth (Recipe B: 77 to 79 ms), and five streams take 597 ms per turn (Recipe B: 287 ms). So Recipe B stays the default for chat, and Recipe C can pay off for one stream with longer replies. Needs vLLM 0.29 or newer and about 4 GB more GPU memory (2.4 GB with the INT8 draft in [Latency](#latency-measured)). K=7 beat K=3/4/5; the extra verification is nearly free at batch 1.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(model="qwen38", max_tokens=128, temperature=0.7,
    messages=[{"role": "system", "content": "You are a concise, friendly assistant."},
              {"role": "user", "content": "I've had a long day. Any quick tip to unwind?"}])
print(r.choices[0].message.content)
```
- No `--quantization` flag needed; vLLM logs `Using MarlinLinearKernel for CompressedTensorsWNA16`. On Hopper (SM 9.0) vLLM picks its Machete kernel for these layers instead, in 0.33 as in earlier versions; we have not measured Hopper.
- Check speculation is really accepting: `curl :8000/metrics | grep spec_decode`. MTP: accepted/draft ≈ 0.6 to 0.75 on chat; DFlash2 K=7: about 0.25, 2.5 to 2.8 tokens per step. A drafter that accepts 0 % is *slower* than speculation off.
- Text-only deployment: `--language-model-only` skips the vision tower (0.9 GB less). Not `--limit-mm-per-prompt` zeros, see [Gotchas](#gotchas).
- On a shared GPU size the cache with `--kv-cache-memory-bytes` rather than `--gpu-memory-utilization`; profiling on a busy card under-estimates it. vLLM 0.31 to 0.33 still check free memory against `--gpu-memory-utilization` (default 0.92) at startup, so lower that too when another process holds a few GB.
- Don't pass `--calculate-kv-scales` on hybrid GDN models ([vllm#37554](https://github.com/vllm-project/vllm/issues/37554)). SGLang: see [SGLang](#sglang).
- Tokenizer and config files are byte-identical to upstream `Qwen/Qwen3.8-27B`.
- vLLM 0.33 needs Python 3.11 or newer, installs torch 2.14 and pins transformers to 5.16.1–5.20 (5.20.2 in our runs). It ships CUDA 13.0 wheels and Docker images only; the CUDA 12.9 builds that 0.32 still had are gone ([vllm#62575](https://github.com/vllm-project/vllm/pull/62575)), so the NVIDIA driver must support CUDA 13 (R580 or newer).

## Latency (measured)

*Conversational prompts (≈1.5 k-token system prompt + 2 turns of history + a user utterance), greedy, thinking off, 18 prompts × 3 runs, streamed over HTTP, medians; "ms/tok" = inter-token latency; \~18-token replies; KV cache fixed at 10 GiB in every row. One RTX 6000 Ada 48 GB. Same weights in every row. Greedy replies are not bit-stable across runs: between any two of our runs, 0 to 6 of the 18 prompts got a different reply, including two restarts of the same vLLM version (torch.compile autotuning picks kernels by timing); persona tags and reply length were not affected.*

| Recipe | vLLM | TTFT | Turn | ms/tok | Accepted / step | Cache hit |
|---|---|---|---|---|---|---|
| MTP off, default prefix caching | 0.29 / 0.30 | 108 / 112 | 437 / 437 | 18.5 | - | 79 % |
| **A** MTP off + `--prefix-match-unit 16` | 0.29 / 0.30 | **51 / 54** | 396 / 398 | 18.2 | - | 99 % |
| MTP K=2, default prefix caching | 0.29 / 0.30 | 261 / 265 | 461 / 462 | 11.6 / 11.9 | 2.46 | 55 % |
| **B** MTP K=2 + `--prefix-match-unit 16` | 0.29 / 0.30 | 141 / **67** | 388 / **273** | 11.2 / 11.0 | 2.43 | 93 to 98 % |
| **B** + one prefix-warming call | 0.29 / 0.30 | 89 / **59** | 276 / **237** | 10.6 / 10.9 | 2.43 | 98 % |
| **C** DFlash2 K=7 | 0.29 / 0.30 | 235 / 236 | 401 / 400 | **10.0 / 10.1** | 2.74 | 59 % |

Multi-turn conversations (5 × 4 turns, each turn appends the model's own reply, 0.30.0): Recipe A 45 ms TTFT on the first turn and 84 to 86 ms after; Recipe B 52 then 81 to 91 ms; Recipe C 251 then 264 to 295 ms.

**Concurrency** (c = 1 / 2 / 5 streams, each sending the 18 prompts, `--prefix-match-unit 16`; per-turn medians, aggregate tok/s at c = 5)

| Recipe | vLLM | c=1 | c=2 | c=5 | tok/s at c=5 |
|---|---|---|---|---|---|
| A, MTP off | 0.30.0 | 428 ms | 415 ms | 414 ms | 223 |
| B, MTP K=2 | 0.30.0 | 426 ms | **250 ms** | **327 ms** | **267** |
| B, MTP K=2 | 0.29.0 | 432 ms | 438 ms | 889 ms | 113 |
| C, DFlash2 K=7 | 0.30.0 | 407 ms | 698 ms | 1556 ms | 65 |

On 0.30 MTP no longer hurts multi-user serving; on 0.29 and older it roughly doubles the turn at 5 streams. DFlash2 is single-stream only on this card.

**vLLM 0.33.0** (released 2026-11-02), measured back to back with a 0.32.0 control on the same card and morning: each 0.32.0 arm ran right before its 0.33.0 arm, and Recipes A and C were repeated with the order reversed. TTFT is the median over all 54 requests; *warm turn* = median of the second and third pass over the 18 prompts (prefix already cached), because the pooled 54-request turn median mixes in the cold first pass and moves with reply length; *first pass* = TTFT median of the first pass, where each prompt's last user message is new and only the system prompt and history are cached, which is what a new question in a running conversation looks like.

| Recipe | 0.32.0 control: TTFT / warm turn / ms·tok | 0.33.0: TTFT / warm turn / ms·tok | First pass TTFT (0.32 / 0.33) | Cache hit (0.32 / 0.33) | Accepted / step (0.32 / 0.33) |
|---|---|---|---|---|---|
| **A** MTP off + `--prefix-match-unit 16` | 47 / 366 / 18.2 | 52 / 361 / 18.2 | 135 / 135 | 95 / 95 % | - |
| **B** MTP K=2 + `--prefix-match-unit 16` | 57 / 235 / 10.9 | 57 / 235 / 10.9 | 285 / **133** | 90 / **96 %** | 2.45 / 2.44 |
| **C** DFlash2 K=7 | 239 / 409 / 9.8 | **84 / 249** / 9.5 | 234 / **83** | 63 / **90 %** | 2.73 / 2.74 |
| **C** with the INT8 draft | 242 / 405 / 9.5 | **80 / 236** / 9.0 | 238 / 80 | 59 / 89 % | 2.73 / 2.74 |
| **C** with the INT4 draft | 241 / 387 / 8.4 | **79 / 229** / 8.7 | 236 / 79 | 59 / 89 % | 2.78 / 2.76 |

- **Recipe B, new questions.** Up to 0.32 a speculative request that hit the cache gave back its last cached block, so the first pass paid 285 ms TTFT, twice Recipe A's 135 ms. 0.33 hashes each block together with the token(s) after it ([vllm#61589](https://github.com/vllm-project/vllm/pull/61589)) and keeps the block: 133 ms. `"disable_lookahead_block_hashes": true` in the speculative config restores the old hashing (286 ms). Repeated prompts hit fully on both versions, which is why the pooled TTFT does not move.
- **Recipe C.** 0.33 also stops dropping the last cached block for DFlash ([vllm#54163](https://github.com/vllm-project/vllm/pull/54163)), so a hit now covers 3 of the prompt's 448-token blocks instead of 2, and every pass, new or repeated, lands at 80 to 85 ms. With 0.33 run first: 84 to 86 ms against 244 to 260 ms. The BF16-draft row ran with `--prefix-match-unit 16` added, the INT8 / INT4 rows with Recipe C's flags as printed; the hit rate was 89 to 90 % either way.
- **Recipe A** reaches the first token 3 to 5 ms later on 0.33 (52 vs 47 ms; 50 vs 46 to 47 with 0.33 run first); warm turn and decode are level, and the logs show the same kernels, block sizes and attention backend.
- **Decode steps are unchanged.** On fixed 192-token replies a step takes 25.8 ms with MTP K=2 on both versions, and 26.9 / 25.1 / 24.1 ms with the BF16 / INT8 / INT4 DFlash2 draft on 0.33 (27.1 / 25.1 / 24.2 on 0.32). The 18-token ms/tok medians in the table move by up to 0.5 ms from run to run.
- Greedy replies were identical on 48 to 51 of the 54 requests between the two versions, against 45 to 54 between two 0.32.0 runs on different days.

Multi-turn on 0.33.0: Recipe A 44 ms TTFT on the first turn and 89 to 95 ms after (0.32: 40 then 79 to 80); Recipe B 49 then 77 to 79 ms at 97 % hit; Recipe C 79 then 90, 111 and 130 ms (0.32: 252 then 261 to 303). Concurrency c = 1 / 2 / 5 on 0.33.0: A 398 / 398 / 416 ms (222 tok/s at c = 5), B 297 / 230 / 287 ms (314 tok/s), C 250 / 337 / 597 ms (158 tok/s); 0.32.0 on 2026-10-23 gave B 430 / 250 / 337 ms (273 tok/s) and C 412 / 712 / 1591 ms (63 tok/s). MTP K=3 is unchanged (10.2 ms/tok, 2.85 accepted per step, warm turn 254 vs 235 ms for K=2), so K=2 stays the recommendation. `--attention-backend FLASHINFER` with MTP K=2 decodes 3 % faster per step on 0.33 (25.1 vs 25.8 ms; [vllm#60532](https://github.com/vllm-project/vllm/pull/60532) gives it full CUDA graphs for speculative decoding below SM 9.0) but costs 7 ms TTFT and 16 ms per warm turn (63 / 251 ms), so the default FLASH_ATTN stays; with DFlash2 it gives 86 / 253 ms, no gain either.

**`lm_head` INT4** (`lm-head-int4` revision) saves 4 to 8 % per token on this card (17.8 vs 18.5 ms MTP off, 9.2 vs 10.0 with DFlash2), less than the 20 % measured on an RTX 5090, and lowers DFlash2 acceptance a little (2.59 vs 2.74 per step); `main` keeps the INT8 head. On 0.33 with the BF16 DFlash2 draft (single run): 82 ms TTFT, 213 ms per warm turn, 25.4 ms per decode step and 2.58 accepted per step (`main`: 84 / 249 ms, 26.9 ms, 2.74).

**Quantized DFlash2 drafts.** [Twu31/Qwen3.8-27B-DFlash2-INT8-LowLatency](https://huggingface.co/Twu31/Qwen3.8-27B-DFlash2-INT8-LowLatency) is the z-lab draft with MLP, `o_proj` and `fc` in INT8 (`main`, 2.30 GB) or INT4 (`int4`, 1.51 GB). With this target at K=7: **9.0 ms/tok** on 0.30 (9.5 / 9.2 on 0.29), acceptance unchanged, 1.4 to 2.2 GB less GPU memory. Replace `"model"` in Recipe C's speculative config to use it. Both load on 0.31.0 with the same acceptance, and 0.31.0 also loads drafts whose q/k/v are quantized (see [Gotchas](#gotchas)). Timed on fixed-length replies, a decode step takes 25.3 ms with the INT8 draft and 24.3 ms with INT4 against 27.1 ms with the BF16 draft (7 % / 10 % less), the same on 0.31.0, 0.32.0 (re-test 2026-10-23 with Nsight Systems, [details on the draft's card](https://huggingface.co/Twu31/Qwen3.8-27B-DFlash2-INT8-LowLatency#per-decode-step)) and 0.33.0 (25.1 / 24.1 against 26.9 ms). An earlier version of this card said 0.32.0 removed the gain; that came from the 18-token medians, which move by up to 0.5 ms/tok from run to run. On 0.33 both drafts also get Recipe C's shorter TTFT: 80 / 236 ms (INT8) and 79 / 229 ms (INT4).

**Quantizing the MTP draft's `lm_head`** (vLLM 0.33, `"draft_lm_head_quantization": "fp8"` or `"nvfp4"` in the speculative config, [vllm#59973](https://github.com/vllm-project/vllm/pull/59973)) needs a target whose `lm_head` is unquantized, so here it works only on `mtp-int4-autoround` (BF16 head). There a decode step drops from 28.8 ms to 26.1 ms with fp8 and 24.6 ms with nvfp4 (Marlin's weight-only NVFP4 kernel on Ada), acceptance stays at 2.41 to 2.43, and TTFT / warm turn go from 60 / 259 ms to 57 / 237 (fp8) and 56 / 225 ms (nvfp4), the shortest MTP turn we measured on 0.33 (`main`: 57 / 235 ms). On `main`, `awq-llmcompressor`, `gptq-llmcompressor` and `lm-head-int4` vLLM refuses the option (`Quantizing an lm_head copy needs an unquantized fp16/bf16 lm_head without bias.`); their MTP draft already shares the target's quantized head.

<details>
<summary><b>Earlier measurements</b>: vLLM 0.32.0 against 0.31.0, vLLM 0.31.0 against 0.30.0, vLLM 0.24.0 recipe ladder and MTP sweep (RTX 6000 Ada), head precision and fidelity on an RTX 5090 (vLLM 0.27.1)</summary>

**vLLM 0.32.0 against a 0.31.0 control** (0.32.0 released 2026-10-18), measured back to back on the same card and night: each 0.31.0 arm ran right before its 0.32.0 arm. TTFT is the median over all 54 requests as above; *warm turn* = median of the second and third pass over the 18 prompts (prefix already cached), because the pooled 54-request turn median mixes in the cold first pass and moves with reply length.

| Recipe | 0.31.0 control: TTFT / warm turn / ms·tok | 0.32.0: TTFT / warm turn / ms·tok | Accepted / step (0.31 / 0.32) |
|---|---|---|---|
| **A** MTP off + `--prefix-match-unit 16` | 49 / 367 / 18.2 | **47 / 358** / 18.3 | - |
| **B** MTP K=2 + `--prefix-match-unit 16` | 58 / 249 / 10.9 | 58 / **237** / 11.0 | 2.46 / 2.45 |
| **B** + one prefix-warming call | 57 / 249 / 10.7 | 57 / 236 / 10.8 | 2.46 / 2.45 |
| **C** DFlash2 K=7 | 249 / 409 / 9.5 | 243 / 404 / 9.5 | 2.76 / 2.77 |

0.32.0 is level with 0.31.0. The 9 to 12 ms shorter warm turns come mostly from replies half a token shorter (median 18.0 vs 18.5 tokens), not from faster steps. The hybrid prefix-cache fixes in 0.32 ([vllm#55601](https://github.com/vllm-project/vllm/pull/55601), [vllm#59759](https://github.com/vllm-project/vllm/pull/59759), [vllm#59536](https://github.com/vllm-project/vllm/pull/59536)) leave Recipe B's TTFT and its 96 to 98 % multi-turn hit rate where 0.31 had them. Greedy replies were identical on 42 to 51 of the 54 requests between the two versions, against 45 to 54 between two 0.31.0 runs on different days.

Multi-turn on 0.32.0: Recipe A 41 ms TTFT on the first turn and 79 to 80 ms after; Recipe B 51 then 81 to 82 ms; Recipe C 256 then 273 to 313 ms. Concurrency c = 1 / 2 / 5 on 0.32.0: A 402 / 396 / 410 ms (226 tok/s at c = 5), B 430 / 250 / 337 ms (273 tok/s), C 412 / 712 / 1591 ms (63 tok/s). MTP K=3 is unchanged (10.3 ms/tok, 2.85 accepted per step, warm turn 256 vs 237 ms for K=2), so K=2 stays the recommendation.

**vLLM 0.31.0 against a 0.30.0 control** (2026-10-07; 0.31.0 released 2026-10-05), measured back to back on the same card and morning. TTFT is the median over all 54 requests as above; *warm turn* = median of the second and third pass over the 18 prompts (prefix already cached), because the pooled 54-request turn median mixes in the cold first pass and moves with reply length.

| Recipe | 0.30.0 control: TTFT / warm turn / ms·tok | 0.31.0: TTFT / warm turn / ms·tok | Accepted / step (0.31) |
|---|---|---|---|
| **A** MTP off + `--prefix-match-unit 16` | 55 / 363 / 18.2 | **49** / 367 / 18.2 | - |
| **B** MTP K=2 + `--prefix-match-unit 16` | 71 / 259 / 10.9 | **56 / 247** / 11.0 | 2.47 |
| **B** + one prefix-warming call | - | 56 / 248 / 10.9 | 2.47 |
| **C** DFlash2 K=7 | - | 246 / 420 / 10.0 | 2.74 |

Multi-turn on 0.31.0: Recipe A 42 ms TTFT on the first turn and 82 to 83 ms after; Recipe B 51 then 79 to 81 ms; Recipe C 256 then 269 to 304 ms. Concurrency c = 1 / 2 / 5 on 0.31.0: A 424 / 407 / 408 ms (231 tok/s at c = 5), B 432 / 250 / 326 ms (278 tok/s), C 412 / 695 / 1581 ms (64 tok/s). MTP K=3 decodes faster (10.3 ms/tok, 2.85 accepted per step) but its warm turn is longer (256 vs 247 ms), so K=2 stays the recommendation.

**Serving-recipe ladder, vLLM 0.24.0, MTP off** (block-aligned prompt padding is the pre-`--prefix-match-unit` way to get the same hit rate)

| Configuration | Prompt tok | Cache hit | TTFT | Turn | ms/tok |
|---|---|---|---|---|---|
| default (`--enable-prefix-caching`) | 1449 | 51 % | 272 ms | 634 ms | 20.4 |
| + `--mamba-ssm-cache-dtype bfloat16` | 1449 | 79 % | 120 ms | 468 ms | 20.1 |
| + block-aligned system prompt (1604 tok) | 1662 | 90 % | 75 ms | 418 ms | 20.1 |
| **+ both** | 1662 | **92 %** | **67 ms** | **415 ms** | 19.9 |
| *ref: Qwen3.6-35B-A3B 4-bit (MoE, 3B active), same card/flags* | 1449 | 67 % | 95 ms | 413 ms | 15.0 |

**MTP, vLLM 0.24.0:** K=0 20.4 ms/tok → K=1 15.9 (79 % acceptance) → **K=2 13.7 (73 %, 1.49×)** → K=3 13.5 (61 %). Calibration data matters here: the same recipe calibrated on pile-10k reaches 57.9 % acceptance, on conversational transcripts 73.1 %; a community export without `mtp` in `ignore` loads a broken drafter and runs at **0 % acceptance, 1.5× slower than MTP off**.

**Head precision, fidelity and concurrency: RTX 5090 32 GB, vLLM 0.27.1** (NLL = per-token negative log-likelihood of the BF16 base's own greedy answers under the quantized model; lower = closer to BF16)

| Variant | MTP off TTFT / turn / ms·tok | MTP K=2 TTFT / turn / ms·tok | K=2 accept | NLL | Repetition / vision / format probes |
|---|---|---|---|---|---|
| INT4 body, BF16 lm_head | 80 / 300 / 11.6 | 145 / 299 / 8.7 | 64.6 % | 0.327 | all pass |
| **INT4 + INT8 lm_head: `main`** | 78 / 281 / **10.8** | 140 / 274 / **7.6** | 64.4 % | 0.326 | all pass |
| INT4 + INT4 lm_head: `lm-head-int4` | 76 / 254 / **10.4** | 136 / 262 / **7.0** | 66.0 % | 0.326 | all pass |
| MTP head also INT4: `mtp-int4-autoround` | 81 / 280 / 11.6 | 144 / 286 / 8.0 | 67.8 % | 0.324 | all pass |

The levers that move decode latency cut bytes per token: INT8 lm_head −13 %, INT4 lm_head −20 %, INT4 MTP head −8 % (MTP on). Concurrency on this box (5 streams, MTP off): 394 ms/turn, 253 tok/s aggregate; with MTP K=2: 750 ms/turn.

</details>

### Prefix caching on a hybrid GDN model
Qwen3.8-27B has 16 attention + 48 Gated-DeltaNet layers. vLLM sizes the attention block so that one attention page holds one GDN state, *ceil(GDN state bytes / KV bytes per token)* rounded to the kernel's alignment: 784 to 832 tokens with the default fp32 state, **400** with `--mamba-ssm-cache-dtype bfloat16` (416 with MTP K=2, 448 with the DFlash2 draft; unchanged through 0.33). The GDN state is only stored at block boundaries, so by default a request reuses whole blocks and recomputes the tail of the prompt.
- `--mamba-ssm-cache-dtype bfloat16` halves the state and the block (51 % → 79 % hit on 0.24, no output change on this workload). Keep it.
- `--prefix-match-unit 16` (works on this model from vLLM 0.26) computes cache keys every 16 tokens and makes the prefill stop at the prompt's last 16-token boundary so the GDN state is stored there too; the next request sharing the prefix hits within 16 tokens of it. This replaces padding the system prompt to a block multiple: the unpadded prompt hits 99 %, TTFT 108 → 51 ms (MTP off) and 265 → 67 ms (MTP, 0.30). 16 divides every block size this model can end up with; 32 does not. vLLM 0.32 checks this at startup and refuses a unit that does not divide every block ([vllm#58021](https://github.com/vllm-project/vllm/pull/58021)); 16 passes.
- **Prefix warming.** The state lands at the end of whatever prompt was last prefilled, so register a fixed system prompt (plus history) once: send the rendered prefix, cut right after the last `<|im_start|>user\n`, to `/v1/completions` with `max_tokens=1`. Every later request starting with it hits there (59 ms TTFT with MTP on 0.30). In a conversation the previous turn's prompt is such a prefix automatically, which is why later turns land at 80 to 90 ms.
- **MTP + prefix cache** fell to 0 % on 0.24 to 0.27 because speculative steps never ended on a block boundary; with `--prefix-match-unit 16` it hits 93 to 98 % on 0.29, and 0.30 also fixes the first turn (67 vs 141 ms); 0.31 lowers it to 56 ms, and 0.32 is level with 0.31 (58 ms for both in a back-to-back run). Up to 0.32, though, a speculative request that hit the cache gave back its last cached block, so a new question after a cached system prompt paid 285 ms TTFT against 135 ms with MTP off; 0.33 hashes each block for EAGLE / MTP together with the next token(s) ([vllm#61589](https://github.com/vllm-project/vllm/pull/61589)), keeps the block and gets 133 ms at 96 % hit (90 % on 0.32). `"disable_lookahead_block_hashes": true` in the speculative config restores the old hashing. On 0.32 the experimental `"disable_eagle_block_drop": true` in the speculative config keeps the block as well: 136 ms for a new question at 96 % hit, but turns whose whole prompt was already cached took about 11 ms longer (246 vs 235 ms; vLLM warns that the flag may affect acceptance). On 0.33 it adds nothing (132 ms) and also switches the lookahead hashes off, so leave it off there. `--enable-mamba-shared-prefix-checkpoint` brings 0.32 to 167 ms and makes no difference on 0.33 (134 ms). We did not test either flag on older versions.
- **DFlash2 + prefix cache**: the draft's five sliding-window layers use vLLM's `SlidingWindowManager`, which has no fine-grained lookup, and vLLM then disables fine-grained hits for the whole model (`Disabling fine-grained prefix-cache hits because these KV cache managers require block-aligned lookups: SlidingWindowManager`). Hits fall back to whole 448-token blocks ([vllm#53786](https://github.com/vllm-project/vllm/issues/53786), [vllm#57032](https://github.com/vllm-project/vllm/issues/57032), both still open). Through 0.32.0 the last of those blocks was also dropped on every hit, about 59 % on a 1.5 k-token prompt; 0.33.0 keeps it for DFlash ([vllm#54163](https://github.com/vllm-project/vllm/pull/54163)): 89 to 90 % and 84 ms TTFT. The log line is unchanged, and the tail past the last 448-token boundary is still recomputed, which is why later turns of a conversation climb from 79 to 130 ms.
- vLLM 0.32 removed `--mamba-cache-mode all` ([vllm#58997](https://github.com/vllm-project/vllm/pull/58997)); it was never implemented for Qwen3.5-family models. `align`, which `--enable-prefix-caching` selects on its own, is the only caching mode left, and in it the GDN state block is set to the derived attention block, so 400 is the floor. vLLM 0.33 deprecates the flag itself ([vllm#60043](https://github.com/vllm-project/vllm/pull/60043)): it has no effect and only logs a warning.

## Compatibility

Same checkpoint and harness, one RTX 6000 Ada. Decode speed is flat from 0.26 to 0.33 (18.5 to 18.7 ms/tok MTP off and 11.5 to 11.9 with MTP K=2 under default prefix caching through 0.31; 0.32 and 0.33 are within 0.1 ms/tok of the previous release in back-to-back runs with `--prefix-match-unit 16`); versions differ in prefix-cache behaviour under speculation.

| vLLM | Runner | MTP K=2 | DFlash2 | `--prefix-match-unit 16` | Notes |
|---|---|---|---|---|---|
| 0.24.0 | V1 | 13.7 ms/tok | no | not measured | ladder above; production-tested |
| 0.26.0 | V1 | 11.6 | no (DFlash1 only) | works, 47 ms TTFT | |
| 0.27.1 | V1 | 8.7 (RTX 5090) | no | not measured | py3.10 needs `from __future__ import annotations` in `flashinfer/comm/fd_exchange.py` |
| 0.28 (nightly, 2026-09-06) | V2 | 11.8 | yes | works, 51 ms | |
| 0.29.0 | V2 | 11.6 | yes | works, 51 ms | MTP TTFT 141 ms; MTP at 5 streams 889 ms/turn |
| 0.30.0 | V2 | 11.9 | yes | works, 54 ms | MTP TTFT 67 ms, 327 ms/turn at 5 streams; adaptive verification rejected on the GDN backend |
| main (2026-09-19) | V2 | 11.5 | yes | works | same as 0.30.0 on every arm run |
| 0.31.0 | V2 | 11.9 | yes | works, 49 ms | MTP TTFT 56 ms, 326 ms/turn at 5 streams; drafts with quantized q/k/v load; adaptive verification still rejected |
| 0.32.0 | V2 | 11.0 with `--prefix-match-unit 16` (10.9 on 0.31.0 the same night) | yes | works, 47 ms; checked at startup | MTP TTFT 58 ms, 337 ms/turn at 5 streams; `--mamba-cache-mode all` removed; needs Python 3.11+; quantized DFlash2 drafts 7 / 10 % faster per step than BF16 (INT8 / INT4), as on 0.31; adaptive verification still rejected |
| 0.33.0 | V2 | 10.9 with `--prefix-match-unit 16` (10.9 on 0.32.0 the same morning) | yes, TTFT 84 ms (239 on 0.32.0) | works, 52 ms | MTP TTFT 57 ms and 133 ms for a new question (285 on 0.32.0), 287 ms/turn at 5 streams; CUDA 13.0 builds only; torch 2.14, transformers up to 5.20; `draft_lm_head_quantization` on `mtp-int4-autoround` only; fully quantized DFlash2 drafts load, with lower acceptance; adaptive verification still rejected |

All five revisions load on 0.30.0 (0.30 removed the GPTQ `g_idx` kernels; the `gptq-llmcompressor` branch carries no `g_idx`): `awq-llmcompressor` 53 / 398 ms MTP off and 61 / 364 ms with MTP, `gptq-llmcompressor` 53 / 397 and 61 / 313 ms, `mtp-int4-autoround` (auto_round format via the `inc` backend, BF16 `lm_head`) 64 / 290 ms with MTP at 12.2 ms/tok. All five also load on 0.31.0 with unchanged MTP acceptance (`awq-llmcompressor` 2.22, `gptq-llmcompressor` 2.31, `mtp-int4-autoround` 2.43 accepted per step); the W4A16 layers still run on Marlin. On 0.32.0 all five load on Marlin with the same block sizes; TTFT / warm turn on an idle card: `awq-llmcompressor` 47 / 375 ms MTP off and 57 / 249 ms with MTP (2.31 accepted per step), `gptq-llmcompressor` 47 / 357 and 58 / 236 ms (2.31), `mtp-int4-autoround` 61 / 260 ms with MTP (2.45), `lm-head-int4` 18.0 ms/tok MTP off and 9.9 with DFlash2 (2.55). On 0.33.0 all five load on Marlin with the same block sizes again: `awq-llmcompressor` 53 / 362 ms MTP off and 56 / 248 ms with MTP (2.26 accepted per step), `gptq-llmcompressor` 51 / 361 and 58 / 235 ms (2.34), `mtp-int4-autoround` 60 / 259 ms with MTP (2.43), `lm-head-int4` 17.9 ms/tok MTP off and 82 / 213 ms with DFlash2 (2.58).

## SGLang

SGLang 0.5.20 (CUDA 13 wheels; tested with Python 3.12) serves this checkpoint without conversion: the compressed-tensors W4A16 body loads through SGLang's compressed-tensors path, the GDN layers get SGLang's radix cache for the recurrent state, and prefix hits stay fine-grained with the DFlash2 draft loaded, so DFlash2 does not pay the TTFT penalty it pays in vLLM ([vllm#53786](https://github.com/vllm-project/vllm/issues/53786)). Two SGLang facts set the recipe:

- SGLang's speculative paths (NEXTN/MTP and DFLASH) read `lm_head.weight` and refuse a packed head. `main` (INT8 `lm_head`) fails with `'ParallelLMHead' object has no attribute 'weight'` (NEXTN) or `DFlash2 selector requires a dense FP16/BF16/FP32 target lm_head` (DFLASH). Use the `mtp-int4-autoround` revision, whose `lm_head` is BF16, for speculative decoding in SGLang; `main` serves without speculation.
- `--disable-overlap-schedule` is the TTFT lever at one to a few streams: 73 to 43 ms TTFT without speculation, 95 to 35 ms with NEXTN, and no loss at five streams.

```bash
uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ sglang

python -m sglang.launch_server \
  --model-path Twu31/Qwen3.8-27B-AWQ-INT4-MTP-LowLatency --revision mtp-int4-autoround \
  --served-model-name qwen38 --context-length 12288 --max-running-requests 8 \
  --attention-backend flashinfer --page-size 64 --mamba-ssm-dtype bfloat16 \
  --disable-overlap-schedule --mem-fraction-static 0.8 --max-mamba-cache-size 48 \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

DFlash2 instead of NEXTN: replace the last line with `--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2` (block size 8 is read from the draft config). Without speculation, drop the last line and `--revision`; `main` then gives 30 ms TTFT and 18.7 ms/tok.

*Same prompts and harness as the vLLM tables, medians of 18 prompts x 3, one RTX 6000 Ada 48 GB, overlap scheduler off, `--attention-backend flashinfer --page-size 64 --mamba-ssm-dtype bfloat16`. Most SGLang rows were measured while another process used the same GPU (about 70 % utilisation), so read them as conservative. The two rows marked ¹ were re-run with the launch command above and the GPU otherwise idle, back to back with vLLM Recipes A and B (TTFT within 4 ms of their tables): NEXTN steps 3 did not move (35 ms, 233 ms, 11.6 ms/tok), and no speculation on `main` went from 43 to 30 ms TTFT and from 21.7 to 18.7 ms/tok. The vLLM rows repeat the tables above, taken with the GPU otherwise idle. Where another arm was run twice the table shows both runs.*

| Revision | Recipe | TTFT | Turn (18 tok) | ms/tok | Accepted / step (incl. bonus) | Cache hit |
|---|---|---|---|---|---|---|
| `main` (INT8 head) ¹ | no speculation | 30 | 364 | 18.7 | - | 96 % |
| `mtp-int4-autoround` | no speculation | 46 to 49 | 487 to 532 | 25.6 to 29.3 | - | 96 % |
| `mtp-int4-autoround` | NEXTN steps 2 | 34 | 262 | 12.4 | 2.63 | 96 % |
| **`mtp-int4-autoround`** ¹ | **NEXTN steps 3** | **34** | **229** | **11.5** | 2.65 | 96 % |
| `mtp-int4-autoround` | DFlash2 block 8 (z-lab BF16 draft) | 31 to 32 | 211 to 236 | 10.9 to 12.8 | 2.2 | 96 % |
| vLLM 0.30.0, `main` | Recipe B (MTP K=2 + `--prefix-match-unit 16`) | 67 | 273 | 11.0 | 2.43 | 93 to 98 % |
| vLLM 0.30.0, `main` | Recipe C (DFlash2 K=7) | 236 | 400 | 10.1 | 2.74 | 59 % |

Multi-turn (5 x 4 turns), overlap off: no speculation 122 then 34 to 51 ms; NEXTN steps 3 50 then 40 to 42 ms; DFlash2 49 then 36 to 52 ms, 95 to 97 % hit throughout. Concurrency c=1/2/5 with `--max-mamba-cache-size 48`: no speculation (`main`) 572 / 461 / 436 ms per turn and 33 / 81 / 216 tok/s aggregate; NEXTN steps 3 394 / 292 / 313 ms and 44 / 111 / 279 tok/s; DFlash2 211 / 314 / 423 ms and 73 / 99 / 196 tok/s (vLLM 0.30 Recipe B: 426 / 250 / 327 ms, 267 tok/s at c=5). The concurrency harness uses 256-token replies and two rounds, so its c=1 row is not the 18-token table above.

Notes:
- The GDN state pool, not the KV cache, caps concurrency: the default `extra_buffer` radix strategy books 4 to 5 state slots per running request and the default pool is 16 slots, so the server logs `max_running_requests is capped to 3`. `--max-mamba-cache-size 48` (about 75 MB per slot, plus verification buffers under speculation) serves 8 streams.
- SGLang sizes its pools from `--mem-fraction-static` times the memory that is free when it starts and measures the weights by the change in free memory while loading. On a GPU shared with a process whose usage moves, the sizing drifts and speculative launches can stop with `Not enough GPU memory for hybrid (mamba/linear-attention) state cache` or a 300-token input limit; 0.8 left enough room here. `--max-total-tokens` is a cap, not a floor, so it does not help.
- Rejected here: `--page-size 16` (82 ms TTFT, 29.3 ms/tok), `--attention-backend triton` (85 ms, 29.3 ms/tok), the `no_buffer` radix strategy (no recurrent-state reuse, 584 ms TTFT), overlap scheduler on with NEXTN (95 ms, 16.8 ms/tok).
- The INT8 DFlash2 draft from the companion repo does not load in SGLang's DFLASH path (device-side assert; the INT4 one was not tried); use the z-lab BF16 draft there.
- First launch compiles FlashInfer kernels with nvcc (about 6 minutes here); later launches reuse the cache. SGLang's `auto-round` kernel decodes slower than the compressed-tensors path without speculation (25.6 vs 21.7 ms/tok, both measured before the idle re-run), which is why `main` is the no-speculation choice.

## Which algorithm

Same base model, same calibration pool, same harness, same box (RTX 5090, vLLM 0.27.1); all rows with INT8 `lm_head`:

| Algorithm (tool) | Calibration | MTP off TTFT / turn / ms·tok | K=2 accept | NLL mean / median | Verbatim / first-sentence match with BF16 (of 18) |
|---|---|---|---|---|---|
| **AutoRound** 0.14.2, 200 iters: `main` | 384 × 2048 | 112 / 293 / 10.6 | 66.6 % | **0.325 / 0.320** | **8 / 11** |
| **AWQ** (llm-compressor 0.13.0): `awq-llmcompressor` | 256 × 2048 | 114 / 307 / 10.6 | 64.7 % | 0.347 / 0.322 | 4 / 7 |
| **GPTQ** (llm-compressor 0.13.0, `actorder=weight`): `gptq-llmcompressor` | 384 × 2048 | 112 / 294 / 10.6 | 67.0 % | 0.328 / 0.342 | 5 / 8 |

Serving speed is identical (same tensors, same kernel). Fidelity to BF16 is best for AutoRound, GPTQ close on NLL, AWQ slightly behind (a few prompts raise its mean; its median is in line; llm-compressor's AWQ mappings leave `o_proj` / `linear_attn.out_proj` unscaled). Treat the ordering as indicative, 18 prompts, coarse verbatim counts, flat MTP acceptance, not as a benchmark. (This box reads \~30 ms higher TTFT than the one used in the table above for the same `main` weights; compare within one table.)

## What is quantized, what is not

| Module | Precision | Why |
|---|---|---|
| `self_attn.{q,k,v,o}_proj` (16 layers), `mlp.{gate,up,down}_proj` (64), `linear_attn.in_proj_qkv / in_proj_z / out_proj` (48) | INT4 sym g128 | bulk of the bytes |
| `linear_attn.in_proj_a`, `in_proj_b` (48-wide gates) | BF16 | 4-bit hurts the recurrence, saves nothing, breaks Marlin under TP |
| Vision tower `model.visual.*` | BF16 | keeps image/video quality |
| **MTP head `mtp.*`** (15 tensors) | BF16, `re:.*mtp.*` in `ignore` | the draft must match the target closely or MTP never accepts |
| `lm_head` | INT8 sym g128 (`main`), INT4 (`lm-head-int4`) | 1.27 B params read every token; INT8 is lossless on our metrics |
| embeddings, norms, `conv1d` | BF16 | not on the decode-bandwidth path |

Sizes: `main` 18.2 GB · `lm-head-int4` 17.6 GB · `mtp-int4-autoround` 19.0 GB · branches 18.2 GB. A 24 GB card serves text with a modest KV budget (`--language-model-only` frees another 0.9 GB); 32 GB+ is comfortable at 12 k context; add about 4 GB for the BF16 DFlash2 draft, 2.4 GB for the INT8 one.

## Calibration & reproduce

530 sequences ≥ 2048 tokens rendered through the model's chat template with thinking off: \~60 % multi-turn conversational transcripts (assistant-persona system prompt + short colloquial turns, predominantly Chinese), \~20 % the same turns without a system prompt, \~20 % ShareGPT-GPT4 / UltraChat (zh/en). No benchmark test sets; the transcripts are not distributed and nothing from them is embedded in the weights. If your traffic differs (code, long reasoning), re-run the recipe on your own data; that is what moved MTP acceptance from 58 % to 73 % here.

```bash
# AutoRound (main)
auto-round --model Qwen/Qwen3.8-27B --scheme W4A16 --bits 4 --group_size 128 \
  --iters 200 --nsamples 384 --seqlen 2048 --batch_size 4 \
  --dataset "/path/calib.jsonl:apply_chat_template=false" \
  --ignore_layers "model.visual,linear_attn.in_proj_a,linear_attn.in_proj_b,re:.*mtp.*" \
  --format llm_compressor --enable_torch_compile --output_dir out/      # 1 h 14 min on one RTX 6000 Ada
```
```python
# AWQ / GPTQ (llm-compressor 0.13.0)
ignore = ["re:.*visual.*", "lm_head", r"re:.*linear_attn\.in_proj_a$", r"re:.*linear_attn\.in_proj_b$"]
recipe = [AWQModifier(duo_scaling=True), QuantizationModifier(ignore=ignore, scheme="W4A16", targets=["Linear"])]   # AWQ
# recipe = [GPTQModifier(ignore=ignore, scheme="W4A16", targets=["Linear"], dampening_frac=0.01, actorder="weight")]  # GPTQ
oneshot(model=model, processor=tokenizer, dataset=ds, recipe=recipe, max_seq_length=2048, num_calibration_samples=256)
model.save_pretrained(out, save_compressed=True, max_shard_size="4GB")
```
Three things the exporters do not do for you: (1) keep the 15 BF16 `mtp.*` tensors in the checkpoint **and** add `re:.*mtp.*` to `quantization_config.ignore`; without both, vLLM's drafter loads garbage and MTP runs at 0 %, or the load fails outright ([vllm#58807](https://github.com/vllm-project/vllm/issues/58807)); (2) copy the upstream tokenizer files back verbatim, because `save_pretrained` in transformers 5.14/5.15 re-serialises `tokenizer.json` with an older pre-tokenizer regex (drops `\p{M}`, changes tokenization of scripts with combining marks); (3) the INT8 `lm_head` is applied post-hoc (RTN g128 via `compressed_tensors`, a `group_1` scheme targeting `re:.*lm_head$`) because AutoRound's `--quant_lm_head` needs the 248 k-vocab logits in memory.

## Gotchas

- **YaRN**: a dict `--hf-overrides` for `rope_scaling` / `rope_parameters` does not reach the MTP draft, and acceptance falls to 0 % past the native window ([vllm#37435](https://github.com/vllm-project/vllm/issues/37435), [vllm#58080](https://github.com/vllm-project/vllm/issues/58080)); extend the context in a local `config.json` or keep MTP off past 32 k.
- **DFlash drafts and context length**: a draft with a shorter `max_position_embeddings` than the target reads out of bounds on vLLM 0.32 and older ([vllm#57941](https://github.com/vllm-project/vllm/issues/57941)); 0.33 raises the draft's limit to the target's context, or lowers the served context to the draft's with a warning ([vllm#63251](https://github.com/vllm-project/vllm/pull/63251)); the z-lab draft carries 262144 like the target.
- **Quantized DFlash drafts**: vLLM builds the draft's context-KV projection from the raw `qkv_proj.weight`, so quantized q/k/v fail to load on 0.29 and 0.30 ([vllm#51581](https://github.com/vllm-project/vllm/issues/51581)); keep q/k/v in BF16 there, as the INT8 draft above does. 0.31.0 runs a quantized `qkv_proj` through its quant method, and fully quantized drafts (INT8 or INT4 q/k/v) load with unchanged acceptance (2.75 per step at K=7); 0.32.0 does the same (2.75 to 2.78). 0.33.0 loads them too, but they accepted 2.66 (INT8) and 2.62 (INT4) per step in our run while the q/k/v-BF16 drafts held 2.74 / 2.76, so prefer the latter.
- **Skipping the vision tower**: `--limit-mm-per-prompt` zeros are left out of the torch.compile cache key (still open at 0.33.0), so a text-only start can load a graph compiled with the tower and fail with `'NoneType' object has no attribute 'size'` ([vllm#50891](https://github.com/vllm-project/vllm/issues/50891)); `--language-model-only` is in the key and works.
- **Adaptive verification** (0.30 to 0.33) is rejected on the GDN attention backend.

## Revisions

| Revision | Content | Load |
|---|---|---|
| `main` | INT4 body (AutoRound) + **INT8 lm_head**, MTP head BF16 | compressed-tensors, zero-config (vLLM 0.24 through 0.33 verified) |
| `lm-head-int4` | INT4 body (AutoRound) + **INT4 lm_head** (fastest decode) | compressed-tensors |
| `mtp-int4-autoround` | INT4 body (AutoRound) + **INT4 MTP head** (`mtp.fc` BF16), lm_head BF16 | `auto_round:auto_gptq` packing, vLLM `inc` backend, no flag; the revision to use for speculative decoding in SGLang (BF16 `lm_head`, see [SGLang](#sglang)), and on vLLM 0.33 the one that takes `draft_lm_head_quantization` (see [Latency](#latency-measured)) |
| `awq-llmcompressor` | INT4 body (**AWQ**) + INT8 lm_head | compressed-tensors |
| `gptq-llmcompressor` | INT4 body (**GPTQ**) + INT8 lm_head | compressed-tensors |

`vllm serve Twu31/Qwen3.8-27B-AWQ-INT4-MTP-LowLatency --revision lm-head-int4 ...`. All revisions carry upstream-identical tokenizer/config files (earlier uploads had a transformers-resaved `tokenizer.json`).

## Limitations
- Built and measured for short-reply, thinking-off, single/few-stream interactive serving; long chain-of-thought and > 32 k-context behaviour were not evaluated beyond the base model's claims. No academic benchmarks were run; the fidelity numbers above are relative to the BF16 base on conversational prompts.
- On vLLM 0.29 and older MTP roughly doubles per-turn latency at 5 concurrent streams (0.30 fixes this); DFlash2 is single-stream only. FP8 KV cache is not worth it here (only 16 of 64 layers keep KV, and it makes the cache block larger, not smaller).
- vLLM 0.32 dropped Python 3.10, and 0.33 dropped the CUDA 12.9 builds. vLLM 0.27.1 on Python 3.10 needs `from __future__ import annotations` at the top of `flashinfer/comm/fd_exchange.py`; fresh machines need `g++` and `python3-dev` for the JIT kernels.

## Acknowledgements
Qwen team for Qwen3.8; z-lab / Inco for the DFlash2 draft; Intel for AutoRound; the vLLM / llm-compressor / compressed-tensors maintainers.
