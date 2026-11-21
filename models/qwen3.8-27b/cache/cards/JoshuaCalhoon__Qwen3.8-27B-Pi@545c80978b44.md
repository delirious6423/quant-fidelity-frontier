---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- gptq
- 4-bit
- qwen
- '3.8'
- agentic
---

# Qwen3.8-27B-Pi, MLX mixed-precision GPTQ (b46)

A 4/6/8-bit MLX quant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the `pi` harness LoRA adapter from [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) **merged in**, GPTQ-calibrated on real multi-turn agentic coding sessions. ~15 GB. The vision tower is included and left at bf16.

## Recipe

- Calibration corpus: 984 sequences / 583K tokens, 85% code / 15% prose, drawn from real software projects' agent sessions. Holdout sequences from non-calibrated projects.
- Per-tensor bit allocation derived from a streaming sensitivity pass (block-ref × local-error scores) over the merged bf16 model on the calibration corpus. 
- Base: affine 4-bit / group 128. Allocator upgrades where the scores demanded: finer groups (4:64 / 4:32) on `o_proj` / `out_proj`, plus:
  - `lm_head` 6-bit / g64
  - full-attention `k_proj` + `v_proj` 8-bit / g64 (16 layers)
  - linear-attention `in_proj_a` + `in_proj_b` 8-bit / g64 (48 layers)
- GPTQ (Hessian-weighted error feedback, MLX-affine packing) applied layer-streamed on the same corpus.
- Target 4.5 bpw (param-weighted, group overhead included). Odd bit-widths avoided in the base: on Splash, wide 3-/5-bit configs cost decode-cycle time (+12–26%); a single 6-bit `lm_head` measured free (75.3 vs 75.2 ms/cycle head-to-head).

## Diagnostic results

Measured against the bf16 merged base on a held-out slice (8 sequences / 5,230 tokens from
projects excluded from calibration; KL over bf16 top-64 + tail bucket):

| | this quant | JoshuaCalhoon/Qwen3.8-27B-MLX-4bit-GPTQ (g4p-v2) |
|---|---|---|
| KL vs bf16 (lower better) | **0.01865** | 0.02336 |
| Top-1 agreement with bf16 | 0.9444 | 0.9426 |

≈ 20% lower KL at the same size.

## Real-world results
Setup: 
- Hardware: M5 MacBook Pro (15-core CPU, 16-core GPU, 48GB unified memory), 140W PD, aggressive fan profile (thanks to the [Vorssaint](https://github.com/vorssaint/vorssaint-utils) project), vertical dock (so no risk of inhaling its own exhaust). So best-case power and thermals for this hardware. 
- Software: Splash inference server, usually 64-256K context depending on task, BF16 KV Cache (occasionally INT8 if RAM constrained). When thinking, usually on low or medium (I find XHigh is quite bad on Qwen, either failing or just taking forever for the same results).

Some notes: I recommend setting `temperature` to 0.7. I find it produces better, more consistent results than the stock 1.0, when working as a coding agent/assistant. I don’t use it for anything else, so maybe the stock 1.0 is better elsewhere. Since you’ll be on a Mac to run an MLX package anyways, I highly recommend using Splash, it is the fastest inference server I have found so far. Quicker than both MTPLX and oMLX in my experience. Much faster than Ollama.

### Thinking:
A real agent session, under the Pi harness: 101 requests, 11.6M input tokens, 82K output, peaked at 175K context. **101/101 requests, zero malformed tool calls, zero length-truncations**, 99.1% aggregate prefix-cache hit. It was running a multi-hour version-sweep → release flow on my tweaked Firefox/Librewolf build (M5 + M1 builds: signed, notarized, stapled, smoke-tested, DMG) and read screenshots mid-task, with mid-run self-correction, at 100K+ context throughout. Ran at medium thinking, temperature 0.7. 

Decode by KV depth:
| KV depth | Decode tok/s | Draft accept | ITL p50 |
|---|---|---|---|
| ~1–46K | 49–50 | 0.19–0.28 | 25 ms |
| ~80K | 44 | 0.31 | 29 ms |
| ~112K | 44 | 0.35 | 28 ms |
| ~151K | 43 | 0.37 | 34 ms |

Decode decays only ~13% from 1K to 151K (GDN architecture: 48 of 64 layers hold constant-size state). Prefill ~321 tok/s at 175K with cache hit at its best (98%). Steady throughput is set by cache hits, not depth: effective gen rate held 38–39 tok/s from 50K to 150K+. Draft acceptance tracks task content (research prose ~0.19, code/build flows ~0.37), not quant quality.

TLDR: No visible degradation in work/output quality when in thinking mode. 

### Non-thinking:
Have not actually run a session yet. This is the point of the LoRA adapter, to reach the same result faster by not burning reasoning tokens. I’ll update with numbers here once I have them. 

## Usage

Tested with the Splash engine, which pairs the DFlash2 draft automatically:

```
splash serve --model JoshuaCalhoon/Qwen3.8-27B-Pi
```

`mlx_lm.load` loads the text model. Other MLX runtimes have not been tested.

Sampling defaults come from the base model's `generation_config.json` (temperature 1.0,
top_p 0.95, top_k 20). Reasoning efforts: `low` / `medium` / `xhigh` (default `low`).
**Note:** with thinking on, budget `max_tokens` generously — a small cap can be consumed
entirely by reasoning tokens, returning empty content; use `reasoning_effort: "none"` for
short deterministic replies.

## License

Apache 2.0, inherited from Qwen/Qwen3.8-27B. A copy of the license is in `LICENSE`.
