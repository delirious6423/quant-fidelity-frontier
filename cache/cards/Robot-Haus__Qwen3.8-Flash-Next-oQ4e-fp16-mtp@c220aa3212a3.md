---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- oq
- quantized
- qwen4-exp
- moe
---

# evsinlb/Qwen3.8-Flash-Next-oQ4e-fp16-mtp

4-bit MLX quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — the `qwen4_exp` architecture: a 512-expert MoE (top-10 routing, 48 layers, hybrid linear/full attention at a 4:1 interval), a **51.2B-parameter PLE n-gram embedding table** (20M rows x 2560), a 1-layer Lightning MTP head, native vision, and a 262,144 context. Made with [oQ](https://github.com/jundot/omlx) (oMLX 0.6.4) mixed-precision imatrix quantization, fp16 non-quant tensors for M1/M2-generation Apple Silicon.

** -original quantization from user : evsinlb/ found to he the most stable and consistent proper fp16 q4 **



**99.82 GiB on disk / 21 shards.** MTP head preserved, vision included. This is a **long-context ingestion** build: prefill is its strength (525 tok/s at 8K on an M2 Ultra, ahead of a 27B dense model on the same machine), with decode at \~23-25 tok/s on oMLX 0.7.0.dev2. It suits batch summarization, extraction, and one-shot long-document work, and is workable for interactive use at shallower contexts.

## Quantization details

- **Model type**: `qwen4_exp` (`Qwen4ExpForConditionalGeneration`)
- **Bits**: 4-bit affine base, group size 64, with per-tensor 5-bit boosts on sensitive tensors (hyper-connection mixers, block inject weights)
- **PLE n-gram table**: quantized at **5.00 bits/param** — 29.82 GiB of the total. oQ's PLE policy is base + 1 bit.
- **Enhanced quantization (oQe)**: imatrix-calibrated, 1024 samples x 512 tokens, `oqe_code_multilingual`, 937 entries. Coverage: **75,243 / 75,264 expert slots active (99.97%)**; 21 slots the calibration corpus never routed to fall back to standard oQ. 3 tensors without imatrix entries (`lm_head`, two `hyper_connection_mixer` weights), 0 mismatched.
- **MTP weights preserved** (`mtp.*` tensors + config) — Lightning MTP works after quantization
- **Non-quant weight dtype**: float16 — about 20% faster prefill on M1/M2 Apple Silicon, which have native fp16 but not native bf16. On M3/M4/M5 a bf16 build is the safer choice.
- **Vision components included** (not a text-only strip)
- **Format**: MLX safetensors

## Memory and PLE runtime modes

| | size |
|---|---:|
| PLE n-gram table | 29.82 GiB |
| everything else (always resident) | 70.00 GiB |
| total on disk | 99.82 GiB |

The PLE table can either live in RAM or be memory-mapped and read by row. oMLX picks automatically (`OMLX_QWEN4_PLE_MODE=auto`, or the "SSD N-gram Offload" toggle), and there are two thresholds worth understanding:

1. **Mode selection** — `mmap` if the checkpoint exceeds **70% of physical RAM**, otherwise `resident`. At 99.82 GiB that means mmap on anything below \~143 GiB, resident above it.
2. **Shard fusion** — resident PLE is fused into a single packed embedding only when physical RAM is **at least 192 GiB**. Unfused, the lookup synchronizes token IDs to the host to select among 128 shard buffers; fused, it becomes an ordinary device-side gather.

Both thresholds read physical RAM, not free memory, so neither can be changed by unloading other models.

On 0.7.0.dev2 the mmap path gains **PLE gather-ahead**, which pipelines the row lookups behind compute (`PLE gather-ahead active: next prefill chunk rows are gathered during the current chunk`). All figures below are from that path.

Hosts at or above 192 GiB take the fused resident path instead — the configuration this architecture was designed around, and the one upstream's benchmarks (a 512 GiB M3 Ultra) were run on. It is untested here.

**KV cache is cheap on this architecture** and is unaffected by quantization level. Only 12 of the 48 layers are full-attention (`full_attention_interval: 4`); the other 36 use linear attention with a fixed recurrent state. With `num_key_value_heads: 2` and `head_dim: 256` at fp16 that is **24 KiB/token** — 0.75 GiB at 32K, 3.0 GiB at 131K, **6.0 GiB at the full 262,144**. Context here is bounded by prefill *time*, not by memory.

## Measured throughput

128 GiB M2 Ultra Mac Studio, mmap PLE mode, Lightning MTP on, thinking on, warm — see the caveats below.

**oMLX 0.7.0.dev2:**

| Prompt depth | prefill tok/s | decode tok/s | peak |
|---|---:|---:|---:|
| 1,024 | 465.5 | **25.2** | 73.7 GB |
| 4,096 | 430.6 | **25.0** | 81.1 GB |
| 8,192 | **525.0** | **23.3** | 84.1 GB |
| 51,394 (real document) | \~480 | **25.0** | — |

Continuous batching at pp1024: **50.7 tok/s aggregate at 4x** (2.0x scaling). Peak memory growth with depth is prefill scratch, not KV.

**Decode is essentially flat from 1K to 51K**, which was not true on 0.6.4 — there the same model fell from 12.5 tok/s at 8K to 8.8 at 32K and **4.0 on a real 51K document**. dev2 logs `PLE gather-ahead active: next prefill chunk rows are gathered during the current chunk`, overlapping the n-gram lookups with compute, and the depth penalty disappears on both axes. A 51K-token document summarised end to end (prefill + 1,709 generated tokens) took **175 s**, against 617 s on 0.6.4 at the same settings.

The prefix cache makes follow-up turns cheap: a second question against the same document restored **49,152 tokens from the paged cache with a 2.5 ms lookup**, leaving only 3,924 tokens to prefill.

**Version note.** On 0.6.4 the same warm ladder gave 488.5 prefill / 12.5 decode at 8K. Prefill moved little; **decode roughly doubled**, and TPOT became far more stable (40–44 ms across 1K–8K, where it previously swung with MTP parking). The 192 GiB PLE fusion threshold is unchanged — the gain is in the unfused path itself, so it should apply to any host below that threshold.

Two measurement caveats, both of which cost us hours:

- **Warm the PLE rows before trusting any prefill number.** The same 4,096-token shape measured **187 tok/s cold and 435.8 warm**. Tell-tale: warm prefill *falls* with depth (normal scaling); a run whose prefill *rises* across successive depths is still warming.
- **Screen every decode number for MTP parking.** A parked request drops to standard single-token decode for 128 tokens before probing re-entry, halving throughput. Parking is triggered by a run of consecutive zero-accept cycles, **not** by poor average acceptance — our parked requests were running at 69% and 82%. Check `server.log` for `parked` before believing a TG figure. This was pronounced on 0.6.4 (nine parks in one 32-minute generation, 73.6% acceptance, 2.32 tokens per cycle). On 0.7.0.dev2 a 1,709-token generation at 51K context ran with **no parks at all, 85.7% acceptance and 3.07 tokens per cycle** — most of the decode gain above comes from this.

## Recommended sampling (per the Qwen3.8 family card)

| Mode | temperature | top_p |
|---|---|---|
| Thinking (default) | 1.0 | 0.95 |
| Instruct (non-thinking) | 0.7 | 0.80 |

The template defaults to `reasoning_effort: xhigh`, which injects an explicit instruction to think thoroughly. At this build's decode rate that default is costly: on a 51K-token document, the same request produced **6,697 tokens at `xhigh` versus 1,476 at `low`** — 4.5x fewer, with a summary covering the same ground. For extraction and summarisation work, start at `low`. Note that `medium` renders no instruction at all; only `xhigh` and `low` add text.

Valid values are `xhigh` (default), `medium` and `low`; anything else raises rather than falling back silently. `thinking_budget` is a different mechanism — it never reaches the template, so the model is truncated mid-reasoning rather than adapting its style. Use it as a backstop, not a primary control.

## Fidelity

**This exact build passed the multi-turn agentic battery** (M2 Ultra Mac Studio, Anthropic-shaped tool loop, thinking blocks fed back into history verbatim each turn): **3/3 scenarios completed with correct answers, 0 stalled turns across 9 agentic tool-call turns, 0 errors.** Scenarios required multi-hop tool chaining — following a pointer from one file to another, and computing a total from CSV data via a calculator tool.

Generation quality spot-checks were clean: correct memoized-Fibonacci code with proper base cases and guards, and a sustained 600-token technical explanation that ran to the token cap without degradation.

**Thinking volume is comparable to a dense 27B, not lower.** On the identical battery this build emitted **1,682 reasoning characters** across the three scenarios, against **1,536** for a Qwen3.8-27B run at full reasoning effort — so it does not save you reasoning tokens. The wall-clock difference between them is decode speed rather than deliberation: **209 s here versus 58 s** for the 27B on the same three scenarios. Budget for that if you are substituting this build into a reasoning-heavy role. The battery's scenarios are short, so behaviour on longer reasoning chains is untested.

## Deployment notes

At 70 GiB always-resident this build will evict most of what else you have loaded on a 128 GiB machine, so it works best as a model you bring up for a job rather than keep warm alongside a panel.

Its prefill advantage and its decode deficit pull against each other, so the workload shape decides whether it wins. Against a Qwen3.8-27B on the same machine (231 tok/s prefill at depth / 35 tok/s decode, versus roughly 360 / 12.5 here), the crossover lands near **30:1 prompt-to-generation**: a 100K-token document summarised in 3K tokens favours this build, while anything that generates at length does not. Those ratios are specific to that pairing and that hardware, but the shape generalises — bring it out for reading a great deal and writing a little, and pair it with a smaller dense model for interactive work.
