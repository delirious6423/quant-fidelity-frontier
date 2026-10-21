---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
library_name: vllm
tags:
- qwen3.8
- nvfp4
- modelopt
- quantized
- fp4
- blackwell
- rtx-5090
- dgx-spark
---

# Qwen3.8-27B — NVFP4, linear attention included (the small one)

**TL;DR** — Same model, same modelopt NVFP4 method as the other public checkpoints. Two choices: the 48 **Gated-DeltaNet (linear attention) layers are in NVFP4** — as in a handful of community uploads since 2026-08-14, unlike the three most-downloaded checkpoints which keep them bf16/FP8 — **and `lm_head` is FP8**, which as of 2026-08-16 makes this the smallest *complete* (vision + MTP) NVFP4 checkpoint of Qwen3.8-27B on the Hub (19.3 GB on disk; the GDN-in-NVFP4 uploads with a bf16 `lm_head` are ~20.6 GB). Result on a 32 GB Blackwell card (RTX 5090): **16.5 GiB of weights loaded (text-only) instead of 18.9–24.3** for unsloth/RadixArk/Inferact, i.e. **+45 % to +67 % KV-cache tokens** at identical settings, **+10–20 % decode speed**, and quality within the noise of the unsloth checkpoint on gsm8k, MMLU-Pro, vision and an agentic task (−0.8 and −1.4 pt on the two larger evals = 10 and 5 items, below what these tests resolve — but consistently on the low side, read the caveat before you decide). MTP draft head and the vision tower are kept (bf16). Calibration tokenizer trap avoided (see *Pitfalls*).

Produced 2026-08-16 with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) 0.45.0 (`NVFP4_DEFAULT_CFG` + FP8 `lm_head`), calibration `abisee/cnn_dailymail` 512 samples × 512 tokens, exported in the HF `modelopt` format (`quant_algo: MIXED_PRECISION`), loaded by vLLM as `quantization=modelopt_mixed`.

## Why this checkpoint exists

The three most-downloaded NVFP4 checkpoints of Qwen3.8-27B leave the linear-attention projections out of 4-bit:

| Checkpoint | Linear attention (`linear_attn.in_proj_qkv / in_proj_z / out_proj`) | On disk | **Loaded weights** (vLLM, text-only) |
|---|---|---|---|
| Inferact/Qwen3.8-27B-NVFP4 | in_proj **bf16**, out_proj NVFP4 | 25.4 GB | 24.3 GiB |
| unsloth/Qwen3.8-27B-NVFP4 | **bf16** | 23.4 GB | 20.3 GiB |
| RadixArk/Qwen3.8-27B-NVFP4 | **FP8** | 21.9 GB | 18.9 GiB |
| **this one** | **NVFP4** (W4A4, block 16) | 18.5 GB | **16.5 GiB** (+0.8 vision +0.85 MTP when loaded) |

**Correction (2026-08-16, same day):** the first version of this card said *every* first-wave checkpoint kept linear attention out of 4-bit. Wrong — I had only compared against those three. Several community uploads since 2026-08-14 already quantize `in_proj_qkv / in_proj_z / out_proj` to NVFP4 with modelopt's default config, `lm_head` left in bf16 (~20.6 GB on disk): [gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) (built for the 5090, 80 tok/s, KV pool figures on its card), [vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt](https://huggingface.co/vroomfondel/Qwen3.8-27B-NVFP4-ModelOpt) (gsm8k 97.4 % on 1 319), [kristianpaul/Qwen3.8-27B-NVFP4](https://huggingface.co/kristianpaul/Qwen3.8-27B-NVFP4), [dbrasdasilva/Qwen3.8-27B-Text-NVFP4-MTP](https://huggingface.co/dbrasdasilva/Qwen3.8-27B-Text-NVFP4-MTP) (text-only), and others (verified from the safetensors headers of every `*Qwen3.8-27B*FP4*` repo on the Hub, 50 of them). Nothing here is a new idea. What this repo adds: the FP8 `lm_head` (−1.3 GB, the difference between ~20.6 and 19.3 GB), the MTP head kept and its acceptance measured, the vision tower kept, and a side-by-side measurement against unsloth on one harness.

Those 48 layers hold ~5.5 B parameters. In bf16 that is 11 GB, in FP8 5.5 GB, in NVFP4 ~3.1 GB — on a 32 GB card the difference is the KV cache. What is still bf16 here, on purpose: `embed_tokens`, `linear_attn.conv1d`, the tiny `in_proj_a` / `in_proj_b` gates, all norms, the vision tower and the MTP head; `lm_head` is FP8 per-tensor — a conservative middle ground; NVIDIA's own Qwen3.6-35B-A3B checkpoint goes further and puts it in weight-only NVFP4 (`W4A16_NVFP4`).

This is not a new idea: the community modelopt NVFP4 checkpoint of Qwen3.6-27B (same hybrid GDN architecture) that we have been serving in production since June 2026 quantizes every `Linear` layer to NVFP4 — linear attention included, only `conv1d`, `lm_head` and the embeddings excluded (modelopt 0.43 `ignore` list) — without visible degradation. This repo applies the same choice to Qwen3.8-27B, and additionally takes `lm_head` down to FP8 (per-tensor).

## Measured on one RTX 5090 at a 460 W power limit (vLLM 0.26.1 dev, KV cache NVFP4, `--max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.92`, no MTP unless stated)

| | this checkpoint | unsloth NVFP4 | RadixArk NVFP4 |
|---|---|---|---|
| Loaded weights (text-only) | **16.5 GiB** | 20.3 GiB | 18.9 GiB |
| KV-cache pool at those settings | **434 K – 498 K tokens** (cold / warm compile cache) | 299 K | 308 K |
| Decode, single stream, 1200 tokens | **78–82 tok/s** | ~66 tok/s | 73 tok/s |
| Decode with MTP (n=3), single stream | **101–106 tok/s** (acceptance 0.67/position, mean length 2.2–2.3) | — | — |
| gsm8k, 200 items, thinking off, T=0 | **97.5 %** (195/200) | 97.5 % (195/200, same 5 misses) | (author's card: 97.3 % on 1 319, thinking on) |
| gsm8k, 80 items, thinking on, greedy | **97.5 %** | 96.25 % | — |
| gsm8k, full 1 319, thinking on, **greedy**, cap 4 096 tokens | **95.9 %** (1 265/1 319; 31 hit the cap) | 96.7 % (1 275/1 319; 20 hit the cap) | (author's card: 97.3 %) |
| gsm8k, full 1 319, thinking on, **official sampling** (T 1.0 / top_p 0.95 / top_k 20), cap 4 096 | **95.5 %** (1 260/1 319; 36 hit the cap; 515 output tokens avg) | 96.4 % (1 272/1 319; 26 hit the cap; 476 avg) | — |
| MMLU-Pro, 350 items (25/category, seeded), thinking on, **greedy**, cap 6 144 | **72.9 %** (255/350; 56 hit the cap) | 74.3 % (260/350; 56 hit the cap) | — |
| Long context, RULER-lite (thinking off, greedy; 16 items × 3 tasks × 8K / 32K / 64K / 120K, identical prompts) — keyed needle · 10 most-frequent words · 5-name variable-tracking chain | **needle 64/64 · words 99.5 % · variable tracking 50/64** (per length 13/15/12/**10** of 16) | needle 64/64 · words 99.8 % · variable tracking 53/64 (14/14/12/**13**) | — |
| Agentic run (Codex CLI, 3-fix security task, blind judge /10) | **complete, 11 min, 9/10** | complete, 12.9 min, 9/10 | — |
| Vision (4 Wikimedia images: landscape, dice, STOP sign, Mona Lisa) | all four described correctly, text read, Mona Lisa named | same on the 3 images tested | — |

**Sampling caveat.** All evals above are greedy (T=0) so that the two checkpoints see identical prompts and no sampling noise — the right instrument for an A/B on weights, but Qwen's own guidance for thinking mode is T=1.0 / top_p 0.95 / top_k 20 (shipped in this repo's `generation_config.json`) and warns that greedy decoding can loop. We re-ran the full gsm8k with the official sampling to check whether the greedy cap-hits were loops: they were not — both checkpoints score within 0.4 pt of their greedy run, and the gap between them is the same (−0.9 pt vs −0.8). The cap-hits are genuinely long reasoning, and this checkpoint reasons ~8 % longer than unsloth's on gsm8k.

Reading the two larger evals honestly: on gsm8k both checkpoints score **98.2 % of the answers they finish** — the 0.8 pt gap is 11 more answers hitting the 4 096-token cap, and this checkpoint does think ~3 % longer (485 vs 468 output tokens on gsm8k, 1 982 vs 1 935 on MMLU-Pro). On MMLU-Pro the gap is 5 items out of 350 (binomial SE ≈ 2.4 pt), same 56 truncations; per category it wins some (other 19/25 vs 15, psychology 22 vs 21, philosophy 17 vs 16) and loses some (engineering 12 vs 15, history 14 vs 16). Numbers are ours, on our hardware, at a 460 W power limit (measured: raising it to 575 W gives +12 % prefill and 0 % single-stream decode); treat them as a first datapoint, not a leaderboard.

The published caveat that *linear attention is less robust to 4-bit* (kaitchup on Qwen3.6; RedHatAI 35B discussion; huginnfork: "quantising the recurrent projections measurably degrades this architecture") is why several checkpoints kept it out. The long-context row above is the one test here that targets the recurrent state: pure retrieval is perfect for both up to 120K, and the only movement is on the multi-hop variable-tracking task at 120K (10/16 vs 13/16 — the three items this checkpoint loses are missed links, unsloth's three are decoy names added). Below the resolution of 16 items, but it is the third eval pointing the same way. Everything here is single-turn: if you serve long multi-turn agents, benchmark your own workload before switching.

## Serve

```bash
vllm serve Mantrah/Qwen3.8-27B-NVFP4-GDN \
  --kv-cache-dtype fp8 \            # or nvfp4 if your build supports it (ours: +70 % pool)
  --max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'   # optional, see note
```
Do **not** pass `--quantization`: auto-detection resolves `modelopt_mixed` (NVFP4 GEMM via FlashInfer Cutlass + FP8 GEMM). Add `--language-model-only` for text-only serving (saves ~0.8 GiB). Requires Blackwell (SM100/SM120) for the native W4A4 kernels; on Hopper vLLM falls back to Marlin emulation.

Thinking: the chat template defaults to `reasoning_effort: xhigh` **and** `preserve_thinking: true`. If your client (Codex CLI does) sends no `reasoning` field you get the heaviest configuration; set `--default-chat-template-kwargs '{"preserve_thinking": false}'` (or `{"enable_thinking": false}`) server-side to control it.

MTP note: on hybrid GDN models the draft verification cost grows with context; we measured MTP as a net win below ~12 K tokens of context and a net loss beyond (Qwen3.6-27B, same architecture). Use it for short-context serving, drop it for long-context agents.

## Pitfalls we hit so you don't

- **Tokenizer truncation.** Calibration calls `tokenizer(text, truncation=True, max_length=512)`, which mutates the tokenizer; saving it afterwards persists `"truncation": {"max_length": 512}` into `tokenizer.json` and silently truncates every prompt at serve time. This is the same class of bug the launch-day unsloth checkpoint shipped with (`max_length: 2048`). This repo copies the original tokenizer files and asserts `truncation == null` before export.
- **MTP head.** The HF `Qwen3_5ForConditionalGeneration` class does not instantiate the MTP module, so the export drops it. The 15 bf16 MTP tensors were re-added from the source checkpoint (`model_mtp.safetensors`, indexed) — vLLM resolves `Qwen3_5MTP` and the acceptance rate above confirms they work.
- **Model behaviour, not quantization** (same on unsloth's checkpoint, not on Qwen3.6-27B): when asked to write a long script *inline inside a PowerShell here-string in a shell tool call*, Qwen3.8 often keeps generating and never closes the tool call (2/7 valid at ~6 K chars requested). The same content through a structured `write_file(path, content)` tool: 6/6 valid. If you drive it with an agent, prefer patch/file tools over inline scripts.

## Reproduce

`quant38.py` in this repo (modelopt 0.45.0, torch 2.11, transformers 5.12): loads bf16 with `device_map=auto` (fits on one 32 GB card + ~30 GB of CPU RAM), calibrates, exports. Dry-run (`--dry-run`, 4 samples): 2 min. Full pass: **18 minutes on one RTX 5090** with half the layers offloaded to CPU; ~5 min on two. `add_mtp.py` then copies the 15 `mtp.*` tensors from the bf16 source into `model_mtp.safetensors` and registers them in the index.

## Files

| File | What |
|---|---|
| `model-0000{1,2}-of-00002.safetensors` | the quantized model (NVFP4 weights + FP8 lm_head + bf16 remainder), 18.5 GB |
| `model_mtp.safetensors` | the MTP draft head, bf16, 0.85 GB — only loaded with `--speculative-config` |
| `hf_quant_config.json` | modelopt per-layer quantization map (`quant_algo: MIXED_PRECISION`) |
| `quant38.py`, `add_mtp.py` | the two scripts that produced this repo |
| tokenizer / preprocessor / chat template | copied verbatim from `Qwen/Qwen3.8-27B` |

## License

Apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). This is a community quantization, not affiliated with Qwen or NVIDIA.
