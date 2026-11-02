---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
language:
  - en
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - flash-next
  - merlin
  - w4a16
  - autoround
  - mtp
  - speculative-decoding
  - fp8-kv
  - long-context
  - vision
  - rtx-3090
  - ampere
  - sm86
---

<div align="center">

<img src="https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin/resolve/main/assets/hero.png" alt="Qwen3.8-Flash-Next W4A16, powered by Merlin: a 4-bit checkpoint, 115 GiB, full 262K context on four 24 GB GPUs. A 924,993-token KV pool in VRAM; 664 tok/s total for 8 agents at once; 197 tok/s decode at 262K; 7,064 tok/s prefill on a 100K prompt." width="100%">

<p>
<a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next"><img src="https://img.shields.io/badge/base-Qwen3.8--Flash--Next-615ced" alt="Base model: Qwen3.8-Flash-Next"></a>
<a href="#the-checkpoint"><img src="https://img.shields.io/badge/quant-W4A16-3987e5" alt="Quant: W4A16"></a>
<a href="#what-it-fits-on"><img src="https://img.shields.io/badge/context-262K-ffb000" alt="Context: 262K"></a>
<a href="#what-it-fits-on"><img src="https://img.shields.io/badge/GPUs-Ampere-76B900" alt="GPUs: NVIDIA Ampere (Ada profiles are in testing or untested)"></a>
<a href="https://github.com/halt95/merlin-engine"><img src="https://img.shields.io/badge/runs_on-Merlin_3.0.1-a78bfa" alt="Runs on Merlin 3.0.1"></a>
<a href="https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin/blob/main/LICENSE"><img src="https://img.shields.io/badge/licence-Qwen_Community_1.0-lightgrey" alt="Licence: Qwen Community 1.0"></a>
</p>

<b><a href="#the-checkpoint">The checkpoint</a> · <a href="#speed">Speed</a> · <a href="#quality">Quality</a> · <a href="#what-it-fits-on">What it fits on</a> · <a href="#use-it">Use it</a> · <a href="#the-base-model">The base model</a> · <a href="https://github.com/halt95/merlin-engine/blob/main/models/flash-next/benchmarks/2026-11-02/BENCH-CARD.md">Bench card</a></b>

</div>

# Qwen3.8-Flash-Next-W4A16-Merlin

A 4-bit build of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Qwen's 125B mixture-of-experts model (6B active per token, with vision), made to run at its full 262,144-token context on four 24 GB GPUs. It is 115 GiB instead of the original 335 GiB: the routed experts are INT4, the 51B-parameter n-gram table is FP8, and attention and the vision tower stay in BF16. It is served by the [Merlin engine](https://github.com/halt95/merlin-engine), a vLLM 0.32.0 fork tuned for Ampere and Ada that keeps the n-gram table in host RAM and the whole KV cache in VRAM. Measured on RTX 3090s; the engine also ships profiles for other Ampere and Ada cards, from RTX 4090s and 48 GB workstation cards to A100s and CMP 170HX (see [What it fits on](#what-it-fits-on)).

<sub>numbers: Merlin 3.0.1 on 4× RTX 3090 w/o NVLINK @ 220W, P2P on, with the gpu-unlock host upgrade, 2026-11-02 bench card (runs of 2026-11-01 to 2026-11-02 on release candidates just before the final commit). KV pool: FP8, all in VRAM; three 262K sessions held at once (3.0.0 run). Agents: eight 16K conversations, then eight simultaneous 512-token follow-ups, thinking off. Decode: one stream, thinking on (effort medium, the server default), exactly 261,888 prompt tokens + 128 out, median of 3. Prefill: one cold 101,875-token prompt. Each figure is also in a table below.</sub>

<a href="https://ko-fi.com/halt95"><img src="https://storage.ko-fi.com/cdn/kofi3.png?v=3" alt="Buy me a coffee on Ko-fi" height="40"></a>

## The checkpoint

| Component | Format |
|---|---|
| Routed experts (512 per layer) | INT4 g128, from [Intel's AutoRound release](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) |
| MTP draft-head experts | INT4 g128, packed in place |
| Gated DeltaNet QKV/Z and output projections | INT8 per channel |
| N-gram (PLE) embedding table, 51B parameters | FP8, kept in host RAM at serve time ([RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)) |
| Attention, norms, embeddings, lm_head, shared expert, routers, QSA indexer, gated-residual weights, DeltaNet a/b and conv, MTP non-expert layers, vision tower | BF16, unchanged |
| KV cache | FP8 E4M3 with calibrated static scales (`qsa_kv_scales_262k.json`) |

**Files in this repository**

| Files | What they hold | Size |
|---|---|---|
| `model-000NN-of-00017.safetensors` (15) | Routed experts, DeltaNet projections, attention, norms, embeddings, shared expert, vision tower | 66.2&nbsp;GiB |
| `model-plefp8-0000N.safetensors` (10) | The n-gram embedding table, FP8; loaded into host RAM at serve time | 47.7&nbsp;GiB |
| `mtp-routed-experts-int4.safetensors` | MTP draft-head experts, INT4 | 1.2&nbsp;GiB |
| `model_extra_tensors.safetensors` | Further tensors listed in the index | 0.2&nbsp;GiB |
| `qsa_kv_scales_262k.json` | Calibrated static scales for the FP8 KV cache | 1.4&nbsp;KiB |
| `model.safetensors.index.json` | Which file holds each tensor | 24.3&nbsp;MiB |
| `chat_template.jinja` | Qwen's own chat template, unchanged (byte-identical to the upstream file). Merlin serves [Froggeric's fixed Qwen3.8 template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v22.5) by default; see Use it | 8.7&nbsp;KiB |
| `config.json`, `quantization_config.json`, tokenizer, processor configs | Model, quantisation, tokenizer and processor settings | 12.3&nbsp;MiB |
| `LICENSE` | Qwen Community License 1.0 | 3.1&nbsp;KiB |

<sub>115.3 GiB in total. There are no shards 00002 and 00016: `model.safetensors.index.json` maps no tensor to them, so nothing is missing.</sub>

## Speed

<img src="https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin/resolve/main/assets/speed.png" alt="Speed on 4x RTX 3090, Merlin 3.0.1. Decode, one stream, thinking on (effort medium): 178 tok/s at 4K, 183 at 131K, 197 at 262K. Cold prefill: 6,628 tok/s at 10K, 7,064 at 100K, 6,803 at 262K. Time to first token: 0.67 s at 4K, 18.5 s at 131K, 38.4 s at 262K. 8 agents at once, total: 664 tok/s with thinking off, 677 with thinking on. The same figures are in the tables below." width="100%">

| Prompt | Decode, one stream | Time to first token |
|---|---|---|
| 4K | 178 tok/s | 0.67 s |
| 32K | 172 tok/s | 4.59 s |
| 131K | 183 tok/s | 18.5 s |
| 262K | 197 tok/s | 38.4 s |

<sub>4× RTX 3090, thinking on at reasoning effort medium (the server default). The needle-prompt ladder (4,046 / 32,566 / 130,386 prompt tokens, every send a cold prefill) and the exact 261,888-token prompt (send 1 cold; sends 2 and 3 reuse the prefix before the needle). Decode: ≤ 256 tokens out (128 at 262K), median of 3. Time to first token: the median of 3 cold sends below 262K, the cold send 1 at 262K. The 262K needle was found in 3 of 3 thinking-off sends (3.0.0 run).</sub>

| Cold prefill, one send | Prefill | Time to first token |
|---|---|---|
| 10,187 tokens | 6,628 tok/s | 1.54 s |
| 101,875 tokens | 7,064 tok/s | 14.4 s |
| 261,888 tokens | 6,803 tok/s | 38.5 s |

<sub>One cold request per prompt, 8 tokens out at 10K and 100K. The 261,888-token row is the 2026-11-01 run's cold thinking-on send, so its time to first token (38.5 s) can differ slightly from the 262K row above (38.4 s). The 101,875-token figure is the hero's prefill number.</sub>

| Agents at once | Total tok/s |
|---|---|
| 2 (3.0.0 run) | 222 |
| 4 (3.0.0 run) | 378 |
| 8, thinking off | 664 |
| 8, thinking on (effort medium) | 677 |

<sub>N 16K conversations, then N simultaneous 512-token follow-ups, total tok/s while all N decode; thinking off unless stated. With thinking on a 512-token follow-up may end inside the reasoning and reasoning tokens count, so that total measures decoded tokens, not finished answers. Overlap windows: 4.55 and 5.22 s for the 3.0.0 rows, 5.90 s (thinking off) and 5.43 s (thinking on) for 3.0.1: a few seconds each, so read these rows loosely.</sub>

| Sessions held at once | Needles found | Re-sent from cache, lowest |
|---|---|---|
| 3 × 262K | 3 / 3 | 99.95 % |
| 6 × 131K | 6 / 6 | 99.91 % |

<sub>N needle prompts (thinking off, greedy) sent at once, then re-sent one by one: every re-send served from the prefix cache means the pool held all N. Peak pool use 100 %, 0 preemptions. 3.0.0 run, 2026-10-20; the KV pool and the sessions it holds are unchanged in 3.0.1.</sub>

<details>
<summary><b>Against the previous release (3.0.0)</b></summary>

| Measure | 3.0.0 | 3.0.1 | Change |
|---|---|---|---|
| Decode, 4K prompt, thinking on | 167.7 | 177.5 | +5.8 % |
| Decode, 131K prompt, thinking on | 170.3 | 182.9 | +7.4 % |
| Decode, 259K prompt, thinking on | 187.4 | 198.0 | +5.7 % |
| Decode, exactly 261,888 tokens, thinking on | 183.6 | 197.2 | +7.4 % |
| Decode, exactly 261,888 tokens, thinking off | 149.0 | 148.4 | −0.4 % |
| Prefill, 10,187-token prompt | 5,660 | 6,628 | +17.1 % |
| Prefill, 101,875-token prompt | 6,051 | 7,064 | +16.7 % |
| Prefill, 261,888-token prompt | 5,838 | 6,803 | +16.5 % |
| 8 agents, thinking off | 638 | 664 | +4.1 % |

<sub>tok/s. All 3.0.1 rows ran with the gpu-unlock host upgrade on; the 3.0.0 rows did not. The engine wheel is unchanged, and the engine's stock-versus-unlocked A/B on Flash-Next found about 16 % faster prefill with the unlock and decode unchanged. Thinking-on decode also compares different efforts: 3.0.0 ran at `low`, 3.0.1 at `medium` (the server default). The 3.0.1 ladder prompts are 30 tokens shorter than the 3.0.0 ones (4,046 against 4,076 at 4K); the bench card sets them side by side.</sub>

</details>

<details>
<summary><b>How it was measured</b></summary>

- Host: 4× RTX 3090 (24 GB, sm_86) at 220 W, PCIe Gen4 with P2P, no NVLink, driver 610.43.02, with the [Zagitta's gpu-unlock](https://github.com/halt95/merlin-engine/blob/main/docs/merlin/hardware-matrix.md#optional-host-upgrade-gpu-unlock-geforce-ga102) host upgrade on (the 3.0.0 runs did not have it).
- Engine: Merlin 3.0.1 (the engine wheel is the same as in 3.0.0), served by `merlin serve` with the shipped defaults, one fresh boot per layout. The runs were on 2026-11-01 to 2026-11-02, on release candidates just before the final commit; none used the final commit.
- Layout: TP2 × PP2 + EP on four cards, MTP K=3, FP8 KV, BF16 draft lm_head, thinking effort `medium`, the server default (the 3.0.0 numbers were measured at `low`).
- Decode: T=0, a needle prompt at depth; every send a cold prefill except sends 2 and 3 at exactly 262K, which reuse the prefix before the needle; 3 sends per cell, median. Thinking-on rows were re-measured on 2026-11-02 on a fresh boot of a later release candidate; thinking-off rows, prefill and the thinking-off agent rows are from the 2026-11-01 runs, some on an earlier candidate. tok/s = tokens after the first streamed event / time from first to last event.
- Streams are client-timed. Full tables, thinking-off rows and ITL: the [bench card](https://github.com/halt95/merlin-engine/blob/main/models/flash-next/benchmarks/2026-11-02/BENCH-CARD.md).

</details>

## Quality

**This checkpoint**, measured on the Merlin engine:

| Check | Result | Setting |
|---|---|---|
| GSM8K-200 | 199 / 200 | 5-shot, thinking off |
| GSM8K-200, thinking on | 197 / 200 on four cards | 5-shot, effort medium, none cut at the 8,192-token cap |
| Needles at 131K | 10 / 10 | single-occurrence needles, an image in the prompt, thinking off, greedy, cold cache |
| Needles at 262K | 23 / 23 | as above |
| Needle at exactly 261,888 tokens | 3 / 3 | thinking off |
| Image, 4096 × 2048 (8,388,608 px) | 1 / 1 · 2 / 2 at once · 1 / 1 with four in one prompt | thinking off; names both colours of a two-colour image (with four, of the last one) |

<sub>Thinking-on GSM8K-200: Merlin 3.0.1, four cards, the same 200 problems and scorer as the thinking-off row, temperature 0. Every other row is a 3.0.0 run, not re-run for 3.0.1. GSM8K and images: the Merlin 3.0.0 release wheel, 2026-10-20. Needle panel: a pre-release engine build, 2026-10-17. These are checks of this checkpoint, not a matched BF16 comparison: no BF16 run of the same checks was made, so how much quality the 4-bit build keeps is not quantified here.</sub>

**The base model**, as reported by Qwen for the BF16 weights:

| Area | Benchmark | Qwen3.8-Flash-Next (BF16) |
|---|---|---|
| Coding | SWE-bench Pro | 62.5 |
| Coding | DeepSWE 1.1 | 58.7 |
| Coding | LiveCodeBench v6 | 91.9 |
| Agents | Toolathlon Verified (Pass@1) | 73.5 |
| Agents | CoWorkBench | 73.9 |
| Reasoning and instructions | GPQA Diamond | 91.7 |
| Reasoning and instructions | HLE | 35.9 |
| Reasoning and instructions | IFBench | 81.3 |
| Vision | AndroidWorld | 84.5 |
| Vision | RealWorldQA | 88.5 |
| Vision | LVBench | 76.6 |

<sub>A selection from the [upstream card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), which has the full tables and comparisons with other models. These were not re-run on this 4-bit build.</sub>

## What it fits on

| Measured on | Context per request | KV pool | Held at once | Images |
|---|---|---|---|---|
| 4× RTX 3090, 24 GB (🟢 Tested) | 262,144 tokens | 924,993 tokens, FP8, all in VRAM | 3 × 262K or 6 × 131K | 8 MP |

<sub>Measured on the reference host (3.0.0 run, 2026-10-20 bench card): three 262K and six 131K sessions held at once with every re-send served from cache; a 4096 × 2048 image (8,388,608 px) with at least 1,026 MiB free on every card.</sub>

**Other GPUs.** The engine also ships profiles for these; the Status column says how far each has been run:

| GPUs | Arch | Profile | Context fit | Status |
|---|---|---|---|---|
| 4× RTX 3090 Ti, RTX A5000, A10 (24 GB) | Ampere, sm_86 | `fn-quad-3090` | ~3 × 262K (as the 3090) | ⚫ Untested |
| 4× RTX 3090 without P2P | Ampere, sm_86 | `fn-quad-3090-nop2p` | ~3 × 262K | ⚫ Untested |
| 8× RTX 3090 | Ampere, sm_86 | `fn-octo-3090-tp2pp4` | ~17 × 262K | ⚫ Untested |
| 4× RTX 4090 (24 GB) | Ada, sm_89 | `fn-quad-24g-sm89` | ~3 × 262K | 🟡 Testing |
| 4× RTX 4080 Super 32 GB (modified) | Ada, sm_89 | `fn-quad-32g-sm89` | ~10 × 262K | ⚫ Untested |
| 2 / 4× 48 GB: RTX A6000, A40 | Ampere, sm_86 | `fn-2x48g-pp2-sm86`, `fn-4x48g-tp2pp2-sm86` | ~5 / ~25 × 262K | ⚫ Untested |
| 2 / 4× 48 GB: RTX 6000 Ada, L40S | Ada, sm_89 | `fn-2x48g-pp2-sm89`, `fn-4x48g-tp2pp2-sm89` | ~5 / ~25 × 262K | ⚫ Untested |
| 4× A10G, L4 (cloud, 22.5 GiB as reported) | Ampere / Ada | `fn-quad-3090-nop2p`, `fn-quad-24g-sm89` | ~1 × 262K | ⚫ Untested |
| 1 / 2 / 4× A100 80 GB | Ampere, sm_80 | `fn-sm80-80g-x1`, `-x2-tp2`, `-x4-tp2pp2` | ~1 / ~11 / ~28 × 262K | ⚫ Untested |
| 4× A100 40 GB, CMP 170HX 40 GB | Ampere, sm_80 | `fn-sm80-40g-x4-pp4` | ~7 × 262K | ⚫ Untested |
| 3 / 4× CMP 170HX 32 GB | Ampere, sm_80 | `fn-sm80-32g-x3-pp3`, `fn-sm80-32g-x4-pp4` | ~1 / ~4 × 262K | ⚫ Untested |
| 4× A30 (24 GB) | Ampere, sm_80 | `fn-quad-24g-sm80` | ~1 × 262K | ⚫ Untested |
| 2× CMP 170HX 64 GB | Ampere, sm_80 | `fn-sm80-64g-x2-pp2` | ~6 × 262K | 🟡 Testing |
| 4× CMP 170HX 64 GB | Ampere, sm_80 | `fn-sm80-64g-x4-pp4` | 17 × 262K (8 at once) | 🟡 Testing |

<sub>`merlin serve` and `./setup.sh` pick the profile from the GPUs they find. A `~` is the engine's estimate, not a measurement. "(8 at once)": the KV pool holds 17 full 262K conversations, and the server admits 8 requests at a time; the rest queue. On sm_80 cards (A100, A30, CMP 170HX) this checkpoint defaults to a BF16 KV cache, so fewer sessions fit; `KV_DTYPE=fp8` opts into FP8 (untested ⚫). 🟢 measured by us · ⚫ a profile ships, not yet run by us (expected to work). Every profile: the engine's [hardware matrix](https://github.com/halt95/merlin-engine/blob/main/docs/merlin/hardware-matrix.md). 🟡 Testing: full-model runs passed on cloud-hosted cards; not yet qualified against the reference host.</sub>

| Need | Details |
|---|---|
| Host RAM | 72 GiB tested minimum on four cards (startup is refused below about 69 GiB, plus the KV RAM cache size when it's on; with `--ple int6` about 60 GiB (estimated, not boot-tested)). 96 GiB is the reference allocation; the FP8 n-gram table alone takes about 47.7 GiB. |
| Disk | About 115 GiB for the checkpoint, plus the engine environment and caches |
| Engine host | Linux x86-64 (glibc 2.38 or newer), NVIDIA driver R580 or later (CUDA 13), Python 3.13, GCC/G++ 13 or newer, `/dev/shm` of at least 1 GB; GPUs headless with no other CUDA process. `merlin doctor` checks the host. |

## Use it

```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine
./setup.sh --model flash-next
```

One command does the rest: after a small uv and Python bootstrap it checks the GPUs, driver, RAM and disk, shows the engine, package and checkpoint downloads with their sizes before fetching them, installs the prebuilt engine, downloads and verifies this checkpoint, and starts an OpenAI-compatible server on `127.0.0.1:8000`; requests name the model `flash-next-v2` (also `flash-next`). Enter takes the recommended answer at each question; `--yes` takes them all, `--dry-run` shows the plan. Run `./setup.sh` again to start the server, and `git pull && ./setup.sh` to update the engine without downloading the model again.

> [!IMPORTANT]
> This checkpoint needs the Merlin engine; stock vLLM was not tested with it.

<details>
<summary><b>Manual install</b></summary>

```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine
curl -LO https://github.com/halt95/merlin-engine/releases/download/merlin-v3.0.1/vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl
MERLIN_WHEEL="$PWD/vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl" \
  bash release/install-env.sh ./venv

venv/bin/hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin --revision cec53e9f90e5995b81691604ec0b32191debf797 \
  --local-dir ~/models/flash-next
(cd ~/models/flash-next && sha256sum -c "$OLDPWD/models/flash-next/checkpoint.sha256")
venv/bin/merlin serve ~/models/flash-next
```

`MERLIN_WHEEL` installs the prebuilt engine (sm_80, sm_86, sm_89; the same wheel file as in Merlin 3.0.0), checked against `release/merlin-wheel.sha256`; without it the installer compiles the engine instead, about 1.5 hours on 8 cores. `merlin serve` recognises the checkpoint, picks the largest tested layout for the GPUs it finds (or, with none, the largest untested one that fits, labelled as such) and serves on `127.0.0.1:8000`. Add `--dry-run` to print the resolved command without starting anything. The first start compiles kernels.

</details>

**Recommended settings** (Qwen's, for this model):

| Mode | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| Thinking (default) | 1.0 | 0.95 | 20 | 0.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 1.5 |

<sub>Qwen's recommendation for this model, with `min_p=0.0` and `repetition_penalty=1.0` in both modes. A `presence_penalty` between 0 and 2 reduces endless repetition; higher values can mix languages.</sub>

- Turn thinking off for a request with `"chat_template_kwargs": {"enable_thinking": false}`. With thinking on, a short `max_tokens` can end inside the reasoning with empty `content`.
- Reasoning effort: Merlin serves `medium` by default; pick another per request with `"chat_template_kwargs": {"reasoning_effort": "low"}` (`low` / `medium` / `xhigh`).
- Chat template: Merlin serves [Froggeric's fixed Qwen3.8 template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v22.5) by default, not the file in this repository. It accepts system messages after the first turn, so coding agents' per-turn notes stay in place and the prefix cache keeps hitting. `CHAT_TEMPLATE=<path>` serves another template; `CHAT_TEMPLATE=` (empty) serves this repository's own.
- With Qwen's own template (`CHAT_TEMPLATE=`), earlier turns' reasoning is kept by default and `"chat_template_kwargs": {"preserve_thinking": false}` keeps only the latest.
- Images: up to 42 per request (the launcher's cap), 8 MP each; four 8 MP images in one prompt were checked on the Flash-Next checkpoint.
- Tool calls use the standard `tools` field, parsed with the Qwen3 coder tool parser.
- To reach it from other machines, answer `network` at setup's Access question (it generates an API key), or start `merlin serve` with `HOST=0.0.0.0` and `VLLM_API_KEY` set; put a TLS reverse proxy in front before opening it beyond your own network.

**Optional Merlin 3.0.1 features**, all off by default ([flags and limits](https://github.com/halt95/merlin-engine/blob/main/docs/how-to-serve.md#turn-on-the-merlin-features)):

- **KV RAM cache (experimental).** `--kv-ram-cache 24g` keeps conversation prefixes the GPU evicts in host RAM and restores them instead of recomputing them, with exactly the same output as a GPU cache hit. On Flash-Next a 262K restore takes 1.2–1.4 s instead of about 45 s to recompute. It works with the MTP drafter only (not with `--spec dflash2`) and adds no context or sessions.
- **More sessions with `--swarm` (experimental).** Raises the session cap on four RTX 3090s from 8 to 24, with a 48 GB KV RAM cache. Tested on this checkpoint with P2P; without P2P it uses the same cap untested (⚫, expected to work).
- **DFlash2 drafter.** `--spec dflash2` on four RTX 3090s with P2P decodes 5–18 % faster than the default MTP drafter across the Flash-Next family; the server then holds three 262K sessions instead of six 131K ones. `./setup.sh --spec dflash2` downloads the 1.2 GB drafter.
- **INT6 n-gram table.** `--ple int6` loads a smaller INT6 copy of the n-gram table and saves about 9 GiB of host RAM per server, with quality checks passed on this checkpoint and on Swift1.5. It is [one shared pack](https://huggingface.co/halt95/Qwen3.8-Flash-Next-PLE-INT6-Merlin) laid over the FP8 files you downloaded, which stay untouched; FP8 stays the default.
- **Past 262K with YaRN.** No profile turns YaRN on; you add it as a vLLM override, and asking for more than 262K without it is refused at startup. Tested on the reference host at 393K and 524K (MTP and DFlash2) and 917K (MTP); 1M is untested. The recipes are in [How to serve](https://github.com/halt95/merlin-engine/blob/main/docs/how-to-serve.md#longer-context-with-yarn-flash-next).

## The base model

Qwen3.8-Flash-Next is Qwen's preview of the architecture behind Qwen4. What changed, in short:

- **Hybrid attention.** Three Gated DeltaNet (linear-attention) layers for every Qwen Sparse Attention layer. QSA attends over micro-blocks instead of single tokens, which keeps long-context latency down.
- **Sparse experts.** 512 experts per layer with 10 routed and 1 shared active per token: 6B of the 125B parameters do the work for each token.
- **N-gram embedding.** A 51B-parameter bigram/trigram table at layer 2 adds parameter capacity with less computation than adding experts would. Merlin keeps it in host RAM, so the VRAM goes to KV cache.
- **Gated residual.** Widened residual streams with a data-dependent read gate and a per-branch write gate.
- **Multi-token prediction.** One MTP layer, trained over several steps; Merlin uses it for speculative decoding.
- **Thinking by default**, with a non-thinking mode per request; Qwen's template keeps earlier turns' reasoning by default (preserved thinking).

| Property | Qwen3.8-Flash-Next |
|---|---|
| Type | Causal language model with a vision encoder (images and video) |
| Parameters | 125B total, 6B active, plus 51B n-gram embedding and 4B MTP |
| Layers | 48: 12 × (3 × Gated DeltaNet → MoE, then 1 × Qwen Sparse Attention → MoE) |
| Experts | 512 per layer; 10 routed + 1 shared active; expert width 640 |
| Attention | QSA: 24 query / 2 KV heads, head dim 256, budget 512 blocks or 2,048 tokens · Gated DeltaNet: 48 V / 16 QK heads, head dim 128 |
| Context | 262,144 tokens native (Qwen: extensible to 1M with YaRN). Merlin serves 262,144 by default and, with a YaRN override, past that: tested at 393K, 524K and 917K (see Use it) |

<sub>Architecture as reported by Qwen on the [upstream card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).</sub>

## Other builds

| Checkpoint | Base | GPUs | Decode at 262K |
|---|---|---|---|
| **Flash-Next (this card)** | **Qwen3.8-Flash-Next, 125B MoE** | **4× 3090** | **197 tok/s** |
| [Swift1.5](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) | Flash-Next fine-tune | 4× 3090 | 186 tok/s |
| [Swift1.5-Huihui](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin) | Swift1.5, abliterated | 4× 3090 | 196 tok/s |
| [Qwen3.8-27B](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin) | Qwen3.8-27B, dense | 1×, 2× or 4× 3090 | 71 / 143 / 238 tok/s |
| [ThinkingCap](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin) | Qwen3.8-27B fine-tune | 2× or 4× 3090 | 160 / 238 tok/s |
| [Nemotron 3.5 Super VL](https://huggingface.co/halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin) | Nemotron 3.5 Super VL, 120B-A12B MoE | 4× 3090 | 93 tok/s |
| [Ornith 1.5](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin) | Qwen3.5-122B-A10B fine-tune (ornith-ai), 122B MoE | 4× 3090 | 105 tok/s |

<sub>One stream, thinking on, 261,888 prompt tokens + 128 out, median of 3; Merlin 3.0.1, 2026-11-02 bench cards. Thinking on at reasoning effort `medium`, the server default, wherever the checkpoint has effort levels; the Nemotron and Ornith 1.5 templates have none, so those two ran at full thinking.</sub>

## Good to know

- Every figure we measured for this checkpoint comes from one setup: 4× RTX 3090 at 220 W, PCIe Gen4 x16 with P2P (the 3.0.1 figures also with the gpu-unlock host upgrade on), except the rented-host row under Other GPUs; upstream results and the Other builds rows use their stated setups. Hosts without P2P use the no-P2P profiles (see [What it fits on](#what-it-fits-on)).
- Only if you shrink the KV pool yourself: keep it larger than `--max-model-len` plus a few blocks. With the shipped settings this cannot occur.
- Coming from the `qwen38-flash-next-3090s` engine (v2.x or earlier)? It is replaced by the [Merlin engine](https://github.com/halt95/merlin-engine); this checkpoint's files are unchanged.

## Changes

| Date | Change to this repository |
|---|---|
| 2026-11-02 | Card updated for Merlin 3.0.1 (speed figures and options). The weights are unchanged. |
| 2026-10-06 | `tokenizer.json` replaced with the upstream file; re-fetch it if you downloaded earlier |
| 2026-09-18 | `config.json` gains `ple_embedding_dtype` in `text_config`, which the engine needs since v2; re-fetch it if you downloaded earlier |
| 2026-09-08 | Weights published; unchanged since |

<sub>Other card-only edits are not listed. Engine releases and their speed changes: the engine's [CHANGELOG](https://github.com/halt95/merlin-engine/blob/main/docs/CHANGELOG.md).</sub>

## Credits and licence

Qwen (model; Qwen Community License 1.0, see `LICENSE`), Intel (AutoRound experts), RadixArk (FP8 PLE table), DominikBucko (MTP packing recipe), [froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (the fixed Qwen3.8 chat template the engine serves), the vLLM community PLE-offload work, aikitoria (peer-to-peer on consumer Ampere), [Neiko2002](https://huggingface.co/Neiko2002), [brandon-nuttall](https://github.com/brandon-nuttall) (YaRN past 262K report), [R-omk](https://github.com/R-omk) (YaRN past 262K crash report and recipes), and a community idle-power report (behind the v2.5.2 engine fix). Upstream fixes are credited in the engine repository.
