---
license: apache-2.0
license_link: LICENSE
language:
  - en
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - merlin
  - w4a16
  - awq
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

<img src="https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin/resolve/main/assets/hero.png" alt="Qwen3.8-27B W4A16, powered by Merlin: a 4-bit checkpoint, 16 GiB, full 262K context on one to four 24 GB GPUs. On four GPUs: a 2,012,105-token KV pool in VRAM; 752 tok/s total for 8 agents at once; 215 tok/s decode at 262K; 2,295 tok/s prefill on a 100K prompt." width="100%">

<p>
<a href="https://huggingface.co/Qwen/Qwen3.8-27B"><img src="https://img.shields.io/badge/base-Qwen3.8--27B-615ced" alt="Base model: Qwen3.8-27B"></a>
<a href="#the-checkpoint"><img src="https://img.shields.io/badge/quant-W4A16-3987e5" alt="Quant: W4A16"></a>
<a href="#what-it-fits-on"><img src="https://img.shields.io/badge/context-262K-ffb000" alt="Context: 262K"></a>
<a href="#what-it-fits-on"><img src="https://img.shields.io/badge/GPUs-Ampere-76B900" alt="GPUs: NVIDIA Ampere (Ada profiles ship untested)"></a>
<a href="https://github.com/halt95/merlin-engine"><img src="https://img.shields.io/badge/runs_on-Merlin_3.0.0-a78bfa" alt="Runs on Merlin 3.0.0"></a>
<a href="https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin/blob/main/LICENSE"><img src="https://img.shields.io/badge/licence-Apache_2.0-lightgrey" alt="Licence: Apache-2.0"></a>
</p>

<b><a href="#the-checkpoint">The checkpoint</a> · <a href="#speed">Speed</a> · <a href="#quality">Quality</a> · <a href="#what-it-fits-on">What it fits on</a> · <a href="#use-it">Use it</a> · <a href="#the-base-model">The base model</a> · <a href="https://github.com/halt95/merlin-engine/blob/main/models/qwen38-27b/benchmarks/2026-10-20/BENCH-CARD.md">Bench card</a></b>

</div>

# Qwen3.8-27B-W4A16-Merlin

A 4-bit build of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Qwen's dense 27B model with vision, made to run at its full 262,144-token context on one, two or four 24 GB GPUs. It is 16 GiB instead of the original 52 GiB: the attention, Gated DeltaNet and feed-forward projections are AWQ INT4, the `lm_head` and the MTP draft head INT4, and the vision tower and embeddings stay in BF16. It is served by the [Merlin engine](https://github.com/halt95/merlin-engine), a vLLM 0.32.0 fork tuned for Ampere and Ada that keeps the token embeddings in host RAM and the whole KV cache in VRAM. Measured on RTX 3090s; the engine also ships profiles for other Ampere and Ada cards, from RTX 4090s and 48 GB workstation cards to A100s and CMP 170HX (see [What it fits on](#what-it-fits-on)).

<sub>numbers: Merlin 3.0.0 on 4× RTX 3090 w/o NVLINK @ 220W, P2P on, 2026-10-20 bench card; the hero is the four-card layout. KV pool: FP8, all in VRAM; seven 262K sessions held at once on four cards. Agents: eight 16K conversations, then eight simultaneous 512-token follow-ups, thinking off. Decode: one stream, thinking on (effort medium, the default), exactly 261,888 prompt tokens + 128 out, median of 3. Prefill: one cold 101,845-token prompt. Each figure is also in a table below.</sub>

<a href="https://ko-fi.com/halt95"><img src="https://storage.ko-fi.com/cdn/kofi3.png?v=3" alt="Buy me a coffee on Ko-fi" height="40"></a>

## The checkpoint

| Component | Format |
|---|---|
| Gated attention (16 layers), Gated DeltaNet `in_proj_qkv` / `in_proj_z` / `out_proj` (48 layers) and feed-forward (all 64) | AWQ INT4, asymmetric, group 128 ([philbert's checkpoint](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ)) |
| `lm_head` | INT4 GPTQ, asymmetric, group 128 |
| MTP draft head (projections, attention, feed-forward) | INT4, asymmetric, group 128, GPTQ-calibrated on drafter inputs |
| Gated DeltaNet `in_proj_a` / `in_proj_b`, conv and norms; vision tower; embeddings; norms | BF16, unchanged |
| KV cache | FP8 E4M3 with per-layer static scales for the 16 attention layers and the MTP layer (`kv_scales.safetensors`) |

**Files in this repository**

| Files | What they hold | Size |
|---|---|---|
| `model.safetensors` | The language model, vision tower and embeddings | 15.7&nbsp;GiB |
| `model-mtp.safetensors` | The MTP draft head, INT4 | 210&nbsp;MiB |
| `kv_scales.safetensors` | Calibrated static scales for the FP8 KV cache | 3.5&nbsp;KiB |
| `model.safetensors.index.json` | Which file holds each tensor | 206&nbsp;KiB |
| `chat_template.jinja` | Qwen's own chat template, unchanged (byte-identical to the upstream file). Merlin serves [Froggeric's fixed Qwen3.8 template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v22.5) by default; see Use it | 8.7&nbsp;KiB |
| `config.json`, `generation_config.json` | Model, quantisation and sampling settings | 21.4&nbsp;KiB |
| `tokenizer.json`, `tokenizer_config.json`, processor configs | `tokenizer.json` byte for byte from Qwen3.8-27B; `tokenizer_config.json` re-serialised by transformers; image / video processor settings | 12.2&nbsp;MiB |

<sub>About 16 GiB in all (the 11 files of the engine's `models/qwen38-27b/checkpoint.sha256`). Also in the repository: `LICENSE` (Apache-2.0) and `NOTICE`.</sub>

## Speed

<img src="https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin/resolve/main/assets/speed.png" alt="Speed on 2x and 4x RTX 3090, Merlin 3.0.0. Decode, one stream: 207 / 173 / 142 tok/s at 4K / 131K / 262K on two cards, 250 / 235 / 215 on four. Cold prefill: 1,679 / 1,190 / 858 tok/s at 32K / 131K / 262K on two cards, 2,842 / 2,131 / 1,596 on four. Time to first token: 2.17 / 109.5 / 305.3 s on two cards, 1.35 / 61.2 / 164.1 s on four. Agents at once, total: 281 / 373 / 517 tok/s for 2 / 4 / 8 on two cards, 361 / 549 / 752 on four. The same figures, and one card, are in the tables below." width="100%">

| Prompt | Decode, 1× 3090 | Decode, 2× 3090 | Decode, 4× 3090 |
|---|---|---|---|
| 4K | 124 tok/s | 207 tok/s | 250 tok/s |
| 32K | 121 tok/s | 196 tok/s | 252 tok/s |
| 131K | 95 tok/s | 173 tok/s | 235 tok/s |
| 262K | 73 tok/s | 142 tok/s | 215 tok/s |

<sub>One stream, thinking on (effort medium), ≤ 256 tokens out (128 at 262K), median of 3; 4,046 / 32,566 / 130,386 / 261,888 prompt tokens. Thinking off at 262K: 49 / 111 / 158 tok/s. The one-card figures predate the single-card memory settings (2026-10-21), which left text output identical.</sub>

| Prompt | Prefill and TTFT, 1× 3090 | 2× 3090 | 4× 3090 |
|---|---|---|---|
| 4K | 1,043 tok/s · 3.88 s | 1,865 tok/s · 2.17 s | 2,997 tok/s · 1.35 s |
| 32K | 930 tok/s · 35.0 s | 1,679 tok/s · 19.4 s | 2,842 tok/s · 11.5 s |
| 131K | 644 tok/s · 202.4 s | 1,190 tok/s · 109.5 s | 2,131 tok/s · 61.2 s |
| 262K | 456 tok/s · 574.1 s | 858 tok/s · 305.3 s | 1,596 tok/s · 164.1 s |

<sub>Prefill: prompt tokens / time to first token (TTFT), the median of 3 cold sends below 262K and the cold send 1 at 262K.</sub>

| Cold prefill, one send | 1× 3090 | 2× 3090 | 4× 3090 |
|---|---|---|---|
| 10,157 tokens | 1,021 tok/s · 9.94 s | 1,823 tok/s · 5.57 s | 3,009 tok/s · 3.38 s |
| 101,845 tokens | 708 tok/s · 143.8 s | 1,297 tok/s · 78.5 s | 2,295 tok/s · 44.4 s |
| 261,888 tokens | 456 tok/s · 574.1 s | 858 tok/s · 305.3 s | 1,596 tok/s · 164.1 s |

<sub>Prefill and time to first token, one cold request per prompt, 8 tokens out; the 261,888-token row is the exact-262K prompt's cold send 1 (128 tokens out).</sub>

| Agents at once | 1× 3090 | 2× 3090 | 4× 3090 |
|---|---|---|---|
| 2 | 164 tok/s | 281 tok/s | 361 tok/s |
| 4 | two at most | 373 tok/s | 549 tok/s |
| 8 | two at most | 517 tok/s | 752 tok/s |

<sub>N 16K conversations, then N simultaneous 512-token follow-ups, thinking off, total tok/s while all N decode. One card runs two sequences at most. One and two cards: a pre-release engine build, 2026-10-17; four cards: the release wheel, 2026-10-20. Overlap windows were short (2.6 to 7.3 s), so read these rows loosely.</sub>

| Sessions held at once | Needles found | Re-sent from cache, lowest |
|---|---|---|
| 1 × 262K or 2 × 131K (one card) | 1 / 1 · 2 / 2 | 99.99 % · 99.98 % |
| 3 × 262K or 6 × 131K (two cards) | 3 / 3 · 6 / 6 | 99.99 % · 99.98 % |
| 7 × 262K or 14 × 131K (four cards) | 7 / 7 · 14 / 14 | 99.99 % · 99.98 % |

<sub>N needle prompts (thinking off, greedy) sent at once, then re-sent one by one: every re-send served from the prefix cache means the pool held all N. 0 preemptions.</sub>

<details>
<summary><b>Against the previous release (v1.3.0, two and four cards)</b></summary>

| Measure | v1.3.0 | 3.0.0 | Change |
|---|---|---|---|
| Prefill, 10,157-token prompt (two cards) | 1,739 | 1,823 | +4.8 % |
| Prefill, 101,845-token prompt (two cards) | 1,267 | 1,297 | +2.4 % |
| Prefill, 10,157-token prompt (four cards) | 2,483 | 3,009 | +21.2 % |
| Prefill, 101,845-token prompt (four cards) | 1,989 | 2,295 | +15.4 % |

<sub>tok/s, one cold send each. v1.3.0's decode ran at T 1.0 with effort `low` and 3.0.0's at T=0 with effort `medium`, so decode tok/s are not like for like; the decode step time is unchanged (the bench card has both). One card is new in 3.0.0.</sub>

</details>

<details>
<summary><b>How it was measured</b></summary>

- Host: 4× RTX 3090 (24 GB, sm_86) at 220 W, PCIe Gen4 with P2P, no NVLink, driver 610.43.02.
- Engine: Merlin 3.0.0 from the release wheel, installed with `release/install-env.sh`, served by `merlin serve` with the release defaults, one fresh boot per layout. One card: TP1; two cards: TP2 on GPUs 0 and 2; four: TP4.
- Layout: MTP K=4, FP8 KV, RecoverSSM on every layout, host-resident token embeddings, thinking effort `medium`.
- Decode: T=0, a needle prompt at depth, every send a cold prefill except send 3 at 262K, which reuses the prefix before the needle; one discarded warm-up, then 3 sends per cell, median. tok/s = tokens after the first streamed event / time from first to last event.
- Streams are client-timed. Full tables, thinking-off rows and ITL: the [bench card](https://github.com/halt95/merlin-engine/blob/main/models/qwen38-27b/benchmarks/2026-10-20/BENCH-CARD.md).

</details>

## Quality

**This checkpoint**, measured on the Merlin engine:

| Check | Result | Setting |
|---|---|---|
| GSM8K-200 | 195 / 194 / 197 of 200 on 1 / 2 / 4 cards | 5-shot, thinking off |
| Needles at 131K | 5 / 5 on each layout | single-occurrence needles, an image in the prompt, thinking off, greedy, cold cache |
| Needles at 262K | 12 / 12 on each layout | as above |
| Needle at exactly 261,888 tokens | 3 / 3 on each layout | thinking off |
| Image, 4096 × 2048 (8,388,608 px) | 1 / 1 · 2 / 2 at once · 1 / 1 with four in one prompt, on 2 and 4 cards | thinking off; names both colours of a two-colour image (with four, of the last one) |

<sub>Four cards and images: the Merlin 3.0.0 release wheel, 2026-10-20. One and two cards' GSM8K and 131K / 262K needle panel: a pre-release engine build, 2026-10-17; the exact-261,888-token needle on every layout: the release wheel, 2026-10-20. These are checks of this checkpoint, not a matched BF16 comparison: no BF16 run of the same checks was made. The FP8 KV scales were calibrated on traffic that included the first 50 GSM8K test questions (activation maxima only); the `lm_head` and MTP head were GPTQ-calibrated on public benchmark prompts (ARC, MMLU-Pro, GPQA, GSM8K train split, HumanEval), synthetic tasks and a small hand-written set. No calibration data is distributed.</sub>

**The base model**, as reported by Qwen for the BF16 weights:

| Area | Benchmark | Qwen3.8-27B (BF16) |
|---|---|---|
| Coding | Terminal Bench 2.1 (Terminus) | 73.0 |
| Coding | SWE-bench Pro | 61.7 |
| Coding | LiveCodeBench v6 | 90.3 |
| Agents | CoWorkBench | 70.7 |
| Reasoning and instructions | GPQA Diamond | 89.2 |
| Reasoning and instructions | HLE | 30.8 |
| Reasoning and instructions | IFBench | 79.5 |
| Vision | OSWorld-Verified | 84.3 |
| Vision | AndroidWorld | 81.9 |
| Vision | RealWorldQA | 85.9 |

<sub>A selection from the [upstream card](https://huggingface.co/Qwen/Qwen3.8-27B), which has the full tables and comparisons with other models. These were not re-run on this 4-bit build.</sub>

## What it fits on

| Measured on | Context per request | KV pool | Held at once | Images |
|---|---|---|---|---|
| 1× RTX 3090, 24 GB (🟢 Tested) | 262,144 tokens | 271,506 tokens | 1 × 262K or 2 × 131K | 1 per prompt, 4 MP |
| 2× RTX 3090, 24 GB (🟢 Tested) | 262,144 tokens | 826,689 tokens | 3 × 262K or 6 × 131K | 8 MP |
| 4× RTX 3090, 24 GB (🟢 Tested) | 262,144 tokens | 2,012,105 tokens | 7 × 262K or 14 × 131K | 8 MP |

<sub>Measured on the reference host (2026-10-20 bench card). KV cache FP8, all in VRAM. Two and four cards need GPU peer-to-peer on RTX 3090s for these rows (hosts without it: the no-P2P profiles below). A 4096 × 2048 image (8,388,608 px) left at least 886 MiB free on two cards and 862 MiB on four. One card takes one 4 MP image per prompt: with the single-card memory settings, one and two requests at once and after a 262K needle left at least 558 MiB free (2026-10-21). The original one-card run failed at 8 MP; with those settings 8 MP still stays below the 524 MiB headroom floor (478 MiB on the reference host).</sub>

**Other GPUs.** The engine also ships profiles for these; the Status column says how far each has been run:

| GPUs | Arch | Profile | Context fit | Status |
|---|---|---|---|---|
| 2 / 4× RTX 3090 without P2P | Ampere, sm_86 | `27b-tp2-3090-nop2p`, `27b-tp4-3090-nop2p` | ~3 / ~7 × 262K (the 3090 pins) | ⚫ Untested |
| 1 / 2 / 4× RTX 3090 Ti, RTX A5000 (24 GB) | Ampere, sm_86 | `27b-tp1-3090`, `-tp2-3090`, `-tp4-3090` | ~1 / ~3 / ~7 × 262K (as the 3090) | ⚫ Untested |
| 1 / 2 / 4× RTX 4090 (24 GB) | Ada, sm_89 | `27b-tp1-sm89-24g`, `-tp2-`, `-tp4-` | ~1 / ~3 / 7 × 262K (four cards: 7 × 262K held on a rented host with Qwen3.8-27B; one card not yet run at the shipped settings) | ⚫ Untested |
| 1 / 2 / 4× 32 / 48 GB Ada: RTX 4080 Super (modified), RTX 4090 48 GB (modified), RTX 6000 Ada, L40S | Ada, sm_89 | `27b-tpN-sm89-32g`, `-sm89-48g` | ~1 / ~3 / ~7 × 262K (the 24 GB pins) | ⚫ Untested |
| 1 / 2 / 4× 48 GB: RTX A6000, A40 | Ampere, sm_86 | `27b-tpN-sm86-48g` | ~3 / ~8 / ~18 × 262K | ⚫ Untested |
| 1 / 2 / 4× A30 (24 GB) | Ampere, sm_80 | `27b-tpN-sm80-24g` | ~1 / ~3 / ~7 × 262K | ⚫ Untested |
| 1 / 2 / 4× CMP 170HX 32 GB | Ampere, sm_80 | `27b-tpN-sm80-32g` | ~1 / ~4 / ~11 × 262K | ⚫ Untested |
| 1 / 2 / 4× A100 40 GB | Ampere, sm_80 | `27b-tpN-sm80-40g` | ~2 / ~6 / ~14 × 262K | ⚫ Untested |
| 1 / 2 / 4× CMP 170HX 64 GB | Ampere, sm_80 | `27b-tpN-sm80-64g` | ~5 / ~11 / ~25 × 262K (needles exact on a rented host) | ⚫ Untested |
| 1 / 2 / 4× A100 80 GB | Ampere, sm_80 | `27b-tpN-sm80-80g` | ~7 / ~15 / ~32 × 262K | ⚫ Untested |

<sub>`merlin serve` and `./setup.sh` pick the profile from the GPUs they find; without P2P the default is the largest layout, and `merlin serve` prints the `--tp 1 --instances N` command, which gives more total throughput for many users. A `~` is the engine's estimate, not a measurement. Every profile uses an FP8 KV cache, on sm_80 cards too (FP8 KV ran on the 4× CMP 170HX 64 GB rental on one, two and four cards, needles exact, GSM8K-50 48/50). 🟢 measured by us · ⚫ a profile ships, not yet qualified by us (expected to work; a rented-host run is noted in the row where one was made). Every profile: the engine's [hardware matrix](https://github.com/halt95/merlin-engine/blob/main/docs/merlin/hardware-matrix.md).</sub>

| Need | Details |
|---|---|
| Host RAM | About 2.4 GiB pinned for the host-resident token embeddings (1.18 GiB per card on two cards, 0.59 on four), on top of normal use; one card also keeps the vision-tower weights in host RAM (about 1 GiB) |
| Disk | About 16 GiB for the checkpoint, plus the engine environment and caches |
| Engine host | Linux x86-64 (glibc 2.38 or newer), NVIDIA driver R580 or later (CUDA 13), Python 3.13, GCC/G++ 13 or newer, `/dev/shm` of at least 1 GB; GPUs headless with no other CUDA process. `merlin doctor` checks the host. |

## Use it

```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine
./setup.sh --model 27b
```

One command does the rest: after a small uv and Python bootstrap it checks the GPUs, driver, RAM and disk, shows the engine, package and checkpoint downloads with their sizes before fetching them, installs the prebuilt engine, downloads and verifies this checkpoint, and starts an OpenAI-compatible server on `127.0.0.1:8100`. Enter takes the recommended answer at each question; `--yes` takes them all, `--dry-run` shows the plan. Run `./setup.sh` again to start the server, and `git pull && ./setup.sh` to update the engine without downloading the model again. Requests use the model name `qwen38-27b`; the setup asks how many cards to use and recommends the largest tested layout (on a multi-card host without P2P it recommends `--tp 1 --instances N`).

> [!IMPORTANT]
> This checkpoint needs the Merlin engine; stock vLLM was not tested with it.

<details>
<summary><b>Manual install</b></summary>

```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine
curl -LO https://github.com/halt95/merlin-engine/releases/download/merlin-v3.0.0/vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl
MERLIN_WHEEL="$PWD/vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl" \
  bash release/install-env.sh ./venv

venv/bin/hf download halt95/Qwen3.8-27B-W4A16-Merlin \
  --local-dir ~/models/27b
(cd ~/models/27b && sha256sum -c "$OLDPWD/models/qwen38-27b/checkpoint.sha256")
venv/bin/python merlin_cli.py serve ~/models/27b
```

`MERLIN_WHEEL` installs the prebuilt engine (sm_80, sm_86, sm_89), checked against `release/merlin-wheel.sha256`; without it the installer compiles the engine instead, about 1.5 hours on 8 cores. `merlin serve` recognises the checkpoint, picks the largest tested layout for the GPUs it finds (or, with none, the largest untested one that fits, labelled as such) and serves on `127.0.0.1:8100`. `--tp 1|2|4` picks a layout. Add `--dry-run` to print the resolved command without starting anything. The first start compiles kernels.

</details>

**Recommended settings** (Qwen's, for Qwen3.8-27B):

| Mode | temperature | top_p | top_k | presence_penalty |
|---|---|---|---|---|
| Thinking (default) | 1.0 | 0.95 | 20 | 0.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 1.5 |

<sub>Qwen's recommendation for this model, with `min_p=0.0` and `repetition_penalty=1.0` in both modes. A `presence_penalty` between 0 and 2 reduces endless repetition; higher values can mix languages.</sub>

- Turn thinking off for a request with `"chat_template_kwargs": {"enable_thinking": false}`. With thinking on, a short `max_tokens` can end inside the reasoning with empty `content`.
- Reasoning effort: Merlin serves `medium` by default; pick another per request with `"chat_template_kwargs": {"reasoning_effort": "low"}` (`low` / `medium` / `xhigh`).
- Chat template: Merlin serves [Froggeric's fixed Qwen3.8 template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (v22.5) by default, not the file in this repository. It accepts system messages after the first turn, so coding agents' per-turn notes stay in place and the prefix cache keeps hitting. `CHAT_TEMPLATE=<path>` serves another template; `CHAT_TEMPLATE=` (empty) serves this repository's own.
- With Qwen's own template (`CHAT_TEMPLATE=`), earlier turns' reasoning is kept by default and `"chat_template_kwargs": {"preserve_thinking": false}` keeps only the latest.
- Images: up to 64 per prompt at 8 MP each on two or four cards; one card takes one image per prompt at 4 MP. The cap is a served default: a client that raises it per request goes past the measured headroom.
- Tool calls use the standard `tools` field, parsed with the Qwen3 coder tool parser.
- To reach it from other machines, answer `network` at setup's Access question (it generates an API key), or start `merlin serve` with `HOST=0.0.0.0` and `VLLM_API_KEY` set; put a TLS reverse proxy in front before opening it beyond your own network.

## The base model

Qwen3.8-27B is Qwen's dense model on the Qwen3.5 architecture, with vision. The architecture, in short:

- **Hybrid attention.** Three Gated DeltaNet (linear-attention) layers for every gated-attention layer: 48 and 16 of the 64 layers.
- **Dense.** All 27B parameters work on every token; no experts.
- **Multi-token prediction.** One MTP layer, trained over several steps; Merlin uses it for speculative decoding (4 draft tokens).
- **Thinking by default**, with a non-thinking mode per request and `reasoning_effort` levels; Qwen's template keeps earlier turns' reasoning by default (preserved thinking).

| Property | Qwen3.8-27B |
|---|---|
| Type | Causal language model with a vision encoder (images and video) |
| Parameters | 27B |
| Layers | 64: 16 × (3 × Gated DeltaNet → FFN, then 1 × gated attention → FFN) |
| Attention | 24 query / 4 KV heads, head dim 256 · Gated DeltaNet: 48 V / 16 QK heads, head dim 128 |
| Feed-forward | Width 17,408 |
| Context | 262,144 tokens native (Qwen: extensible to 1M; this build serves 262,144) |

<sub>Architecture as reported by Qwen on the [upstream card](https://huggingface.co/Qwen/Qwen3.8-27B).</sub>

## Other builds

| Checkpoint | Base | GPUs | Decode at 262K |
|---|---|---|---|
| **Qwen3.8-27B (this card)** | **Qwen3.8-27B, dense** | **2× or 4× 3090** | **142 / 215 tok/s** |
| [Flash-Next](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin) | Qwen3.8-Flash-Next, 125B MoE | 4× 3090 | 184 tok/s |
| [Swift1.5](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) | Flash-Next fine-tune | 4× 3090 | 177 tok/s |
| [Swift1.5-Huihui](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin) | Swift1.5, abliterated | 4× 3090 | 189 tok/s |
| [ThinkingCap](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin) | Qwen3.8-27B fine-tune | 2× or 4× 3090 | 165 / 248 tok/s |
| [Nemotron 3.5 Super VL](https://huggingface.co/halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin) | Nemotron 3.5 Super VL, 120B-A12B MoE | 4× 3090 | 91 tok/s |
| [Ornith 1.5](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin) | Qwen3.5-122B-A10B fine-tune (ornith-ai), 122B MoE | 4× 3090 | 100 tok/s |

<sub>One stream, thinking on, 261,888 prompt tokens + 128 out, median of 3; Merlin 3.0.0, 2026-10-20 bench cards. Effort as each bench card ran it: Flash-Next low; Swift1.5, Swift1.5-Huihui and ThinkingCap xhigh; Qwen3.8-27B medium. Merlin now serves `medium` on every Qwen3.8 checkpoint by default. Nemotron and Ornith 1.5 ran at full thinking: neither template reads `reasoning_effort`. Ornith 1.5 was measured on 2026-10-21 on the reference host with its engine changes (uniform MoE verify on) before they merged into 3.0.0.</sub>

## Good to know

- Every figure we measured for this checkpoint comes from 4× RTX 3090 at 220 W, PCIe Gen4 with P2P, used as one, two or four cards, except the rental rows under Other GPUs; the Other builds table names each build's card count.
- The same engine profiles serve [ThinkingCap](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin), BottleCap AI's fine-tune of this model, which reasons in fewer tokens (under a different licence).
- Coming from the `qwen38-27b-3090s` engine (v1.x)? It is replaced by the [Merlin engine](https://github.com/halt95/merlin-engine); this checkpoint's files are unchanged.

## Changes

| Date | Change to this repository |
|---|---|
| 2026-10-12 | `tokenizer.json` replaced with the upstream Qwen3.8-27B file: combining-mark scripts (Hindi, Thai, Arabic and others) now tokenize as upstream with transformers 5.18.0 or newer (which Merlin pins; 5.17.0 and earlier ignore this file); English, code and CJK are unchanged |
| 2026-09-27 | Weights published; unchanged since |

<sub>Card-only edits are not listed. Engine releases and their speed changes: the engine's [CHANGELOG](https://github.com/halt95/merlin-engine/blob/main/CHANGELOG.md).</sub>

## Credits and licence

Qwen (model; Apache-2.0, see `LICENSE`), [philbert](https://huggingface.co/philbert440) (the AWQ INT4 body), [froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (the fixed Qwen3.8 chat template the engine serves), the vLLM project, FlashInfer, compressed-tensors, aikitoria (peer-to-peer on consumer Ampere), [Neiko2002](https://huggingface.co/Neiko2002), and [CoolShades](https://github.com/CoolShades) (the idle-power report behind the v2.5.2 engine fix). Upstream fixes are credited in the engine repository.
