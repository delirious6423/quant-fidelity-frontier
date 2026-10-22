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

<img src="https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin/resolve/main/assets/hero.png" alt="Qwen3.8-Flash-Next W4A16, powered by Merlin: a 4-bit checkpoint, 115 GiB, full 262K context on four 24 GB GPUs. A 924,993-token KV pool in VRAM; 638 tok/s total for 8 agents at once; 184 tok/s decode at 262K; 6,051 tok/s prefill on a 100K prompt." width="100%">

<p>
<a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next"><img src="https://img.shields.io/badge/base-Qwen3.8--Flash--Next-615ced" alt="Base model: Qwen3.8-Flash-Next"></a>
<a href="#the-checkpoint"><img src="https://img.shields.io/badge/quant-W4A16-3987e5" alt="Quant: W4A16"></a>
<a href="#what-it-fits-on"><img src="https://img.shields.io/badge/context-262K-ffb000" alt="Context: 262K"></a>
<a href="#what-it-fits-on"><img src="https://img.shields.io/badge/GPUs-Ampere-76B900" alt="GPUs: NVIDIA Ampere (Ada profiles ship untested)"></a>
<a href="https://github.com/halt95/merlin-engine"><img src="https://img.shields.io/badge/runs_on-Merlin_3.0.0-a78bfa" alt="Runs on Merlin 3.0.0"></a>
<a href="https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin/blob/main/LICENSE"><img src="https://img.shields.io/badge/licence-Qwen_Community_1.0-lightgrey" alt="Licence: Qwen Community 1.0"></a>
</p>

<b><a href="#the-checkpoint">The checkpoint</a> · <a href="#speed">Speed</a> · <a href="#quality">Quality</a> · <a href="#what-it-fits-on">What it fits on</a> · <a href="#use-it">Use it</a> · <a href="#the-base-model">The base model</a> · <a href="https://github.com/halt95/merlin-engine/blob/main/models/flash-next/benchmarks/2026-10-20/BENCH-CARD.md">Bench card</a></b>

</div>

# Qwen3.8-Flash-Next-W4A16-Merlin

A 4-bit build of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Qwen's 125B mixture-of-experts model (6B active per token, with vision), made to run at its full 262,144-token context on four 24 GB GPUs. It is 115 GiB instead of the original 335 GiB: the routed experts are INT4, the 51B-parameter n-gram table is FP8, and attention and the vision tower stay in BF16. It is served by the [Merlin engine](https://github.com/halt95/merlin-engine), a vLLM 0.32.0 fork tuned for Ampere and Ada that keeps the n-gram table in host RAM and the whole KV cache in VRAM. Measured on RTX 3090s; the engine also ships profiles for other Ampere and Ada cards, from RTX 4090s and 48 GB workstation cards to A100s and CMP 170HX (see [What it fits on](#what-it-fits-on)).

<sub>numbers: Merlin 3.0.0 on 4× RTX 3090 w/o NVLINK @ 220W, P2P on, 2026-10-20 bench card. KV pool: FP8, all in VRAM; three 262K sessions held at once. Agents: eight 16K conversations, then eight simultaneous 512-token follow-ups, thinking off. Decode: one stream, thinking on (effort low, as benchmarked), exactly 261,888 prompt tokens + 128 out, median of 3. Prefill: one cold 101,875-token prompt. Each figure is also in a table below.</sub>

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

<img src="https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin/resolve/main/assets/speed.png" alt="Speed on 4x RTX 3090, Merlin 3.0.0. Decode, one stream: 168 tok/s at 4K, 170 at 131K, 184 at 262K. Cold prefill: 6,081 tok/s at 32K, 6,046 at 131K, 5,838 at 262K. Time to first token: 0.79 s at 4K, 21.6 s at 131K, 44.9 s at 262K. Agents at once, total: 222 tok/s for 2, 378 for 4, 638 for 8. The same figures are in the tables below." width="100%">

| Prompt | Decode, one stream | Prefill, cold | Time to first token |
|---|---|---|---|
| 4K | 168 tok/s | 5,159 tok/s | 0.79 s |
| 32K | 168 tok/s | 6,081 tok/s | 5.36 s |
| 131K | 170 tok/s | 6,046 tok/s | 21.6 s |
| 262K | 184 tok/s | 5,838 tok/s | 44.9 s |

<sub>4× RTX 3090, the needle-prompt ladder (4,076 / 32,596 / 130,416 / 261,888 prompt tokens), every send a cold prefill except at 262K, where sends 1 and 2 are cold and send 3 reuses the prefix before the needle. Decode: thinking on, ≤ 256 tokens out (128 at 262K), median of 3. Prefill: prompt tokens / time to first token (the median below 262K; the cold send 1 at 262K). The 262K needle is found in 3 of 3 thinking-off sends.</sub>

| Cold prefill, one send | Prefill | Time to first token |
|---|---|---|
| 10,187 tokens | 5,660 tok/s | 1.80 s |
| 101,875 tokens | 6,051 tok/s | 16.8 s |
| 261,888 tokens | 5,838 tok/s | 44.9 s |

<sub>One cold request per prompt, 8 tokens out; the 261,888-token row is the exact-262K prompt's cold send 1 (128 tokens out). The 101,875-token figure is the hero's prefill number.</sub>

| Agents at once | Total tok/s |
|---|---|
| 2 | 222 |
| 4 | 378 |
| 8 | 638 |

<sub>N 16K conversations, then N simultaneous 512-token follow-ups, thinking off, total tok/s while all N decode. Each overlap window was short (4.55, 5.22 and 5.97 s), so read these rows loosely.</sub>

| Sessions held at once | Needles found | Re-sent from cache, lowest |
|---|---|---|
| 3 × 262K | 3 / 3 | 99.95 % |
| 6 × 131K | 6 / 6 | 99.91 % |

<sub>N needle prompts (thinking off, greedy) sent at once, then re-sent one by one: every re-send served from the prefix cache means the pool held all N. Peak pool use 100 %, 0 preemptions.</sub>

<details>
<summary><b>Against the previous release (v2.5.3)</b></summary>

| Measure | v2.5.3 | 3.0.0 | Change |
|---|---|---|---|
| Decode, 4,076-token prompt | 162.0 | 167.7 | +3.5 % |
| Decode, 130,416-token prompt | 168.0 | 170.3 | +1.4 % |
| Decode, 259,587-token prompt | 177.2 | 187.4 | +5.8 % |
| Prefill, 10,187-token prompt | 5,020 | 5,660 | +12.7 % |
| Prefill, 101,875-token prompt | 5,417 | 6,051 | +11.7 % |

<sub>tok/s. Decode: thinking on, median of 3, same frozen driver and prompts. Prefill: one cold send each. v2.5.3 added the Swift1.5-Huihui checkpoint and kept the v2.5.2 engine unchanged, so its Flash-Next figures are the v2.5.2 bench card's. The exact 261,888-token row (184 tok/s) is new in 3.0.0 and has no v2.5.3 counterpart.</sub>

</details>

<details>
<summary><b>How it was measured</b></summary>

- Host: 4× RTX 3090 (24 GB, sm_86) at 220 W, PCIe Gen4 with P2P, no NVLink, driver 610.43.02.
- Engine: Merlin 3.0.0 from the release wheel, installed with `release/install-env.sh`, served by `merlin serve` with the release defaults, one fresh boot.
- Layout: TP2 × PP2 + EP on four cards, MTP K=3, FP8 KV, BF16 draft lm_head, thinking effort `low`, the default when benchmarked on 2026-10-20 (Merlin now serves `medium` by default).
- Decode: T=0, a needle prompt at depth, every send a cold prefill except send 3 at 262K, which reuses the prefix before the needle; one discarded warm-up, then 3 sends per cell, median. tok/s = tokens after the first streamed event / time from first to last event.
- Streams are client-timed. Full tables, thinking-off rows and ITL: the [bench card](https://github.com/halt95/merlin-engine/blob/main/models/flash-next/benchmarks/2026-10-20/BENCH-CARD.md).

</details>

## Quality

**This checkpoint**, measured on the Merlin engine:

| Check | Result | Setting |
|---|---|---|
| GSM8K-200 | 199 / 200 | 5-shot, thinking off |
| Needles at 131K | 10 / 10 | single-occurrence needles, an image in the prompt, thinking off, greedy, cold cache |
| Needles at 262K | 23 / 23 | as above |
| Needle at exactly 261,888 tokens | 3 / 3 | thinking off |
| Image, 4096 × 2048 (8,388,608 px) | 1 / 1 · 2 / 2 at once · 1 / 1 with four in one prompt | thinking off; names both colours of a two-colour image (with four, of the last one) |

<sub>GSM8K and images: the Merlin 3.0.0 release wheel, 2026-10-20. Needle panel: a pre-release engine build, 2026-10-17. These are checks of this checkpoint, not a matched BF16 comparison: no BF16 run of the same checks was made, so how much quality the 4-bit build keeps is not quantified here.</sub>

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

<sub>Measured on the reference host (2026-10-20 bench card): three 262K and six 131K sessions held at once with every re-send served from cache; a 4096 × 2048 image (8,388,608 px) with at least 1,026 MiB free on every card.</sub>

**Other GPUs.** The engine also ships profiles for these; the Status column says how far each has been run:

| GPUs | Arch | Profile | Context fit | Status |
|---|---|---|---|---|
| 4× RTX 3090 Ti, RTX A5000, A10 (24 GB) | Ampere, sm_86 | `fn-quad-3090` | ~3 × 262K (as the 3090) | ⚫ Untested |
| 4× RTX 3090 without P2P | Ampere, sm_86 | `fn-quad-3090-nop2p` | ~3 × 262K | ⚫ Untested |
| 8× RTX 3090 | Ampere, sm_86 | `fn-octo-3090-tp2pp4` | ~17 × 262K | ⚫ Untested |
| 4× RTX 4090 (24 GB) | Ada, sm_89 | `fn-quad-24g-sm89` | ~3 × 262K | ⚫ Untested |
| 4× RTX 4080 Super 32 GB (modified) | Ada, sm_89 | `fn-quad-32g-sm89` | ~10 × 262K | ⚫ Untested |
| 2 / 4× 48 GB: RTX A6000, A40 | Ampere, sm_86 | `fn-2x48g-pp2-sm86`, `fn-4x48g-tp2pp2-sm86` | ~5 / ~25 × 262K | ⚫ Untested |
| 2 / 4× 48 GB: RTX 6000 Ada, L40S | Ada, sm_89 | `fn-2x48g-pp2-sm89`, `fn-4x48g-tp2pp2-sm89` | ~5 / ~25 × 262K | ⚫ Untested |
| 4× A10G, L4 (cloud, 22.5 GiB as reported) | Ampere / Ada | `fn-quad-3090-nop2p`, `fn-quad-24g-sm89` | ~1 × 262K | ⚫ Untested |
| 1 / 2 / 4× A100 80 GB | Ampere, sm_80 | `fn-sm80-80g-x1`, `-x2-tp2`, `-x4-tp2pp2` | ~1 / ~11 / ~28 × 262K | ⚫ Untested |
| 4× A100 40 GB, CMP 170HX 40 GB | Ampere, sm_80 | `fn-sm80-40g-x4-pp4` | ~7 × 262K | ⚫ Untested |
| 3 / 4× CMP 170HX 32 GB | Ampere, sm_80 | `fn-sm80-32g-x3-pp3`, `fn-sm80-32g-x4-pp4` | ~1 / ~4 × 262K | ⚫ Untested |
| 4× A30 (24 GB) | Ampere, sm_80 | `fn-quad-24g-sm80` | ~1 × 262K | ⚫ Untested |
| 2× CMP 170HX 64 GB | Ampere, sm_80 | `fn-sm80-64g-x2-pp2` | ~6 × 262K | ⚫ Untested |
| 4× CMP 170HX 64 GB | Ampere, sm_80 | `fn-sm80-64g-x4-pp4` | 17 × 262K (8 at once) | 🟢 Tested (rented host) |

<sub>`merlin serve` and `./setup.sh` pick the profile from the GPUs they find. A `~` is the engine's estimate, not a measurement. "(8 at once)": the KV pool holds 17 full 262K conversations, and the server admits 8 requests at a time; the rest queue. On sm_80 cards (A100, A30, CMP 170HX) this checkpoint runs a BF16 KV cache, since its attention path takes FP8 KV only on sm_86 and sm_89 for now, so fewer sessions fit; FP8 there comes in a 3.0.x update. 🟢 measured by us (the 4× CMP 170HX row on a rented host) · ⚫ a profile ships, not yet run by us (expected to work). Every profile: the engine's [hardware matrix](https://github.com/halt95/merlin-engine/blob/main/docs/merlin/hardware-matrix.md).</sub>

| Need | Details |
|---|---|
| Host RAM | 96 GiB reference allocation; the FP8 n-gram table alone takes about 47.7 GiB. The minimum has not been re-measured on 3.0.0. |
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
curl -LO https://github.com/halt95/merlin-engine/releases/download/merlin-v3.0.0/vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl
MERLIN_WHEEL="$PWD/vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl" \
  bash release/install-env.sh ./venv

venv/bin/hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin \
  --local-dir ~/models/flash-next
(cd ~/models/flash-next && sha256sum -c "$OLDPWD/models/flash-next/checkpoint.sha256")
venv/bin/python merlin_cli.py serve ~/models/flash-next
```

`MERLIN_WHEEL` installs the prebuilt engine (sm_80, sm_86, sm_89), checked against `release/merlin-wheel.sha256`; without it the installer compiles the engine instead, about 1.5 hours on 8 cores. `merlin serve` recognises the checkpoint, picks the largest tested layout for the GPUs it finds (or, with none, the largest untested one that fits, labelled as such) and serves on `127.0.0.1:8000`. Add `--dry-run` to print the resolved command without starting anything. The first start compiles kernels.

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
| Context | 262,144 tokens native (Qwen: extensible to 1M with YaRN; this build serves 262,144) |

<sub>Architecture as reported by Qwen on the [upstream card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).</sub>

## Other builds

| Checkpoint | Base | GPUs | Decode at 262K |
|---|---|---|---|
| **Flash-Next (this card)** | **Qwen3.8-Flash-Next, 125B MoE** | **4× 3090** | **184 tok/s** |
| [Swift1.5](https://huggingface.co/halt95/Swift1.5-Qwen3.8-Flash-Next-W4A16-Merlin) | Flash-Next fine-tune | 4× 3090 | 177 tok/s |
| [Swift1.5-Huihui](https://huggingface.co/halt95/Swift1.5-Huihui-Qwen3.8-Flash-Next-W4A16-Merlin) | Swift1.5, abliterated | 4× 3090 | 189 tok/s |
| [Qwen3.8-27B](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin) | Qwen3.8-27B, dense | 2× or 4× 3090 | 142 / 215 tok/s |
| [ThinkingCap](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin) | Qwen3.8-27B fine-tune | 2× or 4× 3090 | 165 / 248 tok/s |
| [Nemotron 3.5 Super VL](https://huggingface.co/halt95/Nemotron-3.5-Super-VL-120B-A12B-W4A16-Merlin) | Nemotron 3.5 Super VL, 120B-A12B MoE | 4× 3090 | 91 tok/s |
| [Ornith 1.5](https://huggingface.co/halt95/Ornith-1.5-122B-A10B-W4A16-Merlin) | Qwen3.5-122B-A10B fine-tune (ornith-ai), 122B MoE | 4× 3090 | 100 tok/s |

<sub>One stream, thinking on, 261,888 prompt tokens + 128 out, median of 3; Merlin 3.0.0, 2026-10-20 bench cards. Effort as each bench card ran it: Flash-Next low; Swift1.5, Swift1.5-Huihui and ThinkingCap xhigh; Qwen3.8-27B medium. Merlin now serves `medium` on every Qwen3.8 checkpoint by default. Nemotron and Ornith 1.5 ran at full thinking: neither template reads `reasoning_effort`. Ornith 1.5 was measured on 2026-10-21 on the reference host with its engine changes (uniform MoE verify on) before they merged into 3.0.0.</sub>

## Good to know

- Every figure we measured for this checkpoint comes from one setup: 4× RTX 3090 at 220 W, PCIe Gen4 x16 with P2P, except the rented-host row under Other GPUs; upstream results and the Other builds rows use their stated setups. Hosts without P2P use the no-P2P profiles (see [What it fits on](#what-it-fits-on)).
- Only if you shrink the KV pool yourself: keep it larger than `--max-model-len` plus a few blocks. With the shipped settings this cannot occur.
- Coming from the `qwen38-flash-next-3090s` engine (v2.x or earlier)? It is replaced by the [Merlin engine](https://github.com/halt95/merlin-engine); this checkpoint's files are unchanged.

## Changes

| Date | Change to this repository |
|---|---|
| 2026-10-06 | `tokenizer.json` replaced with the upstream file; re-fetch it if you downloaded earlier |
| 2026-09-18 | `config.json` gains `ple_embedding_dtype` in `text_config`, which the engine needs since v2; re-fetch it if you downloaded earlier |
| 2026-09-08 | Weights published; unchanged since |

<sub>Card-only edits are not listed. Engine releases and their speed changes: the engine's [CHANGELOG](https://github.com/halt95/merlin-engine/blob/main/CHANGELOG.md).</sub>

## Credits and licence

Qwen (model; Qwen Community License 1.0, see `LICENSE`), Intel (AutoRound experts), RadixArk (FP8 PLE table), DominikBucko (MTP packing recipe), [froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (the fixed Qwen3.8 chat template the engine serves), the vLLM community PLE-offload work, aikitoria (peer-to-peer on consumer Ampere), [Neiko2002](https://huggingface.co/Neiko2002), and [CoolShades](https://github.com/CoolShades) (the idle-power report behind the v2.5.2 engine fix). Upstream fixes are credited in the engine repository.
