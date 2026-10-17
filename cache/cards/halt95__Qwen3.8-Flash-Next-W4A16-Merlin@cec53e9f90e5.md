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
  - w4a16
  - autoround
  - mtp
  - speculative-decoding
  - fp8-kv
  - long-context
  - rtx-3090
  - ampere
  - sm86
---

# Qwen3.8-Flash-Next-W4A16-Merlin

**Run Qwen3.8-Flash-Next at its full 262K context on four RTX 3090s, with vision, tool calls and fast agent
follow-ups.**

![Flash-Next v2.5.2 at a glance](https://raw.githubusercontent.com/halt95/qwen38-flash-next-3090s/v2.5.2/docs/images/flashnext-v2.5-summary.png)

A 4-bit build of Qwen3.8-Flash-Next (125B MoE) made for consumer Ampere cards, served by the patched vLLM engine in
the companion repository, [halt95/qwen38-flash-next-3090s](https://github.com/halt95/qwen38-flash-next-3090s). The
whole KV cache stays in VRAM: long conversations are never paged out to system RAM.

## Why use it

- **Long context that actually fits.** A 924,993-token FP8 KV pool: three full 262K sessions, or six 131K sessions,
  resident at once.
- **Built for agents.** Follow-up turns resume from cache: in one conversation the next turn starts in **0.96 s**; with
  8 agents at once (temperature 0), 92.5 % of follow-up tokens come from cache and the median follow-up starts in 1.61 s.
- **Fast.** About 162–177 tok/s single-stream (thinking on, MTP speculative decoding), 534 tok/s across 8 concurrent
  requests, and 5,000–5,530 tok/s prefill from 10K to 261K tokens.
- **Quality kept.** Experts in Intel's INT4 AutoRound, attention in BF16, every release gated against the BF16
  model (divergence, 262K needle, GSM8K 198/200, structured output 80/80).
- **Vision and tools.** Up to 42 images per request (4 MP each), Qwen3 tool calling and reasoning parsers, one OpenAI-compatible
  endpoint.

![Prefill and decode over context depth, v2.5.1 against v2.2.0](https://raw.githubusercontent.com/halt95/qwen38-flash-next-3090s/v2.5.1/benchmarks/2026-10-06/flashnext-v2.5.1-ctx-pp-tg-itl.png)

## What you need

- 4× RTX 3090 (24 GB, sm_86), headless, with no other CUDA process on them. PCIe peer-to-peer is recommended (it also
  runs without, with slower prefill); no NVLink needed.
- 96 GB of host RAM (the model's n-gram embedding table lives there), `/dev/shm` of at least 1 GB, and about 250 GB of
  free disk (the checkpoint is 115 GiB).
- Linux x86-64 with glibc 2.34 or newer, an NVIDIA driver 580 or later (CUDA 13.0), Python 3.13 with venv and headers,
  a C/C++ compiler, ninja, git and curl. The build installs its own CUDA 13.0 toolchain into the venv.

## Quick start

The `hf` command is the Hugging Face CLI (`pipx install huggingface_hub`).

```bash
git clone --branch v2.5.2 https://github.com/halt95/qwen38-flash-next-3090s.git && cd qwen38-flash-next-3090s
R=https://github.com/halt95/qwen38-flash-next-3090s/releases/download
curl -fLO "$R/v2.5.2/v2.5.2-from-upstream-v0.30.0.bundle.gz" && gunzip v2.5.2-from-upstream-v0.30.0.bundle.gz
curl -fLO "$R/v2.2.0/build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz"
hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin --local-dir ~/models/Flash-Next-Merlin

BUNDLE=./v2.5.2-from-upstream-v0.30.0.bundle ARTIFACTS=./build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz \
  scripts/build-v2.5.sh ./vllm-v2.5 ./venv-v2.5
TREE=./vllm-v2.5 VENV=./venv-v2.5 CACHE_ROOT=./.vllm-cache-v2.5 scripts/serve-v2.5.sh ~/models/Flash-Next-Merlin
```

It is ready when the log shows `Application startup complete`. It serves on 127.0.0.1:8000; set `VLLM_API_KEY`
before exposing it with `HOST=0.0.0.0`. Then talk to it like any OpenAI endpoint:

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "flash-next", "messages": [{"role": "user", "content": "Hello!"}]}'
```

The first start compiles kernels (about 5–6 minutes); later starts take about 2–3 minutes. Container and Proxmox LXC
recipes, a health check and troubleshooting are in the companion repository's
[reference](https://github.com/halt95/qwen38-flash-next-3090s/blob/main/docs/reference.md).

## New in v2.5.2

The bounds-guard reader now parks once the engine drains, so idle cards drop to P8: about 110 W for four idle cards
instead of 446 W in P2 on the reference host, with every bounds check still on (`VLLM_E2_GUARD_IDLE=0` restores the
always-polling reader). Weights, compiled ops and speed are unchanged; the same engine commit ships in the Qwen3.8-27B
releases. Thanks to [CoolShades](https://github.com/CoolShades) for the idle-power report in
[issue #2](https://github.com/halt95/qwen38-flash-next-3090s/issues/2) and
[PR #3](https://github.com/halt95/qwen38-flash-next-3090s/pull/3), which led to this fix. Measurements: the
[v2.5.2 bench card](https://github.com/halt95/qwen38-flash-next-3090s/blob/main/benchmarks/2026-10-12/BENCH-CARD.md);
release notes: the [CHANGELOG](https://github.com/halt95/qwen38-flash-next-3090s/blob/main/CHANGELOG.md#v252).

## New in v2.5.1

v2.5.1 is an engine release: the weights are unchanged since 2026-09-08. Downloaded before 2026-09-18? Re-fetch
`config.json` (the v2.5 serve script needs its `ple_embedding_dtype` key). `tokenizer.json` is now the upstream file.
v2.5.0 was not published, so this is everything since v2.2.0.

- Agent follow-up turns resume on a 64-token grid instead of a 4,096-token block, and a conversation's cached prefix
  is kept until its next turn: a follow-up starts in 0.96 s (v2.2.0: 1.56 s).
- A larger KV pool (924,993 tokens, up from 806,792) on the same weights.
- Faster cold prefill, correct tokenization of combining marks (transformers 5.18.0), up to 42 images per request.
- Single-stream decode is about 3 % slower than v2.2.0.

Full measurements: the [v2.5.1 bench card](https://github.com/halt95/qwen38-flash-next-3090s/blob/main/benchmarks/2026-10-06/BENCH-CARD.md).

## What's inside

| component | format |
|---|---|
| routed experts (512 per layer) | INT4 g128, from [Intel's AutoRound release](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) |
| MTP draft-head experts | INT4 g128, packed in place |
| Gated DeltaNet projections | INT8 per channel |
| n-gram (PLE) embedding table, 51B parameters | FP8, kept in host RAM at serve time ([RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)) |
| attention, norms, embeddings, lm_head, shared expert, vision tower | BF16, unchanged |
| KV cache | FP8 E4M3 with calibrated static scales (`qsa_kv_scales_262k.json`) |

![How the model is laid out on four RTX 3090s](https://raw.githubusercontent.com/halt95/qwen38-flash-next-3090s/v2.5.2/docs/images/flashnext-v2.5-layout.png)

This checkpoint needs the companion repository's engine; stock vLLM was not tested with it.

## Good to know

- Tested on one setup: 4× RTX 3090 at 220 W, PCIe Gen4 x16. Without peer-to-peer it works, with lower prefill.
- Only if you shrink the KV pool yourself: keep it larger than `--max-model-len` plus a few blocks. With the shipped
  settings (a 924,993-token pool, requests up to 262,144 tokens) this cannot occur.
- With 8 long agent sessions and a full pool, a few follow-ups per 5 minutes still re-read their whole conversation.

## Credits

Qwen (model and licence: Qwen Community License 1.0, see `LICENSE`), Intel (AutoRound experts), RadixArk (FP8 PLE
table), DominikBucko (MTP packing recipe), the vLLM community PLE-offload work, aikitoria (peer-to-peer on consumer
Ampere), and [CoolShades](https://github.com/CoolShades) (idle-power report, behind the guard fix in v2.5.2). Upstream fixes are credited by PR number in the companion repository's commit messages and by author in its `NOTICE`.
