---
license: apache-2.0
language:
  - en
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
  - qwen3.8
  - qwen3_5
  - awq
  - gptq
  - int4
  - w4a16
  - compressed-tensors
  - fp8-kv
  - mtp
  - vision
  - vllm
  - long-context
  - rtx-3090
  - ampere
  - sm86
  - speculative-decoding
---

# Qwen3.8-27B-W4A16-Merlin

Qwen3.8-27B for serving at its full 262K context on dual or quad RTX 3090s: philbert's asymmetric int4 AWQ checkpoint
(group size 128) of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), with three additions:

- an **int4 GPTQ `lm_head`** (group 128, asymmetric);
- an **int4 MTP drafter head**, so MTP speculative decoding runs from int4 weights;
- **calibrated per-layer FP8 E4M3 KV-cache scales** for the full-attention layers.

The vision tower and every other tensor are unchanged from
[philbert440/Qwen3.8-27B-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ).
Weights are stored in the compressed-tensors format (11 checkpoint files, about 17 GB).

## What changed in this revision (v1.3.0)

**`tokenizer.json` is now the upstream Qwen3.8-27B file** (sha256 `0997f410c57a1f4e…`). The file this repository shipped
in v1.0.0 (sha256 `f913b92a26abaa0e…`), inherited from the AWQ base, had a pre-tokenizer split pattern without `\p{M}`,
so combining marks did not join the letters they follow: Hindi, Thai, Arabic and other combining-mark scripts were split
differently from upstream. Vocabulary and merges were already
equal. No weight, scale or config file changed. English, code, JSON and CJK tokenize identically either way.

**Serve with `transformers==5.18.0`.** transformers 5.17.0 and earlier ignore `tokenizer.json` for this tokenizer
and rebuild the pre-tokenizer from a built-in pattern without `\p{M}`, so with them the file swap alone does not change
what is served. The serving repository pins 5.18.0. Other consumers (llama.cpp conversions, `tokenizers` directly, other
frameworks) read `tokenizer.json` and get the upstream behaviour from this revision on.

The v1.3.0 serving repository also moves the engine and the defaults, see below.

## Serving

Use the serving repository, version v1.3.1 (v1.3.0 with documentation fixes):
[halt95/qwen38-27b-3090s](https://github.com/halt95/qwen38-27b-3090s), tag `v1.3.1`. It installs the
Merlin engine v2.5.2 (vLLM 0.30.0 plus the Flash-Next fork commits and the guard fix) with an eleven-patch 27B series (0001-0011) and a FlashInfer
32-row verify tile (hash-locked environment), and serves this checkpoint at 262K context on dual RTX 3090s (TP=2) or
quad (TP=4), with MTP K=4, fp16 SSM state and FP8 KV. Validated on sm_86 (Ampere) only. Single card is parked for this
release and returns in a later one. This release is a bare-metal install; it ships no container image. The default
reasoning effort is `medium` (`THINKING` in the serve scripts; a request can choose its own with
`"chat_template_kwargs": {"reasoning_effort": "..."}`).

```bash
git clone --branch v1.3.1 https://github.com/halt95/qwen38-27b-3090s && cd qwen38-27b-3090s
# the two engine assets pinned in upstream/PIN go in the clone's parent directory (install-env.sh checks both hashes):
(cd .. && curl -fLO https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.5.2/v2.5.2-from-upstream-v0.30.0.bundle.gz \
       && curl -fLO https://github.com/halt95/qwen38-flash-next-3090s/releases/download/v2.2.0/build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz)
release/install-env.sh .venv
HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/hf download halt95/Qwen3.8-27B-W4A16-Merlin --local-dir ./ckpt
(cd ckpt && sha256sum -c ../release/checkpoint.sha256)
MODEL=$PWD/ckpt serve/serve.sh          # serve/serve-tp4.sh for quad card
```

It is working when the log reads `GPU KV cache size: 826,689 tokens` (1,974,072
on quad card; the pool is pinned in bytes, so any other number means a different setup) and `Application startup
complete`, and a one-token generation on `127.0.0.1:8100` (model `qwen38-27b`) answers. Needs Linux x86_64, Python 3.13, a
driver for CUDA 13.0, `nvcc` for FlashInfer's JIT, and **about 2.4 GiB of extra system RAM**: the serving stack's own
host-resident embedding patch keeps the token-embedding table in pinned host RAM (1.184 GiB per GPU on dual card, 0.592
on quad card) and gives that VRAM to the KV cache. Full requirements, troubleshooting and every knob are in the
repository's `docs/reference.md`. Both serve scripts expect working GPU peer-to-peer (P2P).
The serve scripts apply the chat template vendored in the repository (froggeric's Qwen-Fixed-Chat-Templates v22.5,
`serve/templates/`), not this repository's `chat_template.jinja`; set `CHAT_TEMPLATE=` (empty) to use the checkpoint's own.

Second checkpoint of the same release: [halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/ThinkingCap-Qwen3.8-27B-W4A16-Merlin),
BottleCap AI's ThinkingCap fine-tune of this model, which reasons in fewer tokens, in the same format. Its weights are under
the PolyForm Small Business License 1.0.0 plus BottleCap AI's personal-use permission, not Apache-2.0. It is served by the
same repository with `serve/serve-thinkingcap.sh` (`serve/serve-thinkingcap-tp4.sh` for quad card), which default the
reasoning effort to `xhigh`; the repository README's Models table compares the two.

## Measured (v1.3.0, reference host)

Shipped profiles at the release gate (2026-10-08 to 2026-10-09 (UTC), the shipped serve scripts with no knob set). The
255K rung targets 261,120 tokens (259,587 as sent at the release gate).

| | |
|---|---|
| KV pool at K=4, shipped pins | 826,689 tokens on dual card (883 blocks, 3.15 × 262,144); 1,974,072 on quad card (2,470 blocks, 7.53×); both printed by the release gate boots |
| single-stream decode, thinking on (effort low, 256-token generations), K=4, shipped profiles, 4K / 32K / 128K / 255K | dual card **206 / 198 / 172 / 144** tok/s; quad card **258 / 249 / 238 / 226** tok/s (release gate) |
| GSM8K-200 | release gate **196** on dual card and **194** on quad card; dual card with the FlashInfer verify tile off 194 |
| 262K needle | passed on dual card and quad card at the release gate (cold 261,568-token prompt: 307 s dual card, 181 s quad card) |

**Quality and stability (release gate, both shipped profiles).** Residency with cold sessions: dual card 3 × 261,600 tokens
and 6 × 130,800, quad card 7 × 261,600, every session returned its code, 0 preemptions. 8-conversation load: dual card 217
requests and quad card 466, 0 errors, prefix-cache hit 92.6 % and 88.6 %, 0 preemptions. Multi-turn replay median follow-up
TTFT 2.79 s on dual card and 1.78 s on quad card against 22.9 s and 15.3 s for the first turn. Repetition loops: 72
generations per profile, **0 genuine collapses**. Smokes: warm prefix hit, tool call, reasoning split and vision (4 of 4
images) passed; JSON-schema structured output was not tested at the gate. Tokenizer: the served tokenizer equals the upstream
one on the frozen 153-item combining-mark set plus controls (delta-audit tree, transformers 5.18.0). For the previous release
(v1.0.0, stock vLLM 0.30.0, K=3): 0 genuine repetition collapses in 108 long generations (the serving repository's
`docs/loop-check.md`).

The bench card is in the repository: `benchmarks/2026-10-12/BENCH-CARD.md`.

## How it was made

- The `lm_head` and the MTP drafter head were quantized with GPTQ using input Hessians captured
  from the served model; the drafter head was calibrated on drafter inputs. Codes are packed with
  compressed-tensors `pack_to_int32` in the signed range.
- FP8 E4M3 KV scales were calibrated per full-attention layer from observed absolute maxima,
  with a 1.10 margin, on the philbert checkpoint before the head and drafter were quantised; that traffic included
  the first 50 GSM8K test questions (activation maxima only).
- **Calibration data:** public benchmark prompts (ARC, MMLU-Pro, GPQA, GSM8K train split, HumanEval),
  template-generated synthetic tasks and a small hand-written prompt set, together with the
  model's own responses to them. No calibration data is distributed.
- The tools are in the serving repository under `quant/`. The AWQ recipe for the base weights
  is philbert's; see [philbert440/Qwen3.8-27B-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ).
- `tokenizer.json` (this revision): the upstream Qwen3.8-27B file, byte for byte.

## Known behaviours and limits

All measurements are from one reference host (RTX 3090s, PCIe Gen4 x16, P2P-enabled driver, 220 W cap).

- sm_86 only. Other GPUs, non-P2P hosts and narrower PCIe links are unmeasured. Single card is not supported by this
  release.
- The FlashInfer verify tile is not bitwise equal to stock FlashInfer; with it on (the default) or off, outputs can differ
  at near-tied tokens. The same holds for `SPEC_K`.
- Greedy (T=0) output repeats within one compile cache, not across fresh compiles.
- A request reusing prefix-cache blocks can differ from a cold one at near-tied tokens. In the gate's multi-turn replay a follow-up took a median 2.79 s to the first token on dual card against 22.9 s for the first turn; that run did not report cached-token counts.
- The base model can loop occasionally; the release gate's 72-generation check per profile found 0 genuine collapses.
- No long soak was run for this release; the gate ran the 8-conversation load, the multi-turn replay, the loop check and the image loads on both profiles with 0 errors.

## Licence and attribution

Apache-2.0, like the base model and the AWQ checkpoint. Credit to Qwen (Alibaba Cloud) for
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and to philbert for the AWQ checkpoint
[philbert440/Qwen3.8-27B-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ). The serving stack is
[halt95/qwen38-27b-3090s](https://github.com/halt95/qwen38-27b-3090s) (Apache-2.0); its NOTICE credits vLLM, lucifer1004's vLLM
PR #47979, FlashInfer, compressed-tensors, froggeric's chat template, aikitoria's P2P kernel-module patch and CoolShades
(the idle-power report and PR #3 on the Flash-Next repository, behind the engine's guard fix); this repository's own
`NOTICE` covers the checkpoint. Sibling
release for the MoE model: [halt95/qwen38-flash-next-3090s](https://github.com/halt95/qwen38-flash-next-3090s).
