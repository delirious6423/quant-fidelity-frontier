---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - gguf
  - llama.cpp
  - rocm
  - amd
  - rocmfp4
  - rocmfpx
  - strix-halo
  - gfx1151
  - uncensored
  - abliterated
  - mtp
  - speculative-decoding
pipeline_tag: text-generation
---

# Qwen3.8-27B Uncensored — ROCmFP4 for AMD Strix Halo (gfx1151)

Uncensored ROCmFP4 quantisations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), built with the **same ROCmFPX pipeline and the same card ftypes** as [`kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF).

**Research artifact.** Abliteration removes content-refusal. It does not add capability. Do not ship this as a product default. The aligned repo remains the serving default.

| Metric | Result |
| --- | --- |
| Quantization | ROCmFP4 **103 FAST** + **106 STRIX_LEAN**; also Q6 114/116 |
| Source | Qwen3.8-27B BF16 GGUF |
| Hardware | Ryzen AI Max+ 395 / Radeon 8060S / gfx1151 / 128 GB / ROCm 7.2.4 |
| Drafter | `mtp-Qwen3.8-27B-Q4_0.gguf`, `--spec-draft-n-max 4` |
| Unc FAST 103 decode @8K MTP | prose **23.08** · code **42.80** tok/s |
| Unc STRIX_LEAN 106 decode @8K MTP | prose **24.73** · code **42.70** tok/s |
| Aligned STRIX 105 MTP n=4 (published) | **30.30 tok/s** |
| Aligned STRIX_LEAN 106 (published) | 13.59 GiB, PPL 5.8871, same speed as STRIX |

## Why this build?

The aligned card's finding still stands: **the lever is MTP depth, not the ftype.** llama.cpp's default `--spec-draft-n-max 16` is about half the achievable throughput on this model. The knee is **n=4**. These uncensored files are the card's FAST (103) and STRIX_LEAN (106) from an abliterated BF16, plus Q6 AGENT/LEAN.

On the aligned family, FAST is **dominated** (same speed, worst PPL). Prefer **106** here unless you need the smallest file.

Q6 `LEAN` (ftype **116**) is **not** the card LEAN. Card LEAN = **106**.

## Which file should I use?

**Start with `STRIX_LEAN` (106)** — the card-LEAN equivalent. Take `Q6 AGENT` (114) if you want protected heads. Do not default to FAST on this architecture.

Ryzen AI Max+ 395, ROCm 7.2.4, MTP `--spec-draft-n-max 4`, `-fa on`, ctx 8192, batch 1, temperature 0, thinking **off** so the 256-token budget is decode. Warm medians of 3.

| Build | ftype | Size | prose tok/s | code tok/s |
| --- | --- | --- | --- | --- |
| **Unc FAST** | 103 | 13.33 GiB | 23.08 | 42.80 |
| **Unc STRIX_LEAN** | 106 | 13.59 GiB | 24.73 | 42.70 |
| Unc Q6 AGENT | 114 | 23.15 GiB | — | ~23 (no-spec smoke; serve with MTP) |
| Unc Q6 LEAN | 116 | 20.37 GiB | — | ~10 no-spec / ~28 MTP n=4 (earlier sweep) |
| *Aligned STRIX (published)* | 105 | 13.75 GiB | — | **30.30** MTP n=4 |
| *Aligned STRIX_LEAN (published)* | 106 | 13.59 GiB | — | 13.46 no-spec |

## Quick start

```bash
llama-server \
  -m Qwen3.8-27B-Uncensored-ROCmFP4-STRIX_LEAN.gguf \
  --spec-type draft-mtp --model-draft mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-draft-ngl 99 --spec-draft-device ROCm0 \
  --spec-draft-n-max 4 --spec-draft-n-min 0 --spec-draft-p-min 0.0 \
  -ngl 999 -fa on -dio --jinja -fit off --parallel 1 -dev ROCm0 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  -c 65536 --host 127.0.0.1 --port 8080
```

Use the **Q4_0** draft head, not Q8_0. Official `mtp-Qwen3.8-27B-Q4_0.gguf` works on this vocab (248k).

| Flag | Why |
| --- | --- |
| `--spec-draft-n-max 4` | Default 16 lands on the wrong side of the curve. |
| `--spec-draft-ngl 99` | Without it the draft head can sit on CPU and the gain vanishes. |
| `--jinja` | Otherwise `chat_template_kwargs` are silently ignored. |
| `-fit off` | Autofit on iGPU can shrink context after an unload. |
| `enable_thinking: false` **or** `reasoning_effort: low` | Default is `xhigh`. Small `max_tokens` then returns empty `content`. `none` **throws**. |

> ### ⚠️ If the MTP draft command crashes on your build
> The separate-model `draft-mtp` path has known bugs in the legacy `charlie12345/ROCmFPX`
> line — reported on **Windows 11 / gfx1151 / HIP SDK 7.2** (five stacked bugs, ending in an
> h-row width mismatch in the draft's embedding buffer).
>
> **Fix: build the official repo instead — no patches needed.**
>
> ```bash
> git clone https://github.com/ROCmFPX/ROCmFPX.git
> ```
>
> The MTP path was reworked there (unified `n_embd_out` row widths, `t_h_nextn` reset in
> `llm_graph_result::reset()`, `ctx_other` wired centrally), which covers the whole chain.
> Reported and verified on that configuration: 24–31 tok/s, coherent output, tool calling working.
> Legacy-line patch: [PR #109](https://github.com/charlie12345/ROCmFPX/pull/109).
>
> Linux builds on the legacy line are not known to be affected.


## Speculative decoding (MTP)

Same recipe as the aligned card. Qwen3.8-27B has `nextn_predict_layers = 1` as a **separate** draft GGUF, not in-model nextn.

Aligned MTP curve (STRIX, 8K): n=3 30.13 · **n=4 30.30 (knee, accept 0.926)** · n=5 27.52. An earlier sweep on uncensored Q6 LEAN (116) reproduced the same knee: n=3 24.96 · **n=4 27.66** · n=5 25.27.

## Uncensored findings

Content-refusal scoring on a 24 harmful / 12 harmless / 8 quality research set, greedy. **Counts only.** Refusal on the uncensored family was scored on **Q6 AGENT (114)**; 4-bit 103/106 are the same checkpoint and should not restore refusals.

| Model | Harmful 24 | Harmless 12 | Quality 8 |
| --- | --- | --- | --- |
| Qwen3.8 aligned Q8 AGENT | 23 refuse, 1 comply | 11/12 ok (1 over-refuse) | 6/8 |
| Qwen3.8 uncensored Q6 AGENT (114) | 23 comply, 1 broken | 11/12 ok (1 over-refuse) | 6/8 |
| Muse aligned Q6 AGENT (114) | 18 refuse, 6 comply | 11/12 ok (1 over-refuse) | 7/8 |
| Muse uncensored STRIX_LEAN (106) | 24 comply | 12/12 ok (0 over-refuse) | 7/8 |

Reading:

- Aligned Qwen refuses ~all of this set. Abliterated Qwen complies ~all of it.
- Quality is **byte-identical as a score** (6/8 both arms, same two fails). Abliteration here removed refusals without moving the smoke check.
- One harmless over-refuse survived on **both** Qwen arms — the refusal classifier, not a unique unc defect.

## Files

| File | ftype | Size | Role |
| --- | --- | --- | --- |
| `Qwen3.8-27B-Uncensored-ROCmFP4-FAST.gguf` | 103 | 13.33 GiB | smallest 4-bit; dominated on aligned PPL |
| `Qwen3.8-27B-Uncensored-ROCmFP4-STRIX_LEAN.gguf` | 106 | 13.59 GiB | **card-LEAN — recommended 4-bit** |
| `Qwen3.8-27B-Uncensored-Q6_0_ROCMFPX_AGENT.gguf` | 114 | 23.15 GiB | 6-bit, Q8 head/attn |
| `Qwen3.8-27B-Uncensored-Q6_0_ROCMFPX_LEAN.gguf` | 116 | 20.37 GiB | 6-bit throughout |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | — | 1.56 GiB | MTP draft head — **use this** |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | — | 0.59 GiB | vision projector (unmodified) |

Six files. Aligned STRIX/FAST/LEAN are **not** in this repo.

## Quantization

```bash
llama-quantize Qwen3.8-27B-Uncensored-BF16.gguf Qwen3.8-27B-Uncensored-ROCmFP4-FAST.gguf Q4_0_ROCMFP4_FAST 16
llama-quantize Qwen3.8-27B-Uncensored-BF16.gguf Qwen3.8-27B-Uncensored-ROCmFP4-STRIX_LEAN.gguf Q4_0_ROCMFP4_STRIX_LEAN 16
```

No extra `--tensor` flags — matches the aligned recipe. Architecture is `qwen35`; no port required.

## Architecture note (unchanged)

Dense hybrid attention, 64 layers, `full_attention_interval = 4`. KV is cheap. **Prompt caching does not work** on hybrid/recurrent memory in stock llama.cpp — budget full prefill every turn. MTP prompt-cache fix from the aligned repo is optional and not required to load these files.

## Known issues

1. Vulkan/CUDA/CPU cannot load these files.
2. `--spec-draft-n-max` defaults to 16 — set 4.
3. `reasoning_effort: "none"` throws. Use `enable_thinking: false`.
4. Small `max_tokens` + thinking = empty `content`.
5. This is an uncensored research build. Do not deploy it as the public default.

## Not yet measured

| Test | Status |
| --- | --- |
| Perplexity vs aligned STRIX_LEAN 5.8871 | ❓ not re-run on unc |
| Tool-calling 7/7 on unc weights | ❓ not re-run |
| Vision 4/4 spatial on unc | ❓ projector reused |
| Independent reproduction | ❓ none yet |

## License and attribution

Base model: Qwen team, Apache 2.0. MTP draft head redistributed with the Qwen GGUF companions. ROCmFP4 types: [ROCmFPX](https://github.com/charlie12345/ROCmFPX). This repository is quantisation and measurement of an abliterated Qwen3.8-27B checkpoint.

See the aligned card for the MTP depth curve, tool-calling suite, and vision results.
