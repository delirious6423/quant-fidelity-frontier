---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - exl3
  - exllamav3
  - qwen3.8
  - flash-next
  - moe
  - mtp
  - quantized
  - ngram
  - unsloth
language:
  - en
  - zh
pipeline_tag: image-text-to-text
---

# Qwen3.8-Flash-Next — EXL3 3.82 (Unsloth-shaped, h8 / mtp5 / ng6)

EXL3 quantization of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Built with [ExLlamaV3](https://github.com/turboderp-org/exllamav3) **1.5.0** on **3× RTX 3090 24 GB**.

This is **not** a Transformers / vLLM / SGLang checkpoint. The original Qwen README that
landed in the first upload was leftover BF16 documentation. Load it with ExLlamaV3
(TabbyAPI, or any exllamav3 server). `tensor_parallel` is **not** implemented for Qwen4Exp.

## What this is (and why)

The sibling 3.50 HQ cut leaves VRAM unused on a 3×24 GB box. Instead of a uniform `--hq`
run, this build uses a **per-tensor "Unsloth UD" shape** recipe (target **3.821 bpw**) that
spends the ~5 GiB of leftover VRAM where it matters:

- `lm_head` 6 bpw, attention / Gated DeltaNet 8 bpw, shared experts 6 bpw, special `down_proj` 6 bpw
- middle MoE expert `up/gate` 3–4 bpw (`down >= up/gate`)
- MTP head **5** bpw, n-gram (PLE) table **6** bpw, vision tower **6** bpw

The result is a higher effective bitrate (+~0.3 bpw over 3.50 HQ) focused on the sensitive
tensors, still inside the same 3×3090 memory envelope — so it should match or beat 3.50 HQ at
serve time without a bigger card. `--hq` is ignored when `--recipe` is used.

| | |
|---|---|
| Source | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (BF16, 336 GB) |
| Recipe | per-tensor Unsloth UD-IQ4_XS shape, target `3.821` bpw |
| Size | **~104 GB** (9 shards = 65.2 GB + 39.0 GB n-gram table) |
| Effective bitrate (`quantization_config.bits`) | **3.82** |
| `head_bits` / `lm_head` | 8 / 6 bpw |
| n-gram (PLE) table | 6 bpw |
| MTP head | 5 bpw (native `component="mtp"`, 6203 `mtp.*` keys) |
| Vision tower | 6 bpw |
| Codebook / scales | `mul1` / `out_scales=always` |
| Calibration | 250 rows × 2048 cols |
| ExLlamaV3 | 1.5.0 |

## Related models

- **Base (BF16):** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Sibling 3.50 HQ cut:** [groxaxo/Qwen3.8-Flash-Next-exl3-3.50bpw_hq_h6_ng6](https://huggingface.co/groxaxo/Qwen3.8-Flash-Next-exl3-3.50bpw_hq_h6_ng6)
- **Upstream style parent:** [turboderp/Qwen3.8-Flash-Next-exl3 `4.05bpw_h6_ng6`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6)

MTP here is **inside this checkpoint** — do not add an external draft model.

## Provenance and related experiments

The per-tensor "Unsloth shape" is adapted from the **UD-IQ4_XS** GGUF quantization of the
Uncensored sibling. There, Unsloth's UD-IQ4_XS tensor-type map (experts mostly `iq3_s`, PLE
table `iq4_nl`, attention Q/K/MLP at higher precision — 643 overrides) was driven by a live
agentic importance matrix. This EXL3 recipe keeps that *shape* — low middle-expert `up/gate`,
higher `down` / attention / shared experts — re-expressed as EXL3 bpw targets.

Source experiments (host `laguna-s21`, 3× RTX 3090), in
[`groxaxo/experimentos`](https://github.com/groxaxo/experimentos/tree/main/qwen-next-flash):

- **Shape origin + cyber/template A/B** — the abliterated UD-IQ4_XS GGUF pipeline (FP8 →
  imatrix → 643-override UD map → 88 GB GGUF) and the sharp-vs-baked template matrix:
  [`qwen-next-flash/README.md`](https://github.com/groxaxo/experimentos/blob/main/qwen-next-flash/README.md)
- **Sibling 3.50 HQ EXL3 campaign** — plan and journey for the 3.50/3.75 cuts this build
  extends: [`PLAN.md`](https://github.com/groxaxo/experimentos/blob/main/qwen-next-flash/exl3-350hq-375-2026-09-15/PLAN.md),
  [`JOURNEY.md`](https://github.com/groxaxo/experimentos/blob/main/qwen-next-flash/exl3-350hq-375-2026-09-15/JOURNEY.md)
- **Native MTP draft-window sweep** — draft 3 (88.2 tok/s) beat draft 6 (62.4) and MTP-off
  (64.7) on short prose; the serve tuning behind `ENABLE_MTP=1` / `NUM_DRAFT_TOKENS=3`:
  [`BLOG.md`](https://github.com/groxaxo/experimentos/blob/main/qwen-next-flash/exl3-350hq-mtp-draft-q8-2026-09-16/BLOG.md)
- **Stable-prefix variant switch** — why instruct↔thinking switches re-prefill only the last
  page (Sharp template + exllamav3 page cache):
  [`README.md`](https://github.com/groxaxo/experimentos/blob/main/qwen-next-flash/stable-prefix-variant-switch-2026-09-16/README.md)

Copies of the first three are mirrored in this repo under [`experiments/`](./experiments).

## Conversion

```bash
python convert.py \
  --in_dir Qwen3.8-Flash-Next \
  --out_dir Qwen3.8-Flash-Next-exl3-unsloth-shaped-h8-mtp5 \
  --work_dir exl3-work-qwen38-flash-unsloth-fill \
  --recipe recipe-unsloth-shaped-fill.yaml \
  --bits 3.82 \
  --head_bits 8 \
  --mtp_bits 5 \
  --ngram_bits 6 \
  --ngram_file Qwen3.8-Flash-Next-exl3-3.50bpw_hq_h6_ng6/ngram_embedding.safetensors \
  --vision_bits 6 \
  --out_scales always \
  --codebook mul1 \
  --cal_rows 250 \
  --cal_cols 2048 \
  --checkpoint_interval 120 \
  --devices 0,1,2
```

`tensor_parallel` must stay false at serve time.

## Serve (measured on this box)

Host: 3× RTX 3090 24 GB, `gpu_split 24,24,24`, KV cache `fp8_e4m3` (q8), chunk 256,
n-gram table in RAM, native MTP on. `tensor_parallel: false` (Qwen4Exp).

```text
MODEL_PATH=/models/Qwen3.8-Flash-Next-exl3-unsloth-shaped-h8-mtp5
CACHE_KV_BITS=8
CACHE_SIZE=262144
GPU_SPLIT=24,24,24
CHUNK_SIZE=256
NGRAM_RAM=1
ENABLE_MTP=1
```

**Vision is in the checkpoint but not served by this stack.** The vision tower is present
and quantized — 987 `model.visual.*` tensors at 6 bpw, `config.json` has
`language_model_only: false` — but the benchmarked backend (`exl3_qwen38_server.py`) is
**text-only**: it has no mmproj and rejects any image part with
`vision content is not supported by this backend (text-only)`. Every number below is
text-only; the image path was not exercised. Use a multimodal-capable loader to test vision.

## Benchmark (measured 2026-09-17, this checkpoint, :12703)

Speed, single stream, temperature 0.7, **cache-proof unique prompts** (fresh random prefix
per run so nothing is a prefix-cache hit), no competing load. `decode` = tokens/s after the
first token; `aggregate` = `new / (TTFT + decode)`, i.e. **including prompt prefill**. The
native MTP head is on (`NUM_DRAFT_TOKENS=3`, observed `mtp_accept ≈ 0.5`).

| case | prompt tok | new tok | TTFT (prefill) | prefill tok/s | decode tok/s | aggregate |
|---|---|---|---|---|---|---|
| decode512 (count) | 196 | 512 | — | — | **128** | ~128 |
| essay4k | 4,677 | 900 | 4.8 s | 968 | **88.5** | 60.0 |
| essay16k | 18,110 | 900 | 18.8 s | 963 | **89.1** | 31.1 |
| essay48k | 54,020 | 900 | 57.6 s | 937 | **86.0** | 13.2 |
| essay48k (thinking) | 54,006 | 780 | 58.3 s | 926 | **83.1** | 11.5 |

**Why the aggregate looks low at long context:** decode is a steady **~85–89 tok/s** at every
context length — it is *not* a decode slowdown. The low aggregate is **prefill**: a 54k-token
prompt is processed at ~940 tok/s, i.e. ~58 s before the first token, so
`new/(prefill+decode)` reads 13.2. When a prefix is reused, exllamav3's page cache removes
most of that prefill — the same 54k continuation finished in **11.0 s (82 tok/s)** on a warm
cache in this session. Report decode and prefill separately, never as one number.

### Thinking mode used in these tests

The mode is toggled with `chat_template_kwargs.enable_thinking` **directly on the backend**
(`:12703`); no `reasoning_effort`, no proxy preset. The dual-mode re-test used
`enable_thinking=true`, `temperature=0.0`, `max_tokens=4096`; speed rows use `temperature=0.7`.
This is *not* the proxy's thinking preset (T 1.0 / top_p 0.95 / top_k 20 / presence 1.0), so
absolute quality numbers may differ under the proxy presets.

### Quality battery (controlled, greedy, instruct mode)

| suite | score |
|---|---|
| Math (exact numeric) | 9/10 |
| Logic / reasoning | 4/5 |
| Code (executed) | **5/5** |
| Misc (JSON, Spanish ×2) | **5/5** |
| Needle in haystack (~20k, ~60k) | **2/2** |
| **Overall** | **23/25** |

The two misses are genuine instruct-mode reasoning slips on the same two prompts:
`divisible by 3 or 5 but not both` → answered `40` (correct `41`), and the apples word
problem → answered `6` (correct `7`). Both are solved in thinking mode.

### Corrected decoding note (supersedes the first card)

An earlier battery reported **21/25** with four prompts (2 logic, 2 Spanish) returning
**empty content**. That was a **harness artifact, not a weight regression**: the backend
defaults to the `thinking` variant, and the old harness used `max_tokens` 200–250 while
reading only `message.content`. Chain-of-thought consumed the whole budget, so the responses
ended with `finish_reason=length` and `content=""`.

Re-tested with mode control, the four cases pass **8/8 in both modes** (instruct and
thinking), each terminating with `finish_reason=stop`:

| case | instruct | thinking |
|---|---|---|
| "who is youngest" | `Carol` | `Carol` (653 reasoning tok) |
| "steel vs feathers" | `Equal.` | `They are equal.` (1699) |
| es→en (cordobés) | correct | correct (1450) |
| capital of Australia | `Canberra…` | `Canberra…` (2210) |

Lesson for anyone re-running this: either set `chat_template_kwargs.enable_thinking=false`,
or read `reasoning_content` and allow a large `max_tokens`.

### Limits

Single box (3×3090), single checkpoint, one pass per prompt, no BF16 reference run — the
source is 336 GB and does not fit this box, so these are absolute correctness checks, not a
bpw-vs-BF16 delta. They are not a claim about other GPUs. Raw data:
[`bench/BENCH_RESULTS_v2.json`](./bench/BENCH_RESULTS_v2.json),
[`bench/SPEED_V2.json`](./bench/SPEED_V2.json), harness
[`bench/bench_quality_v2.py`](./bench/bench_quality_v2.py).

## License

Same as the source: Qwen Community 1.0. See
[Qwen/Qwen3.8-Flash-Next LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
