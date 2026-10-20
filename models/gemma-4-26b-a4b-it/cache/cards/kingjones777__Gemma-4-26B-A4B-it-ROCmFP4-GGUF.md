---
license: gemma
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags: [gguf, rocmfp4, rocmfpx, strix-halo, gfx1151, amd, rocm, gemma, moe, multimodal, quantized]
---

# Gemma-4-26B-A4B-it — ROCmFP4 / ROCmFPX GGUF

**First ROCmFP4/ROCmFPX quantization of `google/gemma-4-26B-A4B-it`** — the most-downloaded
Gemma-4 model. MoE: 26B total, **~4B active** per token. Four quants, built from the Hub BF16
GGUF (no re-conversion), each verified for text **and** vision on real hardware.

Built for **AMD Strix Halo (gfx1151)** — Ryzen AI MAX+ 395, 128 GB unified, ROCm 7.2.4 — with the
[ROCmFPX](https://github.com/charlie12345/ROCmFPX) llama.cpp fork.

> ⚠️ Needs a **ROCmFPX-capable llama.cpp build**. These will **not** load in stock llama.cpp /
> Ollama / LM Studio — `Q4_0_ROCMFP4_*` and `Q*_0_ROCMFPX*` are not mainline tensor types.

## Variants

| file | ftype | size | BPW | `token_embd` | decode + MTP | decode alone |
|---|---|---|---|---|---|---|
| `gemma-4-26B-A4B-it-Q4_0_ROCMFP4_COHERENT.gguf` | 102 | 13.45 GiB | 4.57 | q6_K | **60.11 t/s** | 54.39 |
| `gemma-4-26B-A4B-it-Q4_0_ROCMFP4_FAST_COHERENT.gguf` | 104 | 12.73 GiB | 4.33 | q6_K | 55.50 t/s | **58.33 t/s** |
| `gemma-4-26B-A4B-it-Q6_0_ROCMFPX_AGENT.gguf` | 114 | 21.81 GiB | 7.42 | q8_0 | 45.33 t/s | 43.39 |
| `gemma-4-26B-A4B-it-Q8_0_ROCMFPX.gguf` | 111 | 24.30 GiB | 8.27 | q8_0 | 46.50 t/s | 41.84 |
| `gemma-4-26B-A4B-it-Q8_0_ROCMFPX_AGENT.gguf` | 115 | 24.67 GiB | 8.39 | q8_0 | **47.58 t/s** | 41.34 |

Also included: `mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (drafter) and `mmproj-BF16.gguf` (vision
projector) — both required to reproduce the numbers above.

**Which to pick — and it depends on whether you can run the drafter:**

| your setup | take | why |
|---|---|---|
| **text + drafter** | **`Q4_0_ROCMFP4_COHERENT` (102)** | 60.11 t/s — fastest with MTP |
| **vision, or no drafter** | **`Q4_0_ROCMFP4_FAST_COHERENT` (104)** | 58.33 t/s — 7.2% faster than 102 without MTP, and 6% smaller |
| want an 8-bit | **`Q8_0_ROCMFPX_AGENT` (115)** | faster than plain Q8 *with* the drafter (47.58 vs 46.50) |

⭐ **Two of these rankings reverse depending on speculation, so pick by your actual setup:**
- **FAST (104) beats COHERENT (102) by 7.2% WITHOUT a drafter, and loses by 8.3% WITH one** —
  even though FAST has *higher* draft acceptance (0.869 vs 0.824). The single-scale layout speeds
  up plain decode but is less suited to the batched verification pass.
- **AGENT (115) beats plain Q8 (111) WITH a drafter and loses WITHOUT it** — the AGENT recipe holds
  attention-output tensors at higher precision to raise draft acceptance, which cannot help if you
  are not speculating.

⚠️ Because **MTP and vision cannot be used together** (see below), multimodal users are forced onto
the no-drafter path — where **FAST (58.33) lands within 3% of COHERENT-with-MTP (60.11)**. If you
are doing image work, 104 is the one to take.

## Measured — not estimated

Hardware: **Ryzen AI MAX+ 395 (Strix Halo, gfx1151)**, 128 GB unified, ROCm 7.2.4.
Decode = 300 tokens, code prompt, warm-up discarded, median of 3, spreads ≤1.015.

| quant | runs (with MTP) | median | draft_n / accepted | acceptance | mean len | correctness |
|---|---|---:|---|---:|---:|---|
| 102 | 63.36 / 60.40 / 60.11 / 59.89 / 59.60 | **60.11** | 238 / 173 | 0.824 | 2.75 | `391` · Tokyo · `366` ✅ |
| **104 FAST** | 56.80 / 55.50 / 55.50 / 55.44 / 55.45 | 55.50 | — | **0.869** | — | `391` ✅ |
| 114 | 45.33 / 45.29 / 45.34 | 45.33 | 241 / 178 | 0.739 | 2.80 | ✅ |
| 111 | 46.96 / 46.50 / 46.43 | 46.50 | 222 / 173 | **0.779** | 2.77 | ✅ |
| 115 | 48.04 / 47.58 / 47.43 | 47.58 | 233 / 177 | 0.760 | 2.82 | ✅ |

Correctness was re-checked **under the shipping MTP config**, not in a separate run. Speculative
decoding is output-preserving — the target verifies every drafted token — and the answers are
identical with and without it.

### Without a drafter (the vision path)
| quant | runs | median |
|---|---|---:|
| 102 COHERENT | 54.44 / 54.39 / 54.27 | 54.39 |
| **104 FAST_COHERENT** | 58.56 / 58.33 / 58.17 | **58.33** |

Spreads 1.003 / 1.007 — clean. This is the honest comparison for image work.

**Vision: 4/4** on a 512² four-colour quadrant image (TL red · TR blue · BL yellow · BR green),
run in a separate server with `-fa off` and no drafter (see the limitation below). A solid-colour
square would pass trivially and proves nothing.

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

```bash
llama-server -m gemma-4-26B-A4B-it-Q4_0_ROCMFP4_COHERENT.gguf \
  --spec-type draft-mtp --model-draft mtp-gemma-4-26B-A4B-it-Q8_0.gguf \
  --spec-draft-ngl 999 --spec-draft-n-max 3 \
  -ngl 999 -c 4096 -fa on -fit off -np 1
```

Measured sweep on the 4-bit — **`n-max 3` is best**, and more depth is worse:

| n-max | 2 | **3** | 4 | 5 |
|---|---:|---:|---:|---:|
| t/s | 53.87 | **55.55** | 53.25 | 54.70 |
| acceptance | 0.792 | 0.756 | 0.693 | 0.676 |

⚠️ **Expect a modest gain here (1.03–1.15×), not the ~3× quoted for dense Gemma-4 models.**
Speculation pays inversely to how fast the target already is: this MoE reads only ~4B active
parameters per token, so it is already fast without a drafter and there is little stall to fill.
Mean accepted length is 2.75–2.82 against a 4.0 ceiling — not saturated, which is why raising
`n-max` past 3 does not help.

### ⛔ MTP and vision cannot be used together
Passing an image while the drafter is loaded aborts the server (`missing MTP boundary` →
`server-context.cpp` fatal; upstream llama.cpp PR #20277). Image embeddings are injected outside
the normal token path and the speculative batch loses its boundary.
**For vision: drop `--spec-type` and use `-fa off`.**

## Verification

- Sizes vs `--dry-run`: constant **~15.09 MiB** GGUF-header delta on all four (spread 6.5 KB) — not truncated
- `token_embd` audited **by exact tensor name**; `tie_word_embeddings: true`, so **no `output.weight` exists** and `--output-tensor-type` is a silent no-op — `--token-embedding-type` is what protects the head
- ⚠️ This MoE has **no `*_shexp` tensors** — its shared FFN is `ffn_{down,gate,up}.weight`; routed experts are `ffn_down_exps` / `ffn_gate_up_exps`
- Bandwidth sanity for MoE must use **active (~4B)** weight, not file size: Q4 ≈ 130 GB/s, Q8 ≈ 173 GB/s

```
14439364064  gemma-4-26B-A4B-it-Q4_0_ROCMFP4_COHERENT.gguf
13674256864  gemma-4-26B-A4B-it-Q4_0_ROCMFP4_FAST_COHERENT.gguf
23417194464  gemma-4-26B-A4B-it-Q6_0_ROCMFPX_AGENT.gguf
26094754272  gemma-4-26B-A4B-it-Q8_0_ROCMFPX.gguf
26486003680  gemma-4-26B-A4B-it-Q8_0_ROCMFPX_AGENT.gguf
  461766816  mtp-gemma-4-26B-A4B-it-Q8_0.gguf
 1194828256  mmproj-BF16.gguf
```

## Credits
- Base model: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) — Gemma Terms of Use
- Quantization stack: [ROCmFPX](https://github.com/charlie12345/ROCmFPX)
