---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - gguf
  - deepseek
  - deepseek-v4
  - ds4
  - dwarfstar
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731 — Q8_0 routed experts (GGUF for ds4)

A DeepSeek V4 Flash GGUF with the **routed MoE experts stored as `Q8_0`**, and every other tensor
carried over unchanged from antirez's `q4-imatrix` build (F16 HC / Compressor / Indexer, Q8
attention, Q8 shared experts, Q8 output).

```
DeepSeek-V4-Flash-Q8Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-0731.gguf
```

- **Size:** 303,146,197,600 bytes (282.33 GiB)
- **Runs on:** [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar) — Metal / CUDA / ROCm
- **Requires:** ~291 GiB of unified memory for weights + headroom, before KV cache
- **Speed:** 30.96 tok/s vs q4-imatrix's 34.85 on an M3 Ultra — only ~11% slower for 1.85× the size

## Read this before downloading: it is not measurably better than q4

**For everyday use, download [antirez's `q4-imatrix`](https://huggingface.co/antirez/deepseek-v4-gguf)
instead.** It is 46% smaller and, on the evidence below, indistinguishable in quality.

This build is a **reference artifact** — a control for measuring quantization damage, not a daily
driver. Published because the measurement is worth having, and because a negative result is still
a result.

### Why it exists

DeepSeek ships V4 Flash with `"expert_dtype": "fp4"` — the routed experts carry **4 bits** in the
official release. Q8_0 is 8.5 bpw, so it adds no information whatsoever. What it removes is the
**second** lossy step: `q4-imatrix` dequantizes the FP4 experts and re-quantizes them onto Q4_K's
different 4-bit grid. Q8_0 stores the dequantized FP4 values essentially exactly.

That second step is real and measurable at the weight level. Comparing `dequant(q4_K)` against
`dequant(q8_0)` across 36 expert tensors spanning the full depth:

| | |
|---|---|
| relative RMSE | **5.34%** |
| cosine similarity | 0.9984 |
| spread across layers 0→42 | 0.73 pp (**flat** — no depth gradient) |
| worst part | `down` (~5.45%) vs `gate` (~5.31%), `up` (~5.23%) |

### But it does not show up in output quality

Scored against the 100 tracked official DeepSeek V4 Flash continuations
(`gguf-tools/quality-testing/data/flash`), metric = target-token NLL:

| metric | q4-imatrix | this build | verdict |
|---|---|---|---|
| avg_nll (token-weighted) | 0.5230 | 0.5209 | −0.0021 |
| avg_nll (per case, 95% CI) | — | — | +0.0127 [−0.008, +0.046] — **n.s.** |
| greedy LCP | 5.070 | 5.560 | +0.49 [−0.38, +1.42] — **n.s.** |
| first-token match | 45 | 49 | McNemar p=0.219 — **n.s.** |
| case wins | 48 | 52 | coin flip |

Nothing reaches significance at n=100. The token-weighted aggregate slightly favours this build,
but the *unweighted per-case* mean slightly favours q4 — the apparent edge comes from weighting
toward longer cases, not a consistent effect. This is a bounded null rather than mere absence of
evidence: the confidence interval rules out this build being more than ~1.6% relatively better in
NLL.

**Conclusion: a 5.3% weight-level RMSE from q4_K's re-quantization does not translate into
measurable output quality on this benchmark.** If you need the fidelity ceiling for controlled
experiments, use this. Otherwise use q4-imatrix.

**It is barely slower than q4.** Measured on an M3 Ultra (400-token greedy completion,
`--ctx 32768`): **30.96 tok/s** versus q4-imatrix's **34.85** — about 11% slower for 1.85× the
size. That surprised me; the reason is that the MoE is sparse. Only 6 of 256 experts fire per
token (`num_experts_per_tok: 6`), so doubling expert precision adds ~75 MB per token, roughly
0.09 ms at M3 Ultra bandwidth, against a ~29 ms per-token budget. Attention, KV and the dense
paths are byte-identical between the two builds.

So the real cost is memory footprint, not throughput: you need ~291 GiB of unified memory to hold
it.

## Why Q8_0 and not Q8_K

ds4's quantizer will happily emit `q8_K` routed experts — its `gguf-tools/README.md` even
documents a "True Q8_K routed experts" recipe — but **the engine cannot load the result**.
`tensor_is_routed_expert_type` (`ds4.c`) accepts only:

```
Q8_0 · IQ2_XXS · Q2_K · Q4_K · Q5_K · Q6_K
```

A q8_K build dies at startup with `expected a routed expert quant type`. Within ds4, Q8_K is an
*activation* format (used when multiplying against Q2_K/IQ2_XXS experts), not expert storage.
Q8_0 is the widest routed-expert type the engine actually supports, and it is what the existing
names already mean by "Q8" (Q8Attn, Q8Shared, Q8Out).

## No imatrix — deliberately

There is no `-imatrix` in the filename because an importance matrix would have no effect here.
In ds4's quantizer (`gguf-tools/quants.c`), the Q8_0 path explicitly discards it:

```c
if (type == DS4Q_TYPE_Q8_0) {
    (void)imatrix;
    ...
```

Only `q2_K`, `q4_K` and `iq2_xxs` consume an imatrix. Since every tensor in this file is Q8 or
F16, nothing in it is imatrix-influenced, and claiming otherwise in the name would be misleading.

## How it was built

With ds4's own offline quantizer, from the official safetensors, using the q4-imatrix GGUF as the
metadata/shape template so all non-expert tensors pass through untouched:

```sh
gguf-tools/deepseek4-quantize \
  --hf <path>/DeepSeek-V4-Flash-0731 \
  --template DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf \
  --out DeepSeek-V4-Flash-Q8Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-0731.gguf \
  --experts q8_0 --threads 20
```

Two checks confirm only the experts changed:

- The quantizer reports **129 type changes** — exactly 43 layers × 3 (gate / down / up). Anything
  else being re-quantized would push that count higher.
- `--compare-tensor` regenerates individual non-expert tensors and byte-compares them against the
  template. `blk.0.attn_q_a.weight`, `blk.0.ffn_gate_shexp.weight` and `output.weight` all match
  with identical FNV-1a hashes.

## Usage

```sh
./ds4-server -m DeepSeek-V4-Flash-Q8Experts-...-chat-v2-0731.gguf --ctx 32768 --metal
```

On a 512 GiB Mac you may need to raise the Metal wired limit:

```sh
sudo sysctl iogpu.wired_limit_mb=516096
```

**Not a general-purpose GGUF.** ds4 is not a generic loader — it expects the specific tensor
layout, quantization mix and metadata of the DeepSeek V4 family. This file will not work with
llama.cpp.

## Credits

- Model weights: [DeepSeek-AI](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- Engine, quantizer and the template GGUF this was derived from:
  [antirez/ds4](https://github.com/antirez/ds4) and
  [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)

MIT, matching upstream.
