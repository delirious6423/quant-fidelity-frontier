---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags: [gguf, llama.cpp, rocm, gfx1151, strix-halo, amd, ryzen-ai-max-395, ai-max-395, rocmfpx, moe, tool-calling]
language: [en]
pipeline_tag: text-generation
---

> ### ⚠️ STOCK `llama.cpp` WILL NOT LOAD THIS MODEL
> `Q8_0_ROCMFPX` is a **ROCmFPX** quant type — it exists only in
> [`charlie12345/ROCmFPX`](https://github.com/charlie12345/ROCmFPX), not upstream llama.cpp.
> Ignore the auto-generated "Use this model" commands above.
>
> 📦 **25.92 GiB, 8.28 bpw** · ✅ **tools 7/7 in BOTH thinking and non-thinking**
> 🚀 **25.07 tok/s with MTP** on Ryzen AI MAX+ 395. ⚠️ The 4-bit build is still faster (38.32).
> ⚠️ **Run it WITH the bundled MTP draft head** — without it you get ~7.9 tok/s, a third of the speed.

# Qwen3.8-27B — ROCmFPX 8-bit (`Q8_0_ROCMFPX`) GGUF

An **8-bit ROCmFPX** quantization of Qwen3.8-27B for **AMD gfx1151 (Ryzen AI MAX+ 395 /
Strix Halo)**, built because 128 GB of unified memory makes 8-bit genuinely affordable on this
hardware. Quantized from the **51.3 GiB BF16** GGUF — not requantized from a lower-bit build.

| | |
|---|---|
| **File** | `Qwen3.8-27B-Q8_0_ROCMFPX.gguf` |
| **Size** | **25.9232 GiB** (27,834,808,672 bytes) |
| **BPW** | **8.28** |
| **ftype** | `Q8_0_ROCMFPX` (111) |
| **sha256** | `960978d5b230485c35c2456988082fa5d1e27a05ac74150604127b0e95b904bf` |

---

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


## Benchmarks — run this WITH the MTP draft head

Ryzen AI MAX+ 395 (gfx1151, 128 GB unified, ROCm 7.2.4), **median of 3, warm-up discarded,
idle box**, shipped serving flags: `--spec-type draft-mtp --model-draft mtp-Qwen3.8-27B-Q4_0.gguf
--spec-draft-n-max 4 -ngl 999 -fa on -fit off`.

| build | size | **decode WITH MTP** | range | draft acceptance |
|---|---|---|---|---|
| **this (`Q8_0_ROCMFPX`)** | 25.92 GiB | **25.07 tok/s** | [25.07 – 25.51] | 0.911 |
| [`Q8_0_ROCMFPX_AGENT`](https://huggingface.co/kingjones777/Qwen3.8-27B-ROCmFPX-Q8_0-AGENT-GGUF) | 26.28 GiB | **26.62 tok/s** | [26.61 – 27.15] | **0.953** |
| [`Q4_0_ROCMFP4_STRIX`](https://huggingface.co/kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF) (4-bit) | 14 GiB | **38.32 tok/s** | [37.91 – 38.61] | 1.000 |

⚠️ **Without the draft head this model runs at ~7.9 tok/s** — MTP is worth **3.2×** here. The
draft head (`mtp-Qwen3.8-27B-Q4_0.gguf`) is included in this repo; use it.

**Choosing:** the 4-bit build is ~1.53× faster and 12 GiB smaller. Take 8-bit for fidelity
headroom, not throughput. And if you are running MTP, prefer the **AGENT** variant — it accepts
more draft tokens (0.953 vs 0.911) and is 6.2% faster despite being marginally larger.

```
llama-server -m Qwen3.8-27B-Q8_0_ROCMFPX.gguf \
  --spec-type draft-mtp --model-draft mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-draft-ngl 99 --spec-draft-n-max 4 \
  -ngl 999 -fa on -fit off --jinja --ctx-size 32768
```

## Verified

| check | result |
|---|---|
| 17 × 23 | ✅ `391` |
| capital of Japan | ✅ `Tokyo` |
| days in 2024 | ✅ `366` |
| **tool calling — thinking** | ✅ **7/7** (multi-arg, nested-object, enum, declines, multi-turn, streaming, parallel) |
| **tool calling — non-thinking** | ✅ **7/7** |

## Per-tensor types (audited, 851 tensors)

`output.weight` **Q8_0** · `token_embd.weight` **Q8_0** · bulk `TYPE_103` (ROCmFPX 8-bit layout).

## 🩹 Prompt caching with the MTP draft head — fixed

**Reported by a user of this repo:** with `--spec-type draft-mtp` loaded, `llama-server` disabled
prefix caching entirely. Every agentic turn reprocessed the whole prompt. Reproduced here on an
8045-token stable prefix:

| config | prompt_n | cache_n | prompt_ms |
|---|---|---|---|
| no draft head | 519 | 7526 reused | 1 908 |
| **draft head (the defect)** | 8045 | **0** | **27 948** |
| **draft head + this patch** | **4** | **5101** | **100** |

**279× less prompt processing per turn**, with MTP still drafting.

### Root cause
The saved speculative state is the **MTP boundary** — the target model's pre-norm hidden row at the
cached prompt's *exact end position*. Any partial-prefix reuse would leave it describing a position
that no longer exists, so the server demanded an exact full-prefix match and otherwise reprocessed
cold, erasing its own context checkpoints on the way.

### The fix
`patches/mtp-prompt-cache-fix.patch` (4 files, applies to `2809dc5`) captures the speculative
boundary **inside the context checkpoint** (`common_prompt_checkpoint::data_spec`).
`create_checkpoint` runs between decode batches — exactly where that boundary is valid — so exact
state is saved and restored **together with the KV**, never rebuilt.

⛔ Two approaches were tried first and **rejected**: rebuilding the boundary from a zero-fill
changed the model's output (deterministically, 3/3), and truncating the KV back to the reuse point
is impossible here — the bounded rollback window is **4 tokens** against the 333 a real turn needs.
Exact state restore is the only shape that preserves output.

### Verification
Independently gated **10/10**: same prompt cold vs warm, `temperature 0`, byte-identical every run,
with the cache genuinely engaged (`cache_n=5101`, not a vacuous pass). The output hash also matches
the **unpatched** build, so behaviour is unchanged. Fails closed — an unreachable rollback logs
`reason=spec-checkpoint-missing` and cold-reprocesses rather than guessing.

### Related upstream
This is the same family as open llama.cpp issues
[#20225](https://github.com/ggml-org/llama.cpp/issues/20225),
[#19794](https://github.com/ggml-org/llama.cpp/issues/19794) and
[#24055](https://github.com/ggml-org/llama.cpp/issues/24055) — checkpoints being invalidated on
hybrid/recurrent models. This patch is **not** upstreamed; it is offered here as-is.

## What was NOT measured

- **No perplexity run**, and **no quality A/B vs BF16 or the 4-bit build.** We show 8-bit is
  *slower*; we have **not** demonstrated it is *better*. If you need proof that 8 bits buys
  accuracy here, that measurement does not yet exist.
- An earlier revision of this card quoted 7.92 vs 13.84 tok/s. Those were measured **without the
  MTP draft head** and understated both builds; the table above supersedes them.
- No long-context testing (model supports 131,072).
- No coding/reasoning benchmark.

Base model licence inherited. Credit for the model goes to Qwen.
