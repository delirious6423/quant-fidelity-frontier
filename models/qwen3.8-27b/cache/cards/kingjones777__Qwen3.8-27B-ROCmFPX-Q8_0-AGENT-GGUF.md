---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags: [gguf, llama.cpp, rocm, gfx1151, strix-halo, amd, ryzen-ai-max-395, ai-max-395, rocmfpx, moe, tool-calling, agent]
language: [en]
pipeline_tag: text-generation
---

> ### ⚠️ STOCK `llama.cpp` WILL NOT LOAD THIS MODEL
> `Q8_0_ROCMFPX_AGENT` is a **ROCmFPX** quant type — it exists only in
> [`charlie12345/ROCmFPX`](https://github.com/charlie12345/ROCmFPX), not upstream llama.cpp.
>
> 📦 **26.28 GiB, 8.39 bpw** · ✅ **tools 7/7 both modes**
> 🚀 **26.62 tok/s with MTP** — **6.2% faster than the plain 8-bit build**, because it accepts
> more draft tokens (**0.953** vs 0.911). ⚠️ Without MTP the two are indistinguishable.

# Qwen3.8-27B — ROCmFPX 8-bit **AGENT** routing (`Q8_0_ROCMFPX_AGENT`) GGUF

The **agent / tool-call coherent** 8-bit ROCmFPX routing of Qwen3.8-27B for **AMD gfx1151
(Ryzen AI MAX+ 395 / Strix Halo)**. Quantized from the **51.3 GiB BF16** GGUF.

| | |
|---|---|
| **File** | `Qwen3.8-27B-Q8_0_ROCMFPX_AGENT.gguf` |
| **Size** | **26.2767 GiB** (28,214,425,952 bytes) |
| **BPW** | **8.39** |
| **ftype** | `Q8_0_ROCMFPX_AGENT` (115) |
| **sha256** | `f3ef2bdbf56e244fff14b72114dc2aa300b84a4cb4eaf54f2e5e4da1cfcae10b` |

---

## ⚖️ Why this variant exists — and when it does NOT

**The routing difference is structural.** Across the 17-tensor `*output.weight` group:

| build | Q8_0 | ROCmFPX 8-bit (`TYPE_103`) |
|---|---|---|
| plain `Q8_0_ROCMFPX` | 1 | 16 |
| **this (`_AGENT`)** | **13** | 4 |

AGENT keeps **12 more attention-output projections at full Q8_0** — that is what the extra
0.35 GiB buys.

### With MTP, that buys real speed

Ryzen AI MAX+ 395 (gfx1151, ROCm 7.2.4), **median of 3, warm-up discarded, idle box**, shipped
flags (`--spec-type draft-mtp`, Q4_0 draft head, `--spec-draft-n-max 4`):

| build | size | **decode WITH MTP** | range | **draft acceptance** |
|---|---|---|---|---|
| plain `Q8_0_ROCMFPX` | 25.92 GiB | 25.07 tok/s | [25.07 – 25.51] | 0.911 |
| **this `_AGENT`** | 26.28 GiB | **26.62 tok/s** | [26.61 – 27.15] | **0.953** |
| `Q4_0_ROCMFP4_STRIX` (4-bit) | 14 GiB | **38.32 tok/s** | [37.91 – 38.61] | 1.000 |

**+6.2% over plain, ranges disjoint.** The mechanism is visible in the acceptance rate: keeping
more attention-output precision makes the MTP draft head's guesses land more often, so more
speculated tokens survive verification.

### ⚠️ Without MTP there is no measurable difference

Run standalone, the two builds are **7.92 vs 7.82 tok/s** (this one marginally slower), and both
score **7/7 tool calling in thinking and non-thinking** — identical across multi-arg,
nested-object, enum, declines, multi-turn, streaming and parallel calls. Our 7-case suite cannot
separate them.

⇒ **Take this variant if you serve with MTP. Take the plain build if you do not.**
And if you want raw throughput, the 4-bit build beats both by ~1.44×.

```
llama-server -m Qwen3.8-27B-Q8_0_ROCMFPX_AGENT.gguf \
  --spec-type draft-mtp --model-draft mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-draft-ngl 99 --spec-draft-n-max 4 \
  -ngl 999 -fa on -fit off --jinja --ctx-size 32768
```

## Verified

17×23 ⇒ ✅ `391` · capital of Japan ⇒ ✅ `Tokyo` · days in 2024 ⇒ ✅ `366` ·
tools **7/7** thinking, **7/7** non-thinking.

`token_embd.weight` **Q8_0**, 851 tensors, `arch=qwen35`.

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

- **No perplexity, no quality A/B vs BF16 or the plain 8-bit build.** We have shown a *structural*
  difference in tensor types and **no measurable behavioural difference**.
- No long-context testing (131,072 supported), no coding/reasoning benchmark.
- Our tool suite is 7 cases. **A 7/7 tie is not proof of equivalence** — it is the limit of what
  this instrument can resolve. The MTP acceptance rate did separate the builds where the tool
  suite could not.
- An earlier revision of this card said no advantage over the plain build could be measured. That
  was true **without MTP** and wrong with it; the table above supersedes it.

Base model licence inherited. Credit for the model goes to Qwen.
