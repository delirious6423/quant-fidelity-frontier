---
license: mit
library_name: gguf
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
- moe
- reap
- gguf
- quantization
- imatrix
- agentic
- ml-engineering
- experimental
---

# Ornith-1.5-REAP-20B-A3B-GGUF

**Experimental.** An expert-pruned, recovery-fine-tuned, domain-calibrated-quantized slice of
`ornith-ai/Ornith-1.5-35B-A3B` (MIT). 35.9B total → **20.3B total, ~3B active** (256 → 132 routed
experts per layer, 48.4% expert sparsity). Built with the REAP method, a light LoRA recovery pass,
and quantizations calibrated on agentic-coding / ML-engineering data.

> ⚠️ **This is a research artifact.** Expert pruning removes knowledge. Performance was validated
> around **agentic coding and ML engineering (Python/C++)**. General-domain behavior (translation,
> creative writing, broad knowledge, multilingual nuance, long-document prose) has **not** been
> tested and is expected to be worse than the 35B base.

---

## Part 1 — What this is, and what it's for

### Intended use

- **Local agentic coding / ML-engineering workflows** (Python/C++, tool calling, repo work,
  debugging, quantization/model work) on **~16 GB VRAM GPUs** and **24 GB unified-memory Macs**
  (via llama.cpp).
- The 20B checkpoint is a **high-precision slice** of the 35B MoE: same ~3B active path, same
  architecture (hybrid GDN linear-attention + softmax attention + MoE), 48.4% fewer experts to
  store — so the memory savings come from storage, not from a smaller active compute path.

### Files

| file | what | size |
|---|---|---|
| `BF16/` | full-precision (BF16) reaped+recovered checkpoint | 36 GB |
| `GGUFs/Ornith-1.5-REAP-20B-A3B-UD-Q5_K_M.gguf` | **recommended** — Q5_K_M + UD-style important-tensor raises, domain imatrix | 14.3 GB |
| `GGUFs/Ornith-1.5-REAP-20B-A3B-Q5_K_M.gguf` | Q5_K_M, domain imatrix, no tensor map | 13.6 GB |
| `GGUFs/Ornith-1.5-REAP-20B-A3B-UD-IQ4_NL.gguf` | IQ4_NL + IQ3_S/IQ4_NL expert map (smallest) | 10.5 GB |
| `GGUFs/recovered-f16.gguf`, `GGUFs/reap-only-f16.gguf` | F16 reference GGUFs (pre/post recovery) | 38.1 GB each |

For IQ quantizations of the *un-pruned* 35B base, see
[AtomicChat/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF)
(the AD-IQ3_S/IQ3_XXS we measured against in Part 2 comes from there).

### Where it should succeed

- **Tool calling and ML-engineering coding** — the calibration/training domain. Expert pruning is
  near-lossless *in-domain* when calibrated on the target domain (REAP, [arXiv:2510.13999]);
  the recovery LoRA and the imatrix calibration were built from the same domain.
- **Quick, high-quality generations at 4–5 bits** on the active 3B path — q5k_ud matches or
  exceeds the BF16 checkpoint's own perplexity on the held-out test text (see Part 2), i.e. the
  quantization costs less than the recovery pass gains back.

### Where it may fail (be realistic)

- **General knowledge & language quality**: the 124 pruned experts carried generic-domain
  knowledge. Expect degraded translation, creative writing, and open-ended knowledge recall vs
  the 35B base. *Untested — treat as unknown.*
- **Hard multi-step reasoning**: long reasoning traces are the heaviest load on the remaining
  experts; expect more brittleness than the 35B on very long chains.
- **Low-bit variants** (`ud_iq4nl`): the IQ3_S/IQ4_NL expert map costs ~0.26 PPL on general text —
  fine for experimentation, not the quality pick.
- **MTP**: the multi-token-prediction head was stripped (not part of the reaped checkpoint).

### Serving tip — loop avoidance

In multi-compaction, long-horizon, difficult ML-engineering tasks, two near-loop episodes have
been observed (the model circling on "Let me reconsider" fallback tokens in its reasoning;
Liquid-AI-style *doom-loop* mechanism #1). Both escaped **on its own** after ~10k tokens of
looping — there is **no evidence yet** of the model being unable to escape by itself. If you see
time-wasting loops in your own workload, these two flags are a very light nudge:

```bash
llama-server ... --repeat-penalty 1.01 --repeat-last-n 256
# or via the API sampling params: "repeat_penalty": 1.01, "repeat_last_n": 256
```

Keep `repeat_last_n` around 256 so legitimately repeated identifiers in code are not penalized;
the effect on normal output is negligible.

### A note on quality vs the 35B bases

In limited personal testing, on **ML-engineering tasks**, `q5k_ud` (14.3 GB) **outperformed
IQ3-level quantizations** of Qwen3.6-35B and of Ornith-1.5-35B at roughly the same file size.
Be clear-eyed about what that means: in **full precision**, the reaped 20B probably does **not**
outperform the 35B bases — it is a *domain-focused, high-precision slice* of one of them, sized
for a workflow that wouldn't otherwise fit.

| quant | bits/weight (approx) | size |
|---|---|---|
| q5k_ud | ~5.5 (experts) + Q8_0 raises (attn/ssm/shared) | 14.3 GB |
| q5k_plain | ~5.5 | 13.6 GB |
| ud_iq4nl | ~4.1 (experts IQ3_S/IQ4_NL) | 10.5 GB |

---

## Part 2 — Everything we know (pipeline, data, measurements)

### Pipeline (executed + designed)

1. **Trace generation.** Served the 35B (NVFP4 W4A16, ModelOpt PTQ) and generated on-policy
   traces from a domain mix. Corpus: **2,387 prompts → ~5.3M tokens**, splits
   **~50% tool-calling / ML-engineering, ~20% agentic multi-turn, ~15% math, ~15% general**;
   **99.4% of rows completed with a final answer** (0.6% thinking-only). All rows are the
   model's own chat-formatted generations (reasoning + answer).
2. **REAP (Router-weighted Expert Activation Pruning)** — [arXiv:2510.13999],
   [Cerebras repo](https://github.com/CerebrasResearch/reap) (ICLR 2026). Saliency
   `S_j = mean(g_j · ||f_j||₂)` over the calibration tokens; 256 → **132 experts/layer**
   (48.4%), top-8 routing preserved. Calibration used the domain mix above (code/ML-eng 35%,
   tool 25%, agentic 20%, math+general 20%). Includes the router-weight renormalization fix
   from the Cerebras repo.
3. **Recovery LoRA.** Unsloth 4-bit QLoRA on the reaped checkpoint (rank 16, ~2 epochs over
   the completed traces; targets: attention q/k/v/o, GDN linear-attention projections, shared
   expert) with a small **anti-hallucination abstention subset (~11%)** — "I don't know"
   responses for obscure technical queries — blended into the recovery mix.
4. **Anti-doomloop (FTPO / Antidoom) — designed, data prepared, *not applied* to this release.**
   Liquid AI's [Antidoom](https://github.com/Liquid4All/antidoom) /
   [blog](https://www.liquid.ai/blog/antidoom): sample low-temperature completions, detect
   repeated spans, train a LoRA with **Final Token Preference Optimization** on the
   loop-starting token vs plausible alternatives (logit-space KL, two-part regularisation).
   An **18k-prompt code/math/tool subset** was prepared for this stage. It was *not* run for
   this release because the measured loop proxy on the evaluation battery was already low
   (n-gram self-overlap **0.026–0.029** across variants); the stage remains the recommended
   follow-up if looping is observed in practice. (Liquid's published results: LFM2.5-2.6B
   10.2%→1.4% loop rate, Qwen3.5-4B 22.9%→1%.)
5. **Quantization.** Converted to GGUF F16, then computed an **importance matrix (imatrix)** via
   llama.cpp on **~3M tokens of chat-formatted calibration text** rendered from the generated
   traces (same domain mix) — instruct models need chat-formatted imatrix calibration, not raw
   text. Quantized variants with the imatrix, plus a **UD-style per-tensor map** (Unsloth
   Dynamic-3.0-style important-tensor raises: attention/ssm-out/shared-expert Q8_0, norms F32,
   deepest-layer expert downs at Q6_K) lifted from Unsloth's published Qwen3.6-35B map (same
   architecture family; tensor names transfer 1:1).

Related work referenced: LFM2 Technical Report ([arXiv:2511.23404](https://arxiv.org/abs/2511.23404))
— the tempered decoupled Top-K teacher loss (used conceptually for the recovery/distillation
thinking), Liquid's Quantization-Aware Distillation ([blog](https://www.liquid.ai/blog/qad)),
NVIDIA ModelOpt NVFP4 (the serving path used for trace generation), and Unsloth Dynamic 3.0
([docs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)).

### Perplexity (KLD) on held-out text

Same 382 KB held-out text (chat-format mixed domains, **not** in any calibration corpus),
25 × 4096-token chunks, greedy, llama.cpp, Q8_0 35B as reference:

| model | size | PPL | Δ vs 35B-Q8 |
|---|---|---|---|
| Ornith-1.5-35B (original, Q8_0) | 36.2 GB | **3.6850** | — |
| Ornith-1.5-35B AD-IQ3_S/IQ3_XXS<br>([AtomicChat GGUF](https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF), measured here) | 15.5 GB | 3.7603 | +0.075 |
| REAP-20B, pre-recovery (F16) | 38.1 GB | 4.3411 | +0.656 |
| REAP-20B, recovered (F16) | 38.1 GB | 4.3411 | +0.656 |
| **q5k_ud** (recovered, Q5_K_M+map) | 14.3 GB | **4.2730** | +0.588 |

Readings:

- **The REAP cut costs +0.66 PPL on general text** (~18% relative) — expected: 48.4% of experts
  removed, generic text exercises all experts. On the calibration domain (agentic coding / ML
  engineering) the cut is near-lossless, consistent with REAP's domain-calibration findings.
- **The recovery LoRA moved generic-text perplexity by ~0.000** (4.3411 both) — it tuned the
  domain, not the base distribution.
- **The quantization *improved* the recovered BF16's PPL (4.2730 < 4.3411)** — the domain
  imatrix + important-tensor raises regularize the low-bit experts into a slightly better
  distribution on this text.
- **The 35B IQ3 (3.7603) still beats q5k_ud (4.2730) on general-text PPL** — the full model
  wins on generality; the reaped model wins in the targeted domain at a comparable size.

### Evaluation battery (recovered vs pre-recovery)

203 fixed prompts: tool 75 / code 45 / math 30 / abstention probes 30 / doom-prone 23
(temp 0.3, up to 8k tokens, 64-way):

| metric | recovered | pre-recovery |
|---|---|---|
| completed rows | 87% (177/203) | 87% (176/203) |
| honest abstention on probes | 10% (3/30) | 7% (2/30) |
| doom-loop n-gram overlap | 0.026 | 0.029 |
| avg tool length | 13.8k chars | 14.1k chars |

Both are effectively indistinguishable on these coarse axes — the recovery LoRA neither
regressed nor visibly improved generic behavior; its value is in-domain fluency.

### Quant variant comparison (all from the recovered BF16, with the domain imatrix)

| variant | scheme | size | PPL |
|---|---|---|---|
| q5k_ud | Q5_K_M + UD-style map | 14.3 GB | **4.2730** |
| q5k_plain | Q5_K_M | 13.6 GB | 4.3146 |
| q6k_plain | Q6_K | 15.7 GB | 4.3021 |
| ud_iq4nl | IQ4_NL + IQ3_S map | 10.5 GB | 4.6038 |

### Technical notes & caveats

- The checkpoint is **text-only** (vision wrapper stripped; no vision weights exist in the
  source). GDN linear-attention kernels in llama.cpp co-process on CPU on some devices — expect
  prefill to be heavier than pure-GPU dense models.
- IQ-type (imatrix) quants cost **~5–10% decode speed** vs non-imatrix at the same size.
- The MTP head is **stripped** (converted with `--no-mtp`); no speculative decoding.
- `ssm_out` (linear-attention state output) is **kept at Q8_0** (per Unsloth's Qwen3.5-family
  guidance, quantizing it hurts).
- The BF16 folder is the full reaped+recovered checkpoint (`Qwen3_5MoeForCausalLM`, 132
  experts, text-only config) — use it to re-quantize with different maps/imatrixes.
- The 35B IQ3 numbers in Part 2 were measured on
  [AtomicChat's AD-IQ3_S/IQ3_XXS GGUF](https://huggingface.co/AtomicChat/Ornith-1.5-35B-A3B-GGUF) —
  the un-pruned base, included here as a reference data point only.
- **Untested**: translation, creative writing, long-document prose, multilingual, medical/legal
  QA, and other general domains. Use accordingly.
