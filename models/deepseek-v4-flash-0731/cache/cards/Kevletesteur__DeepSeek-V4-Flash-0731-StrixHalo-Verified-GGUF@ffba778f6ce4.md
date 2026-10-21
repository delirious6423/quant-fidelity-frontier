---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
language:
- en
- fr
tags:
- gguf
- deepseek
- strix-halo
- rocm
- llama.cpp
- speculative-decoding
- unified-memory
---

# 🌀 DeepSeek-V4-Flash-0731 · **StrixHalo-Verified** (q6kattn GGUF)

> **TL;DR** 🎯 A 284B MoE in **103 GB (2.90 bpw)** that we **verified against the
> full-precision official API** on identical protocols: 🟢 90.8% token-identical,
> 🟢 240/240 paired-QA parity, 🟢 deep-derivation parity (SymPy-checked), 🟢 0.0%
> confident-wrong-answer rate. Tuned end-to-end for **AMD Strix Halo (Ryzen AI Max+ 395,
> gfx1151, 115 GB unified RAM)** on **mainline llama.cpp** — no fork. Every flag below is a
> measurement, not a vibe. 📏

**What's different from the original unsloth UD-IQ3_XXS?** The three attention tensor
families that dominate the per-token byte budget (`attn_q_b`, `attn_output_a`,
`attn_output_b` — **51.9% of all bytes read per token**, more than the 256 experts combined!)
are requantized **Q8_0 → Q6_K** from the clean source. Measured effect: **+23.8% generation**
on long-form output, and draft acceptance jumping **44% → 63%** (less attention noise keeps
the speculative drafter and the target in agreement). 🚀

## 📦 Install (Strix Halo / gfx1151)

```bash
# 1️⃣ One-time: raise the unified-memory GPU ceiling (the single biggest win: ×2.8 !)
#    Default GTT = half your RAM. On a 128 GB box you want ~110 GiB:
echo 'GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT amdgpu.gttsize=112640 ttm.pages_limit=28835840"' \
  | sudo tee /etc/default/grub.d/99-amdgpu-gtt.cfg
sudo update-grub && sudo reboot

# 2️⃣ Build llama.cpp (mainline!) for ROCm/HIP
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3️⃣ Download this repo (model + the DSpark drafter from unsloth's repo)
hf download Kevletesteur/DeepSeek-V4-Flash-0731-StrixHalo-Verified-GGUF
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
#    → requantize the drafter to Q3_K_S (8.5 GB) with llama-quantize, or use it as-is (10.9 GB)

# ✅ Verify integrity BEFORE first launch (we learned this the hard way):
sha256sum -c <(echo "538ec6932c50c6993f8a2cb18dc138e0fb5f6a7c2659e53a7566f1dc8ebb4344  DeepSeek-V4-Flash-0731-UD-IQ3_XXS-q6kattn.gguf")
```

## ⚙️ Launch (every flag measured — see "why" table below)

```bash
GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_NO_FUSE_HC_POST=1 \
./build/bin/llama-server \
  -m DeepSeek-V4-Flash-0731-UD-IQ3_XXS-q6kattn.gguf \
  --no-mmap -fa on -ctk q8_0 -ctv q8_0 \
  -ngl 99 -t 12 -b 2048 -ub 512 -c 262144 \
  --n-cpu-moe 16 \
  --model-draft DSpark-draft-Q3_K_S.gguf --spec-type draft-dspark \
  --spec-draft-n-max 2 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 \
  --temp 1.0 --top-p 0.95 --min-p 0.0
```

| flag | why (measured) |
|---|---|
| `--no-mmap` | 🔴 **mandatory** on unified memory — with mmap the model exists twice (GPU buffer + page cache): 51 min of thrashing vs **40 s** load |
| `GGML_CUDA_DISABLE_GRAPHS=1` | **+3.0%** — a 256-expert MoE re-routes every token; HIP graph re-instantiation costs more than launch amortization saves |
| `LLAMA_NO_FUSE_HC_POST=1` | **+1.5%** — this specific micro-fusion loses on gfx1151 |
| `-ctk/-ctv q8_0` | on THIS quant, f16 KV is **−20.6%** (on the previous quant it was +21%… settings do not transpose — re-measure in your config) |
| `--n-cpu-moe 16` | frees **33 GB of GTT** for 0.1% cost — and prefill *improves* 2.4× (the constraint is memory pressure, not compute) |
| `--spec-draft-n-max 2` | ⚠️ **the optimum depends on context depth**: 3 wins at short context, **2 wins from ~16k on** (+3-4%). Long-context agents → 2. There is a hard cliff at n≥4 (RDNA3 MMVQ `max_batch=4` for these expert types) |
| `--temp 1.0 --top-p 0.95` | the values *sourced* from DeepSeek + unsloth (lower temp is faster via draft acceptance, but unsourced) |
| speed you should see | ~20-27 t/s short ctx (task-dependent!), ~14-16 t/s @16k, ~11 t/s @150k · prefill ~130 t/s |

## 🧨 Errors to avoid (each one cost us hours-to-days)

1. 🥇 **Don't evaluate quality through raw `/completion`.** The auto-injected `<think>` can
   close instantly at temp 0 → the model answers reflexively with 10-char reasoning traces
   and looks broken. We "discovered" a factual-recall deficit that was 100% this artifact.
   **Use `/v1/chat/completions` for anything quality-related.**
2. 🐉 **Never use a 2-bit drafter.** Draft hit-rate collapses to 9-15% and speculation then
   *loses 40%* vs no drafter at all. Q3_K_S (3.44 bpw) minimum. "A degraded drafter only
   costs acceptance" — true, and acceptance IS the entire gain.
3. ⛓️ **Never combine `draft-dspark` with ngram modes.** Syntactically accepted, destroys
   DSpark (+101% → +13% on tool calls). Ngram modes alone lose up to −46%.
4. 📏 **Don't transpose optima across contexts, quants or models.** Batch size swept at 32k
   and deployed at 256k cost us ×3.7. KV f16 flipped from +21% to −20.6% between two quants.
5. 🧊 **Wait for GTT release** after killing a server before starting the next (the driver
   frees buffers *after* the process dies) — poll `mem_info_gtt_used`, don't `sleep 5`.
6. 🎲 Speculative decoding here is **quality-neutral but not bit-exact** (batched
   verification reorders float reductions; near-tie argmax can flip). For token-paired
   benchmarks, bench with the drafter off.
7. 💾 Use `--slot-save-path` + the save/restore API around restarts: **25 ms instead of
   ~6 min** of re-prefill for a 30k-token session (×14,000).

## 🔬 How this was verified (summary)

Paired protocols, identical serving path both sides, instruments validated on synthetic cases
*before* any model run (they caught 9 real bugs — see the harness repo):

| probe | result vs full-precision API |
|---|---|
| 🧬 token-level teacher forcing (17,929 positions, 4 domains × EN/FR) | **90.8%** top-1 agreement · original's token outside local top-20: **0.04%** |
| 🧮 240-item exact-answer bank (McNemar, 6 families) | **240/240 = 240/240, zero discordance** (gap bounded ≲1.2%) |
| 📐 60 multi-step derivations, SymPy oracle | **parity** |
| 🛠️ structured tool-call sequences | **25/25 both sides** |
| 🚨 confident-wrong-answers P(answers ∧ wrong) | **0.0%** |
| 🇫🇷 French | *more* token-faithful than English (88.1% vs 83.0% on physics) |

Also measured: **Q5_K attention has better wikitext perplexity and worse real behavior**
(failed temp-0 probes, one repetition-loop) — our third demonstration that PPL ≠ quality.

## 🔁 Reproduce

`recipe/requant_attention.sh` + `recipe/tensor_types_q6kattn.txt` rebuild this file from
unsloth's UD-IQ3_XXS shards in ~15 min (129 tensors requantized, 1,199 copied bit-identical,
every tensor explicitly pinned — unpinned tensors silently fall into mixture heuristics).

## 🧪 Full methodology, harness & negative results

👉 **[Kevletesteur/quant-vs-api-parity-harness](https://huggingface.co/datasets/Kevletesteur/quant-vs-api-parity-harness)** —
the complete benchmark toolkit (gates, teacher forcing, SymPy oracles, calibration), plus
`NEGATIVE_RESULTS.md`: what did **not** work, with numbers — three heavy engineering efforts
abandoned on measurement, and the 9 instrument traps we fell into so you don't have to. 🪤

## 📜 Provenance & integrity

`deepseek-ai/DeepSeek-V4-Flash-0731` (MIT) → `unsloth/DeepSeek-V4-Flash-0731-GGUF`
(UD-IQ3_XXS, imatrix) → this repo (attention families requantized; experts untouched).

```
sha256  538ec6932c50c6993f8a2cb18dc138e0fb5f6a7c2659e53a7566f1dc8ebb4344
size    103,159,533,696 bytes
```

Benchmarked on: AMD Ryzen AI Max+ 395 (gfx1151/RDNA 3.5), ROCm 7.1, 115.3 GiB unified,
llama.cpp mainline HIP. *Measured, documented, and calibrated — enjoy.* 🚀
