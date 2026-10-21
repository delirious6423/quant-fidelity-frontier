---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - nvfp4
  - modelopt
  - gguf
  - ollama
  - blackwell
  - sm120
  - qwen3.5
  - dlanet
  - hybrid
  - finetune
pipeline_tag: text-generation
language:
  - en
quantized_by: xxchromexx
model_creator: Hermes-ft (fine-tune of Qwen3.8-27B)
---

# Qwermes3.8 — NVFP4 GGUF (Qwen3.8-27B hybrid, fine-tuned)

NVFP4 GGUF of a fine-tuned **Qwen3.8-27B** (dense, 64-layer hybrid: 48 Gated DeltaNet linear-attention + 16 full-attention, Qwen3-VL-style wrapper stripped to text-only):

| File | Quant | Size | Ollama tag | Verified |
|---|---|---|---|---|
| `Qwermes3.8-nvfp4.gguf` | **NVFP4** (ModelOpt, MLP-only; attention/DeltaNet Q8_0) | 20.0 GB | `Qwermes3.8:nvfp4` | ✅ coherent, 262K ctx |

**Why this exists:** this model family (Qwen3.5/3.8 hybrid GDN) has a nasty GPU-kernel gating trap on workstation Blackwell (sm_120). The GGUF works everywhere via llama.cpp/Ollama; NVFP4 additionally runs on Blackwell FP4 tensor cores. Details below — the benchmarks and the root-cause analysis are the useful part for the community.

---

## Benchmarks (RTX PRO 6000 Blackwell 96 GB, sm_120, CUDA 13, Ollama 0.33.1)

### Fine-tune vs stock: Qwermes3.8 NVFP4 vs base Qwen3.8 (12 prompts × 2, num_ctx 8192, seed 42, thinking off)

| Metric | Qwermes3.8 NVFP4 (20 GB) | Qwen3.8 base Q4_K_M (16.8 GB) |
|---|---|---|
| Decode avg | 53.0 t/s | 106.8 t/s |
| Prefill avg | 527.6 t/s | 414.3 t/s |
| Total tokens | 1,694 | 4,820 |

**The 2× decode gap is MTP, not the fine-tune.** The stock model ships with a multi-token-prediction head (`blk.64.nextn.*`); this GGUF was converted `--no-mtp`. Same-quant control (both Q4_K_M): Qwermes 65.1 t/s vs base 84.1 t/s — the fine-tune costs ~29% decode speed (longer generations, same architecture).

> ⚠️ **Scope: directional smoke test, not a scored eval.** 12 hand-written prompts, single run, eyeball quality check. No rubric, no confidence intervals, no contamination check. Scored benchmarks (GSM8K/HumanEval/IFEval via lm-eval-harness) are in progress and will be published here when complete.

### Perplexity (wikitext-2 test, full 1.29 MB, llama-perplexity, ngl 99, c 8192)

| Quant | PPL |
|---|---|
| NVFP4 | 5.9276 ± 0.036 |

### Why NVFP4 is (correctly) slower in vLLM on sm_120 — the GDN kernel gate

Qwen3.8-27B is hybrid: **48/64 layers are Gated DeltaNet (GDN)** linear-attention. vLLM's fast GDN backends (FlashInfer/CuteDSL) gate to hardware with **tcgen05/TMEM tensor-memory** instructions: sm_90 and **sm_10x** (datacenter Blackwell, B200/GB200) with `head_k_dim==128` + CUDA 13. **sm_120 (RTX PRO 6000, RTX 5090, PRO 2000-series) is NOT on the list** → Triton/FLA fallback:

- vLLM 0.27.1: `Using Triton/FLA GDN prefill kernel` — decode 42.2 t/s (vs Ollama 52.7), **~2.2s FIXED TTFT** regardless of prompt length (measured: 1-token 2.24s, ~10k-token 4.97s).
- vLLM 0.22.0: same gate in source (verified). Also rejects unwrapped text-only configs (`Qwen3_5TextConfig` vs required VLM `Qwen3_5Config`).
- Marginal prefill is actually excellent (~5,500–11,000 tok/s measured) — the fixed 2.2s per-request cost dominates short prompts. vLLM is a batch/long-context play here, not an interactive one.
- Community confirmation: [ai-muninn / DGX Spark (sm_121)](https://ai-muninn.com/en/blog/dgx-spark-qwen3-122b-vllm-to-atlas-2x) — same wall; vLLM PR #44700 (GDN decode fix, +24% on B200) = **zero movement** on non-datacenter Blackwell. "The fast road isn't built here."

**llama.cpp/Ollama's GDN kernels need no tcgen05 → full speed on sm_120.** For interactive single-stream use on workstation Blackwell, **Ollama + NVFP4 beats vLLM on every metric.**

---

## NVFP4 build recipe (works — 5 failed attempts distilled)

**Do NOT use llmcompressor for NVFP4 GGUFs on this arch.** Its single-scale packing (`scales`) scrambles every dequant in llama.cpp. ModelOpt's two-level `weight_scale`/`weight_scale_2` is what llama.cpp's `_repack_nvfp4` expects.

1. **Unwrap the VLM wrapper first.** The merged checkpoint has triple-nested keys (`model.language_model.language_model.language_model.*`). Every quantizer loads via `AutoModelForCausalLM` and reports **every tensor UNEXPECTED → randomly initialized**. This silent failure is the #1 garbage-output cause. Rewrite the safetensors headers to clean `model.layers.*` (hardlinks can't fix headers) and drop the vision tower.
2. **ModelOpt 0.46.0** (studio venv, transformers 5.14.1): `quantize()` with `NVFP4_MLP_ONLY_CFG` (MLP→NVFP4 group 16, rest FP8) + calibration forward loop → **`compress()`** (packs real two-level scales; without it you get fake-quant BF16 at 50 GB) → `save_pretrained()`.
3. **Rename scale tensors** for the GGUF converter: `weight_quantizer._scale→weight_scale`, `_double_scale→weight_scale_2`, `input_quantizer._amax→input_scale` (= amax/448, FP8 E4M3 max). Keep `quantization_config` in config.json or the converter never triggers NVFP4 repack.
4. Convert with llama.cpp b10679 `convert_hf_to_gguf.py --outtype q8_0 --no-mtp` → NVFP4 tensors repacked, leftover BF16 (attention/DeltaNet/vision) → Q8_0. Layout matches the known-good [RadixArk Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) reference exactly.
5. **Embed the chat template** in the GGUF (`tokenizer.chat_template` — extract from a working GGUF of the same family or the source config). Without it, Ollama advertises only `['completion']` — no tools, no thinking — and rejects the `enable_thinking` param Hermes sends.

### Dead ends (don't repeat)
- llmcompressor NVFP4 → garbage GGUF every time (single-scale packing).
- unsloth native export → CUDA OOM crash (parent + subprocess both hold 51 GB on a 96 GB card) → run quantize standalone.
- BF16 "control test" through plain transformers → garbage (VLM wrapper confuses `AutoModelForCausalLM`); only llama-cli/Ollama tests are valid.
- Every exit-139 segfault was a resident Ollama model eating VRAM — `ollama stop` before heavy HF loads.

### Ollama registration (Windows)
Manual manifest in `registry.ollama.ai/library/Qwermes3.8/<tag>/` — big local copies wedge `ollama create`. Blob filenames use `sha256-` (dash, Windows). Config blob: `{"model_format":"gguf","model_family":"qwen35","file_type":1,"model_type":"27.3B"}`. Params: `{"temperature":0.7,"top_p":0.8,"num_ctx":262144,"stop":["<|im_start|>","<|im_end|>"]}`.

---

## KV cache facts (DeltaNet is state-heavy!)

Measured: **117 KB/token** → 128K ctx ≈ **14.7 GB**, 262K ≈ **30 GB**. Plan VRAM accordingly; a 16 GB card cannot hold 128K ctx fully in VRAM.

## Thinking-mode quirk

On complex code prompts at small num_ctx, the model thinks to the context limit and emits **zero content** (8,120 tokens of pure thinking — fine-tune behavior, not a quant defect). Fix: `"think": false` in Ollama request bodies (top-level; Ollama ignores it under `options`).

## Files

- `Qwermes3.8-nvfp4.gguf` — 20.0 GB, 192 NVFP4 tensors (64 layers × ffn gate/up/down, group 16), 306 Q8_0 (attention/SSM/vision), chat template embedded

**Verify integrity after download:**
```
sha256 71a78d965f36dcf8bb0669f0344f3bc7b7033b9e963ac6a8abe3e9a6a143378c  Qwermes3.8-nvfp4.gguf
```

## Training data

Fine-tuned with **QLoRA** (r=32, α=64, LR 2e-4, cosine, 32K ctx, packing) on **Hermes-3 multiturn data** — `agentlans/NousResearch-Hermes-3-Dataset-multiturn` (13,259 rows, ~10M tokens, single pass ≈ 2 effective epochs at 32K packing). Base: `Qwen/Qwen3.8-27B`. The fine-tune targets assistant-style conversational behavior (Hermes-3 format); no additional proprietary data was used.

> ⚠️ **Contamination note:** the fine-tune data is conversational (Hermes-3 multiturn), not benchmark-derived — but no formal contamination audit was run. Treat scored results as upper bounds.

## License

Apache-2.0 (base: Qwen/Qwen3.8-27B). Fine-tune weights derived from Hermes-ft project; no additional restrictions.
