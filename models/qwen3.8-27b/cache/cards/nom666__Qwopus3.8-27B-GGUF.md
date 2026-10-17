---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- multi-token-prediction
- speculative-decoding
- qwen3_5
- merge
- task-arithmetic
---

# Qwopus3.8-27B · GGUF (with MTP)

GGUF quants of **Qwopus3.8-27B** — a task-vector transplant that carries the Qwopus reasoning distillation onto the **Qwen3.8-27B** base. Qwopus3.8 keeps Qwen3.8's architecture, 262,144 context length and native MTP speculative head, and layers the Opus-style distilled reasoning trait on top: **materially fewer tokens for the same result**, and a stronger non-thinking (instruct) mode. The **MTP block is exported** (nextn layout), so llama.cpp speculative decoding works out of the box. MLX/Apple-Silicon users: see the MTPLX builds ([4bit-Speed](https://huggingface.co/nom666/Qwopus3.8-27B-MTPLX-4bit-Speed) · [8bit-Quality](https://huggingface.co/nom666/Qwopus3.8-27B-MTPLX-8bit-Quality)).

## Files

| File | Size | Notes |
|---|---|---|
| `Qwopus3.8-27B-UD-Q4_K_XL.gguf` | 17.6 GB | **recommended 4-bit** — dynamic per-tensor allocation (Unsloth Dynamic v3 recipe + imatrix); measurably closer to bf16 than Q4_K_M at near-identical size |
| `Qwopus3.8-27B-Q4_K_M.gguf` | 16 GB | fits 24 GB VRAM with room for context |
| `Qwopus3.8-27B-Q8_0.gguf` | 27 GB | near-lossless; also the fastest llama.cpp quant on Apple Metal (K-quant kernels bottleneck there, not bandwidth) |

### About the UD-Q4_K_XL quant

Built by mirroring the per-tensor type map and calibration imatrix from
[Unsloth's Dynamic v3 Qwen3.8-27B release](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
onto this merge (the models share an identical architecture, so the
importance data transfers). Measured against the merge's own bf16 ground
truth on a held-out code/agentic/prose corpus: top-token agreement
**96.7%** vs 94.2% for Q4_K_M, with tail divergences 2–3× tighter. On a
38-task non-thinking suite it scores **36/38 vs 35/38 for stock Qwen3.8
quantized with the exact same UD v3 recipe** — the merge's
instruction-following edge survives quantization. The MTP head is
retained (Q6_K), so speculative decoding works as with the other files.
Credit to the Unsloth team for the v3 recipe and public imatrix.

## How it was made

Qwen3.6-27B and Qwen3.8-27B share an identical `qwen3_5` skeleton (64 layers, 5120 hidden, 248K vocab, hybrid GDN + full attention, 1-layer MTP head), which makes cross-generation task arithmetic shape-exact:

```
Qwopus3.8 = Qwen3.8 + (Qwopus3.6-27B-v2 − Qwen3.6)
```

All 1,199 source tensors matched and merged (fp32 math, bf16 out), including the `mtp.*` head; converted with llama.cpp's `convert_hf_to_gguf` (MTP exported as nextn) and quantized with `llama-quantize`.

## Results

Benchmarks below were measured on the MTPLX (MLX) builds of this same merge on an Apple M5 Max; GGUF quants at comparable bit-widths were smoke-tested for coherence and MTP drafting but not separately benchmarked.

**Non-thinking mode (the recommended regime):**

| | Qwopus3.8 4-bit | Qwopus3.8 8-bit | Qwen3.8 8-bit |
|---|---|---|---|
| 38-task instruct/code/math suite | **37/38** | 36/38 | 35/38 |
| Instruction-following subset | **4/5** | 3/5 | 2/5 |
| Hard-task suite (executed code + math + logic) | — | 8/13 @ **2,889 tok** | 8/13 @ 3,384 tok |

**Thinking mode — token efficiency:** at high reasoning effort, Qwopus3.8 matches base Qwen3.8 on a 13-task hard suite (13/13 both) while spending **10,050 vs 18,455 reasoning tokens — 45% fewer**.

## Usage

```bash
llama-server -m Qwopus3.8-27B-Q4_K_M.gguf -ngl 99 -c 32768 \
  --alias qwopus38 --spec-type draft-mtp --spec-draft-n-max 2 \
  --temp 0.3 --top-p 0.9 --top-k 40
```

Disable thinking (recommended for agentic/coding loops) via your chat-template kwargs or client; for thinking mode use a generous `max_tokens`. **Do not greedy-decode in thinking mode** (a known Qwen-family failure pattern).

## Notes

- Recommended for **fast agentic/coding loops and instruct-style serving with thinking disabled** — that is where this merge measurably beats its base.
- With thinking enabled, Qwopus3.8 solves hard tasks with ~45% fewer reasoning tokens, though base Qwen3.8 remains the stronger choice for heavy thinking-mode workloads — it is more consistent at concluding very long reasoning inside tight token budgets (Qwopus3.8 can occasionally deliberate past a tight budget on some prompts) and keeps an edge on the hardest reasoning tasks.
- This is a weight-space merge, not a fine-tune: no gradient training was performed on Qwen3.8.
- All credit for the Qwopus distillation to **Jackrong** ([Qwopus3.6-27B-v2](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2)); base model **Qwen3.8-27B** by the Qwen team.
