---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
- mxfp4
- quark
- rdna4
- vllm
- speculative-decoding
- dflash
---

# Ornith-1.5-35B-A3B — MXFP4 (Quark) for RDNA4, with bundled DFlash drafter

Data-free MXFP4 (OCP microscaling) quantization of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
built with AMD Quark for RDNA4 (gfx12) vLLM serving. 67 GB bf16 → **21.4 GB**, plus a
**bundled 737 MB DFlash draft model** for single-stream speculative decoding — the repo is
a complete two-profile serving package.

- All 256-expert MoE layers + shared experts in MXFP4 (30,840 modules); attention, routers,
  norms, embeddings, **vision tower** and the MTP head stay bf16. Vision verified working.
- **Weight-only config on purpose**: `quantization_config.global_quant_config.input_tensors`
  is `null`. Activations run in bf16 — this is what enables vLLM's `triton_unfused` MoE
  backend; a w4a4 reading falls into per-call QDQ emulation on platforms without native
  MXFP4 MoE kernels (~5× slower).
- Built from base revision `10fbf86fed` (2026-08-23) — the revision with the **trained**
  MTP head (earlier snapshots shipped a random-init head; see discussion #10 on the base repo).

## Serving (vLLM on RDNA4 — 2×Radeon AI PRO R9700, TP2)

**Get the vLLM that serves this on RDNA4 (gfx12):** the prebuilt image
[`capicua25x/vllm-rocm-rdna4`](https://hub.docker.com/r/capicua25x/vllm-rocm-rdna4)
(`:0.28.0-rdna4` / `:latest`), source and port notes at
[Capicua25x/vllm-rocm-rdna4](https://github.com/Capicua25x/vllm-rocm-rdna4)
(`RDNA4-PORT.md`). On CUDA hardware, recent stock vLLM works with the same flags.


**Concurrency profile** (default — ~32 concurrent users):

```bash
vllm serve <this-repo> --tensor-parallel-size 2 \
  --moe-backend triton_unfused --attention-backend TRITON_ATTN
```

**Single-stream profile** (bundled DFlash drafter; not for concurrent load):

```bash
vllm serve <this-repo> --tensor-parallel-size 2 \
  --moe-backend triton_unfused --attention-backend TRITON_ATTN \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"<this-repo>/dflash-draft\",\"num_speculative_tokens\":3}"
```

For tool calling, add `--enable-auto-tool-choice --tool-call-parser qwen3_coder` (this
model family emits the qwen3-coder XML function format).

Do **not** serve with the default (`auto`) MoE backend on ROCm gfx12 — every native MXFP4
MoE backend is CUDA/CDNA-gated and the fallback is quantize-dequantize emulation.

## Measured (same-protocol runs, 2×R9700 TP2, apexia concurrency-bench v3)

| profile / shape | c1 | c4 | c8 | c16 | c32 agg | practical ceiling* |
|---|---|---|---|---|---|---|
| concurrency · short | 60.5 | 191 | 326 | 552 | **888** | ~32 users |
| concurrency · 6k prefill | 58.6 | 189 | 314 | 516 | **782** | ~32 users |
| DFlash · short | **92.5** | 165 | 283 | 663 | 913 | single-stream use |
| DFlash · 6k prefill | **64.2** | 178 | 231 | 285 | 329 | single-stream use |

\* per-user ≥ 20 tok/s. Aggregate tok/s shown; the emulation fallback measures ~22 tok/s c1
with a ~1-user ceiling on the same hardware.

**Speculative-decoding notes**: the DFlash drafter degrades under long-context concurrent
load (separate drafter KV pool) — use the concurrency profile there. The native MTP head is
carried in bf16 (`mtp.*`, 785 tensors) and the config ships the exclusions vLLM needs to
build it unquantized (`mtp.fc`, `re:mtp\..*`), but MTP-3 lost to plain decode on this
hardware even with a tuned draft config — treat it as experimental.

## Quality — AA vs the official FP8

Same protocol both arms (lm-eval, temp 0.6/top-p 0.95/top-k 20, thinking off); quant cells
are 5-seed medians [range], seeds 1234–1238; reference cells are single-seed 1234.
reference = [ornith-ai/Ornith-1.5-35B-A3B-FP8](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-FP8)
measured on the same hardware.

| suite | this quant (conc) | this quant (DFlash serve) | official FP8 |
|---|---|---|---|
| GSM8K flexible (n=50) | 0.90 [0.84–0.96]⁵ | 0.88 [0.84–0.92]⁵ | 0.90 |
| GSM8K strict (n=50) | 0.84 [0.80–0.92]⁵ | 0.84 [0.78–0.86]⁵ | 0.88 |
| IFEval inst-strict (n=80) | 0.859 [0.836–0.883]⁵ | 0.875 [0.836–0.883]⁵ | 0.867 |
| IFEval prompt-strict (n=80) | 0.800 [0.763–0.825]⁵ | 0.800 [0.763–0.813]⁵ | 0.813 |
| τ²-bench telecom reward (n=114) | **0.965** (110/114) | same distribution* | not measured |

⁵5-seed median [range]. On the short suites the quant tracks its FP8 reference within
seed noise (±1–2 items per seed); τ² is the deep gate. \*Speculative decoding is
distribution-preserving (rejection sampling), so τ² quality carries across serve profiles.

## Credits

- **ornith-ai** for the base model (MIT).
- **z-lab** for the DFlash draft model, bundled unmodified in `dflash-draft/` from
  [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
  (apache-2.0, revision `f181eece`) — their original README is preserved in the folder.
- **EryriLabs** for the untrained-head diagnosis on the base repo, and **leonardlin
  (shisa-ai)** for the alternative distilled MTP head.
- **AMD Quark** quantization toolkit.
