---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - compressed-tensors
  - blackwell
  - vllm
  - glm5_next
  - moe
  - multimodal
language:
  - en
  - zh
---

# GLM-5.3-Flash — NVFP4 (experts-only, 4/6-optimal scales)

## TL;DR — the plain-English version

**What this is:** GLM-5.3-Flash, one of the strongest open models released this year, shrunk
from **660 GB down to ~196 GB** so it can run on a single multi-GPU machine instead of a
server rack — with quality kept as close to the original as the format allows.

**How:** the model's "experts" (where ~90 % of its bulk lives) are stored in **NVFP4** — a
4-bit number format that NVIDIA's newest (Blackwell) GPUs understand natively. Everything
quality-critical — attention, the vision system, the speculative-decoding head — is left
completely untouched at full precision.

**Is the quality still good?** The compressed weights land within ~8 % relative error of
the originals per-tensor (measured, not guessed — numbers below), which is at the good end
of what 4-bit formats achieve. We also used a scale-picking trick that cuts error ~19 %
compared to how most 4-bit quants are made.

**Can I run it today?**
- ✅ **H100 / B200-class GPUs** (≥ ~4× 80 GB): yes — command below.
- ⏳ **RTX PRO 6000 / consumer Blackwell (SM120)**: not yet — a missing GPU kernel in
  vLLM, being fixed in the open right now
  ([vllm#53963](https://github.com/vllm-project/vllm/issues/53963) — filed by us, with this
  exact checkpoint as the test artifact). Watch that issue; when it closes, this runs.

**New to quantization?** Think of it as saving a photo as a high-quality JPEG instead of
RAW: dramatically smaller, nearly indistinguishable, and this card documents exactly how
much was lost and where — so you don't have to take our word for it.

---

Weight-only **NVFP4 (NVFP4A16)** quantization of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — Z.ai's ~320B/A18B
natively-multimodal MoE (hybrid linear attention + rope-free sparse MLA) — in standard
**compressed-tensors `nvfp4-pack-quantized`** format. ~196 GB.

Quantized from the **true BF16 release**
([zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16), pinned
`b1967181`), not the FP8 export — no double-quantization on the experts.

## What makes this build different

Most NVFP4 quants use plain round-to-nearest with `amax/6` block scales. This build selects
block scales **per 16-element block by exact post-rounding squared error**, choosing between
the classic `amax/6` plan and NVIDIA ModelOpt's *four-over-six* plan (`M ∈ {4, 6}`,
256-normalized global scale) per tensor — never worse than classic RTN by construction.

Measured on this model (48-tensor random sample, against the BF16 source as ground truth):

| Metric | Value |
|---|---|
| Mean relative error (quantized experts vs BF16) | **0.0828** |
| Max relative error | 0.0848 |
| Total squared-error reduction vs plain `amax/6` RTN | **−18.75 %** |
| Blocks selecting the M=4 scale | 44 % (100 % of tensors adopted the 4/6 plan) |

For reference, the other public NVFP4 of this model reports rel-err ≈ 0.0925 with plain RTN
(same experts-only recipe; their card, their measurement).

## What is quantized, what is not

- **NVFP4**: routed-expert weights only — 36,288 tensors (42 MoE layers × 288 experts ×
  3 projections), ≈ 90 % of parameters.
- **BF16 (untouched)**: all attention (MLA incl. `kv_b_proj`, linear-attention layers, DSA
  indexer), shared experts, dense MLPs, router gates, embeddings, `lm_head`, the **full
  vision tower** (byte-identical to source — verified tensor-by-tensor), and the **MTP layer**
  (kept full-precision so vendor-style speculative decoding works:
  `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`).

## Verification (receipts, not vibes)

- Quantization math **bit-exact** vs an llmcompressor `NVFP4A16` oneshot and
  **byte-identical** vs NVIDIA ModelOpt's `NVFP4QTensor` export path (packed weights, E4M3
  block scales, global scales) at the pinned toolchain: `torch 2.13.0` /
  `compressed-tensors 0.18.0` / `safetensors 0.8.0`.
- Exact expert census asserted at build time; all values verified on the FP4 grid.
- Vision tower byte-compared against the source (347 tensors).
- `MANIFEST.sha256` covers every file.
- Not measured: end-to-end perplexity/evals (see serving status below). Weight-fidelity
  metrics only — stated plainly.

## Serving status (read this)

`glm5_next` is not yet in vLLM main (support PR:
[vllm#53906](https://github.com/vllm-project/vllm/pull/53906)). vLLM publishes dedicated
per-model images; this checkpoint **loads cleanly** there (Marlin NVFP4 auto-selected):

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  vllm/vllm-openai:glm53-flash-x86_64-cu130 \
  --model dabsLabs/GLM-5.3-Flash-NVFP4 \
  --tensor-parallel-size 4 \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --max-num-seqs 256
```

- **H100 / B200 / GB200**: expected to serve (vendor-verified hardware for the image).
- **SM120 (RTX PRO 6000 / consumer Blackwell)**: currently **blocked upstream** — the
  image's sparse-MLA kernels don't cover rope-free MLA on SM120. Full three-failure-mode
  dossier and fix tracking: **[vllm#53963](https://github.com/vllm-project/vllm/issues/53963)**
  (filed by us; this exact checkpoint is the test artifact). `--kv-cache-dtype` and backend
  workarounds do **not** help — it's a kernel gap, not a config issue.
- Set `VLLM_ENGINE_READY_TIMEOUT_S=3600`: engine init on a 320B MoE is slow.

## Provenance

Source revision pinned; conversion is a streaming shard-by-shard pass (the model is never
materialized), quantization primitives called directly from `compressed-tensors`. Built and
verified on 2026-08-26. MIT, as is the base model.

*Quantized by dabsLabs.*
