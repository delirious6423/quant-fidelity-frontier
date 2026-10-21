---
language:
- en
- zh
library_name: transformers
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
pipeline_tag: text-generation
base_model: zai-org/GLM-5.3
base_model_relation: quantized
tags:
- exl3
- trellis
- tr3
- glm_moe_dsa
---

# GLM-5.3 — EXL3/TR3 3.25 bpw (mixed K3/K4 trellis, data-free)

> [!NOTE]
> **Weights uploaded and verified; KLD measured** (2026-08-29, full-vocabulary
> teacher-forced KLD vs the sealed BF16 teacher, held-out confirmation windows
> of [brandonmusic/GLM-5.3-BF16-full-logits](https://huggingface.co/datasets/brandonmusic/GLM-5.3-BF16-full-logits)
> — 4 windows x 2,047 positions x 154,880 vocab; reproduction kit + runner in
> this repo).

Trellis (EXL3) quantization of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)
(755B glm_moe_dsa MoE: 78 layers + MTP, 256 routed experts/layer), following the
GLM-5.2 TR3 lineage layout. Fits TP4 on 4x 96 GB (RTX PRO 6000 Blackwell class)
with FP8 KV cache.

From the upstream card: *"GLM-5.3 uses the same base model as GLM-5.2 — every gain
comes from post-training. Compared with GLM-5.2, it is much better at complex
coding and long-horizon tasks."* See the [base model card](https://huggingface.co/zai-org/GLM-5.3)
for benchmarks, serving guides, and the technical report
([arXiv:2602.15763](https://arxiv.org/abs/2602.15763)).

## What is quantized

| component | treatment |
|---|---|
| routed experts (layers 3–78, incl. MTP-78) | EXL3 trellis, per layer 192 experts K3 + 64 experts K4 (avg 3.25 bpw), mcg codebook |
| dense MLP (layers 0–2), all attention, norms, embeddings, lm_head, mlp.gate, eh_proj | BF16, carried byte-exact |
| shared experts | BF16 in-checkpoint (online K6 at serve) |

- **Data-free encode**: identity Hessian (H=I), rotations + trellis search only —
  no calibration capture anywhere. Deterministic: seeds derived from
  (layer, expert, projection, rank), seed_base 20260711.
- **K4 selection**: per layer, the experts with highest relative round-trip MSE
  under the K3 encode are re-encoded at K4 (worst-64 per layer).
- Per-expert tier map in `tier_bitmap.json`; encode provenance in
  `config.json.hybrid_tr3_tail`; file hashes in `MANIFEST.sha256`.

## Serving

TR3 mixed-bit layout (carrier BF16 shards + per-layer trellis payloads) — serve
with an exllamav3-b12x/sparkinfer-lineage stack, TP4 (+ DCP4/MTP3), FP8 KV.
The mixed-K projection-tiers patch is REQUIRED; a stock loader that assumes a
uniform K per layer will produce fluent garbage. Not loadable by vanilla
exllamav3 model loading.


## KLD vs BF16 teacher

Full-vocabulary, teacher-forced `KL(teacher || student)` against sealed BF16
GLM-5.3 logits — 4 held-out windows × 2,047 positions × 154,880 vocabulary,
fp32 log-softmax both sides. Measured independently on two different
4× RTX PRO 6000 (96GB) machines:

| weight quant | KV mode | this work | CN3 ([@dareposte](https://huggingface.co/dareposte)) | Δ |
|---|---|---:|---:|---:|
| **3.42 bpw** | **fp8** | **0.024105** | **0.023966** | −0.6% |
| **3.25 bpw** | **fp8** | **0.026103** | **0.026776** | +2.6% |
| 3.25 bpw | nvfp4 | 0.035741 | 0.036661 | +2.6% |
| 3.42 bpw | nvfp4 | 0.037757 | 0.037060 | −1.8% |
| 3.42 bpw | nvfp4+rope8 | 0.039518 | 0.037695 | −4.6% |
| 3.25 bpw | nvfp4+rope8 | — | 0.039396 | CN3 only |

Readings: the 3.25↔3.42 weight step changes KLD by only ~0.002; the
fp8→nvfp4 KV step costs ~7× more (~0.014) — cache format matters more than
the extra 0.17 bpw. Under nvfp4 KV the weight-quant difference washes out
entirely. The window SD (~0.02–0.03) is corpus heterogeneity — one
citation-dense legal window is uniformly hardest; dialogue, explanatory
prose, and reasoning-trace registers measure near-transparent.

<details>
<summary>Per-window means (dialogue / legal / prose / reasoning-trace)</summary>

| config | source | w0000 | w0001 | w0002 | w0003 |
|---|---|---:|---:|---:|---:|
| 3.42 fp8 | this work | 0.0141 | 0.0537 | 0.0137 | 0.0148 |
| 3.42 fp8 | CN3 | 0.0148 | 0.0542 | 0.0135 | 0.0134 |
| 3.25 fp8 | this work | 0.0188 | 0.0579 | 0.0138 | 0.0139 |
| 3.25 fp8 | CN3 | 0.0183 | 0.0592 | 0.0142 | 0.0154 |
| 3.42 nvfp4 | this work | 0.0199 | 0.0816 | 0.0221 | 0.0274 |
| 3.42 nvfp4 | CN3 | 0.0198 | 0.0775 | 0.0210 | 0.0300 |
| 3.25 nvfp4 | this work | 0.0256 | 0.0799 | 0.0184 | 0.0190 |
| 3.25 nvfp4 | CN3 | 0.0255 | 0.0804 | 0.0189 | 0.0218 |
| 3.42 nvfp4+rope8 | this work | 0.0212 | 0.0812 | 0.0257 | 0.0300 |
| 3.42 nvfp4+rope8 | CN3 | 0.0218 | 0.0761 | 0.0236 | 0.0292 |
| 3.25 nvfp4+rope8 | CN3 | 0.0268 | 0.0883 | 0.0204 | 0.0221 |

</details>

### Method (reproducible)

- **Teacher**: [brandonmusic/GLM-5.3-BF16-full-logits](https://huggingface.co/datasets/brandonmusic/GLM-5.3-BF16-full-logits),
  `reference-full-panel` **confirmation** lane (held out from every
  calibration fit), revision `427368f1`.
- **Student**: this checkpoint, loaded by the digest-pinned r17 serving
  image (`sha256:c5e96c5b…`) — the real trellis kernels and online-K6 path,
  offline `vllm.LLM`, TP4, one teacher-forced prefill per window.
- Full runbook + runner: [`kld/`](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw/tree/main/kld) in this repo
  (`KLD-REPRODUCTION.md`, `prefill_kld_53.py`, `fetch-teacher.sh`).
- Independent reproduction bundle (receipts, unedited logs, checksums,
  pinned revisions): [`kld/cn3/`](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw/tree/main/kld/cn3).

> **Capacity caveat** (via the CN3 report): KV-pool token figures printed by
> KLD-profile boots (TP4/DCP1, 4,096-token envelope) are logical pool
> values for that profile only — they are **not** maximum context length or
> production serving capacity.

## Credits

- **[brandonmusic](https://huggingface.co/brandonmusic)** — thank you for the
  [GLM-5.3-BF16-full-logits](https://huggingface.co/datasets/brandonmusic/GLM-5.3-BF16-full-logits)
  teacher captures that make this measurement possible without a 1.5TB BF16
  forward, and for the TR3 quantization references this release follows: the
  [GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78)
  method card, runbook, and the r10 reproducibility bundle whose encoder
  lineage (`encode_tr3_v31.py`) this checkpoint was produced with.
- **[dareposte](https://huggingface.co/dareposte)** — thank you for the
  independent CN3 reproduction: all six weight/KV configurations on separate
  hardware, within ±5% of our means, published here with receipts under
  [`kld/cn3/`](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw/tree/main/kld/cn3).
- [willfalco/GLM-5.2-EXL3-TR3-3.25bpw](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
  — the 3.25 tier recipe and mixed-K checkpoint-format lineage.
- [local-inference-lab](https://github.com/local-inference-lab/rtx6kpro) —
  the qualified r17 serving stack these artifacts boot on.

## Status

Measured and independently reproduced 2026-08-29: mean KLD 0.026103 (fp8 KV; CN3 reproduction 0.026776) on held-out confirmation windows. Serving-qualified on the r17 stack; measured 590k-token KV pool at GMU 0.965 — the context-breadth artifact of this release pair.

Quantized with `encode_tr3_53.py` (exllamav3 v0.0.43 vendored math, MIT) on
4x RTX PRO 6000 Blackwell. Encode + tooling notes ship in the repo.
