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

# GLM-5.3 — EXL3/TR3 3.42 bpw (mixed K3/K4 trellis, data-free)

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
| routed experts (layers 3–78, incl. MTP-78) | EXL3 trellis, per layer 148 experts K3 + 108 experts K4 (avg 3.42 bpw), mcg codebook |
| dense MLP (layers 0–2), all attention, norms, embeddings, lm_head, mlp.gate, eh_proj | BF16, carried byte-exact |
| shared experts | BF16 in-checkpoint (online K6 at serve) |

- **Data-free encode**: identity Hessian (H=I), rotations + trellis search only —
  no calibration capture anywhere. Deterministic: seeds derived from
  (layer, expert, projection, rank), seed_base 20260711.
- **K4 selection**: per layer, the experts with highest relative round-trip MSE
  under the K3 encode are re-encoded at K4 (worst-108 per layer: 64 from the 3.25 pass + next-44 delta pass, identical seeds/machinery).
- Per-expert tier map in `tier_bitmap.json`; encode provenance in
  `config.json.hybrid_tr3_tail`; file hashes in `MANIFEST.sha256`.

## Serving

TR3 mixed-bit layout (carrier BF16 shards + per-layer trellis payloads) — serve
with an exllamav3-b12x/sparkinfer-lineage stack, TP4 (+ DCP4/MTP3), FP8 KV.
The mixed-K projection-tiers patch is REQUIRED; a stock loader that assumes a
uniform K per layer will produce fluent garbage. Not loadable by vanilla
exllamav3 model loading.



### Independent turnkey serving qualification (2026-08-29)

A separate live qualification on four RTX PRO 6000 Blackwell Server Edition
GPUs (97,887 MiB each; NVIDIA 595.58.03 / CUDA 13.2) passed with the
fail-closed turnkey image and public profile below:

- Image: [`ghcr.io/malaiwah/glm52-exl3-vast@sha256:aec4075e02241b71321b5601763eba35e86ea02b13c3c3c43ac34fae30161160`](https://github.com/malaiwah/glm52-exl3-vast/pkgs/container/glm52-exl3-vast)
- Profile: `MODEL_PROFILE=glm53-3.42bpw`
- Checkpoint revision: `8bef807a0fcdd180e984a26b50e731cdba9a8ff2`
- Topology: TP4 + DCP4, online shared-expert K6, native probabilistic MTP3,
  mixed K3/K4 routed experts, dynamic-token NVFP4 KV with FP8 RoPE, C8
- Qualified request envelope: 393,216 tokens and 3,415,867,392 KV-cache bytes
  per GPU. This is a measured safe profile, not the architecture's maximum.

The inherited 520,192-token GLM-5.2 envelope passed deterministic startup
checks but OOMed during the first temperature-1 sampler compile; it is not a
qualified setting for this checkpoint. The selected arm retained at least
665 MiB of observed physical free memory after first-use compilation and C8
sampling.

Observed gates: startup arithmetic/factual/instruction plus strict JSON and
32K retrieval; all applicable OpenAI chat/streaming/reasoning/structured-output/
tool-use contracts; 72/72 temperature-1 decode requests across C1/C4/C8
(227.55 output tok/s at C8); and 15/15 five-depth retrieval facts through a
389,959-token tokenizer-exact document. The post-stress verifier and 3,469-line
runtime-log audit were clean. A second boot from the exact published image
passed authentication (401 without a key, 200 with a key), chat inference, and
the token-gated dashboard while running with no source-code bind mounts.

Full measurements, rejected-arm evidence, and artifact names:
[qualification record](https://github.com/malaiwah/glm52-exl3-vast/blob/main/TEST_RESULTS.md#glm-53-full-model-342bpw-qualification-jarvislabs-2026-08-29).
The image carries the required mixed-K projection-tier runtime rather than
silently loading this checkpoint with a uniform-K path.

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
- Full runbook + runner: [`kld/`](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw/tree/main/kld) in this repo
  (`KLD-REPRODUCTION.md`, `prefill_kld_53.py`, `fetch-teacher.sh`).
- Independent reproduction bundle (receipts, unedited logs, checksums,
  pinned revisions): [`kld/cn3/`](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw/tree/main/kld/cn3).

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
  [`kld/cn3/`](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw/tree/main/kld/cn3).
- [willfalco/GLM-5.2-EXL3-TR3-3.25bpw](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
  — the 3.25 tier recipe and mixed-K checkpoint-format lineage.
- [local-inference-lab](https://github.com/local-inference-lab/rtx6kpro) —
  the qualified r17 serving stack these artifacts boot on.

## Status

Measured and independently reproduced 2026-08-29: mean KLD 0.024105 (fp8 KV; CN3 reproduction 0.023966) — the quality artifact of this release pair. Per-layer K4 set is a strict superset of the 3.25 release.

Quantized with `encode_tr3_53.py` (exllamav3 v0.0.43 vendored math, MIT) on
4x RTX PRO 6000 Blackwell. Encode + tooling notes ship in the repo.
