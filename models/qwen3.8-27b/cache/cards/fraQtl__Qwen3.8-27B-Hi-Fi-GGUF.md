---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - qwen
  - qwen3.8
  - gguf
  - q4_k_m
  - calibration-aware-quantization
  - fraqtl
---

# Qwen3.8-27B Q4_K_M GGUF — fraQtl Hi-Fi (calibration-aware quantization)

**Text-only GGUF** (the vision tower is not converted). Iso-size Q4_K_M-class
build of `Qwen/Qwen3.8-27B` with fraQtl calibration-aware per-tensor
quantization: byte-matched to the leading community Q4_K_M (within 0.001%),
measurably closer to the original model's output distribution on both locked
evaluation slices.

## Evaluation — fidelity to the original model

Metric: symmetric top-20 KLD on the support of a Q8_0 teacher **quantized from
the original pinned bf16 checkpoint** (fidelity-to-original — the axis a user
migrating from the original model experiences; teacher-framing caveat applies).
3 independent runs, run-to-run drift ≤1.4e-17 (byte-deterministic rig), one
shared teacher and one pinned eval slice for every arm.

| Arm | bytes | KLD code/math ↓ | KLD general ↓ |
|---|---|---|---|
| **fraQtl Hi-Fi (this repo)** | 16,464,274,816 | **0.12297 (−45.3%)** | **0.23120 (−18.0%)** |
| [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_M (canonical) | 16,464,440,224 | 0.22493 | 0.28209 |
| [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) Q4_K_M (second) | 18,973,870,432 | 0.19074 | 0.24274 |

Margins are vs the canonical arm. Sizes are not iso across all three arms:
this build is byte-matched to the unsloth canonical by construction
(Δ165,408 B, 0.001%); the ggml-org file is a genuinely different, larger
build (18.97 GB, no NextN block — disclosed). Read every row as
fidelity-at-stated-size.

**Disclosed losses:** Wikitext-2 perplexity is measurably worse for this build
(+0.7% vs canonical) and general-slice top-1 token agreement is marginally
worse (−0.2pp). Both are in the build receipt.

### Task benchmarks — ties, stated as ties

GSM8K (n=200) and MATH-500 (n=500), 0-shot instructed-CoT, identical harness
for all three arms:

| Arm | GSM8K | MATH-500 |
|---|---|---|
| fraQtl Hi-Fi | 95.0% (190/200) | 87.6% (438/500) |
| unsloth canonical | 94.0% (188/200) | 88.2% (441/500) |
| ggml-org second | 94.0% (188/200) | 87.4% (437/500) |

**Every pairwise difference is inside the 95% confidence interval (GSM8K
±6.9pp at n=200; MATH-500 ±4.4pp at n=500) — these are statistical ties in
every direction, not wins for any arm.** A per-item paired audit of GSM8K
sharpens this: ours vs canonical differ on only 4/200 items (3–1 split,
exact two-sided sign test p = 0.625 — indistinguishable from a coin flip).
The KLD margins above are the only metrics on this card claimed as wins.

### Long-context retrieval (needle-in-a-haystack)

Grid: contexts {8K, 32K} × depths × 3 passkeys, plus 131K cells × 2 passkeys —
63 cells per arm per run, multi-run. **All three arms are perfect on every
cell of every run (270 individual passkey checks total)** — reported as
parity, not a differentiator.

## Provenance

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0) |
| This artifact | `Qwen3.8-27B-fraQtl-HiFi-Q4_K_M.gguf`, 16,464,274,816 B, sha256 `43bd8f235bd7843b6bb06d632235b3e2430c23ad3f54b2fb1e56f35e74eebd8d`, true bpw 4.8210, 866 tensors (matches the canonical's tensor set, incl. NextN block) |
| Toolchain | llama.cpp @ `4df29be4f4c3673f428170fda944a5b19f743bb8` — single commit for conversion, quantization, and every eval, all arms. Conversion-correctness gate (bf16-GGUF vs HF reference, hybrid GDN arch): top-1 0.983, KLD 2.4e-4 — PASS |
| Calibration | A1-selector winner `mix_cm70_gen30`, llama-imatrix at ctx 8192; `imatrix.dat` (sha256 `c36c3d06…3dbd0d`) included in this repo |
| Teacher | one Q8_0-from-our-bf16 instance, used identically for all three arms |
| Baselines pinned | unsloth `Qwen3.8-27B-GGUF` @ `990216cf3125` (`322e194f…3482`); ggml-org `Qwen3.8-27B-GGUF` @ `0669b98607d4` (`31629f53…3d34`) |
| Eval runtime | llama-cpp-python 0.3.35, A100-80GB, seed 0 |

## Fine print

- One MATH-500 cell (ours) was accidentally run twice due to a preemption
  that looked like a wedge: an isolated re-fire and the original delayed run
  both completed independently and converged on the **exact same score
  (438/500)** — an incidental determinism receipt. Full detail in the
  taskbench receipt.
- 262K is this model's native context; the numbers on this card are a
  quality-tier GGUF claim, not a KV/serving claim.
- Receipts for every number are in [`receipts/`](./receipts/) (JSON: pins,
  conversion gate, calibration selection, build + 3-run KLD, NIAH grid,
  task benches).

## Credit

- [Discussion Qwen/Qwen3.8-27B#65](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/65)
  (d4nnyy/AtomicChat): prior day-0 stored-logits KLD across 36 quants
  (4096-ctx, no retrieval gate) — credited as prior art; we do not claim
  first KLD receipts on this model. Our scope differs: independent teacher,
  32K–131K contexts, needle-gated, 3-run.
- [unsloth](https://huggingface.co/unsloth): the canonical community build
  (4.3M downloads/30d), UD-Q4_K_M class.
- [ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp): toolchain
  and second-arm GGUF.

## Files

| File | What |
|---|---|
| `Qwen3.8-27B-fraQtl-HiFi-Q4_K_M.gguf` | the artifact (16.46 GB) |
| `imatrix.dat` | calibration importance matrix (reproducibility) |
| `receipts/` | receipt JSONs for every number on this card |

## The fraQtl ladder

| Tier | Job | Example |
|---|---|---|
| **Hi-Fi** | maximum fidelity at standard 4-bit size | this repo |
| **Hi-Fi Phone** | fits phone/laptop RAM, fidelity-first | [E2B Phone, 2.86 GB](https://huggingface.co/fraQtl/Gemma-4-E2B-it-Hi-Fi-GGUF) |
| **SmartEdge** | smallest footprint that survives | [E2B IQ3XXS, 2.45 GB](https://huggingface.co/fraQtl/Gemma-4-E2B-it-Hi-Fi-GGUF) |

Same discipline at every tier: pinned provenance, measured numbers, losses
disclosed.

## More from fraQtl

The serving lane — KV-cache compression sidecars for vLLM — holds
**nine concurrent ≈128K-context users on a single A100** (134.1 tok/s
aggregate, 9/9 per-user retrieval checks, receipt 2026-08-14):
[fraQtl/qwen3-4b-instruct-2507-kv-sidecars](https://huggingface.co/fraQtl/qwen3-4b-instruct-2507-kv-sidecars).
Org page: [huggingface.co/fraQtl](https://huggingface.co/fraQtl).
