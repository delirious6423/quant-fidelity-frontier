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

# Qwen3.8-27B Q4_K_M GGUF: fraQtl Hi-Fi (calibration-aware quantization)

**Q4_K_M at 16,464,274,816 bytes (unsloth UD-Q4_K_M: 16,464,440,224, within 0.001%): KLD to the Q8_0 teacher 0.12297 code/math and 0.23120 general, against 0.22493 and 0.28209 for the canonical build (−45.3%, −18.0%).**

**Text-only GGUF** (the vision tower is not converted). Method: fraQtl calibration-aware per-tensor quantization of `Qwen/Qwen3.8-27B`, byte-matched to the leading community Q4_K_M.

## Fidelity to the original model

Metric: symmetric top-20 KLD on the support of a Q8_0 teacher quantized from the original pinned bf16 checkpoint. 3 independent runs, run-to-run drift ≤1.4e-17 (byte-deterministic rig), one shared teacher and one pinned eval slice for every arm.

| Arm | bytes | KLD code/math ↓ | KLD general ↓ |
|---|---|---|---|
| **fraQtl Hi-Fi (this repo)** | 16,464,274,816 | **0.12297** (−45.3%) | **0.23120** (−18.0%) |
| [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) UD-Q4_K_M (canonical) | 16,464,440,224 | 0.22493 | 0.28209 |
| [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) Q4_K_M (second) | 18,973,870,432 | 0.19074 | 0.24274 |

Margins are vs the canonical arm. This build is byte-matched to the unsloth canonical by construction (Δ165,408 B, 0.001%); the ggml-org file is a different, larger build (18.97 GB, no NextN block). Each row is fidelity at its stated size.

Wikitext-2 perplexity is +0.7% for this build vs canonical, and general-slice top-1 token agreement is −0.2pp; the canonical is lower on perplexity and higher on top-1. Both are in the build receipt.

### Task benchmarks

GSM8K (n=200) and MATH-500 (n=500), 0-shot instructed-CoT, identical harness for all three arms:

| Arm | GSM8K | MATH-500 |
|---|---|---|
| fraQtl Hi-Fi | 95.0% (190/200) | 87.6% (438/500) |
| unsloth canonical | 94.0% (188/200) | 88.2% (441/500) |
| ggml-org second | 94.0% (188/200) | 87.4% (437/500) |

95% confidence interval: GSM8K ±6.9pp at n=200; MATH-500 ±4.4pp at n=500. Every pairwise difference is inside it. Per-item paired GSM8K audit, ours vs canonical: 4/200 items differ, a 3–1 split, exact two-sided sign test p = 0.625.

### Long-context retrieval (needle-in-a-haystack)

Grid: contexts {8K, 32K} × depths × 3 passkeys, plus 131K cells × 2 passkeys, for 63 cells per arm per run, multi-run. All three arms pass every cell of every run (270 individual passkey checks total).

## Scope

- Task benches are n=200 (GSM8K) and n=500 (MATH-500).
- Calibration: text imatrix from the A1-selector winner `mix_cm70_gen30` at ctx 8192; the vision tower is not converted.
- 262K is this model's native context; the numbers on this card are a quality-tier GGUF measurement, not a KV/serving measurement.

## Methods note

- One MATH-500 cell (ours) ran twice after a preemption that looked like a wedge; the re-fire and the original delayed run both completed and returned the same score (438/500). Detail in the taskbench receipt.

## Provenance

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0) |
| This artifact | `Qwen3.8-27B-fraQtl-HiFi-Q4_K_M.gguf`, 16,464,274,816 B, sha256 `43bd8f235bd7843b6bb06d632235b3e2430c23ad3f54b2fb1e56f35e74eebd8d`, true bpw 4.8210, 866 tensors (matches the canonical's tensor set, incl. NextN block) |
| Toolchain | llama.cpp @ `4df29be4f4c3673f428170fda944a5b19f743bb8`, single commit for conversion, quantization, and every eval, all arms. Conversion-correctness gate (bf16-GGUF vs HF reference, hybrid GDN arch): top-1 0.983, KLD 2.4e-4, PASS |
| Calibration | A1-selector winner `mix_cm70_gen30`, llama-imatrix at ctx 8192; `imatrix.dat` (sha256 `c36c3d06…3dbd0d`) included in this repo |
| Teacher | one Q8_0-from-our-bf16 instance, used identically for all three arms |
| Baselines pinned | unsloth `Qwen3.8-27B-GGUF` @ `990216cf3125` (`322e194f…3482`); ggml-org `Qwen3.8-27B-GGUF` @ `0669b98607d4` (`31629f53…3d34`) |
| Eval runtime | llama-cpp-python 0.3.35, A100-80GB, seed 0 |

Receipts for every number are in [`receipts/`](./receipts/) (JSON: pins, conversion gate, calibration selection, build + 3-run KLD, NIAH grid, task benches).

## Credit

- [Discussion Qwen/Qwen3.8-27B#65](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/65) (d4nnyy/AtomicChat): prior day-0 stored-logits KLD across 36 quants (4096-ctx, no retrieval gate), credited as prior art. Our scope differs: independent teacher, 32K–131K contexts, needle-gated, 3-run.
- [unsloth](https://huggingface.co/unsloth): the canonical community build (4.3M downloads/30d), UD-Q4_K_M class.
- [ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp): toolchain and second-arm GGUF.

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

## More from fraQtl

The serving lane (KV-cache compression sidecars for vLLM) holds nine concurrent ≈128K-context users on a single A100 (134.1 tok/s aggregate, 9/9 per-user retrieval checks, receipt 2026-08-14): [fraQtl/qwen3-4b-instruct-2507-kv-sidecars](https://huggingface.co/fraQtl/qwen3-4b-instruct-2507-kv-sidecars). Org page: [huggingface.co/fraQtl](https://huggingface.co/fraQtl).
