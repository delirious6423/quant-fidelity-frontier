---
license: apache-2.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3_5_moe
  - autoround
  - w4a16
  - int4
  - moe
  - intel
  - arc-pro-b70
  - vllm
  - mtp
  - vision
---

# Ornith-1.5-35B-A3B AutoRound W4A16 G128 (MTP BF16, Vision BF16)

Self-reported Intel Arc Pro B70 measurements (E2) with raw evidence.
Isolated C1, cache off, greedy diagnostic. Not independently reproduced.

Local AutoRound W4A16 conversion of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
for Intel Arc Pro B70 / vLLM XPU. This is **not** an official Ornith release.
The MTP draft head and the full vision tower ship **unquantized BF16** — this
single checkpoint serves text, agentic, and image workloads.

## Why AutoRound — measured equal-or-best vs the GPTQ reference

The previous reference is
[`…GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2`](https://huggingface.co/SergiioB/Ornith-1.5-35B-A3B-GPTQ-Int4-sym-G128-MTP-BF16-MixedCal-v2)
(10.37% of expert projections round-to-nearest fallback; the first Aug-19
GPTQ had 24.76%). AutoRound signSGD-tunes **every** expert projection:

| Conversion | RTN fallback | Speed class (p512/g128) |
|---|---:|---|
| Original GPTQ (Aug-19) | 24.76% | 69.0 t/s n=3 |
| MixedCal-v2 GPTQ | 10.37% | 70.74 t/s n=5 @ 150 W |
| **AutoRound (this repo)** | **0%** (30,720/30,720 tuned) | 70.1 t/s n=3, MTP1 92.4–96.3 |

Reference parity, teacher-forced against the BF16 source on 315 positions /
12 out-of-calibration prompts (2026-08-22) — **equal-or-best on all three
primary metrics**:

| Metric (lower/upper is better) | AutoRound | MixedCal-v2 GPTQ |
|---|---:|---:|
| mean \|Δlogprob\| ↓ | **0.20059** | 0.20250 |
| top-1 agreement ↑ | **0.9143** | 0.9048 |
| top-20-union KL ↓ | **0.27747** | 0.31911 |
| p95 \|Δlogprob\| ↓ | 0.702 | **0.663** |

Claim scope: "equal-or-better calibrated rounding" is measured; end-task
superiority is not claimed (logprob parity on 12 prompts, not a task suite).
Speed cells are n=3 screens inside the n=5 reference spread; the formal
serving-reference swap awaits the AutoRound n=5 set (ADR-0007). Same
`--quantization gptq` load path, same KV footprint (169,622 tokens @ 32K),
MTP1 acceptance 78.5% pos0 (GPTQ: 80.3%), decode t/s unaffected.

## Conversion contract

- **Converter:** AutoRound **0.14.2** (pinned), `quantize_and_save` streaming
  save, forced `is_model_patched`. Wall cost ~3.5 h on the B70 host (GPTQ ~6.4 h).
- **Calibration:** MixedCal-v2 corpus (sha `e88ccd5f…`), identical dynamic
  scope to the GPTQ reference: attention, MoE gates and embeddings excluded;
  the entire `mtp.*` tree and the 333-tensor vision tower stay **BF16**.
- **Packing:** `auto_gptq` / gptq qweight-qzeros-scales, symmetric **G128**,
  `desc_act=false` — vLLM loads it with plain `--quantization gptq`.
- `model_extra_tensors.safetensors` carries the BF16 MTP head and is part of
  the sharded index — download all files.

## Serve on Intel Arc Pro B70

Same stack as the GPTQ reference — do not substitute:

```text
vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
vLLM 0.27.2rc1.dev77+gac7509e2b · vllm-xpu-kernels 0.1.12.3 · MoE backend: XPU WNA16
```

Research default is **MTP1** (single MTP layer; acceptance past pos0 collapses).
Required patches: `patch_mtp_nightly.py` + `patch_mtp_boundary.py` (exact
131,072-token MTP completions).

```bash
vllm serve /model \
  --quantization gptq --dtype float16 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype auto --block-size 64 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --language-model-only --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### Vision serving (full VLM)

Drop `--language-model-only` to accept image inputs — everything else stays
identical. The BF16 tower (~0.89 GB) plus multimodal profiling take ~0.9 GiB
from the KV budget: with fp8 KV, util 0.90 and MTP1, `--max-model-len 160000`
fails the startup KV check (1.82 GiB needed vs 1.77 GiB available); use
**131072** (~1.49 GiB needed) or raise `--gpu-memory-utilization`.

Verified end-to-end on **this artifact** 2026-08-23 (MTP1, fp8 KV, 180 W): a
64×64 red-PNG request returns `red` with ~78 image prompt tokens; text-only
requests unchanged.

Copy-paste recipes and evidence:
[ORNITH-VLLM-XPU.md](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/ornith15-35a3/ORNITH-VLLM-XPU.md)
(vision: §3b).

## License

Follow the Ornith-1.5 source license (Apache-2.0 on the recorded card) plus
any additional terms of `ornith-ai/Ornith-1.5-35B-A3B`.
