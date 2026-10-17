---
license: other
license_name: qwen
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
- nvfp4
- modelopt
- quantized
- vllm
- multimodal
- mtp
- dgx-spark
- gb10
---

# Qwen3.8-Flash-Next — NVFP4 · mmap PLE · BF16 KV · MTP (DGX Spark)

A derivative of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
built for a single NVIDIA DGX Spark (GB10, `sm_121`, TP=1):

- **NVFP4 (W4A16)** weights for the compute-heavy GEMMs;
- the **51.2B-parameter PLE / n-gram table stored outside the model index** as
  packed NVFP4 and read from disk with `mmap`, one requested row at a time;
- **BF16 QSA KV cache** — no FP8 KV;
- **MTP retained**, multimodal (vision) retained, all 31 MTP
  source tensors preserved;
- served by **unmodified upstream vLLM** plus an out-of-tree plugin.

## Provenance

| | |
|---|---|
| Source model | `Qwen/Qwen3.8-Flash-Next` |
| Source revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Quantizer | NVIDIA Model-Optimizer `8810eb5e31ddacc438b08f17317db182e833ef66` |
| Qualified vLLM | `7c5dc571cbd1064ecc8a9b1045637ff647aa22cb` |
| Plugin | `qwen38-nvfp4-mmap-ple` `0.1.0+e80abe78d2ae7010` |
| Target | DGX Spark / GB10, `sm_121`, TP=1 |

## Layout

```text
model-*.safetensors        248 shards, 228018 tensors, 69.9 GiB
ple_nvfp4/                 128 shards, 26.822 GiB, mmap-only
runtime/                   plugin source, preflight, serve profiles
quantization/              tensor map, fused scales, recipe, SHA256SUMS
validation/                10 JSON artifacts: the evidence behind every claim below
```

The external PLE table is **deliberately absent** from
`model.safetensors.index.json`; listing it would make the ordinary vLLM weight
loader allocate 320001536 × 160 rows in BF16
(~95 GiB).

## Precision

`W4A16_NVFP4`, block size 16, global scale `amax / (6.0 * 448.0)`.

NVFP4: routed MoE experts, shared-expert projections, QSA q/k/v/o and indexer,
GDN in-projections and out-projection, PLE key/value projections, MTP linears,
`lm_head`.

BF16: vision tower and projector, token embedding, norms, routers and
shared-expert gates, HyperConnection parameters, GDN and PLE convolution/state
tensors — i.e. every non-GEMM or numerically sensitive tensor. The model is
NVFP4-*dominant*, not uniformly FP4.

**Fused-layer global scales.** vLLM keeps one global scale per fused Linear
(`weight_scale_2.max()` for `qkv_proj` / `gate_up_proj` / `in_proj_qkvz` /
`in_proj_ba`, and `w13_weight_scale_2[:, 0]` for routed experts). Every member of
a fused group is therefore quantized against a shared scale; the validator
checks this exactly.

## Evidence

| Gate | Result | Artifact |
|---|---|---|
| Upstream vLLM has the required Qwen4Exp APIs | passed | `validation/vllm-main-audit.json` |
| Serve flags exist at the pinned revision | passed | `validation/serve-flags-audit.json` |
| PLE row fidelity (10000 random rows) | cosine 0.995540, finite=True | `validation/ple-row-validation.json` |
| PLE mmap: anonymous RSS on open | 0.39 MiB for a 26.822 GiB mapping | `validation/ple-mmap-validation.json` |
| PLE mmap: decode identity | 0/8192 rows differ from the in-memory decoder | `validation/ple-mmap-validation.json` |
| Body structure | 228018/228018 keys, 0 missing, 0 duplicate | `validation/body-nvfp4-validation.json` |
| Body fidelity | cosine mean 0.995620, min 0.995146 | `validation/body-nvfp4-validation.json` |
| Fused-scale consistency | 0 violations across 134 groups and 25088 expert pairs | `validation/body-nvfp4-validation.json` |
| Published checkpoint integrity | 228018 tensors across 248 shards, size exact, 0 dtype errors | `validation/checkpoint-verification.json` |
| Plugin unit tests | 64 passed, 0 failed | `validation/plugin-tests.json` |
| 32K vLLM instance (text · vision · MTP) | not run | `validation/vllm-instance-32k.json` |
| 262K native context | pending | — |
| 393K YaRN 1.5x context | pending | — |
| GB10 memory envelope | pending | — |

**Not yet qualified on hardware.** Quantization, packaging and the static
compatibility audits are complete and reproducible. Nothing in this repository
has been booted on a GB10: the 32K instance test, the long-context profiles and
the memory envelope are pending, and no serving claim should be made until the
corresponding artifact exists under `validation/`.

## Serving

```bash
python -m pip install -e runtime/qwen38_nvfp4_mmap_ple
export VLLM_PLUGINS=qwen38_nvfp4_mmap_ple
export QWEN38_PLE_MMAP=1 QWEN38_PLE_THREADS=8
unset VLLM_PLE_CPU_OFFLOAD

MODEL=/path/to/this/checkpoint ./runtime/serve-32k.sh
```

The launchers run `runtime/preflight.py` first, which refuses to start unless
the checkpoint, PLE manifest, plugin, vLLM revision, device and serving flags
all match what was qualified. The PLE table must live on **local NVMe**: it is
paged in on every lookup, and preflight rejects network filesystems.

`--kv-cache-dtype bfloat16`, TP=1 and disabled prefix caching are requirements,
not defaults — see `plan.md` sections 24-30.

## License

Inherits the upstream `Qwen/Qwen3.8-Flash-Next` license; see `LICENSE`.
