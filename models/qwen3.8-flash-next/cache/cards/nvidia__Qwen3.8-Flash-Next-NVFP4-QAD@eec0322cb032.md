---
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
library_name: modelopt
tags:
- nvidia
- ModelOpt
- Qwen3.8
- quantized
- nvfp4
- mxfp8
- qad
---

# Qwen3.8-Flash-Next-NVFP4-QAD

This is an experimental mixed-precision QAD checkpoint of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), exported after
100 optimizer updates using NVIDIA Model Optimizer and NeMo AutoModel with FSDP2.
It contains the full multimodal model, including the frozen vision encoder,
PLE n-gram embedding, and MTP module. Weight files occupy approximately 129.7 GB
(120.8 GiB).

**This checkpoint uses a more aggressive quantization recipe than
[Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).**
In addition to routed experts, it quantizes shared experts to NVFP4 and selected
GDN, QSA, and indexer projections to MXFP8. The repository name's `-QAD` suffix
identifies the training method; it does not mean the original NVFP4 recipe was
kept unchanged. Task accuracy gains from QAD are not statistically established.

Weight shards were consolidated for distribution. Every source file was verified
against its export SHA256, and consolidated files passed a complete readback.
Tensor names, shapes, dtypes, and stored values are unchanged.

## Quantization recipe

| Component | Stored precision | QAD treatment |
|---|---|---|
| Main LM routed experts, 48 MoE groups | NVFP4, block size 16 | Trained |
| Main LM shared experts, 144 linear matrices | NVFP4, block size 16 | Trained |
| GDN projections, 180 linear matrices | MXFP8, block size 32 | Trained |
| QSA q/k/v/o projections, 48 linear matrices | MXFP8, block size 32 | Trained |
| QSA indexer projections, 12 linear matrices | MXFP8, block size 32 | Frozen |
| PLE n-gram embedding | Per-tensor FP8 E4M3, BF16 scale | Frozen |
| MTP routed experts | FP8 E4M3, 128 x 128 block scales | Frozen |
| MTP shared experts and attention | BF16 | Frozen |
| Vision encoder, routers, gates, convolutions, output head, other excluded tensors | Original precision | Frozen |

The main model uses native NVFP4 and MXFP8 computation in the validated serving
configuration. Selected excluded operations retain their original FP32 contracts.
The KV cache was BF16 during evaluation; KV-cache quantization is not part of
this recipe. See `hf_quant_config.json`, `config.json`, and
`quantization_recipe.json` for machine-readable metadata.

The main LM was quantized from the original BF16 model. The preserved MTP and
PLE state derives from the released NVIDIA mixed-precision checkpoint. QAD did
not further quantize or train MTP. All benchmark measurements below disabled MTP.

## QAD training

The student used fake-quantized computation with floating master weights and a
frozen BF16 teacher. Training used text inputs; the vision encoder stayed frozen.

| Setting | Value |
|---|---|
| Completed optimizer updates | 100 |
| Maximum training sequence length | 16,384 tokens |
| Global batch size | 512 sequences |
| Consumed sequence examples | 51,200, including any repeated samples |
| Microbatch size | 1 sequence per rank |
| FSDP2 world size | 64 |
| Optimizer | Adam, betas (0.9, 0.95), epsilon 1e-8, weight decay 0 |
| Peak / minimum learning rate | 1e-5 / 1e-6 |
| Warmup / scheduler horizon | 50 / 500 updates; this export stops at update 100 |

16,384 is a sequence-length ceiling. It does not mean every example contained
16,384 non-padding tokens. The exact training data files are not distributed in
this repository. This is an inference checkpoint; optimizer state is not included.

Export validation compared native training-reference and exported-model
log probabilities at input lengths 256, 4,096, 8,192, and 16,384. Across 28,924
scored positions, mean absolute log-probability error was 1.21e-6, maximum error
was 1.00e-5, and top-1 agreement was 99.25%. This establishes export parity within
the tested limits, not parity with BF16 model accuracy.

## Experimental evaluation

The baseline is the original
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) model in BF16.
All entries are our measurements on a 0–100 scale. Benchmark metrics and
evaluation protocols can differ from published model-card results.

Validation scope: checkpoint identity, native quantization kernels, score
coverage, and task-specific grading were checked. Historical full-run request
retry, timeout, and output-termination accounting is incomplete, particularly
for Terminal-Bench. The numbers are descriptive; unresolved limit effects
prevent a clean attribution of differences to quantization or training.

| Benchmark / metric | Original model (BF16) | QAD update 100 |
|---|---:|---:|
| GPQA-Diamond accuracy | 92.05 | 91.60 |
| IFBench instruction-following score | 79.60 | 79.93 |
| SciCode subtask success | 50.22 | 51.11 |
| AA-LCR accuracy | 77.81 | 77.13 |
| MMMU-Pro accuracy | 77.75 | 77.34 |
| Terminal-Bench 2.1 native pass@1, qualified | 85.96 | 85.96 |

SciCode reports mean subtask success over eight operational runs per model,
with 338 scored subtasks per run (2,704 per model). Repeats used the same configured
request seed and are not independently reseeded trials. A confidence interval
against BF16 is not reported here. SciCode subtask success must not be compared
with published main-problem scores.

Terminal-Bench retains all 712 outcomes per model, including protocol-defined
runtime failures scored as zero: 6/712 for BF16 and 4/712 for QAD. Both models
scored 612/712. The exploratory task-cluster bootstrap 95% interval for QAD minus
BF16 is [-2.95, +3.23] percentage points. Historical allocation-wide timeout/retry
and output-cap rates remain unknown; later two-task logging diagnostics do not
retroactively fill those gaps.

These measurements do not establish an overall task-accuracy improvement over
the BF16 baseline.

## Serving

Validated hardware was NVIDIA GB300, using native Blackwell kernels, vLLM 0.30.0,
four-way tensor parallelism, and expert parallelism. This recipe additionally
needs the compatibility helpers in [`serving/`](serving/README.md): narrow
MXFP8 output padding/cropping and the canonical sparse-indexer policy used for
training/export validation. The helpers check the ModelOpt runtime source hash
and refuse an unvalidated build.

Follow [`serving/README.md`](serving/README.md) for the tested settings and
compatibility requirements. Generic `transformers.from_pretrained()` loading
and arbitrary vLLM versions have not been validated for this mixed format.
The model configuration preserves the base model's context settings; QAD export
parity was checked through 16K, and task-serving configurations used up to 262,144.
This release does not establish accuracy at the base model's maximum context.

MTP is preserved for completeness. MTP-enabled accuracy, acceptance rates, and
performance were not validated for this QAD checkpoint. With standard exact
speculative rejection sampling, draft quantization primarily changes acceptance
and speed; approximate verification can change the output distribution.

## Architecture and intended use

The upstream model combines a vision encoder, Gated DeltaNet, Qwen Sparse
Attention, MoE layers, gated residual streams, and n-gram embeddings. Inputs may
contain text, images, or video, and outputs are text. Refer to the
[base model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for its
architecture, training information, intended uses, and limitations. This release
is intended for quantization and QAD research and deployment validation.

## License, attribution, and limitations

Use is governed by the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
and the upstream [Qwen Community License 1.0](LICENSE). Qwen is the upstream
model developer. This checkpoint does not replace or waive upstream terms.
The serving helper code carries the separate license in `serving/LICENSE`.

The model can generate incorrect, biased, toxic, or otherwise unsuitable content.
Evaluate it on the intended application, including multimodal inputs and long
contexts, before deployment. Quantization and short QAD training do not remove
the base model's limitations.

References: [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer),
[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel),
[released NVFP4 model](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
