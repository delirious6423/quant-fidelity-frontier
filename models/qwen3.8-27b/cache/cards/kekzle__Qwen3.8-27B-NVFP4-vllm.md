---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
- en
tags:
  - nvfp4
  - fp4
  - quantized
  - compressed-tensors
  - vllm
  - imp
---

# Qwen3.8-27B-NVFP4 (compressed-tensors, for vLLM and imp)

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized to NVFP4 in
the compressed-tensors `nvfp4-pack-quantized` layout, so it loads in **vLLM** and
in [imp](https://github.com/kekzl/imp). 51.75 GiB of BF16 becomes **19.2 GiB**,
which fits one 32 GB card with room for a real context.

The published NVFP4 exports of this model do not run on consumer Blackwell
(`sm_120`): they are mixed-precision with FP8 attention, and that card has no FP8
GEMM. This one is weight-only NVFP4 (W4A16), so it does.

## Use it

```python
# vLLM (verified on 0.27.1)
from vllm import LLM
llm = LLM(model="kekzle/Qwen3.8-27B-NVFP4-vllm", max_model_len=4096)
```

```bash
# imp
docker run --rm --gpus all -p 8080:8080 -v /path/to/models:/models \
  ghcr.io/kekzl/imp:latest \
  imp-server --host 0.0.0.0 --model /models/Qwen3.8-27B-NVFP4-vllm
```

Two vLLM flags this model needs on a 32 GB card:

- **`--max-num-seqs 64`** (or anything <= 169). This is a GDN hybrid, so vLLM
  takes one Mamba cache block per decode sequence; with these weights resident
  only 169 blocks fit, and the default 256 aborts startup with
  `max_num_seqs (256) exceeds available Mamba cache blocks (169) ... CUDA graph
  capture cannot proceed`. Do not "fix" that with `--enforce-eager`: it hides the
  error and costs most of the decode speed.
- **`VLLM_WSL2_ENABLE_PIN_MEMORY=1`** on WSL2, or it aborts with
  `RuntimeError: UVA is not available`. Keep `--gpu-memory-utilization` at ~0.92
  or below there, because WSLg holds ~1.2 GB.

vLLM also has no native FP4 kernel on `sm_120` and falls back to Marlin, so its
throughput here is not what it would be on a datacenter card: measured
single-stream on this checkpoint, 69.7 tok/s against imp's 81.6.

## What is quantized

| tensors | precision |
|---|---|
| linear weights (attention, MLP, GDN projections) | NVFP4, group size 16 |
| embeddings, `lm_head`, norms | source precision (BF16) |
| vision tower, MTP draft head | source precision (BF16) |

Everything left at source precision is listed in `ignore` inside
`quantization_config`, and `input_activations` is null: this quantizes weights
only, which vLLM reads as NVFP4A16.

Layers an engine merges into one linear (`q/k/v`, `gate/up`, and the GDN
`in_proj_*` pairs) **share one tensor scale**. That is not cosmetic: an engine
keeping a single scale for the merged weight would otherwise dequantize the other
members against the wrong one, and the amax spread inside those groups reaches
3.7x on this model.

## Give it room to think

This model reasons before it answers, and the reply shares the token budget with
that reasoning. In a long conversation the thinking grows, so a small
`max_tokens` can return an **empty** answer: the budget is gone before the reply
starts. Measured over one 74-turn session grown to ~8k tokens:

| `max_tokens` | result |
|---|---|
| 260 | several turns with an empty answer, one reply that was a single non-Latin token |
| 600 | 74 of 74 turns clean, every recall correct |

Not specific to one engine: the same conversation degenerates under vLLM too,
where the thinking is streamed into `content` instead. Set `max_tokens` to at
least ~400 for multi-turn use.

## Measurements

RTX 5090, batch 1, greedy, imp 0.27.

| | |
|---|---|
| Perplexity (`ppl_corpus_45k.txt`, 13.5k tokens, deterministic GEMM) | **4.6158** |
| Decode `tg128` | **87.4 tok/s** (median of 3 processes, 0.06 % spread) |
| Prefill `pp512` | 7 566 tok/s |
| Weights resident | 17.9 GiB, leaving ~7.7 GiB for the KV cache |
| Quality suite (`degen_suite.py`, 45 checks) | **45/45** |

Text and vision both work; the vision tower stays at source precision.

## Provenance and limitations

Converted from the official BF16 release with imp's in-tree quantizer:

```bash
imp-quantize --model ./Qwen3.8-27B --out ./Qwen3.8-27B-NVFP4-vllm --format vllm
```

Quantization is round-to-nearest, without activation calibration, and
`imp-quantize` is **experimental**. Use it for models that have no usable
published export, and prefer a calibrated vendor export where one exists and runs
on your card. Calibration is not available for this architecture anyway: AWQ
folds its scales into the RMSNorm, which is only valid for a plain multiplicative
norm, and this family uses the unit-offset convention.
