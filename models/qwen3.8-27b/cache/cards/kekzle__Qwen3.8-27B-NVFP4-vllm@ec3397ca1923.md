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
  - awq
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
# vLLM (verified on 0.27.1 and 0.31.0)
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
single-stream on this checkpoint, 71.3 tok/s (vLLM 0.31.0) against imp's 96.9
without speculation and 207.8 with its MTP draft head.

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

RTX 5090, batch 1, greedy, imp main of 2026-11-21 (after 0.69.1), one run per cell.

| | this revision (AWQ) | previous revision (round-to-nearest) |
|---|---:|---:|
| Perplexity, prefill route (`ppl_corpus_45k.txt`, 13 811 tokens, deterministic) | **4.5392** | 4.6018 |
| Perplexity, per-step decode route (rows 64 and up) | **4.4705** | 4.5044 |
| Decode `tg`, no speculation | 96.91 tok/s | 96.87 tok/s |
| Prefill `pp512` | 11 474 tok/s | 11 446 tok/s |
| Server with MTP draft head, 30 chat prompts x 300 tokens, end to end | 207.81 tok/s | 207.66 tok/s |
| MTP draft acceptance on those prompts | 45.1 % | 44.9 % |
| vLLM 0.31.0, single stream, 400 tokens | 71.29 tok/s | 70.58 tok/s |
| Quality suite (`degen_suite.py`, 52 checks), with MTP and without | **52/52** | 52/52 |

The round-to-nearest revision stays in this repository's history.

Text and vision both work; the vision tower stays at source precision.

## Provenance and limitations

Converted from the official BF16 release with imp's in-tree quantizer, with AWQ
activation calibration since 2026-11-21. BF16 (51.75 GiB) does not fit a 32 GB
card, so the activation statistics come from a round-to-nearest export of the
same model:

```bash
imp-quantize --model ./Qwen3.8-27B --out ./rtn
imp-cli --model ./rtn --perplexity calib.txt --calibrate cal.bin --set speculative.mtp_k=0
imp-quantize --model ./Qwen3.8-27B --out ./Qwen3.8-27B-NVFP4-vllm --calib cal.bin --format vllm
```

`calib.txt` is 150 000 bytes of English prose (37 235 samples, 496 calibration entries), disjoint from the perplexity corpus.
The exact settings are in `imp_recipe.json`, the per-tensor error in
`quant_report.json`. `imp-quantize` is **experimental**: prefer a calibrated
vendor export where one exists and runs on your card.
