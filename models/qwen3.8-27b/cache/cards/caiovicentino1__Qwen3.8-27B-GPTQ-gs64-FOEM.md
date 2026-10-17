---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gptq
- quantization
- foem
- moe
- hybrid
- vision
- text-generation
- transformers
language:
- en
pipeline_tag: text-generation
---

# Qwen3.8-27B-GPTQ-gs64-FOEM

4-bit **GPTQ** quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — a hybrid MoBA + full-attention multimodal 27B model. Built with **group_size 64 + FOEM** and a config that targets BF16-equivalent quality.

## Quantization config

| Parameter  | Value                                  |
|------------|----------------------------------------|
| method     | GPTQ (`gptqmodel` 6.0.3)               |
| bits       | 4                                      |
| group_size | 64                                     |
| desc_act   | True                                   |
| sym        | True                                   |
| FOEM       | alpha=0.25, beta=0.2, device=auto      |
| license    | apache-2.0                             |

### Skip-list (layers kept in fp16/bf16)
`linear_attn` (MoBA blocks), `visual` (vision encoder), `mtp`, `lm_head`, `model.language_model.embed_tokens`, and all `norm` layers.

> **Total size: 27.29 GB.** Larger than pure 4-bit (~14GB) because the layers above are preserved in fp16/bf16 to keep fidelity in the sensitive parts of the hybrid model.

## Usage

### With gptqmodel
```python
from gptqmodel import GPTQModel
from transformers import AutoTokenizer

model = GPTQModel.from_pretrained(
    "caiovicentino1/Qwen3.8-27B-GPTQ-gs64-FOEM",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("caiovicentino1/Qwen3.8-27B-GPTQ-gs64-FOEM")
```

## Generation details
- **Stack**: `transformers==5.15.0` (required for the `qwen3_5` / `Qwen3_5ForConditionalGeneration` architecture) + `gptqmodel==6.0.3`, on an NVIDIA RTX PRO 6000 (G4).
- **Calibration**: 128 samples from `allenai/c4` (`en`), batch_size 1.
- Quantization took ~16 min; each shard was uploaded via LFS.

## License

This quantization is distributed under the **Apache 2.0** license, matching the original model [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), which is also **Apache 2.0**. See [`LICENSE`](https://www.apache.org/licenses/LICENSE-2.0).

## Original model citation

If you use this model, please cite the original work:

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```

## Evaluation

Measured on a single **NVIDIA RTX PRO 6000 (Colab G4, 96GB)** with
`gptqmodel 6.0.3` + `transformers 5.15.0`. Numbers are for this **quantized
checkpoint only** (no BF16 baseline run here yet).

### Quality

| Benchmark | Setting | Result |
|---|---|---|
| Perplexity (C4) | 40 docs, 16,683 tokens | **9.90** |
| HellaSwag | 20 samples, MC acc | **0.35** |
| ARC-Challenge | 20 samples, MC acc | **0.45** |
| MMLU (avg, 5-shot) | 8 q/sup × 4 sups, MC acc | **0.47** |
|   abstract_algebra | | 0.25 |
|   global_facts | | 0.375 |
|   computer_security | | 0.75 |
|   high_school_mathematics | | 0.50 |
| GSM8K (8-shot) | 12 samples, exact-match | **0.75** |
| Generation sanity | 3 prompts | coherent ✔ |

> Sub-samples are small (20/12/8 per benchmark) for fast CI-style runs; treat
> as indicative, not publication-grade.

### Speed / throughput

Greedy decoding, `batch=1`, warm. `prefill_tok/s` = prompt tokens/s on the
full-prompt forward; `gen_tok/s` = decoded tokens/s; `peak` = max VRAM.

| prompt_len | max_new | prefill_tok/s | gen_tok/s | peak VRAM |
|---|---|---|---|---|
| 70   | 32  | 108   | **11.98** | 27.66 GB |
| 560  | 128 | 1363  | **12.18** | 27.80 GB |
| 2270 | 64  | 2031  | **11.02** | 28.77 GB |

> Small-prompt `prefill` is dominated by launch overhead; larger contexts show
> the true prefill throughput. Peak VRAM ~27–29GB vs ~54.7GB for BF16 FP16
> weights, plus the quantized saved size of 27.29GB.

## Status / roadmap
- ✅ Published and complete (7 shards + config + model card).
- ✅ Quality + speed smoke battery (see Evaluation).
- ⬜ Full BF16-vs-quantized comparison on identical splits, and HumanEval pass@1.
