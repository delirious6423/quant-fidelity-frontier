---
license: apache-2.0
pipeline_tag: text-generation
library_name: mlx
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - mlx
  - aquaduck
  - qwen
  - qwen3
  - conversational
  - layer-shards
---

# Model Card for aquaduck/Qwen3.8-27B-MLX
Pinned 4bit MLX of Qwen3.8-27B (`qwen/qwen3.8-27b`), plus midpoint layer shards for staged / multi-node loading (Aquaduck Arc mlx-package-v1).
The shard files are not a new quantization. They are contiguous midpoint packages cut from the full 4bit MLX in this repo.

## Model lineage
Qwen/Qwen3.8-27B
  └── quantized → Qwen/Qwen3.8-27B (4bit)
        └── full MLX + midpoint shards → aquaduck/Qwen3.8-27B-MLX (this repo)
- Base weights: https://huggingface.co/Qwen/Qwen3.8-27B (apache-2.0)
- Quantization source: https://huggingface.co/Qwen/Qwen3.8-27B (tag 4bit)
- This repo: full 4bit MLX (`model-00001-of-00003.safetensors`) and midpoint MLX shards

## Model Details

| | |
|---|---|
| Catalog id | `qwen/qwen3.8-27b` |
| Quantization | 4bit |
| Parameters | 27.8B |
| Native context | 262,144 tokens |
| License | apache-2.0 |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Ingest MLX | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |

### Model Description
- Hosted by: Aquaduck (hosting and layer packaging only; base model by Qwen Team / Alibaba Cloud; MLX quant by mlx-lm)
- Shared by: Aquaduck AI
- Model type: Causal language model (Qwen3.8-27B), MLX 4bit
- Language(s): Multilingual (same as base)
- License: apache-2.0 (inherits from Qwen/Qwen3.8-27B)
- Finetuned from model: N/A — not a fine-tune
- Derived from: Qwen/Qwen3.8-27B ← Qwen/Qwen3.8-27B

### Model Sources
- Base model card: https://huggingface.co/Qwen/Qwen3.8-27B
- Quantized MLX source: https://huggingface.co/Qwen/Qwen3.8-27B

## Files
| File | Role | Approx. size |
|------|------|--------------|
| `model-00001-of-00003.safetensors` | Full-model MLX (4bit) | ~15.13 GB |
| `layers-0-32/model.safetensors` | Split shard (layers 0–31) | ~7.57 GB |
| `layers-32-64/model.safetensors` | Split shard (layers 32–63) | ~8.28 GB |
- Total layers: 64
- Valid split boundaries: 32
Filenames use exclusive end indices (layers-{start}-{endExclusive}/model.safetensors).

## Uses

### Direct Use
- **Full `model-00001-of-00003.safetensors`:** standard single-file 4bit MLX (mlx-lm-compatible). Use this for single-node / local runs.
- **`layers-*-*/model.safetensors`:** Aquaduck / Arc staged loading only. These are not drop-in complete models for stock mlx-lm.
Use the base model’s chat template (including thinking / instruct modes as documented on the base model card); other formats will not work correctly.

### Out-of-Scope Use
- Expecting any one shard to run as a complete model
- Treating this repo as a new training run or re-quant
- Uses prohibited by the apache-2.0 license or the base model’s model card guidance

## Bias, Risks, and Limitations
Same capabilities, biases, and risks as Qwen/Qwen3.8-27B. 4bit quantization can degrade quality vs. the original higher-precision releases. Layer sharding does not change weights beyond packaging.

### Recommendations
Follow the base model’s docs for chat template, thinking vs instruct modes, and sampling. Prefer `model-00001-of-00003.safetensors` in this repo when you do not need staged loading.

## How to Get Started
These files are meant to be loaded automatically by the **Aquaduck desktop app**.
1. Download the Aquaduck desktop app and sign in.
2. Devices connected to the internet will receive a model assignment from the model catalog (`qwen/qwen3.8-27b`).
3. Download the model from the Home view. The app will:
   - download only the assigned file from this repo (full `model-00001-of-00003.safetensors` or one midpoint half)
   - keep that stage ready for serving
You do not need to pick files by hand, but you may for local serving. Assignment and download are driven by model catalog metadata.
The full `model-00001-of-00003.safetensors` is a standard 4bit MLX. The `layers-*-*/model.safetensors` files are not.

## Training Details
No training. Weights come from Qwen Team / Alibaba Cloud; 4bit MLX from Qwen/Qwen3.8-27B; this repo hosts that MLX and (when split) packages it into midpoint layer shards.

## Evaluation
No separate evals for the hosted MLX or shards. See Qwen/Qwen3.8-27B.

## Technical Specifications
- Architecture: Qwen3.8-27B (~27.8B params, GQA (24 Q / 4 KV heads), 64 layers, hidden dim 5120)
- Quantization: 4bit
- Packaging: pinned full 4bit MLX; optional Arc midpoint shards (`layers-{start}-{endExclusive}/model.safetensors`)
- Package format: mlx-package-v1
- Split: 2 stages at layer 32 (maxStages: 2)

## Citation
```bibtex
@misc{qwen3827b,
    title  = {Qwen3.8-27B},
    author = {Qwen Team / Alibaba Cloud},
    year   = {2026},
    url    = {https://huggingface.co/Qwen/Qwen3.8-27B}
}
```

Credit:
- The MLX quantization source (https://huggingface.co/Qwen/Qwen3.8-27B)
- mlx-lm (https://github.com/ml-explore/mlx-lm) for MLX support

## Attribution

Quantized MLX ingested from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Original weights: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Redistributed under the base model's license.

Hosted by [Aquaduck](https://huggingface.co/aquaduck).

## Model Card Contact
Aquaduck AI — https://huggingface.co/aquaduck
