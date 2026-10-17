---
license: apache-2.0
pipeline_tag: text-generation
library_name: gguf
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - gguf
  - aquaduck
  - qwen
  - qwen3
  - conversational
  - layer-shards
---

# Model Card for aquaduck/Qwen3.8-27B-GGUF
Pinned Q4_K_M GGUF of Qwen3.8-27B (`qwen/qwen3.8-27b`), plus midpoint layer shards for staged / multi-node loading (Aquaduck Arc layer-package-v1).
The shard files are not a new quantization. They are contiguous midpoint packages cut from the full Q4_K_M GGUF in this repo.

## Model lineage
Qwen/Qwen3.8-27B
  └── quantized → unsloth/Qwen3.8-27B-GGUF (Q4_K_M)
        └── full GGUF + midpoint shards → aquaduck/Qwen3.8-27B-GGUF (this repo)
- Base weights: https://huggingface.co/Qwen/Qwen3.8-27B (apache-2.0)
- Quantization source: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF (tag Q4_K_M)
- This repo: full Q4_K_M GGUF (`Qwen3.8-27B-UD-Q4_K_M.gguf`) and midpoint GGUF shards

## Model Details

| | |
|---|---|
| Catalog id | `qwen/qwen3.8-27b` |
| Quantization | Q4_K_M |
| Parameters | 27.8B |
| Native context | 262,144 tokens |
| License | apache-2.0 |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Ingest GGUF | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |

### Model Description
- Hosted by: Aquaduck (hosting and layer packaging only; base model by Qwen Team / Alibaba Cloud; GGUF quant by Unsloth / llama.cpp ecosystem)
- Shared by: Aquaduck AI
- Model type: Causal language model (Qwen3.8-27B), GGUF Q4_K_M
- Language(s): Multilingual (same as base)
- License: apache-2.0 (inherits from Qwen/Qwen3.8-27B)
- Finetuned from model: N/A — not a fine-tune
- Derived from: unsloth/Qwen3.8-27B-GGUF ← Qwen/Qwen3.8-27B

### Model Sources
- Base model card: https://huggingface.co/Qwen/Qwen3.8-27B
- Quantized GGUF source: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF

## Files
| File | Role | Approx. size |
|------|------|--------------|
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | Full-model GGUF (Q4_K_M) | ~16.46 GB |
| `Qwen3.8-27B-UD-Q4_K_M-layers-0-32.gguf` | Split shard (layers 0–31) | ~7.62 GB |
| `Qwen3.8-27B-UD-Q4_K_M-layers-32-64.gguf` | Split shard (layers 32–63) | ~9.22 GB |
- Total layers: 64
- Valid split boundaries: 32
Filenames use exclusive end indices (layers-{start}-{endExclusive}).

## Uses

### Direct Use
- **Full `Qwen3.8-27B-UD-Q4_K_M.gguf`:** standard single-file Q4_K_M GGUF (llama.cpp-compatible). Use this for single-node / local runs.
- **`*-layers-*.gguf`:** Aquaduck / Arc staged loading only. These are not drop-in complete models for stock llama.cpp.
Use the base model’s chat template (including thinking / instruct modes as documented on the base model card); other formats will not work correctly.

### Out-of-Scope Use
- Expecting any one shard to run as a complete model
- Treating this repo as a new training run or re-quant
- Uses prohibited by the apache-2.0 license or the base model’s model card guidance

## Bias, Risks, and Limitations
Same capabilities, biases, and risks as Qwen/Qwen3.8-27B. Q4_K_M quantization can degrade quality vs. the original higher-precision releases. Layer sharding does not change weights beyond packaging.

### Recommendations
Follow the base model’s docs for chat template, thinking vs instruct modes, and sampling. Prefer `Qwen3.8-27B-UD-Q4_K_M.gguf` in this repo when you do not need staged loading.

## How to Get Started
These files are meant to be loaded automatically by the **Aquaduck desktop app**.
1. Download the Aquaduck desktop app and sign in.
2. Devices connected to the internet will receive a model assignment from the model catalog (`qwen/qwen3.8-27b`).
3. Download the model from the Home view. The app will:
   - download only the assigned file from this repo (full `Qwen3.8-27B-UD-Q4_K_M.gguf` or one midpoint half)
   - keep that stage ready for serving
You do not need to pick files by hand, but you may for local serving. Assignment and download are driven by model catalog metadata.
The full `Qwen3.8-27B-UD-Q4_K_M.gguf` is a standard Q4_K_M GGUF. The `*-layers-*.gguf` files are not.

## Training Details
No training. Weights come from Qwen Team / Alibaba Cloud; Q4_K_M GGUF from unsloth/Qwen3.8-27B-GGUF; this repo hosts that GGUF and (when split) packages it into midpoint layer shards.

## Evaluation
No separate evals for the hosted GGUF or shards. See Qwen/Qwen3.8-27B.

## Technical Specifications
- Architecture: Qwen3.8-27B (~27.8B params, GQA (24 Q / 4 KV heads), 64 layers, hidden dim 5120)
- Quantization: Q4_K_M
- Packaging: pinned full Q4_K_M GGUF; optional Arc midpoint shards (`*-layers-{start}-{endExclusive}.gguf`)
- Package format: layer-package-v1
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
- The GGUF quantization source (https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- llama.cpp (https://github.com/ggml-org/llama.cpp) for GGUF support

## Attribution

Quantized GGUF ingested from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Original weights: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Redistributed under the base model's license.

Hosted by [Aquaduck](https://huggingface.co/aquaduck).

## Model Card Contact
Aquaduck AI — https://huggingface.co/aquaduck
