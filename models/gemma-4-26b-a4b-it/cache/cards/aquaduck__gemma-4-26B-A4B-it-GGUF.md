---
license: apache-2.0
pipeline_tag: text-generation
library_name: gguf
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
  - gguf
  - aquaduck
  - google
  - conversational
  - layer-shards
---

# Model Card for aquaduck/gemma-4-26B-A4B-it-GGUF
Pinned Q4_K_M GGUF of Gemma-4-26B-A4B (`google/gemma-4-26b-a4b-it`), plus midpoint layer shards for staged / multi-node loading (Aquaduck Arc layer-package-v1).
The shard files are not a new quantization. They are contiguous midpoint packages cut from the full Q4_K_M GGUF in this repo.

## Model lineage
google/gemma-4-26B-A4B-it
  └── Q4_K_M GGUF + midpoint shards → aquaduck/gemma-4-26B-A4B-it-GGUF (this repo)
- Base weights: https://huggingface.co/google/gemma-4-26B-A4B-it (apache-2.0)
- Quantization source: https://huggingface.co/aquaduck/gemma-4-26B-A4B-it-GGUF (tag Q4_K_M)
- This repo: full Q4_K_M GGUF (`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`) and midpoint GGUF shards

## Model Details

| | |
|---|---|
| Catalog id | `google/gemma-4-26b-a4b-it` |
| Quantization | Q4_K_M |
| Parameters | 25.8B |
| Native context | 262,144 tokens |
| License | apache-2.0 |
| Base model | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Ingest GGUF | [aquaduck/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/aquaduck/gemma-4-26B-A4B-it-GGUF) |

### Model Description
- Hosted by: Aquaduck (hosting and layer packaging only; base model by Google; GGUF quant by Unsloth / llama.cpp ecosystem)
- Shared by: Aquaduck AI
- Model type: Causal language model (Gemma-4-26B-A4B), GGUF Q4_K_M
- Language(s): Multilingual (same as base)
- License: apache-2.0 (inherits from google/gemma-4-26B-A4B-it)
- Finetuned from model: N/A — not a fine-tune
- Derived from: aquaduck/gemma-4-26B-A4B-it-GGUF ← google/gemma-4-26B-A4B-it

### Model Sources
- Base model card: https://huggingface.co/google/gemma-4-26B-A4B-it
- Quantized GGUF source: https://huggingface.co/aquaduck/gemma-4-26B-A4B-it-GGUF

## Files
| File | Role | Approx. size |
|------|------|--------------|
| `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | Full-model GGUF (Q4_K_M) | ~16.95 GB |
| `gemma-4-26B-A4B-it-UD-Q4_K_M-layers-0-15.gguf` | Split shard (layers 0–14) | ~8.83 GB |
| `gemma-4-26B-A4B-it-UD-Q4_K_M-layers-15-30.gguf` | Split shard (layers 15–29) | ~8.92 GB |
- Total layers: 30
- Valid split boundaries: 15
Filenames use exclusive end indices (layers-{start}-{endExclusive}).

## Uses

### Direct Use
- **Full `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`:** standard single-file Q4_K_M GGUF (llama.cpp-compatible). Use this for single-node / local runs.
- **`*-layers-*.gguf`:** Aquaduck / Arc staged loading only. These are not drop-in complete models for stock llama.cpp.
Use the base model’s chat template (including thinking / instruct modes as documented on the base model card); other formats will not work correctly.

### Out-of-Scope Use
- Expecting any one shard to run as a complete model
- Treating this repo as a new training run or re-quant
- Uses prohibited by the apache-2.0 license or the base model’s model card guidance

## Bias, Risks, and Limitations
Same capabilities, biases, and risks as google/gemma-4-26B-A4B-it. Q4_K_M quantization can degrade quality vs. the original higher-precision releases. Layer sharding does not change weights beyond packaging.

### Recommendations
Follow the base model’s docs for chat template, thinking vs instruct modes, and sampling. Prefer `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` in this repo when you do not need staged loading.

## How to Get Started
These files are meant to be loaded automatically by the **Aquaduck desktop app**.
1. Download the Aquaduck desktop app and sign in.
2. Devices connected to the internet will receive a model assignment from the model catalog (`google/gemma-4-26b-a4b-it`).
3. Download the model from the Home view. The app will:
   - download only the assigned file from this repo (full `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` or one midpoint half)
   - keep that stage ready for serving
You do not need to pick files by hand, but you may for local serving. Assignment and download are driven by model catalog metadata.
The full `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` is a standard Q4_K_M GGUF. The `*-layers-*.gguf` files are not.

## Training Details
No training. Weights come from Google; Q4_K_M GGUF from aquaduck/gemma-4-26B-A4B-it-GGUF; this repo hosts that GGUF and (when split) packages it into midpoint layer shards.

## Evaluation
No separate evals for the hosted GGUF or shards. See google/gemma-4-26B-A4B-it.

## Technical Specifications
- Architecture: Gemma-4-26B-A4B (~25.8B params, GQA (16 Q / 8 KV heads), 30 layers, hidden dim 2816)
- Quantization: Q4_K_M
- Packaging: pinned full Q4_K_M GGUF; optional Arc midpoint shards (`*-layers-{start}-{endExclusive}.gguf`)
- Package format: layer-package-v1
- Split: 2 stages at layer 15 (maxStages: 2)

## Citation
```bibtex
@misc{gemma426ba4bit,
    title  = {Gemma-4-26B-A4B},
    author = {Google},
    year   = {2026},
    url    = {https://huggingface.co/google/gemma-4-26B-A4B-it}
}
```

Credit:
- The GGUF quantization source (https://huggingface.co/aquaduck/gemma-4-26B-A4B-it-GGUF)
- llama.cpp (https://github.com/ggml-org/llama.cpp) for GGUF support

## Attribution

Quantized GGUF ingested from [aquaduck/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/aquaduck/gemma-4-26B-A4B-it-GGUF). Original weights: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). Redistributed under the base model's license.

Hosted by [Aquaduck](https://huggingface.co/aquaduck).

## Model Card Contact
Aquaduck AI — https://huggingface.co/aquaduck
