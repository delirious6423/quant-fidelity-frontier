---
library_name: transformers
base_model:
- Qwen/Qwen3.8-27B
tags:
- gguf
- llama.cpp
- image-text-to-text
- vision
- multimodal
- text-generation-inference
- transformers
- unsloth
- conversational
- text-generation
- qwen3_8
- qwen3
- qwen
- 27b
- fine-tuned
- instruction-tuned
- reasoning
- agent
- agentic
- tool-use
- function-calling
- code-generation
- mtp
- speculative-decoding
- local-inference
license: apache-2.0
language:
- en
- zh
- es
- ru
- ja
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-GGUF

**Author:** [Feiteng](https://x.com/FeitengLi) this work is sponsored by **[Draw Things](https://drawthings.ai/)**.

**The primary goal of this repository is to provide Qwen3.8-27B model quantizations below 4 bits per weight (bpw) in GGUF format.**

The below-4-bpw target applies to language model weights. Companion vision encoder files are provided in the formats listed below, with their final quality metrics reported separately.

**We also publish compact quantized vision encoders with high output fidelity, resolving GGUF quantization compatibility issues for the Qwen3.8-27B vision encoder. The Q4_K, Q5_K, and Q6_K files support inference with unmodified llama.cpp.** Compared with our own **BF16 export (888.01 MiB)**, our **Q4_K** file is **314.46 MiB**, a **64.59% size reduction**, with **0.99537 cosine similarity** against the original Hugging Face FP32 vision outputs.


## DT Checkpoints

| Checkpoint | File | Size (GiB) | Size without MTP (GiB) |
| --- | --- | ---: | ---: |
| DT-bpw2.96-IQ3_XXS | [Qwen3.8-27B-DT-IQ3_XXS.gguf](Qwen3.8-27B-DT-IQ3_XXS.gguf) | 9.71 | 9.29 |
| DT-bpw2.48-IQ2_S | [Qwen3.8-27B-DT-IQ2_S.gguf](Qwen3.8-27B-DT-IQ2_S.gguf) | 8.20 | 7.78 |

Both DT checkpoints score 100.00% on AIME 2026 in the reported evaluation.

### bpw vs. Score

![Benchmark scores at 2.48 and 2.96 bpw for the two DT checkpoints, with BF16 reference scores.](assets/bpw-vs-score.png)

The plot uses the bpw values indicated by the DT checkpoint names. Dashed lines show BF16 reference scores. Each benchmark is shown separately, with a shared score scale.

### Benchmark Comparison

Scores are percentages; higher is better. AIME 2026, GPQA Diamond, and BFCL v4 (1k) report accuracy. IFEval reports prompt-level strict accuracy (`prompt_level_strict`).

| Checkpoint | Size (GiB) | Includes MTP | Size without MTP (GiB) | AIME 2026 | GPQA Diamond | IFEval | BFCL v4 (1k) |
|---|---:|:---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B-BF16 | 46.55 | No | 46.55 | 100.00% | 90.91% | 94.09% | 68.36% |
| UD-Q2_K_XL | 9.15 | Yes | 8.83 | 100.00% | 85.35% | 93.89% | 69.46% |
| UD-IQ2_S | 7.80 | No | 7.80 | 96.67% | 79.29% | 90.39% | 68.26% |
| AD-IQ2_XS | 9.21 | Yes | 8.94 | 96.67% | 76.77% | 90.20% | 68.46% |
| AD-IQ1_M | 7.91 | Yes | 7.64 | 90.00% | 51.01% | 78.37% | 62.08% |
| **DT-bpw2.96-IQ3_XXS** | 9.71 | Yes | **9.29** | **100.00%** | 86.36% | **93.90%** | **68.06%** |
| **DT-bpw2.48-IQ2_S** | 8.20 | Yes | **7.78** | **100.00%** | 77.27% | **93.16%** | **69.56%** |

Sizes are reported in GiB (1 GiB = 2^30 bytes). "Includes MTP" indicates whether the file contains multi-token prediction weights; it does not indicate whether MTP was enabled during evaluation. "Size without MTP" excludes those weights.

### MTP

DT-bpw2.96-IQ3_XXS retains higher precision in its MTP component. With the maximum draft length set to **3**, preliminary measurements show an overall draft acceptance rate of approximately **60%**.

### Vision Encoder

The files below include our own BF16 and FP16 exports alongside the four quantized formats. BF16 and FP16 files are prepared for upload; their download links will become available after upload.

| Format | File | Size (MiB) | MSE ↓ | Cosine Similarity ↑ | RealWorldQA (100 samples) ↑ | OCRBench (100 samples) ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | [mmproj-Qwen3.8-27B-DT-BF16.gguf](mmproj-Qwen3.8-27B-DT-BF16.gguf) | 888.01 | 0.000056675555 | 0.999955598992 | 73% | 86% |
| FP16 | [mmproj-Qwen3.8-27B-DT-F16.gguf](mmproj-Qwen3.8-27B-DT-F16.gguf) | 888.01 | 0.000128205964 | 0.999899941413 | 73% | 86% |
| Q8_0 | [mmproj-Qwen3.8-27B-DT-Q8_0.gguf](mmproj-Qwen3.8-27B-DT-Q8_0.gguf) | 481.42 | 0.000868770063 | 0.999319249587 | 71% | 86% |
| Q6_K | [mmproj-Qwen3.8-27B-DT-Q6_K.gguf](mmproj-Qwen3.8-27B-DT-Q6_K.gguf) | 441.82 | 0.000878257184 | 0.999312708268 | 73% | 86% |
| Q5_K | [mmproj-Qwen3.8-27B-DT-Q5_K.gguf](mmproj-Qwen3.8-27B-DT-Q5_K.gguf) | 335.94 | 0.005477580664 | 0.995707853142 | 74% | 86% |
| Q4_K | [mmproj-Qwen3.8-27B-DT-Q4_K.gguf](mmproj-Qwen3.8-27B-DT-Q4_K.gguf) | 314.46 | 0.005899684403 | 0.995368459336 | 71% | 85% |

Sizes are reported in MiB (1 MiB = 1,048,576 bytes). Lower MSE is better; higher cosine similarity and benchmark accuracy are better.

MSE and cosine similarity compare full vision outputs, including the merger, from llama.cpp CUDA against the original Hugging Face FP32 outputs. All six reported formats use the same evaluation conditions.

MSE is the total squared difference between the GGUF and reference vision encoder outputs (including the merger), divided by the total number of output elements across all evaluation images. Cosine similarity is computed over the flattened, concatenated outputs of those images. MSE and cosine similarity measure vision output fidelity.

RealWorldQA and OCRBench report end-to-end accuracy (higher is better), using the same `Qwen3.8-27B-DT-IQ3_XXS.gguf` language model with each vision encoder on llama.cpp CUDA. Each dataset has 100 fixed test examples; OCRBench includes 10 per category. These are quick-subset scores, not full benchmark results.

Thinking is disabled and responses are limited to 1,024 tokens. Responses reaching that limit remain included in the scores. RealWorldQA limit counts: BF16: 5; FP16: 2; Q8_0: 7; Q6_K: 2; Q5_K: 2; Q4_K: 5. No OCRBench responses reached the limit.

All 1,200 examples have scored responses. FP16 required one server restart during OCRBench; completed examples were retained and the remaining examples were evaluated with the same configuration. One answer changes a dataset score by one percentage point.

## Reproduce the Benchmarks

#### Reproduce the DT Results

Use `profiles/qwen3.8-thinking.json` to evaluate the DT checkpoints on AIME 2026, GPQA Diamond, IFEval, and BFCL v4 1K.

Clone the benchmark runner and install its evaluation dependencies in your Python 3.10+ environment:

```bash
git clone https://github.com/drawthingsai/benchmarks.git
cd benchmarks
python3 -m pip install 'evalscope[bfcl,ifeval]==1.11.0' 'soundfile==0.13.1' huggingface_hub
```

A compatible `llama-server` must be available on `PATH`; see the benchmark repository's [setup instructions](https://github.com/drawthingsai/benchmarks#install). Before running GPQA, accept the [dataset access terms](https://huggingface.co/datasets/Idavidrein/gpqa) and authenticate:

```bash
hf auth login
```

The commands below use the final DT language model filenames and can be run once the files are uploaded to this Hugging Face repository. The `mmproj` files are companion vision encoders and cannot be used as the language model in these commands.

```bash
model_dir="$PWD/models/drawthingsai/Qwen3.8-27B-GGUF"
gguf_296='Qwen3.8-27B-DT-IQ3_XXS.gguf'
gguf_248='Qwen3.8-27B-DT-IQ2_S.gguf'

hf download drawthingsai/Qwen3.8-27B-GGUF \
  "$gguf_296" "$gguf_248" \
  --local-dir "$model_dir"
```

Run both evaluations from the cloned `benchmarks` directory, in the same shell:

```bash
python3 benchmark.py run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf "$model_dir/$gguf_296" \
  --model-name DT-bpw2.96-IQ3_XXS \
  --run-id DT-bpw2.96-IQ3_XXS \
  --parallel 4

python3 benchmark.py run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf "$model_dir/$gguf_248" \
  --model-name DT-bpw2.48-IQ2_S \
  --run-id DT-bpw2.48-IQ2_S \
  --parallel 4

python3 benchmark.py compare \
  runs/DT-bpw2.96-IQ3_XXS \
  runs/DT-bpw2.48-IQ2_S \
  --output results/dt-comparison.md
```

These commands evaluate all four language benchmarks with MTP speculative decoding disabled. Add `--dry-run` to a `run` command to inspect its configuration before evaluation, or `--resume` to continue an interrupted run with the same settings.

BFCL v4 1K evaluates 1,002 examples without Web Search; this is a subset score, not the official BFCL v4 Overall score. The current comparison profile replaces the former `qwen3.8-thinking-bfcl-1k.json` profile.

#### Reproduce the Community Baselines

See [Reproduce the community baselines](https://github.com/drawthingsai/benchmarks#reproduce-the-community-baselines) for pinned community model downloads and evaluation commands. Use the same `profiles/qwen3.8-thinking.json` profile and llama.cpp build as for the DT runs above.

