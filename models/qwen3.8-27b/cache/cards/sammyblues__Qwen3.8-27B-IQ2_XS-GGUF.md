---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
tags:
  - qwen3.8
  - gguf
  - llama.cpp
  - ollama
  - quantized
  - iq2-xs
  - 2-bit
  - text-generation
---

# Qwen3.8-27B IQ2_XS for llama.cpp and Ollama

This is a text-only, very-low-bit GGUF conversion of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), built to run a
27-billion-parameter model on a 12 GB consumer GPU or, more slowly, on CPU.

The recommended file is **`Qwen3.8-27B-64L-IQ2_XS.gguf`**. It keeps all 64
language-model decoder layers, uses llama.cpp's importance-matrix-aware
`IQ2_XS` quantization, occupies 8.47 GiB on disk, and used 9.07 GiB of GPU
memory in our 4K-context test. It passed deterministic inference through both
llama.cpp and Ollama.

## Plain-language summary

Quantization is similar to storing each model weight with a much smaller set of
numbers. This file compresses a roughly 50.11 GiB BF16 text model to 8.47 GiB,
an 83% reduction. The tradeoff is quality: on our test data, perplexity rose
from 6.71 to 8.03 and HellaSwag accuracy fell from 83% to 75%. Those measurements
show meaningful degradation, but ordinary text generation remained coherent.

We also tested deleting 16 and 24 complete decoder layers. Those versions were
smaller and faster, but generated gibberish or repetitive text. They are not
included in this repository. Their results are documented below because a
failed pruning experiment is useful evidence: for this model, one-shot layer
deletion without recovery training is much more destructive than 2-bit
quantization alone.

This artifact is suitable for experimentation and memory-constrained local
inference. Extreme 2-bit quantization is not the best choice when a higher-bit
model fits your hardware, and this small benchmark suite is not a substitute
for task-specific evaluation.

## What model surgery was performed?

The published 64-layer model received exactly two transformations:

1. **The optional one-layer MTP head was removed.** Qwen3.8 contains an
   additional multi-token-prediction (MTP) layer after the 64 text decoder
   layers. It is an optional speculative-prediction head, not one of the 64
   ordinary autoregressive decoder layers. llama.cpp's importance-matrix pass
   does not exercise it during normal next-token generation, so it had no
   calibration statistics for safe IQ2_XS quantization. We removed that head
   with llama.cpp's layer-pruning facility and changed the GGUF metadata field
   `qwen35.nextn_predict_layers` from `1` to `0`. The same MTP removal was
   applied to the BF16 benchmark reference for a like-for-like comparison.
2. **The remaining text model was quantized to IQ2_XS.** An importance matrix
   was generated from 128 chunks of WikiText-2 training text, each using a
   512-token context. llama.cpp then quantized the calibrated text-model
   tensors with the `IQ2_XS` preset. IQ2_XS is commonly described as a roughly
   2.31-bits-per-weight preset; the GGUF contains a mixture of tensor types, so
   “2-bit” should not be interpreted as every tensor using exactly two bits.

No text decoder blocks were removed from the file being published. There was
no fine-tuning, LoRA merge, distillation, vocabulary change, tokenizer change,
or chat-template replacement. The optional vision/projector components are not
present, so this GGUF is text-only.

## Artifact details

| Property | Value |
|---|---|
| Source model | `Qwen/Qwen3.8-27B` |
| Runtime architecture | Qwen3.8/Qwen3.5 hybrid text decoder in llama.cpp |
| Text decoder layers | 64 retained out of 64 |
| Parameters reported by llama.cpp | 26,895,998,464 |
| Quantization | `IQ2_XS`, importance-matrix calibrated |
| File size | 9,090,591,456 bytes / 8.466 GiB |
| SHA-256 | `eaeb705c72e5b7a55268f9f9a84b855ec3c1dd1b1f8c414cfc2d9ea52ad9eb23` |
| Context used for fit test | 4,096 tokens |
| Peak GPU allocation in fit test | 9,288 MiB / 9.070 GiB |
| llama.cpp revision | `0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0` |
| Conversion host | Ubuntu 24.04, CUDA 12.8, NVIDIA A100 80GB PCIe |
| Intended runtimes | llama.cpp, llama-cpp-python, Ollama |

Peak memory is configuration-dependent. Longer context, larger batch sizes,
different llama.cpp builds, and other processes using VRAM can push usage above
12 GB. The measured 9.07 GiB leaves useful headroom at the tested 4K context,
but it is not a guarantee for every configuration.

## Benchmark results

All GPU measurements used one NVIDIA A100 80GB PCIe, full layer offload, flash
attention, and 28 host threads. Peak allocation came from a 4K-context
generation. CPU throughput used 28 threads on an AMD EPYC 9554. WikiText-2
perplexity used all 453 512-token chunks of the held-out test file. HellaSwag
used a fixed 100-task subset; its confidence intervals are wide, so treat it as
directional rather than definitive.

| Variant | File GiB | Peak GPU GiB | GPU prompt t/s | GPU gen t/s | CPU gen t/s | WikiText-2 PPL | HellaSwag-100 | Generation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 64L BF16 reference | 50.11 | 48.75 | 2384.97 | 26.77 | not run | 6.7061 | 83% | coherent reference |
| **64L IQ2_XS (published)** | **8.47** | **9.07** | **934.67** | **51.54** | **4.67** | **8.0306** | **75%** | **GPU and CPU pass** |
| 48L IQ2_XS experiment | 6.65 | 7.15 | 1245.97 | 67.85 | 5.92 | 144,585.2858 | 22% | failed: digit-like gibberish |
| 40L IQ2_XS experiment | 5.75 | 6.19 | 1493.95 | 80.41 | 8.08 | 8,201.4746 | 26% | failed: single-letter repetition |

For HellaSwag-100, the 95% confidence intervals were 74.45–89.11% for BF16,
65.70–82.45% for 64L IQ2_XS, 15.00–31.07% for 48L, and 18.40–35.37% for 40L.
The small subset cannot establish a precise eight-point BF16-to-IQ2_XS gap on
its own. The perplexity result covers much more text and independently shows
the cost of quantization.

The A100 generation speed of the quantized model is higher than BF16 because
less weight data must be read, while prompt processing is lower because the
very-low-bit kernels have different compute costs. These figures should not be
projected directly to an RTX 3060; they establish correctness, relative model
behavior on the test host, and the measured memory requirement.

## The pruning experiment and why it failed

Qwen3.8-27B uses 64 text decoder layers in repeating four-layer hybrid groups:
three Gated DeltaNet/linear-attention layers followed by one full-attention
layer. We measured each layer's block influence on 16 calibration prompts as
the mean residual-stream change `1 - cosine(block_input, block_output)`. Low
scores were considered candidates for deletion. To preserve the hybrid
attention cadence, deletion was restricted to complete four-layer groups.

- The 48-layer plan removed source layers 4–19 (groups 1–4).
- The 40-layer plan additionally removed source layers 28–31 (group 7) and
  40–43 (group 10).
- llama.cpp removed the tensors, renumbered surviving layers, and updated GGUF
  block-count metadata. Each resulting BF16 checkpoint received a separate
  WikiText-2 importance matrix before IQ2_XS quantization.

This was deliberately aggressive, one-shot depth pruning with no recovery
fine-tuning or distillation. Both pruned models crossed from “some loss” into
catastrophic failure. HellaSwag fell close to random four-choice performance,
perplexity increased by three to four orders of magnitude, and deterministic
smoke prompts produced malformed output. Their greater speed is therefore not
useful. A future pruning attempt should add teacher distillation or
continued-pretraining recovery and evaluate smaller reductions, such as one
group at a time, before committing to 48 or 40 layers.

The machine-readable scores and aggregate benchmark data are included under
`experiment/`. The failed GGUFs themselves are intentionally not published.

## llama.cpp usage

Download the GGUF and use a recent llama.cpp build:

```bash
./llama-cli \
  --model Qwen3.8-27B-64L-IQ2_XS.gguf \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --flash-attn on \
  --jinja \
  --reasoning off \
  --prompt "Explain why the sky is blue in two sentences."
```

For CPU-only inference, use `--n-gpu-layers 0` and set a suitable thread count,
for example `--threads 12`. The included `inference/llama_cli_smoke.sh` performs
a deterministic CLI check. `inference/llama_cpp_smoke.py` is a minimal
`llama-cpp-python` example:

```bash
python -m pip install llama-cpp-python
python inference/llama_cpp_smoke.py \
  Qwen3.8-27B-64L-IQ2_XS.gguf \
  --gpu-layers -1

# CPU only
python inference/llama_cpp_smoke.py \
  Qwen3.8-27B-64L-IQ2_XS.gguf \
  --gpu-layers 0
```

## Ollama usage

Download `Modelfile` and the GGUF into the same directory, then run:

```bash
ollama create qwen38-27b-iq2xs -f Modelfile
ollama run qwen38-27b-iq2xs
```

The model was imported and smoke-tested successfully with Ollama 0.33.2. The
provided Modelfile defaults to a 4K context and Qwen-recommended sampling values
used in this experiment.

## Reproducibility

The `experiment/` directory contains the pipeline, benchmark driver, aggregate
results, and block-influence scores. The original build used the ephemeral work
root `/ephemeral/qwen38-pruning`. The major stages were:

1. Download the official Hugging Face checkpoint.
2. Convert it to BF16 GGUF with llama.cpp.
3. Score all 64 layers and construct complete-group pruning plans.
4. Build 48L and 40L negative-test checkpoints.
5. Remove the optional MTP head from all runtime/reference variants.
6. Build a separate WikiText-2 importance matrix for each variant.
7. Quantize with `llama-quantize ... IQ2_XS`.
8. Run llama.cpp GPU/CPU smoke tests, performance tests, perplexity,
   HellaSwag-100, peak-memory measurement, and an Ollama import/smoke test.

## Limitations and responsible use

- This is an unofficial derivative and is not endorsed by the Qwen team.
- It is text-only. Vision inputs and the optional MTP speculative head are not
  supported by this artifact.
- IQ2_XS is an extreme compression format. Factuality, reasoning, coding,
  multilingual ability, and long-context behavior were not comprehensively
  evaluated and may degrade more than these aggregate tests suggest.
- The fit test used a 4K context, not the source model's maximum context.
- HellaSwag-100 is intentionally small. Do not use its point estimate alone to
  make high-stakes quality claims.
- Outputs can be incorrect, biased, or unsafe. Evaluate the model for your own
  application and keep human oversight where errors matter.

The source checkpoint is Apache-2.0 licensed. The upstream license is included
with this derivative; users remain responsible for complying with applicable
terms and laws.
