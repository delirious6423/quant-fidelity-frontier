---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- quantization
- trellis
- qtip
- ptq
- 2p5-bpw
---

# Qwen3.8-27B, trellis-quantized to 2.5 bits per weight

A weight-only post-training quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at
**2.4998 bits per weight** over the whole text model (embedding and lm_head included, every scale counted).
It is coded in a format meant to be decoded directly by a GEMM kernel at tensor-parallel degree 4.

## What is in this repository

- **The root folder** holds the dequantized weights in BF16: exactly the weights every result below was measured with.
  They load like the base model in transformers, SGLang or vLLM. They save no memory and give no speed-up; they are
  here for evaluation and reproduction.
- **`native/`** holds the quantized model in the serving format, 13.8 GB in all:
  - `model-native-NN.safetensors` (7.74 GB, one file per decoder layer): the packed trellis codes and their scales for
    the 496 coded linears, at 2 / 3 / 4 / 5 bits for 188 / 166 / 25 / 117 modules.
  - `model-other.safetensors` (6.01 GB): every other tensor in BF16, with the same values as in the root folder: the
    embedding, lm_head, norms, the vision tower and the DeltaNet small tensors (`A_log`, `dt_bias`, `conv1d`, `norm`).
    The PTQ also coded the embedding and lm_head; this export stores both as their BF16 values, and lm_head's band
    codes are not in this format version. The multi-token-prediction head is not included in `native/`.
  - `config.json` (the model's configuration plus `quantization_config`, one record per coded module),
    `model.safetensors.index.json` and the tokenizer files.
  - `FORMAT.md` describes every tensor, the bitstream, state, codebook and tile order, and the Hadamard sides.
  - `decode_reference.py` (PyTorch and safetensors, on the CPU) rebuilds any coded module's weight. From inside
    `native/`: `python decode_reference.py . MODULE --out f.safetensors` (`--list` names the modules).
  - `check.json` records 36 modules, every module kind at its first, middle and last layer, checked bit-exact against
    the exact decode of the quantization tables.
- `conversion-record.json` names every tensor's source and hash.

The vision tower (333 tensors) and the multi-token-prediction head (15 tensors) are the base model's,
unquantized, and not counted in the bit budget.

## Method

- **Trellis-coded quantization** (QTIP-style): every 16 x 16 tile is one tail-biting bitstream with a 16-bit state, at
  k bits per weight; the codebook value is computed from the state with integer operations (an INT8 grid).
- **Rotations a kernel can apply**: one random-sign Hadamard per tensor-parallel shard on each non-residual side,
  128-point blocks on the residual-stream side, so no transform crosses a GPU at TP4.
- **Error feedback**: block LDLQ against input Hessians, and two-sided against output-side Fisher Hessians where it
  helps; each output row is normalised before coding and its scale folded into a per-row output scale.
- **Mixed widths**: per-module widths of 2-6 bits chosen by measured sensitivity; lm_head is split into 16
  frequency bands with their own widths. Width histogram (modules): {"2": 203, "3": 167, "4": 25, "5": 117, "6": 1}.
- **Scale and norm tune**: per-row and per-column scales and the norm weights trained end to end on KL to the BF16 model,
  over self-generated coding-agent traces plus general text (math, code, science, chat, long context) at 30% weight.

## Results

Single attempts, so small differences are noise. BF16 is the base model under the same harness.

| | this model | BF16 |
| --- | --- | --- |
| DeepSWE (113 agentic coding tasks, mean feature-test score) | 83.08% (46 fully solved) | 82.97% (44) |
| DeepSWE, paired difference against BF16 (points) | +0.1 (SE 3.2) | |
| AIME 2025 + 2026 (60 problems x 8 samples) | 96.25% | 97.7% |
| RealWorldQA (765 questions x 3 samples) | 80.96% | 82.35% |
| MGSM (11 languages, 2,750 problems) | 92.69% | 93.96% |
| KL(BF16 \|\| model), held-out coding-agent traces | 0.01170 | 0 |
| KL(BF16 \|\| model), held-out general text (weighted) | 0.03583 | 0 |

Fully solved: every new and existing test passes; the earlier count (49) took tasks whose new tests all pass (corrected 2026-11-12).

All 113 DeepSWE tasks were graded natively. DeepSWE ran with a Claude Code agent scaffold against the served model (SGLang, TP4, speculative
decoding), 24 tasks at a time, a 3-hour limit per task; the coding-agent calibration repositories are disjoint from
the evaluation tasks.

## Serving

All published benchmark results were measured with the BF16 copy at the repository root. That copy was produced with
two BF16 roundings, so it is close to, but not bit-identical with, the exact decode of `native/`: it equals the exact
decode in about 75% of elements, almost all the rest differ by one BF16 step, and about 1 in 100,000 differ by more,
all of them values within 1e-5 of zero.

Kernels that run `native/` directly are not yet released. Current measurements (unscored, H100 GPUs):

- at TP4, end-to-end decode runs at 0.72-0.94x the BF16 checkpoint's speed, depending on load;
- on a single GPU (TP1), the linear layers alone, measured with synthetic 3-bit codes, run 1.06-1.31x faster than BF16
  at 1-64 rows. This is not an end-to-end measurement.

The native model is 13.8 GB, against the BF16 copy's 56 GB.

## License

Apache-2.0, as the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), from which these weights
are derived.
