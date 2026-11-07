---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - RedHatAI/Qwen3.8-27B-INT4
  - z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
library_name: vllm
datasets:
  - HuggingFaceH4/ultrachat_200k
  - openai/gsm8k
tags:
  - qwen3.8
  - 3-bit
  - gptq
  - compressed-tensors
  - dflash2
  - speculative-decoding
  - draft-model
  - w4a16
  - atdev-serve
---

## Up to 14.2× faster Qwen3.8-27B inference on a single A100

# Qwen3.8-27B atdev c1 (W3 overlay + DFlash2 d16 drafter)

**c=1: 1 client (single user, one request at a time).** C1 refers to client concurrency, and both C1 profiles below are tuned for this single-client workload.

> **Easy setup on vLLM — let your AI coding agent handle it.**
> atdev-serve installs alongside unmodified **vLLM 0.31.0** and serves the familiar OpenAI-compatible API.
> **[Open the AI installation & setup guide](https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md)** — give this link to your AI coding agent and ask it to install, launch your chosen C1/C32 profile, and verify the server on your GPU machine.
> You can also follow the guide yourself. **Requires a supported NVIDIA GPU and CUDA environment**; the published profiles target an A100 80GB.



> **Model files for [atdev-serve](https://github.com/atdev-company/atdev-serve), C1 fast (W3) and C1 accuracy (W3 disabled)** (c=1 = 1 client: a single user, one request at a time): **C1 fast: 408.9 tok/s** decoding vs 28.9 for the original bf16 model on stock vLLM, and **2.08×** the best vLLM settings (196.5 tok/s), while passing our accuracy gates (GSM8K within 1 point of the stock vLLM reference, long-context 36/36).
>
> **Code & install:** [github.com/atdev-company/atdev-serve](https://github.com/atdev-company/atdev-serve)
>
> **More models coming:** we are optimizing other widely used open models the same way. Follow [atdev-company](https://huggingface.co/atdev-company) for new releases.

![Single-user (c=1) decode speed: bf16 vLLM default 28.9, INT4 vLLM default 74.9, best vLLM 196.5, ATDEV engine only 284.4, ATDEV full stack 408.9 tok/s](assets/ladder_c1.png)

The model-side files of the atdev-serve **c1** stack for
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) served as
[`RedHatAI/Qwen3.8-27B-INT4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) on one A100 80GB.
**c1** means the stack is tuned for c=1: one client, i.e. single-user latency (one request at a time).
The C32 stack now uses a separate [Hadamard W4A4 sidecar](https://huggingface.co/atdev-company/Qwen3.8-27B-atdev-c32) and built-in MTP; it does not use the drafter files in this repo.

This repo is **not a standalone model**. It holds two parts that atdev-serve loads next to the INT4
target:

* **`w3/`: 3-bit weight overlay.** Replaces 400 linear layers of the INT4 checkpoint at load time
  (fewer weight bytes per decode step).
* **`drafter/`: DFlash2 d16 drafter (W4A16).** A re-trained and quantized version of the DFlash 2
  block-diffusion draft model [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2),
  tuned for atdev-serve's 16-node tree verification (more accepted tokens per step).

## Two C1 profiles, distinct speed and accuracy

**c=1: 1 client (single user, one request at a time).** Both profiles report per-request decode speed for this workload.

The **up to 14.2x / 2.08x maximum-speed headline remains the original C1 fast W3 profile**.

| Profile | Command | Files downloaded from this repo | Mean c=1 decode speed |
|---|---|---|---:|
| C1 fast (default) | `./serve_c1.sh` | `w3/` + `drafter/` | published **408.9 tok/s**, 14.2x / 2.08x headline |
| C1 accuracy | `./serve_c1_accuracy.sh` | **`drafter/` only**; stock RedHatAI INT4 target | **357.17 tok/s**, mean of two runs |

C1 accuracy disables only W3 and retains the same existing d16 drafter and remaining C1 engine/verification settings. It requires **no new trained target weights**. Both existing W3 files and the existing drafter file are unchanged. The accuracy mode has better measured broad scores and PPL relative to C1 fast, with a speed cost; it is not a losslessness or bit-identical-to-INT4 claim.

| Metric | Stock INT4 reference | c1_fast | c1_accuracy |
|---|---:|---:|---:|
| MMLU-Pro | 71.71% | 69.14% | 71.43% |
| KMMLU Korean | 66.84% | 63.91% | 66.22% |
| IFEval prompt strict | 82.90% | 82.07% | 82.99% |
| IFEval prompt loose | 86.60% | 84.84% | 86.32% |
| IFEval instruction strict | 87.89% | 87.65% | 88.01% |
| IFEval instruction loose | 90.89% | 89.69% | 90.77% |
| HumanEval | 95.73% | 94.51% | 95.73% |
| HumanEval+ | 90.85% | 88.41% | 91.46% |
| MBPP | 91.27% | 89.68% | 92.06% |
| MBPP+ | 76.46% | 75.66% | 76.98% |
| WikiText-2 PPL (lower is better) | 7.298721 | 8.219982 | 7.300041 |

GSM8K: fast 290/300 and 1253/1319; accuracy 291/300 and 1250/1319. Both passed long-context 36/36. Stock broad reference values are the mean of two runs; stock GSM1319 is a separate c=32 run (1248/1319), while C1 modes use c=1. Broad score denominators: MMLU1400, KMMLU1125, IFEval541 prompts/834 instructions, HumanEval164, MBPP378; PPL40920 tokens. Saved EvalPlus code tests were executed. Passing the original GSM/long-context gates does not assert all broader accuracy gaps are closed.

[Full results and measurement limits](https://github.com/atdev-company/atdev-serve/blob/main/docs/ACCURACY.md) · [Machine-readable scores](accuracy_results.json)

## Files

| file | size | contents |
|---|---:|---|
| `w3/w3_gptqact_d30_mlpo.safetensors` | 9.86 GB | 256 layers: `mlp.{gate,up,down}_proj` of all 64 decoder layers, `linear_attn.out_proj` (48 GDN layers), `self_attn.o_proj` (16 attention layers) |
| `w3/w3_gptqact_d30_qkv.safetensors` | 2.68 GB | 144 layers: `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z` (48 GDN layers), `self_attn.{q,k,v}_proj` (16 attention layers) |
| `drafter/model.safetensors` | 1.35 GB | the d16 drafter, W4A16 + int8 `fc` |
| `drafter/config.json` | 3 KB | drafter config (`DFlash2DraftModel`, compressed-tensors) |
| `SHA256SUMS.txt` | | sha256 and byte sizes of the files above |

Total for C1 fast on top of the INT4 target: about 13.9 GB. C1 accuracy downloads only the existing roughly 1.35 GB drafter from this repo, not the W3 overlay.

## Results

**c=1: 1 client (single user, one request at a time).** The C1 speed figures below are per-request decode tok/s, not aggregate throughput across 32 clients.

The table below is the original **C1 fast W3** headline measurement, not the W3-disabled accuracy profile. Measured on the complete atdev-serve c1 fast full stack (both parts of this repo + engine
features), 1x A100 80GB, vLLM 0.31.0, greedy decoding, 96 fixed prompts, up to 1024 output tokens,
one request at a time (c=1).

| | best stock vLLM 0.31.0 (RedHatAI INT4, stock DFlash2, k=7) | atdev-serve c1 full stack |
|---|---:|---:|
| decode tok/s per request, mean | 196.5 | **408.9** (**2.08x**) |
| decode tok/s per request, median | 189.3 | 388.9 |
| accepted tokens per verify step (tau) | - | 4.69 (16-node tree, depth 15) |
| GSM8K, first 300 test problems | 0.973 | 0.967 |
| GSM8K, full test (1319) | 0.9462 | 0.9500 |
| long-context retrieval (passkey / multi-key, 2k-6.5k tokens) | 36/36 | 36/36 |

For reference, the original bf16 model with vLLM defaults decodes at 28.9 tok/s on the same GPU
(14.2x). In development, going from 4-bit to 3-bit weights on the MLP layers gave +6.5% c=1 decode
speed and extending it to the remaining 208 linear layers a further +4.3%; the re-trained block-16
drafter raised served tokens per step from 4.52 to 4.68 (+3.8%) (same-session A/B runs). Full tables
are in the [atdev-serve README](https://github.com/atdev-company/atdev-serve#results-at-a-glance).

The overlay changes the target model's weights, so outputs are not identical to the INT4 base model;
the accuracy gates above are the acceptance criterion. The drafter only proposes tokens and the target
verifies every one, so with greedy decoding it changes speed, not the content of the answer.

The released C32 Hadamard W4A4 stack (32 concurrent clients) uses its separate sidecar and measured **2,688.13 aggregate tok/s** in one run: about **4.0x** the historical bf16 baseline and **1.53x** the best-vLLM baseline. No external C32 drafter is needed. The following image is the **historical unrotated C32 ladder (2,730.9 tok/s), not the shipped C32 profile**:

![32 concurrent clients (c=32) aggregate throughput ladder](assets/ladder_c32.png)

## Context length

**Up to 262,144 total context tokens (input + output)** are supported by the C1 Fast W3 and C32 Hadamard W4A4 profiles. Each was tested on a **single A100 80GB** with **261,583 input tokens + 21 output tokens = 261,604 total tokens**, correctly retrieving all three codes placed at the beginning, middle and end of a synthetic long input.

Pass `--max-model-len 262144` to the selected launch script. The default remains 8,192 total context tokens; the published speed benchmarks use an output cap of 1,024 tokens. This record is a single long-request functional test for each profile.

## Default sampling

The GitHub launch scripts `serve_c1.sh`, `serve_c1_accuracy.sh` and `serve_c32.sh`
now set the server default to **`temperature: 0` (greedy decoding)** using
`--override-generation-config '{"temperature":0}'`. An API request that omits
`temperature` uses 0; an explicitly supplied request value takes precedence.
The published speed and accuracy measurements used `temperature: 0`; higher
temperatures have not been validated for those results. The original model
`generation_config.json` is unchanged. Update the serving repository checkout to
obtain this script change; an already running server needs to be restarted by its
operator before a changed launch default takes effect.

## Usage

**c=1: 1 client (single user, one request at a time).** Choose either C1 launch command below for the single-client workload.

The files are loaded by [atdev-serve](https://github.com/atdev-company/atdev-serve). Stock vLLM cannot
use them: the 3-bit layers need atdev-serve's W3 Marlin kernels, stock vLLM does not run the depth-15
tree verification, and its loader expects the original drafter id in `--speculative-config`.

```bash
git clone https://github.com/atdev-company/atdev-serve && cd atdev-serve
./install.sh && source atdev-env.sh       # or: docker compose --profile c1 up --build
./serve_c1.sh                             # C1 fast: W3 + drafter
# ./serve_c1_accuracy.sh                   # C1 accuracy: same drafter, no W3
```

`serve_c1.sh` (fast) sets the following. `serve_c1_accuracy.sh` retains the drafter setting but removes `w3_mixed` and `ATDEV_W3_SIDECAR` entirely:

* `ATDEV_W3_SIDECAR=atdev-company/Qwen3.8-27B-atdev-c1:w3/` (every `*.safetensors` in `w3/`, sorted by
  name). It also accepts `repo:path/file.safetensors` or comma-separated local file paths
  (`W3_OVERLAY=<...> ./serve_c1.sh`). The c1 configuration uses all 400 overlay layers.
* `ATDEV_DRAFTER_QUANT_PATH=atdev-company/Qwen3.8-27B-atdev-c1:drafter/` (a repo subfolder or a local
  directory), which replaces the `z-lab/Qwen3.8-27B-DFlash2` drafter named in `--speculative-config`
  with this checkpoint. Override it with `DRAFTER=<spec or dir> ./serve_c1.sh`.

Offline: `huggingface-cli download atdev-company/Qwen3.8-27B-atdev-c1 --local-dir /models/atdev-c1`,
then pass `W3_OVERLAY=/models/atdev-c1/w3/w3_gptqact_d30_mlpo.safetensors,/models/atdev-c1/w3/w3_gptqact_d30_qkv.safetensors DRAFTER=/models/atdev-c1/drafter`
(see the atdev-serve user guide).

## How the files are used

**W3 overlay (`w3/`).**

* 3-bit symmetric, group size 128, values `q` in [-4, 3], stored in the compressed-tensors int4 layout
  of the base checkpoint (nibbles `q + 8`, tensors `*.weight_packed` / scales), so the overlay drops
  into the RedHatAI INT4 module structure. atdev-serve repacks it at load time into a 3-bit planar
  layout for its W3 Marlin kernel.
* Quantized from the bf16 [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) weights with
  GPTQ (static activation order, dampening 0.3), in the AWQ-smoothed space of the RedHatAI INT4
  checkpoint (`awq_smoothed=True` in the safetensors metadata), so the INT4 checkpoint's other
  parameters stay consistent.
* Embeddings, `lm_head`, norms, the MTP head and every layer not listed above come from the INT4 base
  checkpoint unchanged. Keep the file names: atdev-serve loads the files sorted by name.

**Drafter (`drafter/`).**

* **d16**: block size 16 (`num_speculative_tokens: 15`); the engine turns the drafter's candidate
  scores into a 16-node token tree (depth up to 15) that the target verifies in one pass.
* **W4A16**: decoder-layer linears (attention q/k/v/o, MLP gate/up/down) stored as symmetric int4,
  group size 128 (compressed-tensors, `pack-quantized`); the `fc` projection as int8 g128 ("fc8");
  conv `kernel_projection`, candidate selector, norms and embeddings stay bf16. Activations bf16.
* Same architecture and config fields as the base drafter (`DFlash2DraftModel`, 5 layers, target
  layer taps `[5, 19, 33, 47, 61]`).

## Training and calibration data

Both parts use the same prompt pool and responses generated by the target itself:

* **Prompts**: only
  [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
  (`train_sft` split, MIT License) and [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)
  (`main`, `train` split, MIT License), decontaminated (exact and n-gram overlap) against the
  evaluation prompts, the GSM8K test split and MATH-500. Some prompts are wrapped in short system
  prompts / answer-language instructions written by the authors (agent-style system prompts, "answer
  in Korean"); no other dataset is used.
* **Responses**: generated by Qwen3.8-27B (served as RedHatAI/Qwen3.8-27B-INT4 under vLLM 0.31.0),
  greedy and sampled (T=0.6, top-p 0.95, top-k 20), with several thinking modes.
* **Drafter training**: self-distillation from the target, starting from the stock z-lab DFlash2
  drafter. Targets are the target model's hidden states at the drafter's tap layers and its next-token
  argmax (teacher labels).
* **W3 calibration**: 400 randomly drawn prompt + response pairs from this pool (ultrachat 200,
  ultrachat + Korean instruction 105, ultrachat + agent system prompt 46, GSM8K train 43, GSM8K train +
  Korean instruction 6), each truncated to 1024 tokens; 348,936 tokens in total. GPTQ uses 8192
  activation rows per linear layer drawn from these tokens.

C32 calibration uses additional FineWeb-2 Korean/FineWeb-Edu web text; that separate recipe and ODC-By source attributions are on the [C32 model card](https://huggingface.co/atdev-company/Qwen3.8-27B-atdev-c32) and [data/license provenance](https://github.com/atdev-company/atdev-serve/blob/main/docs/DATA_AND_LICENSES.md). It is not the C1 clean-pool recipe above. No experimental recovery-trained weights are included.

## About ATDEV

These files were built by [ATDEV](https://www.atdev.co.kr). We optimize LLM inference for any model
on your target GPUs, with custom kernels, speculative decoding, quantization and serving-stack
tuning. Results vary by model and hardware; this release is one example, running Qwen3.8-27B **up to
14.2x faster** on a single A100. ATDEV also delivers end-to-end enterprise AI: an experienced team of
forward-deployed engineers (FDEs), full-lifecycle AI agent development, in-house RAG, document parsing
and OCR solutions, and fine-tuning of LLMs, VLMs, OCR and other deep learning models. For business
inquiries, contact [atdev@atdev.co.kr](mailto:atdev@atdev.co.kr).

## License

Apache-2.0, same as the base models: `Qwen/Qwen3.8-27B`, `RedHatAI/Qwen3.8-27B-INT4` (W3 overlay) and
`z-lab/Qwen3.8-27B-DFlash2` (drafter).
