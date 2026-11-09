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

## Qwen3.8-27B on one A100: c1m (multi-turn) and c1s (single-turn)

# Qwen3.8-27B atdev c1: model files for the c1m and c1s profiles (W3 overlay + DFlash2 d16 drafter): up to 14.2x single-turn, up to 9.0x multi-turn

> **Easy setup on vLLM: let your AI coding agent handle it.**
> atdev-serve installs alongside unmodified **vLLM 0.31.0** and serves the familiar OpenAI-compatible API.
> **[Open the AI installation & setup guide](https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md)**: give this link to your AI coding agent and ask it to install, launch your chosen profile, and verify the server on your GPU machine.
> You can also follow the guide yourself. **Requires a supported NVIDIA GPU and CUDA environment**; the published profiles target an A100 80GB.

**Naming.** `c` = concurrent clients, the number is how many the profile is tuned for; the letter is the workload: `m` = multi-turn (growing conversations, prefix cache), `s` = single-turn (independent requests).
This repo serves **c1m** (1 client, multi-turn) and **c1s** (1 client, single-turn); its sibling [atdev-company/Qwen3.8-27B-atdev-c32](https://huggingface.co/atdev-company/Qwen3.8-27B-atdev-c32) serves **c32m** and **c32s**.
Launch with `./serve.sh c1m` or `./serve.sh c1s` ([atdev-serve](https://github.com/atdev-company/atdev-serve)).

- **9.0x faster than the original bf16 model on stock vLLM, multi-turn (c1m):** the same agent trace (2 sessions x 38 turns, context to 111k tokens) takes **301.2 s** against 2,716.3 s at `max_tokens` 2048 (7.3x at 512, 5.1x at 128). Condition: bf16 is a single run of stock vLLM 0.31.0 (MTP k=3, prefix caching on) on the other A100 of the same machine; c1m is the median of 3 cold runs; full stack (this repo's W3 overlay and drafter).
- **3.77x faster agent conversations than the fastest stock setting (c1m):** a scripted agent session (2 sessions x 38 turns, context to 111k tokens) takes **301.2 s** against 1,136.3 s for the fastest stock vLLM 0.31.0 setting. Condition: `max_tokens` 2048 per turn, cold start, median of 3 runs, same A100, full stack. 2.32x at `max_tokens` 128, 3.04x at 512.
- **Long context:** c1m serves a 131,072-token window with prefix caching (hit rate 0.95 on the 111k-token trace).
- **Less energy per token:** at `max_tokens` 2048 the c1m run used 133 kJ against 405 kJ for stock vLLM (2.22 vs 6.10 J per output token).
- **c1s, single-turn:** **408.9 tok/s** decoding vs 28.9 for the original bf16 model on stock vLLM (**14.2x**) and **2.08x** the best vLLM settings (196.5 tok/s; 96 prompts, up to 1,024 output tokens), while passing our accuracy gates.
- All speedups are **full stack** (this repo's W3 overlay and drafter plus the engine). The engine-only single-turn figure (same model files as vLLM) is 1.45x; no engine-only multi-turn number exists.

> **Code & install:** [github.com/atdev-company/atdev-serve](https://github.com/atdev-company/atdev-serve)
>
> **More models coming:** we are optimizing other widely used open models the same way. Follow [atdev-company](https://huggingface.co/atdev-company) for new releases.

## Install

Manual (Linux, one A100 80GB, CUDA 13.0 toolkit with `nvcc`, Hugging Face access):

```bash
git clone https://github.com/atdev-company/atdev-serve && cd atdev-serve
./install.sh && source atdev-env.sh
./serve.sh c1m                           # or c1s
```

Or paste a prompt into your AI coding agent on the GPU machine:

*c1m*

```text
Install atdev-serve on this machine and launch the c1m profile (one user, multi-turn). Follow https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md and use STACK=c1m.
Verify that the server answers a chat request, then tell me the endpoint URL and any problems you ran into.
```

*c1s*

```text
Install atdev-serve on this machine and launch the c1s profile (one user, single-turn). Follow https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md and use STACK=c1s.
Verify that the server answers a chat request, then tell me the endpoint URL and any problems you ran into.
```

Everything else is in [AI_SETUP.md](https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md).

![c1m: whole-trace wall time at max_tokens 2048: bf16 stock vLLM 2716.3 s, fastest stock INT4 1136.3 s, atdev-serve c1m 301.2 s](assets/speed_c1m.png)

![c1s: decode tok/s per request: bf16 default 28.9, INT4 default 74.9, best vLLM 196.5, atdev engine only 284.4, atdev c1s full stack 408.9](assets/speed_c1s.png)

The model-side files of the atdev-serve **c1** profiles for
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) served as
[`RedHatAI/Qwen3.8-27B-INT4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) on one A100 80GB.
The c32 profiles use a separate [Hadamard W4A4 sidecar](https://huggingface.co/atdev-company/Qwen3.8-27B-atdev-c32) and built-in MTP; they do not use the drafter files in this repo.

This repo is **not a standalone model**. It holds two parts that atdev-serve loads next to the INT4
target:

* **`w3/`: 3-bit weight overlay.** Replaces 400 linear layers of the INT4 checkpoint at load time
  (fewer weight bytes per decode step).
* **`drafter/`: DFlash2 d16 drafter (W4A16).** A re-trained and quantized version of the DFlash 2
  block-diffusion draft model [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2),
  tuned for atdev-serve's 16-node tree verification (more accepted tokens per step).

## Profiles served from this repo

| Profile | Launch | Files downloaded from this repo | Speed |
|---|---|---|---|
| **c1m**: 1 client, multi-turn (coding agent, long chat) | `./serve.sh c1m` | `w3/` + `drafter/` | 394.4 tok/s single-request decode; 2.32x / 3.04x / 3.77x vs the fastest stock vLLM on the 111k-token agent trace at `max_tokens` 128 / 512 / 2048 |
| **c1s**: 1 client, single-turn (one-shot, short unrelated prompts) | `./serve.sh c1s` | `w3/` + `drafter/` | published **408.9 tok/s**, 14.2x / 2.08x headline |
| c1s accuracy | `./serve_c1_accuracy.sh` | **`drafter/` only**; stock RedHatAI INT4 target | **357.17 tok/s**, mean of two runs |

c1m is 3.6-3.7% slower than c1s on independent short prompts (394.4 vs 409.5 mean decode tok/s over 96 prompts) and much faster whenever a conversation prefix repeats; its KV cache is smaller (about 284,600 tokens at 131,072 context against about 542,000 for c1s).
c1s accuracy disables only W3 and keeps the same d16 drafter and the remaining engine settings; it requires **no new trained target weights**. Both W3 files and the drafter file are unchanged by this release.

### Accuracy of c1s (published release tree)

| Metric | Stock INT4 reference (mean of 2) | c1s fast | c1s accuracy |
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

GSM8K: fast 290/300 and 1253/1319; accuracy 291/300 and 1250/1319. Both passed long-context 36/36. Stock broad reference values are the mean of two runs; stock GSM1319 is a separate c=32 run (1248/1319), while c1 modes use c=1. Broad score denominators: MMLU1400, KMMLU1125, IFEval541 prompts/834 instructions, HumanEval164, MBPP378; PPL40920 tokens. Saved EvalPlus code tests were executed. Passing the original GSM/long-context gates does not assert all broader accuracy gaps are closed. **c1s fast keeps a gap of about 2.6 points on MMLU-Pro and 2.9 on KMMLU against stock INT4.**

[Full results and measurement limits](https://github.com/atdev-company/atdev-serve/blob/main/docs/ACCURACY.md) · [Machine-readable scores](accuracy_results.json)

## c1m: multi-turn and agent use

The model files are unchanged. With atdev-serve, `./serve.sh c1m` caches the conversation state (attention KV and the recurrent GDN state), prefills only what is new in each turn, uses faster prefill kernels, serves a 131,072-token window and runs a start-up warmup that compiles the Triton kernels before the first request.

Scripted agent conversation (2 sessions x 38 turns, context 5k to 111k tokens): wall-clock seconds for the whole trace, cold start, median of 3 runs, same A100 for both arms; stock = fastest stock vLLM 0.31.0 setting found (RedHatAI INT4, prefix caching on, no speculative decoding).

| `max_tokens` per turn | stock vLLM | c1m | speedup | c1m tok/s | stock tok/s | original bf16 + vLLM (c1m is faster by) |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 268.6 s | **116.0 s** | **2.32x** | 82.2 | 34.7 | 593.4 s (**5.12x**) |
| 512 | 623.1 s | **205.2 s** | **3.04x** | 159.4 | 52.0 | 1,497.5 s (**7.30x**) |
| 2048 | 1,136.3 s | **301.2 s** | **3.77x** | 198.6 | 58.4 | 2,716.3 s (**9.02x**) |

The bf16 column is the original Qwen3.8-27B bf16 model on stock vLLM 0.31.0 (MTP k=3, prefix caching on): **a single run**, on the other A100 of the same machine as the c1m runs (all other arms ran on one GPU). Its KV cache holds 248,680 tokens at 131,072 context; its prefix-cache hit rate was 0.95, the same as the other arms. The multiples against bf16 are the largest multi-turn numbers (9.02x at `max_tokens` 2048); against the fastest stock setting they are 2.32x / 3.04x / 3.77x.

Why the gap grows with output length: summed time-to-first-token (prefill) is 84.3-84.7 s for c1m against 130.2-131.0 s for stock, a flat 1.55x, while the generation phase is 4.1x-4.6x faster (31.7 / 120.7 / 216.5 s against 138.2 / 492.9 / 1,005.3 s). Prefill is 73% of the c1m run at `max_tokens` 128 and 28% at 2048, so the overall ratio rises as outputs get longer.
(The bf16 column is one run on the other A100 of the same machine.)

Energy for the same runs (NVML, 1 Hz): 48.2 / 89.4 / 133.0 kJ against 98.1 / 223.5 / 405.0 kJ for stock (0.49x / 0.40x / 0.33x); average power 415-442 W against 356-365 W; 5.05 / 2.73 / 2.22 J per output token against 10.5 / 6.9 / 6.1.

**Accuracy of c1m** (research runtime, the frozen bundle the profile was built from; stock = one stock INT4 run of the same harness): GSM8K-1319 94.24% (stock 94.31%), GSM8K-300 96.67%, long-context 36/36, MMLU-Pro 69.0% (-2.5), KMMLU 63.8% (-3.3), IFEval prompt strict 81.3% (-2.1), HumanEval+ 90.2%, MBPP+ 75.4%, WikiText-2 PPL 8.28 (stock 7.30). The 3-bit overlay's gap is the same as c1s fast; caching adds none.
**Provenance:** speed and the broad scores were measured on our research runtime; the release tree reproduced the speed (115.0 / 204.4 s at `max_tokens` 128 / 512), the generated text of all 76 turns (identical) and GSM8K-300 (0.963).

Caveats: c1m is 3.6-3.7% slower than c1s on short unrelated prompts; the numbers are full-stack (W3 overlay and this drafter), not engine-only; the fast path uses a bf16 KV cache and holds about 284,600 KV tokens at 131,072 context; prefill uses FlashAttention-2 / FlashInfer and int8 activations, so outputs differ at rounding level from c1s; only greedy decoding is qualified; one fresh 64k-100k prompt is not faster than stock vLLM (the gain comes from re-used history and decode speed).
Details: [docs/MULTITURN.md](https://github.com/atdev-company/atdev-serve/blob/main/docs/MULTITURN.md) and [docs/TECHNIQUES.md](https://github.com/atdev-company/atdev-serve/blob/main/docs/TECHNIQUES.md).

## Files

| file | size | contents |
|---|---:|---|
| `w3/w3_gptqact_d30_mlpo.safetensors` | 9.86 GB | 256 layers: `mlp.{gate,up,down}_proj` of all 64 decoder layers, `linear_attn.out_proj` (48 GDN layers), `self_attn.o_proj` (16 attention layers) |
| `w3/w3_gptqact_d30_qkv.safetensors` | 2.68 GB | 144 layers: `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z` (48 GDN layers), `self_attn.{q,k,v}_proj` (16 attention layers) |
| `drafter/model.safetensors` | 1.35 GB | the d16 drafter, W4A16 + int8 `fc` |
| `drafter/config.json` | 3 KB | drafter config (`DFlash2DraftModel`, compressed-tensors) |
| `SHA256SUMS.txt` | | sha256 and byte sizes of the files above |

Total for c1m / c1s on top of the INT4 target: about 13.9 GB. c1s accuracy downloads only the existing roughly 1.35 GB drafter from this repo, not the W3 overlay.

## Results

**c=1: 1 client (single user, one request at a time).** The C1 speed figures below are per-request decode tok/s, not aggregate throughput across 32 clients.

The table below is the original **c1s** (single-turn, W3) headline measurement, not the W3-disabled accuracy flavor and not c1m. Measured on the complete atdev-serve c1 fast full stack (both parts of this repo + engine
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

**Up to 262,144 total context tokens (input + output)** are supported by c1s and c32s. Each was tested on a **single A100 80GB** with **261,583 input tokens + 21 output tokens = 261,604 total tokens**, correctly retrieving all three codes placed at the beginning, middle and end of a synthetic long input.

Pass `--max-model-len 262144` to the launcher (`./serve.sh c1s --max-model-len 262144`; c1m defaults to 131,072 and was not tested beyond). The default of c1s remains 8,192 total context tokens; the published speed benchmarks use an output cap of 1,024 tokens. This record is a single long-request functional test for each profile.

## Usage

**c=1: 1 client (single user, one request at a time).** Choose either C1 launch command below for the single-client workload.

The files are loaded by [atdev-serve](https://github.com/atdev-company/atdev-serve). Stock vLLM cannot
use them: the 3-bit layers need atdev-serve's W3 Marlin kernels, stock vLLM does not run the depth-15
tree verification, and its loader expects the original drafter id in `--speculative-config`.

```bash
git clone https://github.com/atdev-company/atdev-serve && cd atdev-serve
./install.sh && source atdev-env.sh       # or: docker compose --profile c1 up --build
./serve.sh c1m                            # multi-turn: W3 + drafter, prefix caching
# ./serve.sh c1s                           # single-turn: W3 + drafter, no prefix caching
# ./serve_c1_accuracy.sh                   # c1s accuracy: same drafter, no W3
```

`serve_c1.sh` (c1m / c1s fast) sets the following. `serve_c1_accuracy.sh` retains the drafter setting but removes `w3_mixed` and `ATDEV_W3_SIDECAR` entirely:

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
