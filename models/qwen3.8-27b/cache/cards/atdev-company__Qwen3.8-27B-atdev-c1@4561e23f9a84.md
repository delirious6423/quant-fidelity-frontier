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

# Qwen3.8-27B atdev c1 (W3 overlay + DFlash2 d16 drafter)

## Up to 14.2× faster Qwen3.8-27B inference on a single A100

> **Model files for [atdev-serve](https://github.com/atdev-company/atdev-serve), c=1 stack** (c=1 = 1 client: a single user, one request at a time): **408.9 tok/s** decoding vs 28.9 for the original bf16 model on stock vLLM, and **2.08×** the best vLLM settings (196.5 tok/s), while passing our accuracy gates (GSM8K within 1 point of the stock vLLM reference, long-context 36/36).
>
> **Code & install:** [github.com/atdev-company/atdev-serve](https://github.com/atdev-company/atdev-serve)
>
> **More models coming:** we are optimizing other widely used open models the same way. Follow [atdev-company](https://huggingface.co/atdev-company) for new releases.

![Single-user (c=1) decode speed: bf16 vLLM default 28.9, INT4 vLLM default 74.9, best vLLM 196.5, ATDEV engine only 284.4, ATDEV full stack 408.9 tok/s](assets/ladder_c1.png)

The model-side files of the atdev-serve **c1** stack for
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) served as
[`RedHatAI/Qwen3.8-27B-INT4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) on one A100 80GB.
**c1** means the stack is tuned for c=1: one client, i.e. single-user latency (one request at a time).
(The atdev-serve c32 stack, tuned for 32 concurrent clients, needs no extra model files.)

This repo is **not a standalone model**. It holds two parts that atdev-serve loads next to the INT4
target:

* **`w3/`: 3-bit weight overlay.** Replaces 400 linear layers of the INT4 checkpoint at load time
  (fewer weight bytes per decode step).
* **`drafter/`: DFlash2 d16 drafter (W4A16).** A re-trained and quantized version of the DFlash 2
  block-diffusion draft model [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2),
  tuned for atdev-serve's 16-node tree verification (more accepted tokens per step).

## Files

| file | size | contents |
|---|---:|---|
| `w3/w3_gptqact_d30_mlpo.safetensors` | 9.86 GB | 256 layers: `mlp.{gate,up,down}_proj` of all 64 decoder layers, `linear_attn.out_proj` (48 GDN layers), `self_attn.o_proj` (16 attention layers) |
| `w3/w3_gptqact_d30_qkv.safetensors` | 2.68 GB | 144 layers: `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z` (48 GDN layers), `self_attn.{q,k,v}_proj` (16 attention layers) |
| `drafter/model.safetensors` | 1.35 GB | the d16 drafter, W4A16 + int8 `fc` |
| `drafter/config.json` | 3 KB | drafter config (`DFlash2DraftModel`, compressed-tensors) |
| `SHA256SUMS.txt` | | sha256 and byte sizes of the files above |

Total download for c1 on top of the INT4 target: about 13.9 GB.

## Results

Measured on the complete atdev-serve c1 full stack as released (both parts of this repo + engine
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

For reference, the atdev-serve c32 stack (32 concurrent clients, no extra model files) reaches **2,730.9 tok/s**, 4.1× the original bf16 model on stock vLLM:

![32 concurrent clients (c=32) aggregate throughput ladder](assets/ladder_c32.png)

## Usage

The files are loaded by [atdev-serve](https://github.com/atdev-company/atdev-serve). Stock vLLM cannot
use them: the 3-bit layers need atdev-serve's W3 Marlin kernels, stock vLLM does not run the depth-15
tree verification, and its loader expects the original drafter id in `--speculative-config`.

```bash
git clone https://github.com/atdev-company/atdev-serve && cd atdev-serve
./install.sh && source atdev-env.sh       # or: docker compose --profile c1 up --build
./serve_c1.sh                             # downloads this repo on first start
```

`serve_c1.sh` sets:

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
