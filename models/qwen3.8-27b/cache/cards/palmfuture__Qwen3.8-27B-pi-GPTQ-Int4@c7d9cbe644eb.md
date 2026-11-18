---
license: apache-2.0
language:
- en
- th
- zh
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- qwen3.8-27b
- gptq
- int4
- 4-bit precision
- lora-merged
- agents
- pi
- mtp
- speculative-decoding
- conversational
- vision
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
---

# Qwen3.8-27B-pi-GPTQ-Int4

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the
[harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) LoRA merged in,
then quantized to GPTQ Int4 (`group_size=32`). It is the coding-agent sibling of
[palmfuture/Qwen3.8-27B-GPTQ-Int4](https://huggingface.co/palmfuture/Qwen3.8-27B-GPTQ-Int4): same quantization
settings, same file layout, same MTP (multi-token prediction) weights for speculative decoding.

Use this one if you want the pi adapter's behaviour at Int4 speed with speculative decoding on. A LoRA loaded
at runtime in vLLM could not be combined with speculative decoding on our server and ran at a third of the
speed, which is why the adapter is merged before quantizing here.

**What has and has not been measured**

- The adapter's authors report large gains under the [pi](https://github.com/earendil-works/pi) coding agent
  (Terminal-Bench 2.1 39.5% → 55.8%, SWE-bench Pro hard-51 29.4% → 62.7%). Those are their numbers for the
  unquantized adapter. **They have not been reproduced on this quantized merge.** Our own check was six easy
  agent tasks on which this model and the plain one both scored 18/18, which shows nothing either way.
- What we did measure is below: quantization loss, perplexity-style scores in Thai and English, and speed.

## Quality

| Metric | This model | [Qwen3.8-27B-GPTQ-Int4](https://huggingface.co/palmfuture/Qwen3.8-27B-GPTQ-Int4) |
|---|---|---|
| GPTQ modules quantized / RTN fallback | 400 / 0 | 400 / 0 |
| GPTQ loss, mean | 1.52e-04 | 1.59e-04 |
| GPTQ loss, max | 8.86e-04 | 9.22e-04 |
| Thai text, mean token NLL (lower is better) | 1.535 | 1.575 |
| English text, mean token NLL | 1.750 | 1.745 |
| Short factual/arithmetic QA (6 items) | 4/6 | 5/6 |

- NLL was measured on our own small Thai and English text sets, so compare the two columns with each other,
  not with other models' numbers.
- The Thai difference most likely comes from the calibration set (below), not from the adapter: loading the
  adapter at runtime on the other checkpoint did not move Thai NLL.
- The QA difference is one arithmetic item out of six. Too few items to conclude anything; check arithmetic on
  your own data if it matters to you.

## How it was made

1. **Merge.** `W += (alpha / r) · B·A` with `r=32`, `alpha=64`, computed in float32 and stored as bf16, for the
   adapter's 256 target weights: `gate_proj` / `up_proj` / `down_proj` in all 64 layers and
   `q_proj` / `k_proj` / `v_proj` / `o_proj` in the 16 full-attention layers. The adapter does not touch
   `mtp.*`, `lm_head`, the embeddings or the vision tower.
2. **Quantize** the merged model with GPTQModel 7.0.0: 4 bit, `group_size=32`, symmetric, `desc_act=False`,
   `damp_percent=0.05`, `true_sequential=True`.
3. **Add back** what GPTQModel's `save()` leaves out: `mtp.safetensors` (taken unchanged from the base model)
   and its 15 entries in `model.safetensors.index.json`.

**Quantized (int4, g32):**
- Self-attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Linear attention (Gated DeltaNet layers): `in_proj_qkv`, `in_proj_z`, `out_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

**Kept bf16:**
- Linear-attention state dynamics: `A_log`, `conv1d`, `dt_bias`, `in_proj_b`, `in_proj_a`, `in_proj_ba`,
  `linear_attn.norm`
- `q_norm`, `k_norm`, `input_layernorm`, `post_attention_layernorm`
- Multi-token prediction head `mtp.*` (15 tensors, in `mtp.safetensors`)
- Vision encoder `model.visual.*`
- Embeddings and `lm_head`

### Calibration

220 samples, cut to 1,024 tokens each, with native Thai text included (the intended use is Thai and English):

| Source | Samples | Purpose |
|---|---|---|
| [allenai/c4](https://huggingface.co/datasets/allenai/c4) | 50 | General English text |
| [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | 60 | Instruction following |
| [codeparrot/codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid) | 35 | Code |
| [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 15 | Mathematical reasoning |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.th` | 60 | Native Thai text |

### Hardware and toolchain

4× NVIDIA RTX 3060 12GB, Intel i9-7900X, 32 GB RAM. Merge about 10 minutes on CPU, quantization 58 minutes.

| Component | Version |
|---|---|
| GPTQModel | 7.0.0 |
| Transformers | 5.9.0 |
| PyTorch | 2.12.0+cu130 |
| TorchAO | 0.17.0 |
| Triton | 3.7.0 |
| Flash Linear Attention | 0.5.0 |
| Python | 3.13.13 (free-threading) |

On this hardware GPTQModel's parallel calibration forward crashed in five of six runs
("CUDA error: an illegal memory access"). `auto_forward_data_parallel=False` in `QuantizeConfig` avoided it
at no cost in time.

> If you re-quantize this model with GPTQModel, register `mtp.*` again in `model.safetensors.index.json`,
> pointing at `mtp.safetensors`. Without those entries speculative decoding is silently not used.

## Usage

We serve it with HyperQwen (below). The stock vLLM command here follows the sibling repository's card and has
not been run against this repository as uploaded:

```bash
vllm serve palmfuture/Qwen3.8-27B-pi-GPTQ-Int4 \
  --served-model-name Qwen3.8-27B-pi-GPTQ \
  --quantization gptq \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Adjust `--tensor-parallel-size` and `--max-model-len` to your GPUs. The weights need about 20 GB of VRAM
before the KV cache.

Notes from the adapter's card that matter when serving: it was trained with thinking off, and under pi's
`openai-completions` API rather than the Responses API. See
[the adapter's card](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted#use-with-pi) for the pi
configuration.

### Speed on the hardware above

Measured with HyperQwen (`ghcr.io/syv-ai/hyperqwen`, a patched vLLM 0.33), tensor parallel over the four cards, MTP with 3 speculative tokens, 120 W per card:

| | tok/s |
|---|---|
| One request, greedy | about 105 |
| 8 concurrent requests, total | about 430 |
| Prompt processing | about 1,300 |

These runs also used a reduced-vocabulary draft head built for that server, which is not part of this
repository. On the sibling checkpoint that head took single-request speed from 91.8 to 107.1 tok/s, so expect
less from the files here alone, and different numbers on other hardware or other vLLM versions.

## Licence and credits

Apache-2.0, as are both parents.

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team.
- Adapter: [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) by
  harness-hub. Its training data and method are described on its card.
- Quantization: [GPTQModel](https://github.com/modelcloud/gptqmodel).
