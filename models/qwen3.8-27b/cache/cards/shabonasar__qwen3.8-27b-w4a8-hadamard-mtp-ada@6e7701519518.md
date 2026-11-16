---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
datasets:
- Vikhrmodels/GrandMaster-PRO-MAX
- IlyaGusev/saiga_scored
- wikimedia/wikipedia
- mlabonne/open-perfectblend
- NousResearch/hermes-function-calling-v1
tags:
- qwen3.8
- compressed-tensors
- w4a8
- int4
- int8
- hadamard
- quip
- gptq
- mtp
- speculative-decoding
- vllm
---

# Qwen3.8-27B W4A8 Hadamard + MTP (for Ada / SM89)

A quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) optimized for the NVIDIA RTX 6000
Ada Generation (SM89, 48 GB). Built and tested on vLLM nightly `0.33.0rc1.dev1030+gcf5cea2da`, commit
[`cf5cea2da`](https://github.com/vllm-project/vllm/commit/cf5cea2da).

Weights: 18.5 GB (backbone 18.3 GB + MTP 0.25 GB) vs 52 GB for bf16. Format: `compressed-tensors` (`pack-quantized`).

## What was changed

- Language model linear layers: int4 per-channel symmetric weights with int8 dynamic per-token activations (W4A8).
  On SM89 this runs on the Humming INT8 path, about 1.6× faster prefill than W4A16.
- Hadamard rotation of the layer input (QuIP-style, block 128, online inverse via `hadacore`), then GPTQ with static
  act-order. The recipe is in `recipe.yaml`.
- GDN input projections `linear_attn.in_proj_qkv` / `in_proj_z`: W4A8 without rotation, because the fused
  `in_proj_qkvz` path in vLLM does not support the rotated variant.
- Kept in bf16: GDN `in_proj_a` / `in_proj_b`, `embed_tokens`, `lm_head`, vision tower.
- The built-in MTP layer (`mtp.*`) is kept and quantized separately: int4 group 32 asymmetric weights, bf16
  activations (W4A16), round-to-nearest with MSE clipping search. Per-channel W4A8 did not work for it, because
  `mtp.fc` takes the concatenated embedding and hidden state. The model runs speculative decoding with its own draft
  head.
- KV cache scheme in config: fp8 static per-tensor scales, used with `--kv-cache-dtype fp8_e4m3` and ignored with
  `nvfp4`. Recommended KV cache is nvfp4: about 1.7× more tokens than fp8 at the same quality.

## Precision by layer

The language model has 64 decoder layers in a repeating pattern of three Gated DeltaNet (GDN, linear attention)
layers and one full attention layer: 48 GDN and 16 full attention layers. Every layer has a dense MLP.

Weights stored in the checkpoint:

| module | layers | weight dtype | weight scale | activation dtype | Hadamard |
|---|---|---|---|---|---|
| `mlp.gate_proj`, `up_proj`, `down_proj` | all 64 | **INT4** symmetric, per channel | BF16 per channel | **INT8** dynamic, per token | yes |
| `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` | 16 full attention | **INT4** symmetric, per channel | BF16 per channel | **INT8** dynamic, per token | yes |
| `linear_attn.out_proj` | 48 GDN | **INT4** symmetric, per channel | BF16 per channel | **INT8** dynamic, per token | yes |
| `linear_attn.in_proj_qkv`, `in_proj_z` | 48 GDN | **INT4** symmetric, per channel | BF16 per channel | **INT8** dynamic, per token | no |
| `linear_attn.in_proj_a`, `in_proj_b` | 48 GDN | BF16 | — | BF16 | — |
| `linear_attn.conv1d`, `A_log`, `dt_bias`, `linear_attn.norm` | 48 GDN | BF16 | — | BF16 | — |
| `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, final `norm` | all | BF16 | — | BF16 | — |
| `self_attn.k_scale`, `v_scale` (scales for FP8 KV) | 16 full attention | BF16 | — | — | — |
| `embed_tokens`, `lm_head` | — | BF16 | — | BF16 | — |
| `mtp.fc`, `mtp` attention `q/k/v/o_proj`, `mtp` MLP `gate/up/down_proj` | MTP | **INT4** asymmetric, group 32 | BF16 + INT4 zero point per group | BF16 | no |
| `mtp` norms | MTP | BF16 | — | BF16 | — |
| vision tower: all blocks, `merger`, `patch_embed`, `pos_embed` | — | BF16 | — | BF16 | — |

Created at runtime by the flags in "How to run":

| what | dtype |
|---|---|
| KV cache of the 16 full attention layers, `--kv-cache-dtype nvfp4` | **NVFP4**: FP4 E2M1 values + FP8 E4M3 scale per 16 values |
| KV cache, alternative `--kv-cache-dtype fp8_e4m3` | **FP8 E4M3** with the stored per-tensor scales |
| GDN recurrent state, `--mamba-ssm-cache-dtype bfloat16` | BF16 |
| MTP draft head, copy of `lm_head` (`draft_lm_head_quantization`) | **NVFP4** |
| main `lm_head` | BF16 |

There are no FP8 or FP4 weights in the checkpoint: FP8 and NVFP4 are used only for the KV cache and the draft head.
The Hadamard matrices (128×128) are stored in FLOAT64 as `*.v_input`.

## Calibration datasets

512 samples × 2048 tokens, teacher outputs from the bf16 original. Evaluation texts were excluded.

- [Vikhrmodels/GrandMaster-PRO-MAX](https://huggingface.co/datasets/Vikhrmodels/GrandMaster-PRO-MAX), about 25%
- [IlyaGusev/saiga_scored](https://huggingface.co/datasets/IlyaGusev/saiga_scored), about 10%
- [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), Russian, about 10%
- [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend), about 15%
- [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
  in the native Qwen tool format, about 20%
- small synthetic instruction and tool-call data, not published, about 20%

## How to run

Requirements:
- vLLM with the Humming linear backend and the `hadacore` transform, tested on the build above. Older releases do not
  load the rotated W4A8 layout. The exact build is the public image
  `vllm/vllm-openai:nightly-cf5cea2da37d3dfed27d2e0aa3041c3e553c2d57`.
- INT8 activations on SM89 run through Humming only for per-channel int4 weights, which is what this checkpoint uses.
  Other GPUs were not tested.

```bash
vllm serve shabonasar/qwen3.8-27b-w4a8-hadamard-mtp-ada \
  --linear-backend humming \
  --attention-backend FLASHINFER \
  --mamba-ssm-cache-dtype bfloat16 \
  --mamba-cache-mode align \
  --kv-cache-dtype nvfp4 \
  --max-num-batched-tokens 2943 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-model-len 262144 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_lm_head_quantization":"nvfp4","draft_sample_method":"probabilistic"}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
```

Notes:
- `--mamba-ssm-cache-dtype bfloat16` halves the GDN state page. The attention block is aligned to that page: 832
  tokens with fp8 KV, 1472 with nvfp4. `--max-num-batched-tokens` works best at a multiple of the block minus one:
  2495 for fp8, 2943 for nvfp4.
- `--kv-cache-dtype nvfp4` stores KV as E2M1 with E4M3 scales per 16 values and computes attention in 16 bits
  through FlashInfer. Use `fp8_e4m3` if you prefer the smaller block.
- With bf16 KV on FlashInfer we got broken output on this vLLM build. Use fp8 or nvfp4, or the Triton backend for bf16.

## Measurements (single RTX 6000 Ada 48 GB, vLLM commit cf5cea2da)

Speed with the flags above (greedy, MTP k=2):

| | fp8 KV | nvfp4 KV |
|---|---|---|
| decode, 1 request, tok/s | 90.0 | 88.7–90.2 |
| decode, 8 requests total, tok/s | 621 | 618–622 |
| prefill 42k tokens, tok/s | 3834–3862 | 3832–3862 |
| time to first token, cold 100k prompt, s | 35.1 | 35.2 |
| decode at 64k context, 1 / 4 requests, tok/s | 81.7 / 226 | 86.5–87.6 / 249–250 |
| KV cache capacity, tokens (util 0.90) | 623 391 | 1 071 908 |
| MTP acceptance length | 2.15–2.21 | 2.18 |

For reference, an AWQ int4 g32 (W4A16) checkpoint of the same model on the same card prefills 42k at about 2400
tok/s.

Closeness to bf16 (teacher-forced log-probabilities):

| | bf16 | this model |
|---|---|---|
| perplexity, Russian Wikipedia (165k tokens) | 4.694 | 4.856 (+3.4%) |
| mean \|Δ log-prob\| vs bf16 | — | 0.314 |
| top-1 agreement with bf16 | — | 91.3% |
| tool calls, 280 cases: correct name / correct arguments | — | 272 / 255 |
| tool calls: parseable | — | 280 / 280 |

For the AWQ int4 g32 checkpoint above, the same metrics are 4.749 (+1.2%), 0.183 and 94.5%. This model trades
some closeness to bf16 for about 1.6× faster prefill. On tool calls it scored no worse than the AWQ checkpoint
(250 correct arguments of 280).

KV cache precision on long context. Ten documents of 131 072 tokens (PG19 books and joined Russian Wikipedia
articles), this model's weights, against a bf16 KV reference (Triton backend):

| position | 0–2k | 8–16k | 32–64k | 96–128k |
|---|---|---|---|---|
| perplexity, bf16 KV | 6.915 | 6.610 | 6.811 | 6.786 |
| fp8 KV | 6.916 | 6.611 | 6.815 | 6.785 |
| nvfp4 KV | 6.916 | 6.626 | 6.823 | 6.800 |

The nvfp4 cost stays at +0.1–0.2% and does not grow with context length.

## Limitations

- Only tested on SM89 with the vLLM build above.
- The GDN input projections are not rotated, so they carry most of the remaining quantization error.
- The numbers above come from our own test sets and one GPU model; check them on your workload.

## License

Apache-2.0, same as the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The calibration
datasets are used only for calibration and are not redistributed; see their pages for their own terms.
