---
pipeline_tag: text-generation
base_model:
- nvidia/Qwen3.8-27B-NVFP4
- Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
tags:
- Qwen3.8
- quantized
- NVFP4
- FP8
- ModelOpt
- GPTQ
- vllm
---

# Qwen3.8-27B-NVFP4-GDN

[nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) with the projections of the 48 Gated DeltaNet
(linear attention) layers also in NVFP4. The MLP, `lm_head`, attention layers, embeddings, MTP head and vision tower are
those of the NVIDIA checkpoint, unchanged.

It sits between the NVIDIA checkpoint (FP8 GDN and attention projections) and all-NVFP4 checkpoints such as
[QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) (every linear layer
in NVFP4, quantization-aware training): it removes 77% of the projection bytes QUASAR removes and gets most of its speed,
but keeps the quality of the NVIDIA checkpoint, including parallel tool calls, where all-NVFP4 checkpoints lose ground.

|  | nvidia/Qwen3.8-27B-NVFP4 | **this model** | QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 |
| --- | --- | --- | --- |
| MLP (64 layers) | NVFP4 | NVFP4 (NVIDIA's tensors) | NVFP4 (QAT) |
| GDN `in_proj_qkv` / `in_proj_z` / `out_proj` (48 layers) | FP8 | **NVFP4 (GPTQ)** | NVFP4 (QAT) |
| attention `q/k/v/o_proj` (16 layers) | FP8 | FP8 (NVIDIA's tensors) | NVFP4 (QAT) |
| `lm_head` | NVFP4 | NVFP4 (NVIDIA's tensor) | BF16 |
| checkpoint size | 21.92 GB | 19.50 GB | 20.56 GB |
| language-model weights read per token (NVFP4 `lm_head`) | 16.40 GiB | 14.14 GiB | 13.46 GiB |

The format is the NVIDIA Model Optimizer one (`quant_method: modelopt`, `MIXED_PRECISION`); `config.json` and
`hf_quant_config.json` list the GDN projections as NVFP4 with group size 16.

## Why only the GDN projections

Quantizing *all* projections to NVFP4 costs parallel tool calls (several calls in one answer): on BFCL v3 `parallel`,
QUASAR scores 65.5% and our own GPTQ all-NVFP4 build 73.5%, against 79.0% for the NVIDIA checkpoint. Leaving parts in FP8
showed the loss comes from the **attention** projections (spread over all 16 attention layers), not from the GDN ones:
with only the attention projections in FP8, the model keeps calling on after its first tool call as often as the
NVIDIA checkpoint. Quantizing only the GDN projections therefore keeps tool calling and benchmark scores at the level
of the NVIDIA checkpoint while reading 2.4 GB less weight per decode step.

## Quantization

- Source weights: the BF16 original [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (revision `1d4bf0f2`).
- Method: GPTQ (column-wise error compensation, damping 0.01) with NVFP4 block scales (16 elements, e4m3) chosen per block
  among the e4m3 codes near amax / 6 (three below to one above) by the Hessian-weighted error. The weight global scale is shared by
  `in_proj_qkv` and `in_proj_z` (engines run them as one GEMM; vLLM keeps only the largest global scale of a fused layer).
- Hessians: inputs of each projection with the NVIDIA checkpoint (FP8 projections, NVFP4 MLP), about 398K tokens of
  calibration text: non-public Japanese technical documents, source code (C++/CUDA/Python), and coding-agent requests with
  tool definitions and tool calls. Evaluation data (GSM8K, BFCL, the needle and Japanese task sets) was not used.
- Activation scales (`input_scale`): the NVIDIA checkpoint's calibrated FP8 input scale divided by 6 (same amax, NVFP4 range).

## Evaluation

All runs on one RTX 5090 at a 400 W power limit, FP8 KV cache unless stated, greedy unless stated.

**vLLM v0.29.0** (MTP 3 speculative tokens, the same settings for all three):

|  | nvidia | **this model** | QUASAR |
| --- | ---: | ---: | ---: |
| GSM8K (300, thinking off) | 94.7% | 95.7% | 95.0% |
| BFCL v3 all (1,000) | 89.1% | 89.8% | 87.5% |
| BFCL simple / multiple / parallel / parallel_multiple | 97.0 / 96.5 / 76.0 / 79.0 | 96.5 / 96.5 / **79.5** / 80.0 | 97.0 / 97.0 / 67.0 / 79.5 |
| decode tok/s, single stream (code / json / prose) | 156 / 181 / 130 | **180 / 205 / 139** | 151 / 174 / 123 |
| time to first token, 19.5K-token prompt | 2.62 s | 2.23 s | 2.14 s |
| decode tok/s at that context | 106 | 102 | 91 |

On vLLM this model decodes fastest of the three; QUASAR, although the smallest, decodes slower than the NVIDIA checkpoint
(likely its BF16 `lm_head`, about 2.3 GB read per token against 0.6 GB in NVFP4).

**A custom inference engine** (the numbers this model was selected with; for QUASAR the engine uses the NVIDIA NVFP4
`lm_head`, since QUASAR's is the BF16 original):

|  | nvidia | **this model** | QUASAR |
| --- | ---: | ---: | ---: |
| KL vs the BF16 original (35,328 positions, ja/en/code prompts up to 16K, FP8 KV cache) | 0.0362 | 0.0445 | 0.0475 |
| BFCL v3 all (1,000) | 90.1% | 89.7% | 87.0% |
| BFCL parallel (200) | 79.0% | 78.5% | 65.5% |
| BFCL parallel + parallel_multiple, temperature 0.7, 3 seeds (correct of 400, mean) | 315.0 | **318.0** | 288.7 |
| decode tok/s without speculation (128 / 16K context) | 87.1 / 83.5 | 96.6 / 92.2 | 99.9 / 95.1 |
| prefill tok/s at 19K / 73K / 194K | 9,897 / 7,697 / 5,157 | 10,956 / 8,329 / 5,435 | 11,595 / 8,654 / 5,571 |

Against the NVIDIA checkpoint this model gets 74% of QUASAR's decode gain and 62–67% of its prefill gain. In quality its
KL sits between the two, but on the task benchmarks it stays with the NVIDIA checkpoint: same BFCL (the sampled runs
vary by 8–16 answers between seeds; QUASAR is 26–29 below), and QUASAR loses 13 points on `parallel`.

With the predecessor of this build (identical except separate `in_proj_qkv` / `in_proj_z` global scales, KL 0.0436) the
same engine scored GSM8K 94.7% (NVIDIA 94.3%, QUASAR 94.7%), passed a Japanese reading / translation / summarization set
judged by Gemma 4 within its noise, found 16/16 needles at 190K tokens and 58/64 at 256K (NVIDIA 59/64, QUASAR 53/64),
and produced no runaway or looping responses in 200 low-effort thinking samples at temperature 0.6 (NVIDIA also none).

**The same engine with an NVFP4 KV cache** (this model; the KV cache 44% smaller than FP8, so the pool at 262K context
holds 589,888 tokens instead of 339,968; the prefill attention reads the NVFP4 keys with FP4 MMAs, the decode attention
with FP8 MMAs):

|  | FP8 KV | NVFP4 KV |
| --- | ---: | ---: |
| decode tok/s without speculation (128 / 16K / 32K / 131K context) | 96.6 / 92.2 / 89.5 / 75.8 | 96.5 / 93.1 / 91.0 / **81.9** |
| speculative step with a DFlash draft, ms (4K / 32K / 131K context) | 12.80 / 13.43 / 15.81 | 12.86 / 13.15 / **14.56** |
| prefill tok/s at 19K / 73K / 194K | 10,913 / 8,225 / 5,349 | 11,040 / 8,734 / **5,857** |

Batched speculative decode (one step verifies the draft of every sequence; default sampling, temperature 1.0), ms per
step for 1 / 2 / 4 / 8 sequences and the total tok/s of 8:

|  | 1 / 2 / 4 / 8 sequences | total, 8 sequences |
| --- | ---: | ---: |
| nvidia, NVFP4 KV, short context | 13.69 / 15.18 / 17.84 / 24.04 | 1,377–1,458 |
| **this model**, NVFP4 KV, short context | **12.40 / 13.76 / 16.39 / 21.99** | **1,496–1,547** |
| this model, FP8 KV, short context | 12.43 / 13.79 / 16.44 / 22.13 | 1,553–1,565 |
| this model, NVFP4 KV, 32K context each | 13.17 / 15.05 / 19.21 / 27.79 | 1,083–1,104 |
| this model, FP8 KV, 32K context each | 13.54 / 15.81 / 20.62 / 30.55 | 1,027–1,040 |
| this model, NVFP4 KV, 64K context each | 13.73 / 16.16 / 21.45 / 32.36 | 984–1,002 |

The totals vary with the sampled acceptance length; compare the step times. Through the engine's OpenAI-compatible
server, 8 concurrent streams of 4,096 reasoning tokens (temperature 1.0, top-p 0.95, top-k 20) total 1,404–1,411 tok/s
(4 streams: 946–954) with this model and NVFP4 KV, against 1,262 (846) for the NVIDIA checkpoint with FP8 KV measured a
day earlier with the same settings.

The NVFP4 KV cache passes the same quality checks (GSM8K 95.3%, BFCL 88.9%, the Japanese set, needles 256K 56/64) with
one measurable cost: on short BFCL `parallel` / `parallel_multiple` prompts at temperature 0.7 it gets 306.0 of 400
correct against 317.3 with FP8 KV (3 seeds each), by stopping a parallel call list early. After 32K or 96K tokens of
coding-agent history the difference disappears (−4 and +2.4 of 400).

These are small, saturated benchmarks; differences of a point or two are within their noise. Not evaluated: SWE-style
agent benchmarks, other languages, the vision tower (unchanged from the NVIDIA checkpoint).

## Usage

vLLM (tested with v0.29.0):

```bash
vllm serve <this repo> --kv-cache-dtype fp8_e4m3 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

vLLM logs "Detected ModelOpt fp8 checkpoint" and "Detected ModelOpt NVFP4 checkpoint"; it must **not** warn that "the
global scale for input or weight are different for parallel layers". Other engines that read modelopt mixed-precision
checkpoints (e.g. SGLang) should work the same way but were not tested.

## License and attribution

Apache-2.0, as the base models. Derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen and
[nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) by NVIDIA. Modifications: the GDN projection
tensors (`linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj` of the 48 linear-attention layers) were replaced by NVFP4
tensors quantized from the BF16 original as described above; `config.json`, `hf_quant_config.json` and the safetensors
index were updated accordingly. All other tensors and files are those of the NVIDIA checkpoint. Not affiliated with Qwen
or NVIDIA.
