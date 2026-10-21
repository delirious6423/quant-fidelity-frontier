---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- en
- lt
- multilingual
library_name: vllm
pipeline_tag: text-generation
tags:
- qwen3.8
- qwen
- nvfp4
- fp8
- int8
- mixed-precision
- compressed-tensors
- llm-compressor
- gptq
- vllm
- rtx-5090
- blackwell
- sm120
- 256k
- long-context
- mtp
- speculative-decoding
- agentic
- tool-calling
- structured-output
- digisensus
---

# Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K

**Qwen3.8-27B on one RTX 5090 with the full 262,144-token (256K) context.**

This is the model [Digisensus.com](https://digisensus.com) uses for testing on its RTX 5090 GPU cluster. We are
sharing the weights, the serving setup and the way it was made.

**Made for:** agentic use (tool calling, multi-step agents), structured output (JSON with a
schema) and reasoning with thinking on.

**On a single RTX 5090 (32 GB):** 256K context, NVFP4 KV cache, MTP speculative decoding,
about 115 tokens/s for one user (92 at the full 256K).

## How close is it to the original model?

We measure how far the quantized model's next-token predictions are from the original bf16
Qwen3.8-27B ("stock"): the KL divergence, averaged over every token of 125 held-out prompts
(prose, tool calls, JSON, reasoning, code). **Lower is closer to stock.** "Top-1" is how often
the quantized model picks the same next token as stock.

For comparison we include another NVFP4 quant of the same model that was also adjusted for
the RTX 5090. Both use the same amount of GPU memory.

| model | KL vs stock (lower is better) | top-1 agreement with stock |
|---|---|---|
| another 5090-adjusted NVFP4 quant | 0.01520 | 95.27% |
| **Qwen3.8-27B-Digisensus** | **0.01148** | **96.07%** |

Digisensus is 24.5% closer to stock (95% confidence interval 21.8%–26.9%). By task: prose 26%
closer, reasoning 26%, code 22%, tool calls 19%.

Thinking on, 24 real requests (JSON scoring, tool agents, reasoning): Digisensus wrote 8% fewer
reasoning tokens, hit the token limit 2 times instead of 5 (of 48 runs), and gave 46 valid
answers instead of 43. MTP accepted 67.9% of draft tokens instead of 65.9%.

## How it was made

We measured, for every part of the network, how much quality it loses at FP8 and at NVFP4,
then gave more bits to the parts that lose the most and fewer to the parts that lose almost
nothing — at the same total memory:

- **embeddings: bf16 → INT8 per row.** Costs almost nothing, frees 1.18 GiB. This paid for
  everything else.
- **MLP layers 28–35 and 48–55: NVFP4 → FP8.** These 4-bit MLPs lost the most.
- MLP layers 56–59: FP8 → NVFP4. They lose less than 48–55.
- layer-63 attention `o_proj`: bf16.
- Gated-DeltaNet (linear attention) projections stay at FP8: quantizing them harder hurts long
  reasoning. MTP head bf16. Vision tower bf16 (not used with `--language-model-only`).

| part | format |
|---|---|
| MLP layers 0–27, 36–47, 56–59 | NVFP4 W4A4 (groups of 16, FP8 scales), GPTQ |
| MLP layers 28–35, 48–55, 60–63 (`down` of 28–31: NVFP4) | FP8 W8A8 |
| attention q/k/v/o, GDN in_proj_qkv/z/out_proj, lm_head | FP8 W8A8 (layer-63 o_proj: bf16) |
| embed_tokens | INT8 per row, weight only |
| KV cache | served as NVFP4 (FP8 scales in the checkpoint) |
| MTP head (`model_mtp.safetensors`) | bf16 |

The full method, every script and all raw numbers:
[GitHub — Digisensus/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K](https://github.com/Digisensus/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K).

## How to run it

You need the vLLM v0.28.0 image and patches from the GitHub repo. Stock vLLM cannot load
these weights (INT8 embeddings) and cannot run the NVFP4 KV cache on an RTX 5090. The repo
also fixes stop strings firing inside thinking, leading newlines in answers, and JSON quote
truncation.

```bash
git clone https://github.com/Digisensus/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K.git
cd Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K
./image/build.sh
hf download Digisensus/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K --local-dir models/Qwen3.8-27B-Digisensus-NVFP4-RTX5090-256K
cd compose && cp .env.example .env && docker compose up -d
```

Main flags: `--max-model-len 262144 --kv-cache-dtype nvfp4 --kv-cache-memory-bytes 5694957226
--gpu-memory-utilization 0.94 --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
--language-model-only --reasoning-parser qwen3 --tool-call-parser qwen3_xml`.

Sampling for tool calls and JSON: `presence_penalty=0` (Qwen's suggested 1.5 makes the model
leave repeated tool arguments empty), temperature 0.7 or lower.

## Speed (one RTX 5090, 480 W and 600 W, the compose config above)

**Prefill** (reading the prompt), one request at a time, random-token prompts:

| context | prefill tokens/s | time to first token |
|---|---|---|
| 1k | 4,475 | 0.23 s |
| 8k | 8,517 | 0.96 s |
| 16k | 10,703 | 1.5 s |
| 32k | 6,494 | 5.1 s |
| 64k | 4,986 | 13 s |
| 128k | 3,347 | 39 s |
| 261,600 (max) | 1,661 | 158 s |

**Decode** (writing the answer) on real text: a long document of real conversations plus
"summarise", 384 output tokens, thinking off, temperature 0.7 (`quantization/decode_ctx.py`).
Random-token prompts are not used for decode because MTP speed depends on how predictable the
text is. **Prefix caching was switched off** for these runs so nothing is reused between
requests. Same engine, same test, only the GPU power limit changed. Single-user numbers are
the engine's own counters (tokens / decode-step time; two runs each agree within noise).

| context | 1 user: tokens/s, 480 W | 1 user: tokens/s, 600 W | step time 480 / 600 W | users at once | **all users together: tokens/s, 480 W** | **600 W** | 600 W vs 480 W |
|---|---|---|---|---|---|---|---|
| 1k | 113 | 118 | 23.0 / 23.1 ms | 8 | **692** | **807** | +17% |
| 8k | 119 | 120 | 23.3 / 23.2 ms | 8 | **339** | **351** | +4% |
| 16k | 122 | 118 | 23.3 / 23.3 ms | 8 | **195** | **206** | +6% |
| 32k | 119 | 117 | 23.9 / 24.1 ms | 8 | **87** | **93** | +7% |
| 64k | 132 | 116 | 24.7 / 24.7 ms | 4 | **40** | **42** | +5% |
| 128k | 113 | 114 | 26.4 / 26.3 ms | 2 | **22** | **24** | +9% |
| 258,000 | 98 | 92 | 30.0 / 29.8 ms | 1 | — | — | — |

Peak power draw: 495 W at the 480 W limit, 610 W at the 600 W limit.

What the table says:

- **One user needs no power.** At 480 W a single request runs at the same speed as at 600 W
  at every context (the differences are run-to-run noise). A decode step for one user is
  limited by reading the 21 GB of weights, not by compute, so the extra 120 W buys nothing.
- **Many users at short context are where 600 W helps:** 8 users at 1k get 807 tokens/s at
  600 W vs 692 at 480 W (+17%). At 8k and beyond the gain is 4–9%.
- **Decode barely depends on context** because 48 of the 64 layers are Gated-DeltaNet
  (linear attention), whose cost per token does not grow with context; only the 16
  full-attention layers read the (NVFP4) KV cache. The step time grows from 23.0 ms at 1k to
  30.0 ms at 258k (+30%); MTP then turns each step into 2.6–3.3 tokens.
- "All users together" = all output tokens divided by the time during which at least one user
  was receiving tokens. Per-user speed drops at long context because the engine mixes other
  users' prompt reading into the decode steps. The number of users at once is limited by KV
  memory: min(max-num-seqs, KV pool / (context + output)).

### More than 8 users (480 W, high-concurrency profile)

The config above allows 8 requests at once and the full 256K context. To see whether more
users add throughput we ran a second profile at 480 W: `--max-num-seqs 32`,
`--max-model-len 139264` (136K), KV pool sized by vLLM (198,767 tokens),
`--gpu-memory-utilization 0.90` (`compose/docker-compose.high-concurrency.example.yaml`).
Why not 64 users or 256K here: every sequence reserves a Gated-DeltaNet recurrent state
(about 75 MB across the 48 GDN layers) up front, so 64 sequences cost 4.8 GB and 32 sequences
no longer leave room for one 256K request.

| context | users at once | per user: tokens/s | **all users together: tokens/s** |
|---|---|---|---|
| 1k | 8 | 111 | **777** |
| 1k | 16 | 93 | **752** |
| 1k | 32 | 85 | **633** |
| 8k | 8 | 73 | **358** |
| 8k | 16 | 51 | **316** |
| 8k | 23 | 48 | **307** |
| 16k | 11 | 34 | **113** |
| 32k | 5 | 53 | **83** |
| 64k | 3 | 41 | **51** |

More users do not add throughput: with MTP every step checks 4 draft tokens per user, so the
batch grows fast and past about 8 users the total goes down, not up. On this card and this
quant, 8 users is the sweet spot; more users only spread the same throughput thinner.

## Calibration data

About 641k tokens, rendered with the model's own chat template: 50% private [Digisensus.com](https://digisensus.com)
test traffic (contact-centre QA, audit, PII and scoring prompts, tool-calling agents, with the
model's own answers; **not released**), 18%
[NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
tool calls (Apache-2.0), 12% hermes json-mode (Apache-2.0), 20%
[HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
(MIT). Calibration only looks at activations; nothing was trained.

## Limits

- Needs the patched vLLM from the GitHub repo.
- Tuned and judged for agentic use, structured output and reasoning.
- Long-context retrieval tests (needle, RULER) at 256K with the NVFP4 KV cache are not
  published yet.

## License

Apache-2.0, derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The
serving stack builds on vLLM (Apache-2.0); third-party notices are in the GitHub repo's NOTICE file.
