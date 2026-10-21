---
library_name: vllm
pipeline_tag: text-generation
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
tags:
- lfm2.5
- moe
- nvfp4
- auto-round
- compressed-tensors
- fp8
- bf16
- reasoning
- tool-calling
datasets:
- NeelNanda/pile-10k
---

# LFM2.5-24B-A2B · AutoRound NVFP4

A compact edition of [Liquid AI's LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) from **CoolJohn AI Labs**, with NVFP4 feed-forward weights, original BF16 attention and convolution projections, and calibrated scales for FP8 KV caching.

- **24B parameters, about 2.45B active per token.** Text-only, with reasoning and tool calling.
- **14.75 GB / 13.73 GiB of weights**, tested on two RTX 5060 Ti 16GB GPUs.
- **Recommended context: 96K per session** (96,384 tokens in the tested profile), shared by input, reasoning and the final answer.
- **Three concurrent sessions** with the profile below; approximately **125.9 tokens/sec** in the short single-request decode check.

## How it was quantized

[Intel AutoRound 0.17.0](https://github.com/intel/auto-round) was used to quantize the feed-forward layers and MoE experts to **NVFP4 W4A4, group size 16**. Calibration used **500 SignRound iterations per block**, 256 samples of 2,048 tokens from `NeelNanda/pile-10k`, seed 42 and an effective batch size of eight.

All **40 attention** and **60 convolution input/output projections** are byte-identical to the pinned BF16 source. Embeddings, routing gates, normalization and short convolution filters also keep source precision. The serving profile uses **FP8 E4M3 KV cache**, with 20 calibrated key/value scales included in the checkpoint. Architecture, tokenizer and the official chat template are unchanged. The [quantization recipe](quantization_recipe.json) records provenance and build details.

## Context Limitations

We recommend **96K total tokens per session**: set `max_model_len` to **96384**, allowing **80,000 input tokens + 16,384 output tokens**. Input includes conversation history and tool definitions; output includes reasoning and the final answer, with thinking capped at 8,192 tokens.

This was the largest tested window to pass both the initial retrieval screen and confirmation with new seeds. Starting at 100,000 input tokens, we reduced the input budget in 5,000-token steps after a failure, keeping the same 16,384-token output reserve. At 80,000 input tokens, the model passed **36/36 retrieval cases across four seeds**, plus **2/2 checks using a separate fixture**. These cover single-record lookup, three-record collection and two-hop lookup, with facts near the beginning, middle and end.

The next step up, 85,000 input tokens, passed its initial screen but failed confirmation on three-record lookup. All larger windows tested also had retrieval failures, including the architecture's native 131,072-token window. The 96K recommendation therefore reflects measured retrieval accuracy. These finite synthetic checks are not a guarantee for every document; validate exact retrieval on your own material too. [Results and method](retrieval_tests.json).

## vLLM Settings

The tested runtime is **vLLM 0.31.0**, CUDA 13, on two RTX 5060 Ti 16GB GPUs. Install a compatible [vLLM CUDA build](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/), then download the support files:

```bash
hf download CoolJohn-AI/LFM2.5-24B-A2B-NVFP4-AutoRound \
  apply_vllm_lfm_fixes.py lfm_serving_policy.py --local-dir ./lfm-support
python ./lfm-support/apply_vllm_lfm_fixes.py --apply

PYTHONPATH="$PWD/lfm-support" NCCL_BUFFSIZE=1048576 \
  vllm serve CoolJohn-AI/LFM2.5-24B-A2B-NVFP4-AutoRound \
  --served-model-name LFM2.5-24B-A2B --host 127.0.0.1 --port 8080 \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len 96384 --max-num-seqs 3 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.60 --kv-cache-memory-bytes 2030043136 \
  --kv-cache-dtype fp8_e4m3 \
  --kernel-config '{"moe_backend":"cutlass","enable_jit_warmup":false}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3]}' \
  --enable-auto-tool-choice --tool-call-parser lfm2 --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --middleware lfm_serving_policy.LFMServingPolicy \
  --override-generation-config '{"temperature":0.1,"top_k":50,"top_p":1.0,"min_p":0.0,"repetition_penalty":1.05,"max_new_tokens":16384}'
```

Use the included **official Liquid AI chat template**. The loader helper applies version-checked fixes for expert names and FP8 KV scales. The [serving policy](lfm_serving_policy.py) forces reasoning on and enforces the tested sampling settings, 8,192-token thinking cap and 16,384-token output limit. Clients can request shorter answers; input validation still reserves the full output allowance to preserve the 80,000-token input cap.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="LFM2.5-24B-A2B",
    messages=[{"role": "user", "content": "What is 37 plus 58?"}],
)
print(reply.choices[0].message.content)
```

Three simultaneous sessions at the recommended budget, tool-call/result round trips and short decode speed were checked. Speed counts reasoning tokens and varies with hardware and workload. [Serving measurements](benchmarks.json).

## Credits and license

Base model: **Liquid AI**. Quantization and packaging: **CoolJohn AI Labs**, using **Intel AutoRound**. The [LFM Open License v1.0](LICENSE) applies to the model; support code uses [Apache 2.0](CODE_LICENSE). See [NOTICE](NOTICE) and [SHA256SUMS](SHA256SUMS).
