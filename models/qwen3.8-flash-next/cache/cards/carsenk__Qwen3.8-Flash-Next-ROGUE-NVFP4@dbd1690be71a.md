---
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
library_name: transformers
tags:
  - qwen
  - qwen4-exp
  - mixture-of-experts
  - text-generation
  - quantized
  - nvfp4
  - modelopt
  - vllm
  - rogue
  - dgx-spark
---

# Qwen3.8-Flash-Next ROGUE NVFP4

This is a text-only Qwen3.8-Flash-Next deployment produced after a targeted ROGUE residual-stream intervention. The main model uses activation-calibrated ModelOpt NVFP4, the 51B-parameter PLE lookup table uses a packed NVFP4 representation, and the checkpoint retains the native BF16 MTP layer for speculative decoding.

Qwen3.8-Flash-Next is a 176B-parameter model with about 6B parameters activated per token. “6B active” does not mean the checkpoint has a 6B memory or routing footprint.

## ROGUE method

ROGUE estimates refusal-associated activation directions from balanced harmful and harmless prompt contrasts. Candidate directions are ranked using deterministic train/holdout scoring, projected away from a benign-preservation subspace, and applied only to evidence-ranked residual writers through a low-rank orthogonal projection.

The accepted intervention modified exactly **32 tensors across eight layers**:

- Layers: `19, 22, 23, 27, 28, 29, 31, 32`
- MLP/MoE writers: 24
- Attention writers: 4
- SSM writers: 4

ROGUE is a behavioral intervention, not a safety certification. Independent safety, factuality, instruction-following, and application-specific evaluation remains necessary.

## Artifact

- Architecture: `Qwen4ExpForCausalLM`
- Deployment model type: `qwen4_exp_text`
- Hidden layers: 48
- Native advertised context: 262,144 tokens
- ModelOpt provenance: 0.46.0, activation-calibrated W4A4 NVFP4
- Main-model NVFP4 group size: 16
- Packed PLE modules: 128 (`qwen38-nvfp4-ple-v1`)
- Native MTP: 31 BF16 tensors, one prediction layer
- Ignored/compatibility modules: 431
- Safetensors shards: 174
- Indexed tensors: 296,397
- Indexed tensor bytes: 111,858,152,472
- Vision tensors: omitted
- Source revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`

## Validated runtime

The checkpoint passed bounded tensor-parallel runtime tests on two NVIDIA DGX Spark systems connected directly through ConnectX-7:

- vLLM Qwen3.8-capable current source
- TP2 with one GB10 GPU per node
- `b12x` 1.2.6 for the main NVFP4 linear and MoE paths
- Triton for the BF16 native-MTP MoE and GDN prefill
- Native MTP with three speculative tokens
- Eager and non-eager execution tested; vLLM skipped CUDA graph capture for the native-MTP run, so no CUDA-graph claim is made
- Validation profile: 4,096 maximum sequence length, one active sequence, 2 GiB fixed KV cache per rank

Coherent generation and a controlled harmful-request refusal smoke test passed. The warm 256-token single-stream case measured a median **41.07 decode tokens/s** across three runs, with 768 of 768 draft tokens accepted for that deterministic sequence prompt. This is a narrow runtime measurement, not a universal throughput claim.

The same artifact without speculation measured 12.82 decode tokens/s in the matched case. Broader prompts produced lower native-MTP rates because acceptance depends on content: 30.17 tokens/s for the coherence prompt and 27.38 tokens/s for the refusal prompt in the optimized follow-up.

See `RUNTIME_VALIDATION.json` and `VALIDATION.md` for the exact evidence.

## vLLM usage

As of the validation date, released vLLM does not natively understand this checkpoint's packed PLE layout. The `runtime/` directory contains the reviewed Qwen4Exp PLE and native-MTP compatibility modules used for validation. Apply those modules to the matching paths in a current Qwen3.8-capable vLLM source tree, install `b12x==1.2.6` on SM120/SM121 systems, and validate startup backend selection before serving.

The essential serving options used for the validated two-node run were:

```text
--tensor-parallel-size 2
--distributed-executor-backend mp
--linear-backend b12x
--moe-backend b12x
--gdn-prefill-backend triton
--mamba-cache-dtype bfloat16
--mamba-ssm-cache-dtype bfloat16
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

For non-reasoning chat output, pass `enable_thinking=false` through chat-template arguments:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen3.8-Flash-Next-ROGUE-NVFP4",
    messages=[{"role": "user", "content": "Explain photosynthesis in one sentence."}],
    max_tokens=128,
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

## Caveats

- The native 262,144-token context is preserved in configuration, but this release was runtime-tested at 4,096 tokens. Do not infer a completed 262K stress test.
- The packed PLE path currently requires the included vLLM compatibility modules.
- The measured throughput is specific to TP2 DGX Spark, ConnectX-7, the stated backend versions, prompts, and concurrency.
- The validated single-request path does not reach 100 tokens/s. Published figures near 100 tokens/s for DGX Spark refer to aggregate multi-request throughput, not one decoded stream.
- Vision is not included.
- Keep normal safety controls in place and evaluate the modified behavior before production use.

## Integrity

`SHA256SUMS` covers every published artifact file other than the checksum file itself.
