---
license: mit
library_name: mlx
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
inference: false
tags:
  - deepseek
  - deepseek-v4
  - deepseek-v4-flash-0731
  - mixture-of-experts
  - mla
  - reasoning
  - tool-use
  - dsml
  - mlx
  - apple-silicon
  - quantized
  - jang
  - affine
  - qat
quantization_config:
  family: jang
  weight_format: affine
  profile: affine-awq-diag-imatrix-nr8-gptq-94.995gib
  routed_default_bits: 2
  routed_lifted_bits: 3
  lifted_gate_layers: [5, 14, 30, 34, 37, 42]
  attention_bits: 8
  token_bookend_bits: 8
  critical_controls: source-f32
  code_optimization: gptq-error-compensated
  drop_mtp: true
---

> [!IMPORTANT]
> **This bundle requires vMLX Python 0.22 or newer.** Earlier runtimes do not
> implement the DSV4 composite-cache and pool-quantization contract this
> bundle is stamped for.

<p align="center"><img src="jangq-logo.png" width="300" alt="JANG"/></p>

<p align="center">
  <a href="https://vmlx.net"><img src="vmlx-app.png" alt="vMLX — run JANG models on Apple Silicon" width="820"></a>
</p>

<h3 align="center">⚡ All JANG models are meant to be run in <a href="https://vmlx.net">vMLX</a></h3>

# DeepSeek-V4-Flash-0731-JANG

Dynamic affine JANG quantization of the official
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
release for DSV4-aware Apple Silicon MLX runtimes, with **QAT-grade
error-compensated weight codes** on all routed experts.

This is the **0731 release**, not the earlier DeepSeek-V4-Flash preview. The
source is pinned to immutable commit
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.

| | |
|---|---|
| Source | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Source revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| License | MIT, inherited from upstream |
| Format | JANG mixed affine, GPTQ-optimized codes |
| Bundle size | 102.00 GB / 94.995 GiB |
| Weight shards | 102 |
| Indexed tensor keys | 101,295 |
| Context configuration | 1,048,576 tokens with YaRN |
| Runtime cache schema | `deepseek_v4_v9` |
| Measured decode | ~20 tok/s steady on a 128 GB M5 Max, stock OS config |
| MTP / DSpark | Dropped from this runtime artifact |

## What this quant is

The 94.995 GiB policy protects the non-routed attention, Compressor, Indexer,
shared-expert, embedding, and output paths at 8-bit. Routed down projections
use group 32, gate/up use group 64, and the gate projection is lifted to
3-bit in six selected layers. Folded AWQ and diagonal-imatrix scales improve
the weight fit, and every routed projection additionally carries
error-compensated (GPTQ-family) codes fitted against real routed activation
statistics instead of plain nearest rounding — same grids, same storage
format, same kernels, no runtime sidecar.

| Tensor role | Bits | Group size | Policy |
|---|---:|---:|---|
| Routed expert gate / `w1` | 2 | 64 | Default |
| Routed expert gate / `w1` in layers 5, 14, 30, 34, 37, 42 | 3 | 64 | Quality lifts |
| Routed expert down / `w2` | 2 | 32 | All routed layers |
| Routed expert up / `w3` | 2 | 64 | All routed layers |
| Attention, Compressor, Indexer, shared expert | 8 | 64 | Non-routed fidelity floor |
| Token embedding and output head | 8 | 64 | Bookends |
| Norms, router, mHC, sinks and controls | Source dtype | — | Critical F32 retained |

The index records 11,008 tensors at 2b/G32, 20,480 at 2b/G64, 1,536 at
3b/G64, and 512 at 8b/G64. This is affine JANG, **not JANGTQ**.

## JANG versus uniform MLX quantization

| | This repository | Uniform MLX quant |
|---|---|---|
| Weight policy | Per tensor role, projection, and selected layer | One global/default policy |
| Routed gate/down/up groups | G64 / G32 / G64 | Not independently represented |
| Sensitive gate lifts | Six layers at 3-bit | Not represented by a global bit width |
| Weight codes | Error-compensated against routed activations | Nearest rounding |
| DSV4 controls | Source dtype; critical F32 retained | Loader-dependent |
| Required runtime | DSV4-aware vMLX mixed-affine path | Stock uniform loader |

No like-for-like accuracy benchmark against a uniform MLX artifact is claimed.
A stock loader that applies one bit width to every quantized tensor cannot
faithfully interpret this bundle's plan.

## Runtime requirement

Use vMLX Python 0.22 or newer, supporting:

- per-tensor JANG affine bits and group sizes;
- DSV4 SWA + CSA + HCA composite cache state;
- Compressor and Sparse Indexer state;
- the bundled official 0731 Python encoder and DSML output parser;
- native `low`, `high`, and `max` reasoning effort, with **Low as the
  reasoning default**.

Preview-era adapters that expose only High/Max or map omitted reasoning to
Instruct are incompatible with the 0731 contract.

## Native 0731 chat contract

The official release has no Jinja chat template. This repository does not
synthesize `chat_template.jinja`; `tokenizer_config.json.chat_template` is
unset. Use `encoding/encoding_dsv4.py`:

```python
from encoding.encoding_dsv4 import (
    encode_messages,
    parse_message_from_completion_text,
)

messages = [{"role": "user", "content": "Explain why 17 is prime."}]

# Native default reasoning: thinking mode, Low effort.
prompt = encode_messages(messages, thinking_mode="thinking")

# Non-reasoning / Instruct behavior.
chat_prompt = encode_messages(messages, thinking_mode="chat")

# Explicit 0731 reasoning rails.
high_prompt = encode_messages(
    messages, thinking_mode="thinking", reasoning_effort="high"
)
max_prompt = encode_messages(
    messages, thinking_mode="thinking", reasoning_effort="max"
)
```

The four intended surfaces are Instruct, Reasoning Low, Reasoning High, and
Reasoning Max. Low is the default when `reasoning_effort` is omitted in
thinking mode. Tool calls use native DSML, and tool results are merged into
user messages as `<tool_result>...</tool_result>` blocks. See
[`encoding/README.md`](encoding/README.md) for the full contract.

## Generation and stop contract

The deployment `generation_config.json` uses `do_sample=true`,
`temperature=0.6`, `top_p=0.95`, `top_k=0`, BOS `0`, and EOS `1`, with no
non-neutral repetition-penalty override.

`temperature=0.6` is this bundle's deliberate deployment default, tuned for
coding and agentic use (also DeepSeek's DSV4 pass@1 coding-eval setting). The
upstream card documents `1.0` as its general default; clients may explicitly
select any policy per request. The same defaults are declared in
`jang_config.json` chat metadata so both declarations agree.

DSV4-aware servers should recognize EOS `1` and the 0731 role-boundary tokens
User `128803`, Assistant `128804`, and latest-reminder `128828` where the API
surface uses boundary stopping.

## Cache, context, and speculative decoding

The bundle preserves the 1M-token YaRN configuration, sliding window 128, and
the layerwise compression schedule. Its `deepseek_v4_v9` metadata names SWA,
CSA, HCA, Compressor, and Indexer cache state. Generic TurboQuant KV is
disabled and native q8 pool-cache quantization defaults on. Set
`DSV4_POOL_QUANT=0` only for an explicit diagnostic comparison.

MTP / DSpark weights are intentionally absent from this runtime artifact.
Speculative decoding would require a separate drafter plus atomic rollback of
the complete DSV4 composite cache; this repository does not claim that path.

## Validation

Verified on this exact artifact:

- source identity, the complete 102-shard index, all 101,295 safetensor
  header keys, the per-tensor affine plan, tokenizer metadata, generation
  defaults, and all four official encoder fixtures;
- live generation in vMLX Python 0.22 with pool-cache quantization on:
  exact-instruction following, reasoning-Low arithmetic, a 400-token code
  generation row with exact requested identifiers, and a tool-call row — all
  coherent, naturally stopped, with no degenerate repetition;
- ~20 tok/s steady decode, ~7 s load, ~98 GB peak unified memory on a
  128 GB M5 Max with stock OS configuration.

Not exhaustively re-verified on this exact artifact: the full multi-turn DSML
tool matrix, 30K+ long-context recall, and cache trim/restart rows. No claim
beyond the verified rows is made.

## Download

```bash
hf download JANGQ-AI/DeepSeek-V4-Flash-0731-JANG \
  --local-dir ~/models/DeepSeek-V4-Flash-0731-JANG
```

## Korean summary

이 모델은 공식 `deepseek-ai/DeepSeek-V4-Flash-0731` 릴리스를 Apple Silicon
용으로 AWQ 및 diagonal imatrix가 적용된 affine JANG 양자화에 GPTQ 계열
오류-보상 코드 최적화를 더한 94.995 GiB 번들입니다. 저장 포맷과 커널은
기존과 동일하며, 배포 기본 샘플링은 코딩에 맞춘 temperature 0.6 / top-p
0.95입니다. 기본 추론 모드는 Reasoning Low이고 Low/High/Max와 비추론
Instruct 모드를 지원합니다. vMLX Python 0.22에서 이 번들 그대로 일관된
생성(정확한 지시 수행, 코드 식별자 재현, 도구 호출)과 128 GB M5 Max 기준
약 20 tok/s 디코드를 확인했습니다. 장문 컨텍스트 전체 매트릭스는 아직
완전히 재검증되지 않았습니다.

## Contact

eric@jangq.ai
