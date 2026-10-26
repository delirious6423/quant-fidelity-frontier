---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
language:
  - ko
  - en
tags:
  - gguf
  - llama.cpp
  - moe
  - expert-pruning
  - gsq
  - quantization
  - korean
  - code
---

# Qwen3.8-Flash-Next GSQ-RCO Ko-Coder (256 experts)

Qwen3.8-Flash-Next with half of its routed experts pruned (512 → 256 per layer) and every quantized tensor
re-quantized with GSQ on a Korean + coding + agent calibration mix. The GGUF type of each tensor is the one
[ISTA-DASLab's GSQ-RCO Q2_0 build](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) gives it
(routed experts Q2_0); the quantization parameters (codes and scales) are fitted here, not copied. Text only.

### 한국어 소개

Qwen3.8-Flash-Next의 라우팅 전문가를 층마다 절반(512 → 256)으로 줄이고, 양자화되는 모든 텐서를 한국어·코딩·에이전트
보정 데이터로 GSQ를 써서 다시 양자화한 모델입니다. 텐서마다의 GGUF 양자화 타입은
[ISTA-DASLab의 GSQ-RCO Q2_0 빌드](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)와 같고
(라우팅 전문가는 Q2_0), 양자화 파라미터(코드와 스케일)는 복사하지 않고 이 보정 데이터로 직접 맞췄습니다.
파라미터는 125.7B에서 65.3B로 줄었고(n-gram 임베딩 51.2B 별도), 모델 파일은 37.6 GB에서 20.6 GB가 되었습니다.
토큰당 활성 파라미터는 원본과 같은 약 6B입니다. 텍스트 전용입니다.

| | Original (ISTA GSQ-RCO Q2_0) | This model |
|---|---|---|
| Routed experts per layer | 512 | 256 (top-10 routing kept) |
| Routed expert parameters | 120.8B | 60.4B |
| Parameters (shard 1) | 125.7B | 65.3B |
| n-gram embedding (shard 2) | 51.2B | 51.2B |
| Active parameters per token | ~6B | ~6B |
| GGUF size | 37.6 GB + n-gram shard | 20.6 GB + 28.8 GB n-gram shard |

## Files

| File | |
|---|---|
| `Qwen3.8-Flash-Next-GSQfull-gumbel256-RCOtypes-00001-of-00002.gguf` | the model |
| `Qwen3.8-Flash-Next-GSQfull-gumbel256-RCOtypes-00002-of-00002.gguf` | the n-gram embedding table (`per_layer_token_embd`, IQ4_NL); keep it next to shard 1 |
| `Qwen3.8-Flash-Next-GSQfull-gumbel256-RCOtypes.experts.json` | the original indices of the experts each layer keeps |

## How it was made

- **Pruning.** Each layer keeps 256 experts chosen by a learned Gumbel mask that minimizes the error of the layer's
  MoE output on the calibration tokens (it swaps up to ~80 experts per layer against REAP's saliency ranking, at a
  lower output error); the router keeps only their rows.
- **Quantization.** Every tensor ISTA-DASLab's RCO allocation quantizes is quantized here in the same GGUF type, with
  codes and scales of its own (see [Quantization](#quantization)). Layers go in order, each from the hidden states of
  the already quantized layers before it.
- **Calibration.** 1,584 sequences of 2,048 tokens (1,536 for fitting, 48 held out): Korean chat, writing, wiki and
  coding questions; English code with and without thinking; tool use; agent sessions and chats of the original model
  ([Vemloir/Qwen3.8-Flash-Next-traces](https://huggingface.co/datasets/Vemloir/Qwen3.8-Flash-Next-traces), ODbL);
  coding-assistant conversations.

## Quantization

**Types** follow ISTA-DASLab's RCO per-tensor allocation exactly. **Parameters** (every code and block scale) are this
model's own: GPTQ initialization, then GSQ on per-batch Hessians of each tensor's inputs from the calibration set.
Nothing is taken from ISTA's file but the types and the unquantized BF16/F32/F16 tensors, which are the original
checkpoint's as llama.cpp converts them (bit-identical).

| Tensors | Parameters | GGUF types (tensor count) | Bits/weight | Size |
|---|---|---|---|---|
| Routed experts (gate, up, down; 48 layers × 256) | 60.40B | Q2_0 (144) | 2.25 | 16.99 GB |
| Attention and gated delta-net projections | 2.67B | Q3_K (56), IQ4_XS (45), Q4_K (26), Q5_K (12), Q6_K (10), Q2_0 (7) | 3.98 | 1.33 GB |
| Shared experts | 0.24B | Q2_0 (50), Q3_K (34), Q4_0 (16), Q4_K (12), IQ4_XS (11), Q5_0 (7), IQ4_NL (7), Q5_K (3), Q6_K (2), Q8_0 (2) | 3.60 | 0.11 GB |
| `token_embd` | 0.64B | Q3_K | 3.44 | 0.27 GB |
| `output` | 0.64B | Q5_K | 5.50 | 0.44 GB |
| PLE key projection | 0.03B | Q2_0 | 2.25 | 0.01 GB |
| Not quantized: routers, hyper-connections, indexer, PLE value, delta-net α/β/conv, norms | 0.71B | BF16 (483), F32 (292), F16 (1) | 16.06 | 1.42 GB |
| **Shard 1 total** | **65.31B** | | **2.52** | **20.56 GB** |
| n-gram embedding `per_layer_token_embd` (shard 2) | 51.20B | IQ4_NL | 4.50 | 28.80 GB |

GSQ settings (all quantized tensors; the paper's Table 8 schedule):

| | |
|---|---|
| Initialization | GPTQ, damping 0.1, block 128 |
| Schedule | 10 epochs × 64 steps, last epoch kept |
| Batch | 4 sequences × 6 accumulation steps |
| Gumbel-softmax temperature | 2.0 → 0.05 |
| Optimizer | Lion, betas (0.9, 0.95), weight decay 1.0, cosine to 0.1× |
| Learning rates | grid logits 2e-4; scales 1e-5 (routed experts), 7.4e-4 relative to the GPTQ scale (other tensors) |
| Routed experts | 32 experts per chunk |
| `token_embd`, n-gram rows | rows weighted by how often the calibration set reads them; n-gram rows it never reads are rounded to nearest from the checkpoint |
| `output` | from the head's inputs |

## Evaluation

llama.cpp (HIP build, Radeon AI PRO R9700). ISTA = ISTA-DASLab's unpruned GSQ-RCO Q2_0 GGUF.

Perplexity (2,048-token windows, lower is better):

| Text | ISTA (512 experts) | This model |
|---|---|---|
| Korean (held-out calibration rows) | 4.41 | **4.25** |
| Code / tool use (held-out) | **2.66** | 2.81 |
| wikitext-2 test (40 windows) | **3.83** | 6.72 |

HumanEval (lm-eval `humaneval_instruct`, greedy, thinking off):

| | ISTA | This model |
|---|---|---|
| repeat_penalty 1.1 | 93.9 | 87.2 (12 empty answers) |
| repeat_penalty 1.0 | — | 90.2 |

## Limitations

- General English and science knowledge suffered most from the pruning (wikitext-2 perplexity +75% over ISTA).
  On GPQA-style questions the model often cannot recall basic constants or reagents and keeps thinking until the
  token limit.
- A multiplicative repetition penalty (e.g. 1.1) makes replies stop early, right after a heading or before a code
  body; use DRY against loops instead.

## Usage (llama.cpp)

Tested with upstream llama.cpp at commit `d8d9297` (2026-10-20). The n-gram table stays on the CPU (memory-mapped):

```sh
llama-server -m Qwen3.8-Flash-Next-GSQfull-gumbel256-RCOtypes-00001-of-00002.gguf \
  -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 131072 \
  -ot 'per_layer_token_embd\.weight=CPU' \
  --temp 1.0 --top-p 0.95 --top-k 20 --repeat-penalty 1.0 \
  --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 \
  --logit-bias 248046-2
```

The whole model fits a 32 GB GPU; on a 12 GB GPU put the experts of 32 layers on the CPU (`-ncmoe 32`).
Thinking is on by default (reasoning effort xhigh); `--chat-template-kwargs '{"enable_thinking": false}'` turns it
off. `--logit-bias 248046-2` lowers `<|im_end|>`, which the pruned model otherwise emits too early.

## License and data

Derived from Qwen3.8-Flash-Next under the [Qwen Community License 1.0](LICENSE). The tensor-type allocation and the
unquantized tensors come from ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (Apache-2.0). The calibration data includes
outputs of Claude (Anthropic) from the author's own coding sessions and the original model's agent traces from
Vemloir/Qwen3.8-Flash-Next-traces: this model contains information from
[Vemloir/Qwen3.8-Flash-Next-traces](https://huggingface.co/datasets/Vemloir/Qwen3.8-Flash-Next-traces), which is made
available under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/).
