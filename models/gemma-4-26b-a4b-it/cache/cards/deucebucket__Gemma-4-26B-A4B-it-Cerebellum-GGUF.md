---
license: gemma
library_name: gguf
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
model_name: Gemma-4-26B-A4B-it-Cerebellum-v5-GGUF
model_creator: google
model_type: gemma4
quantized_by: deucebucket
pipeline_tag: text-generation
tags:
  - GGUF
  - gemma4
  - gemma
  - google
  - quantized
  - cerebellum
  - imatrix
  - moe
  - 3-bit
  - conversational
model-index:
- name: Gemma-4-26B-A4B-it-Cerebellum-GGUF
  results:
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: AI2 Reasoning Challenge
      type: ai2_arc
      config: ARC-Challenge
      split: test
    metrics:
    - name: normalized accuracy
      type: acc_norm
      value: 0.954
    source:
      name: Local benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-GGUF/tree/main
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: HellaSwag
      type: hellaswag
      split: validation
    metrics:
    - name: accuracy
      type: acc
      value: 0.847
    source:
      name: Local benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-GGUF/tree/main
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: MMLU-Redux
      type: cais/mmlu
      config: all
      split: test
    metrics:
    - name: accuracy
      type: acc
      value: 0.714
    source:
      name: Local benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-GGUF/tree/main
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: HumanEval (pass@1)
      type: openai_humaneval
      split: test
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.713
    source:
      name: Local benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-GGUF/tree/main
  - task:
      name: Text Generation
      type: text-generation
    dataset:
      name: WikiText-2 Perplexity
      type: wikitext
      config: wikitext-2-raw-v1
      split: test
    metrics:
    - name: perplexity
      type: perplexity
      value: 12356
    source:
      name: Local benchmark run (RTX 3090, llama.cpp)
      url: https://huggingface.co/deucebucket/Gemma-4-26B-A4B-it-Cerebellum-GGUF/tree/main
---

<p align="center">
  <img src="cerebellum_banner.png" alt="Cerebellum" width="640">
</p>

# Gemma 4 26B-A4B-it -- Cerebellum GGUF

Current release: **v5**.

Cerebellum v5 is an ablation-guided mixed-precision GGUF quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

This is a 26B-parameter MoE model with 4B active parameters per token and 128 experts per layer. This release uses tensor-level precision overrides selected from 110+ ablation experiments across five internal iterations.

## At a Glance

- **Current file**: `Gemma-4-26B-A4B-it-Cerebellum-Q3_K_M.gguf`
- **Current release**: v5
- **File size**: 11 GB
- **Base model**: `google/gemma-4-26B-A4B-it`
- **Format**: GGUF, mixed precision, imatrix-guided
- **Test hardware**: RTX 3090, llama.cpp

## Evaluation

Measured on RTX 3090 with llama.cpp at temperature 0. HumanEval below is the
legacy raw-completions number; for Gemma 4 the current protocol is the chat-completions
harness (raw completions understate Gemma 4 coding), so treat it as legacy. Summary JSONs
are in `benchmark_results/`.

| Benchmark | v5 Result |
|-----------|:---------:|
| WikiText PPL | 12,356 |
| HumanEval pass@1 | 71.3% |
| ARC-Challenge | 95.4% |
| HellaSwag | 84.7% |
| MMLU-Redux | 71.4% |

All benchmark results above were measured on the same local setup: RTX 3090 and llama.cpp. PPL was measured on the WikiText-2 test set with 2048 context and 128 chunks.

PPL is high in absolute terms for this model and appears consistent across Gemma 4 26B quant levels tested locally.

### Version History

| Version | PPL | HumanEval | What Changed |
|---------|-----|-----------|-------------|
| v1 | 20,614 | 65.2% | Group-level ablation only |
| v2 | 19,826 | 65.9% | + attn_q layer ablation (30 layers) |
| v3 | 19,826 | 67.1% | + PLE (norms/scales) protection |
| v4 | 12,614 | 69.5% | + ffn_up layer ablation + precision rebalance |
| **v5** | **12,356** | **71.3%** | + attn_k layer ablation (30 layers, 7 promoted) |

## Usage

```bash
llama-server -m Gemma-4-26B-A4B-it-Cerebellum-Q3_K_M.gguf \
  -ngl 99 -c 24576 --jinja --reasoning auto
```

Gemma 4 requires `--jinja`. For non-thinking output, pass request-level
`chat_template_kwargs: {"enable_thinking": false}` and `thinking_budget_tokens: 0`;
do not set a fixed server `--reasoning-budget` (it can burn output into hidden
reasoning until the length cap, which looks like a repetition loop).
Fits in 24 GB VRAM at full GPU offload.

## How Cerebellum Works

Cerebellum uses ablation testing to assign different quantization types to different tensor groups and layers.

### Level 1: Group Ablation

Each tensor category was tested at Q2_K and measured by perplexity impact:

| Group | Tensors | PPL Delta | Action |
|-------|---------|-----------|--------|
| attn_q | 30 | +13.4% | Tested per layer |
| ffn_gate | 30 | -1.2% | Left at Q3_K |
| expert_gate_up | 30 x 507M | -5.5% | Set to Q2_K |
| attn_k | 30 | -12.1% | Tested per layer |
| ffn_up | 30 | -18.2% | Set to Q2_K |

Three of five tested groups had lower measured PPL at Q2_K in this setup.

### Level 2: Layer Ablation (attn_q)

The `attn_q` group had a +13.4% PPL delta at Q2_K. All 30 layers were then tested individually. No individual layer exceeded +3.9%, so v5 keeps 9 of 30 layers at Q5_K.

### Level 3: Reverse Layer Ablation (attn_k)

v4 had all `attn_k` layers at Q2_K. Each layer was tested individually at Q3_K:

- 4 layers (5, 11, 16, 29) had higher PPL at Q3_K and stayed Q2_K.
- 7 layers (1, 6, 17, 18, 23, 24, 28) had lower PPL at Q3_K.
- Layer 23 changed by -3.8%; layer 18 changed by -2.8%.

Those 7 layer changes are included in v5.

### Router/Gate Investigation

We also tested the MoE routing tensors:

| Test | What | Result |
|------|------|--------|
| Crush all routers (F32 to Q2_K) | 30 ffn_gate_inp tensors | +30.7% PPL |
| Promote all gates (Q3_K to Q6_K) | 30 ffn_gate tensors | +36.1% PPL |

The v5 precision map leaves router tensors at F32 and gate tensors at Q3_K.

### Final Precision Map (v5)

| Tensor Type | Precision | Count | Rationale |
|-------------|-----------|-------|-----------|
| attn_q (9 sensitive layers) | Q5_K | 9 | Layer-validated critical |
| attn_k (7 promoted layers) | Q3_K | 7 | Reverse ablation: improve when promoted |
| attn_k (remaining) | Q2_K | 23 | Group-level demotable |
| ffn_up | Q2_K | 30 | Group PPL delta: -18.2% |
| expert_gate_up | Q2_K | 30 | Demotable (-5.5%) |
| ffn_gate | Q3_K | 30 | Tolerant (-1.2%) |
| ffn_gate_inp (router) | F32 | 30 | Group PPL delta: +30.7% when set to Q2_K |
| Norms, scales | F32 | 392 | Structural -- always full precision |

**91 tensor-level overrides** on top of Q3_K_M base with [bartowski's imatrix](https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF).

## Technical Details

- **Architecture**: Gemma 4 26B -- 26B total params, 4B active per token, 128 experts/layer, 30 layers
- **Base quant**: Q3_K_M with bartowski imatrix
- **Ablation experiments**: 110+ total across 5 iterations
- **Quantizer**: llama.cpp `llama-quantize` with `--tensor-type-file` overrides
- **Hardware**: RTX 3090 (24 GB VRAM)

## Credits

- **Base model**: [Google Gemma Team](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Imatrix**: [bartowski](https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF)
- **Method & quantization**: [deucebucket/cerebellum](https://github.com/deucebucket/cerebellum) -- Cerebellum pipeline
