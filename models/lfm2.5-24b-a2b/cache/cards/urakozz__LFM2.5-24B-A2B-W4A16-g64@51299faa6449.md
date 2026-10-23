---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
- autoround
- gptq
- w4a16
- int4
- moe
- mixture-of-experts
- lfm2
- liquid
- quantized
---

# LFM2.5-24B-A2B — W4A16 (AutoRound GPTQ)

4-bit weight-only (W4A16) [AutoRound](https://github.com/intel/auto-round) GPTQ quantization of
[LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), a 24B-total / ~2.45B-active-parameter
hybrid (short-convolution + attention) Mixture-of-Experts model post-trained for agentic and tool-use workloads,
with 131K-token context.

This is an unofficial, community-produced quantization. All credit for the base model goes to Liquid AI — see the
[base model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for details and citation.

> [!IMPORTANT]
> **License:** this is a Derivative Work of LFM2.5-24B-A2B and is distributed under the
> **LFM Open License v1.0** (copy in [`LICENSE`](LICENSE)). It is Apache-2.0-shaped with one addition —
> **§5 Commercial Use Limitation**: commercial use is licensed only to legal entities below USD 10 million annual
> revenue. Read the license before use.
>
> **Modification notice (§4):** the weights of `LiquidAI/LFM2.5-24B-A2B` (revision `3fcfdbfa`) were quantized to
> int4 (group size 64, symmetric) with AutoRound, and the per-expert projections were unfused from transformers' fused
> layout. Configuration, tokenizer and chat template are unchanged apart from the added `quantization_config`.

## Model Details

| | |
|---|---|
| Base model | [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) (revision `3fcfdbfa`) |
| Architecture | `Lfm2MoeForCausalLM` (`model_type: lfm2_moe`, native in transformers ≥ 5.x) |
| Parameters | 24.0B total, 2.45B active per token |
| Layers | 40: 30 short-convolution + 10 full-attention (GQA, 32 query / 8 KV heads) |
| FFN | layers 0–1 dense (intermediate 11776); layers 2–39 MoE: 64 experts, top-4, expert intermediate 1536, no shared expert |
| Router | sigmoid with a selection-only expert bias |
| Hidden size / vocab | 2048 / 128,000 (tied embeddings) |
| Context length | 131,072 tokens |

## Quantization Details

Quantized with [Intel AutoRound](https://github.com/intel/auto-round) on one Intel Arc Pro B70 (XPU), ~75 minutes.

| Setting | Value |
|---|---|
| AutoRound | `intel/auto-round` main `710dcaef` (reports 0.17.0) + two local fixes, see below |
| Scheme | W4A16 (4-bit weights, 16-bit activations) |
| Format | GPTQ (`auto_round:auto_gptq`) |
| Group size | 64 |
| Symmetric | Yes |
| Iterations | 200 (sign-SGD) |
| Calibration | [`opencode-instruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), 512 samples × 2048 tokens |
| Model dtype | bf16 |
| Quantized (int4) | every expert projection (`experts.E.{gate,up,down}_proj`, unfused), dense FFN, conv `in_proj` / `out_proj`, attention `q/k/v/out_proj` — 7,402 linears |
| Kept bf16 | routers (`feed_forward.gate`, listed in `extra_config`), expert bias, depthwise conv kernels, norms, embedding / tied `lm_head` |
| Size | 13.3 GB (12.8 GB int4 + scales) |

The two local AutoRound fixes (not upstream as of `710dcaef`): the calibration loaders reset the BOS/EOS counter per
sample (without it every sample ending in EOS shortens all later calibration rows), and `WrapperLinear` exposes
`in_features` / `out_features`.

Reproduce with:

```bash
python -m auto_round \
  --model LiquidAI/LFM2.5-24B-A2B \
  --scheme W4A16 \
  --group_size 64 \
  --format auto_round:auto_gptq \
  --model_dtype bf16 \
  --iters 200 \
  --nsamples 512 \
  --seqlen 2048 \
  --dataset opencode-instruct \
  --low_gpu_mem_usage \
  --output_dir <output_dir>
```

**Export check:** all 7,402 int4 linears present with the expected expert naming; max relative dequantization error
0.103 over 64 sampled tensors (typical for int4 g64); the untouched tensors bitwise equal to the base checkpoint.

## Benchmarks

Not measured yet — accuracy (KL / top-1 against the bf16 original, tool-call scoring) and speed numbers will be added
here once they are run.

## Usage

Recommended generation settings (from the base model): `temperature=0.1`, `top_k=50`, `repetition_penalty=1.05`.
LFM2.5 always reasons before answering and writes Pythonic tool calls between `<|tool_call_start|>` and
`<|tool_call_end|>`; see the base model card for the chat and tool-use format.

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "urakozz/LFM2.5-24B-A2B-W4A16-g64-AutoRound-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
```

Loading GPTQ / AutoRound weights in transformers needs a GPTQ-capable backend (e.g. `auto-round` or `gptqmodel`).

### vLLM

Not yet tested with this checkpoint; vLLM's GPTQ path and its `lfm2_moe` support are the expected route
(`vllm serve urakozz/LFM2.5-24B-A2B-W4A16-g64-AutoRound-GPTQ --quantization gptq`).

## Limitations

- Accuracy against the bf16 original has not been measured yet.
- The calibration set is code-focused (OpenCodeInstruct), not the base model's agentic data.
- Not recommended for knowledge-heavy tasks on its own (inherited from the base model).

## License

[LFM Open License v1.0](LICENSE), inherited from the base model — including §5's commercial-use threshold. This
repository is a modified Derivative Work (int4 quantization, see the modification notice above).
