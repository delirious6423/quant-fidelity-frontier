---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- qwen
- qwen3
- multimodal
- vision-language
- gptq
- int4
- w4a16
- compressed-tensors
- llm-compressor
- vllm
---

# Qwen3.8-27B-GPTQ-Int4

4-bit (W4A16) GPTQ quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [LLM Compressor](https://github.com/vllm-project/llm-compressor). The weight
footprint drops to roughly a third of the BF16 release; architecture, tokenizer, chat
template and processor configuration are unchanged.

> [!Important]
> **Unofficial community quantization.** Not produced, reviewed or endorsed by the Qwen
> team or Alibaba Cloud. For the original model and its capabilities, see the
> [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B).

## Evaluation

Paired comparisons against the unquantized base model: same prompts, same settings,
greedy decoding. Higher is better.

| Benchmark | Base (BF16) | This checkpoint |
|---|---|---|
| OmniDocBench, text block | 93.2% | **93.8%** |
| OmniDocBench, reading order | 92.6% | **92.8%** |
| OmniDocBench, display formula | 72.7% | **72.9%** |
| MathVista, visual maths | 85.7% | **88.3%** |
| MMMU, visual reasoning | 81.0% | 81.0% |
| V*Bench, visual perception | 88.7% | 87.0% |
| MMLU-Pro, English knowledge | 86.3% | 85.5% |
| MMLU-DE, German knowledge | 82.7% | 79.5% |
| BFCL, tool calling | 57.7% | **58.0%** |
| IFEval, instruction following | 40.1% | 38.4% |


## What is quantized

400 linear modules carry 4-bit weights: `mlp.{gate,up,down}_proj` across all 64 language
layers, `self_attn.{q,k,v,o}_proj` across the 16 full-attention layers, and
`linear_attn.{in_proj_qkv,in_proj_z,out_proj}` across the 48 linear-attention layers.

Left at BF16:

- **The vision tower** (`model.visual.*`). A structural constraint, not a choice:
  `visual.blocks.*.mlp.linear_fc2` has 4304 input columns and group quantization needs
  `columns % group_size == 0`. 4304 = 16 x 269 with 269 prime, so no practical group
  size divides it. Any 4-bit quantization of this architecture leaves the vision tower
  unquantized.
- `lm_head` and the embedding table. The output projection costs accuracy out of
  proportion to its size; the embedding table is not a linear operator.
- `linear_attn.in_proj_a` and `in_proj_b`, the gated-delta-net gate and decay
  projections. Tiny, and they govern the numerically delicate state-space recurrence.
  Qwen's own FP8 release excludes them too.
- The `mtp` multi-token-prediction head, unused outside speculative decoding.

## Quantization details

| | |
|---|---|
| Scheme | `W4A16`, group size 128, symmetric |
| Algorithm | GPTQ (`GPTQModifier`), not round-to-nearest |
| Calibration | 512 samples from `HuggingFaceH4/ultrachat_200k`, max sequence length 2048 |
| Seed | 0, fixed calibration slice |
| Produced with | llm-compressor 0.13.0, compressed-tensors 0.18.0, transformers 5.14.1 |

Deterministic given the base checkpoint and the recipe in `recipe.yaml`. This
architecture (`model_type: qwen3_5`) requires a recent `transformers`.

## Serving with vLLM

```bash
vllm serve ionos/Qwen3.8-27B-GPTQ-Int4
```

Sizing flags depend on your hardware and workload and are deliberately not prescribed.

On compute capability 8.9 (Ada) this is served by vLLM's `MarlinLinearKernel`. Marlin is
mixed-precision: it unpacks the 4-bit weights to 16 bits in registers and runs the GEMM
on FP16/BF16 tensor cores. No INT4 tensor-core instruction is issued on any
architecture, so the gain is weight-memory bandwidth, not 4-bit arithmetic. The benefit
is largest at low concurrency and narrows as batches grow and compute dominates.
Benchmark your own workload shape before assuming 4-bit beats 8-bit.

> [!Important]
> **You may need to lower `--max-num-seqs`.** Qwen3.8-27B is hybrid: three
> gated-delta-net layers for every full-attention layer. Every decode sequence needs a
> recurrent-state ("Mamba") cache block *in addition to* its KV cache pages, from the
> same memory budget. If `max_num_seqs` exceeds the blocks that fit, the engine refuses
> to start:
>
> ```
> ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (<available>).
> ```
>
> Set `--max-num-seqs` at or below the number it reports, or give the cache more room.
> This ceiling comes from the architecture and memory budget, not from quantization.

### Thinking is on by default

`reasoning_effort` defaults to `xhigh`, so the model may emit several thousand reasoning
tokens before its answer. Allocate output length accordingly, or reduce it per request:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# levels: xhigh (default), medium, low
resp = client.chat.completions.create(
    model="ionos/Qwen3.8-27B-GPTQ-Int4",
    messages=[{"role": "user", "content": "..."}],
    reasoning_effort="low", max_tokens=1024,
)

# or disable entirely
resp = client.chat.completions.create(
    model="ionos/Qwen3.8-27B-GPTQ-Int4",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}, max_tokens=1024,
)
```

Sampling defaults, image input and the 262,144-token context (including YaRN guidance
beyond it) follow the base model card unchanged. The vision tower is unquantized, so
multimodal behaviour should track the base model more closely than the language path.

## Modifications to the original model

As Apache 2.0 Section 4(b) requires, the changes to the original work are:

- The linear operators listed above were converted from BF16 to 4-bit integers with
  group-128 symmetric scales using GPTQ, and saved in the `compressed-tensors`
  `pack-quantized` format. Activations are **not** quantized: this is W4A16, so weights
  are dequantized to 16-bit at inference and the matmul runs in 16-bit.
- `config.json` gained a `quantization_config` block.
- A `recipe.yaml` recording the LLM Compressor recipe was added.
- No other file was altered.

## License

Derivative work of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), released
under the same licence, the **Apache License 2.0** ([`LICENSE`](./LICENSE)).

Apache 2.0 requires no specific attribution phrase and the base model ships no `NOTICE`
file. "Qwen" is used only to identify the original model; no trademark rights are
granted (Section 6) and nothing here implies endorsement.

## Citation

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```
