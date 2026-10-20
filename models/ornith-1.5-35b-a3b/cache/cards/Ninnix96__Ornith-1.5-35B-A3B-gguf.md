---
base_model: ornith-ai/Ornith-1.5-35B-A3B
license: mit
pipeline_tag: text-generation
tags:
  - gguf
  - moe
  - quantized
  - antirez-recipe
---

# antirez-recipe quants

MoE quants following the [antirez recipe](https://huggingface.co/antirez/deepseek-v4-gguf):
routed experts quantized hard, everything else protected.

| Tensor class | Quant | Notes |
| --- | --- | --- |
| `blk.*.ffn_gate_exps`, `blk.*.ffn_up_exps` (`ffn_gate_up_exps` if fused) | **GU_TYPE** (imatrix) | routed-expert gate/up |
| `blk.*.ffn_down_exps` | **DOWN_TYPE** (imatrix) | routed-expert down (K-quant for quality) |
| `blk.*.ffn_{gate,up,down}_shexp`, all `attn_*`, `output.weight` | `Q8_0` | shared experts / attention / head |
| `token_embd.weight` | `F16` | embeddings |
| router (`ffn_gate_inp`), norms, sinks, biases | source precision (`BF16`/`F32`) | never quantized by llama.cpp |

Base type `Q8_0` + `--tensor-type` overrides, imatrix calibration
(see SPEC per-model imatrix links). Runs on upstream llama.cpp — no fork or
patch needed.

Validation per file: `gguf-dump` tensor audit (experts = GU/DOWN, protection intact),
`llama-cli` smoke load, `llama-perplexity` spot ppl + `--kl-divergence` vs BF16 logits.

## Files

| File | Size | SHA-256 |
| --- | ---: | --- |
| `Ornith-1.5-35B-A3B-Q2-IQ2_XXS-Q2_K.gguf` | 14215041696 | `f6848c148c1d4644f103c0e1b735545c262f9779f83e9489b95235fadb345916` |

Imatrix: https://huggingface.co/mradermacher/Ornith-1.5-35B-A3B-i1-GGUF

## Recipe

Antirez recipe on ornith-ai/Ornith-1.5-35B-A3B (BF16 source, ornith-ai/Ornith-1.5-35B-A3B):

- `ffn_gate_exps` / `ffn_up_exps` / `ffn_gate_up_exps` -> IQ2_XXS
- `ffn_down_exps` -> Q2_K
- attention + shared experts + repr-MTP blocks blk.38/39/40 + output -> Q8_0
- token embeddings -> F16
- imatrix: mradermacher/Ornith-1.5-35B-A3B-i1-GGUF

Notes: mradermacher's imatrix does not cover the MTP/draft blocks (blk.38-40); they are forced to Q8_0 with a first-position llama-quantize rule (first matching regex wins). ~14.2 GB, about 2.9 BPW.

## Validation (llama.cpp, qwen35moe arch)

- 12-case hard-smoke eval (temp 0, ctx 15360, model's own embedded chat template): **9/12 PASS**; ~27 t/s generation (AMD Vulkan/UMA); fails cases 10567, 1606, juliet-cwe121. Passes both big-code cases.
- Exact output tokens (own tokenizer) over the 9 solved: 51,528 = 5,725 tok/solved.
- KL divergence vs official ornith-ai Q8_0 (ds4 short prompts, ctx 60, 2 chunks, CPU): per-chunk KL 0.214 / 0.262 (avg 0.238), delta-p RMS ~10.8%, same-token top-1 86.2 / 81.0%.

