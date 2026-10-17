---
library_name: transformers
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
language:
- en
base_model:
- Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
tags:
- qwen
- exl3
- 3inst
- uncalibrated
---

# Qwen3.8-Flash-Next EXL3, 3inst, uncalibrated

> **Uncalibrated. WIP.** No LDLQ. Weights were encoded with Viterbi only, so this is not a quality release. It exists to test the [opengfx1030/vllm-rdna](https://github.com/opengfx1030/vllm-rdna) fork. Do not treat outputs as representative of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

EXL3 trellis of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Team). Codebook `3inst`. Input Hadamard only (`hadamard: in`). The part split follows turboderp's mul1 `3.05bpw` recipe (`-b 3` experts, `-hq -hb 5` on the dense group). 3inst is weaker than mul1, so that dense group is K=6 instead of K=5. `index_qk_proj` stays bf16.

Widths that are not a multiple of 128 are zero-padded to the next block (n-gram 160→256, vision `linear_fc2` 4304→4352). The patch embed is stored as the flattened matrix `(1152, 1536)`.

## What is quantized

| Tensors | Codebook | K |
|---|---|---:|
| Routed experts, including MTP (`mlp.experts.*`) | 3inst | 3 |
| Attention `q/k/v/o_proj` | 3inst | 6 |
| Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj` | 3inst | 6 |
| Shared expert | 3inst | 6 |
| `lm_head` | 3inst | 6 |
| PLE `key_proj`, `value_proj`; MTP `fc_hidden`, `fc_embedding` | 3inst | 6 |
| N-gram table, 128 shards | 3inst | 6 |
| Vision linears (`qkv`, `proj`, `linear_fc1`, `linear_fc2`, merger) and patch embed | 3inst | 6 |
| Embeddings (`embed_tokens`, `pos_embed`), norms, `conv1d`, `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`, hyper-connection, MoE router (`.mlp.gate.`), `index_qk_proj`, biases | bf16 | — |

## Credits

- **Model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Qwen Team. License [Qwen Community 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). Architecture, data, and intended use are on that card.
- **Format and codebook:** [exllamav3](https://github.com/turboderp-org/exllamav3), turboderp. EXL3 is the QTIP trellis (Tseng et al.) with the `3inst` codebook and the Hadamard rotation used by that project. The bit allocation above is that project's mul1 3.05bpw split, shifted up one K on the dense group.
