---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - omlx
  - oq
  - quantized
  - qwen3.8
---

# Qwen3.8-27B-oQ6e-fp16-mtp

oQ6e mixed-precision MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
made with [oMLX](https://github.com/jundot/omlx) oQe (v0.7.1.dev1). This is the **fp16** build.
The multi-token-prediction (MTP) head is kept and the vision tower is included.

## Variants

The same family at every bit width and dtype (this repo in bold):

| Variant | dtype | Size | KL vs bf16 ↓ | Top-1 agreement ↑ | Perplexity |
|---|---|---|---|---|---|
| oQ4e | bf16 | 17.0 GB | 0.0358 | 90.6% | 10.47 |
| oQ4e | fp16 | 17.9 GB | 0.0351 | 90.5% | 10.47 |
| oQ5e | bf16 | 20.3 GB | 0.0106 | 94.9% | 10.28 |
| oQ5e | fp16 | 21.2 GB | 0.0103 | 94.9% | 10.28 |
| oQ6e | bf16 | 23.7 GB | 0.0033 | 97.2% | 10.23 |
| **oQ6e** | **fp16** | **24.6 GB** | **0.0028** | **97.1%** | **10.22** |
| oQ8e | bf16 | 30.0 GB | 0.0010 | 98.6% | 10.21 |
| oQ8e | fp16 | 30.9 GB | 0.0007 | 98.4% | 10.20 |
| *original* | bf16 | 56 GB | 0 | 100% | 10.20 |

**Which dtype?** `bf16` (no suffix) is the default. `fp16` builds store the language model's
non-quantized weights and the quantization scales as float16 instead of bfloat16. They are marginally closer to the original at every level. oMLX notes
that float16 can give faster prefill on M1/M2, which lack native bfloat16. They are about
0.9 GB larger because the vision tower is kept in float32; vision quality was not measured.

**How these were measured:** mean KL divergence of each quant against the original bf16 model,
over 48 windows x 2048 tokens of English prose, Python code, Japanese and Korean text. Less than
0.3% of this text (60-character chunks) appears in oQe's calibration data. KL is computed on the
reference's top-256 tokens plus one bucket for the remainder, so it slightly underestimates the
full KL; compare numbers from this method only with each other. These are fidelity measurements,
not task benchmarks, and they apply to this 27B model only.

## About oQe

oQe sets a base bit width per level and raises sensitive layers above it (for example, `oQ4e`
uses a 4-bit base with some projections at 5 bits), and chooses each weight group's scale and
bias using an importance matrix from calibration data. These builds include oMLX's weighted
least-squares refit of group scales and biases
([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)). Compared with earlier oQe builds of this model (pre-oMLX 0.7.1 builds), these builds lower KL by 14-16% at every level (for example, oQ4e 0.0428 to 0.0358). The per-layer bit allocation is identical to the earlier builds at every level, so the difference comes from how the weights and scales were fitted, not from which layers got more bits.

- Group size 64, affine quantization
- Calibration: oMLX built-in `oqe_code_multilingual`, 128 samples x 512 tokens
- MTP head: preserved (1 layer); its linear layers are quantized with the rest
- Vision tower: not quantized; weights unchanged apart from the patch-embed conv, which is
  transposed to MLX's channels-last layout

## Chat template

`chat_template.jinja` is **froggeric's v22.5**
([froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates),
Apache-2.0), not Qwen's official template. Qwen's original is included as
`chat_template.jinja.official`.

**Default reasoning effort differs.** Qwen's official 3.8 template adds a system instruction to every chat by default ("Reasoning effort is set to xhigh. Please think carefully through the task..."). The froggeric template defaults to `medium`, which adds nothing, so by default the prompt asks for less extended reasoning than Qwen's template does. Passing `reasoning_effort="xhigh"` makes the two templates produce identical prompts for plain chat and multi-turn chat (`medium` and `low` also match).

Apart from that default, they differ in:
- **Tool calling:** the tool-instruction text is rewritten, and JSON-string tool arguments (as sent
  by OpenAI-style clients) render instead of raising an error.
- **System messages:** several leading system messages are merged instead of erroring.

Tool-calling quality with the modified template has not been evaluated. To use Qwen's template,
replace `chat_template.jinja` with `chat_template.jinja.official`.

## Credits

Base model: Qwen team. Quantizer: oMLX project. Chat template: froggeric.
