---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
tags:
- quantized
- compressed-tensors
- gptq
- w4a16
- vllm
- ampere
- atlas-tuned
quantized_by: aleada
license: other
license_name: see-base-model-license
license_link: https://huggingface.co/Qwen/Qwen3.8-27B
---

<img src="assert-logo-light.png" alt="ASSERT" height="26">

# Qwen3.8-27B-W4A16-atlas-tuned

A **W4A16** quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced with
[llm-compressor](https://github.com/vllm-project/llm-compressor) inside a
reproducible container, in the `compressed-tensors` format that vLLM loads
first-class. It carries a live **MTP** (multi-token-prediction) speculative
head, grafted back at source precision after quantization.

It differs from a plain quantization of the base model in one deliberate
way, described next.

## What "atlas-tuned" means here

**The weights were edited before quantization — this is not a straight
quantization of the base model.** Specifically, a single direction in the
model's activation space was projected out of the weights that write to the
residual stream at one intermediate layer (layer 57 of 64) and out of the
token-embedding matrix:

```
W' = W - r (rᵀ W)      # r is a unit vector in hidden space
```

This is a rank-1 edit applied to those matrices only; everything orthogonal
to `r` is untouched, and the other 63 layers' residual-writing weights are
left as they were. The direction `r` was estimated from the difference of
the model's mean hidden state over two small prompt sets.

**What this means for you, stated plainly:**

- It is a **good, working general model** — coherent, fluent, and it keeps
  the base model's knowledge (it answers the same factual questions
  correctly).
- Its behaviour stays **close to the base model**: it still declines the
  same broad categories of requests the base model declines. It is **not**
  an "uncensored" or safety-stripped model.
- **Measured on GSM8K** (full test set, 1319 questions, 5-shot, the same
  harness and machine for both packs): this pack scored **90.0%**
  (1187/1319) versus a plain W4A16 quantization of the same base at
  **87.8%** (1158/1319) — a **small edge of about +2.2 points**, at the
  margin of statistical significance. It is **on par with the plain
  quantization, not dramatically better.** An early 200-question probe had
  suggested a much larger gap; the full run did not bear that out, which is
  why the honest figure here is small. The scoring uses a lenient
  last-number extraction, so these absolutes run higher than a stricter
  harness (e.g. lm-eval) would report — the comparison *between the two
  packs* is the meaningful part, not the absolute values.

If you want the base model's behaviour with none of this edit, use a plain
quantization of `Qwen/Qwen3.8-27B` instead.

## Quantization

- **Scheme:** W4A16 (4-bit weights, 16-bit activations), GPTQ via
  llm-compressor, `compressed-tensors` on disk.
- **Left at higher precision:** embeddings, the output head, the norms, and
  the vision tower are not 4-bit — the honest "fraction actually 4-bit" is
  a fraction of the bytes, not all of them.
- **MTP head:** grafted back at source precision after quantization and
  exempted from the quant config, so vLLM's speculative decoding finds a
  real head rather than a randomly-initialised one.
- vLLM auto-detects the quantization config from the embedded
  `config.json` at load time; no `--quantization` flag is needed.

## Serving (vLLM)

```bash
vllm serve aleada/Qwen3.8-27B-W4A16-atlas-tuned \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --reasoning-parser qwen3
```

`--mamba-cache-mode align` is required, not tuning: this model's gated
delta-net layers make prefix caching select a cache mode the MTP path does
not implement, and the two must agree.

## License

Inherits the license of the base model. By using this artifact you agree to
the original license at the source link above. Produced by **atlas**, a
self-hosted model project.
