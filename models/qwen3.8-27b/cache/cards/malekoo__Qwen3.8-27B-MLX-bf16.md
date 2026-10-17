---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- qwen3_5
- conversational
---

<!-- base_model_relation is set explicitly: with no quantization key in
     config.json the Hub would default this repo into the base model's
     "Finetunes" tree, which is wrong (mlx-community/Qwen3.6-27B-bf16 sits
     there today). "quantized" is the Hub's bucket for inference-format
     conversions — GGUF F16 repos live there too.

     UPLOAD NOTE: mlx_lm.convert writes its own stub README.md into the
     output directory — replace it with this card when uploading. -->

# Qwen3.8-27B — MLX, bf16

This model was converted to MLX format from
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) using
[mlx-lm](https://github.com/ml-explore/mlx-lm).
Refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B)
for more details on the model.

The conversion is lossless for the language model (~54 GB on disk): every
weight is bit-identical to the source, except the RMSNorm gains, which are
stored in the mathematically equivalent `1 + w` form that the reference
computes at runtime (verified exact: embeddings, lm_head, linear and A_log
tensors bit-identical; every checked norm exactly `source + 1.0`). No
quantization.

**This conversion is text-only.** The base model is a native vision-language
model; mlx-lm's `qwen3_5` implementation strips the vision encoder
(`vision_tower.*`) and the multi-token-prediction drafter (`mtp.*`) at load
time, so this artifact does not accept images or videos and does not do
speculative decoding. For vision use, run the original checkpoint with
Transformers, vLLM, or SGLang. (The `pipeline_tag` here is `text-generation`
for that reason, while the base card is `image-text-to-text`.)

## Model description (inherited from the base card)

Qwen3.8-27B is the compact dense model of the Qwen3.8 generation, built on the
Qwen3.5 architecture:

- 27B parameters, 64 layers: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated
  Attention → FFN)) — a 3:1 hybrid of linear attention and full attention
- Gated DeltaNet: 48 V-heads / 16 QK-heads, head dim 128
- Gated Attention: 24 Q-heads / 4 KV-heads, head dim 256, RoPE on 64 dims
- Vocab 248,320; context 262,144 native (extensible to 1M — extended-context
  serving beyond 256K is not configured in this conversion)
- Thinking mode on by default, `reasoning_effort` control
  (`xhigh` default / `medium` / `low`), `preserve_thinking` on by default

See the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B) for
benchmark results. Text benchmarks apply to this conversion in principle
(identical bf16 numerics); vision benchmarks do not (no vision tower).

## Use with mlx-lm

```bash
pip install -U mlx-lm
```

```bash
mlx_lm.generate --model malekoo/Qwen3.8-27B-MLX-bf16 --prompt "Explain KV caches briefly." --max-tokens 2048
```

```python
from mlx_lm import load, generate

model, tokenizer = load("malekoo/Qwen3.8-27B-MLX-bf16")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain KV caches briefly."}],
    add_generation_prompt=True,
)
text = generate(model, tokenizer, prompt=prompt, max_tokens=2048, verbose=True)
```

OpenAI-compatible server:

```bash
mlx_lm.server --model malekoo/Qwen3.8-27B-MLX-bf16 --port 8080
```

In requests, set `"model"` to the id reported by `/v1/models` (the model path)
— mlx-lm treats an unrecognized name as a Hub repo to download.

Thinking control goes through `chat_template_kwargs`, as on the base model:
`enable_thinking` (default true), `preserve_thinking` (default true), and
`reasoning_effort` (`xhigh` default, `medium`, `low`) — all verified effective
against this repo's chat template. The generation prompt ends inside an open
`<think>` block (thinking mode), so reasoning-channel splitting requires a
parser aware of the pre-opened tag.

## Recommended sampling (from the base card)

- Thinking mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`,
  `presence_penalty=0.0`, `repetition_penalty=1.0`
- Instruct (non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`,
  `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`

Qwen notes the model operates in thinking mode by default, emitting
`<think>...</think>` before the answer; mlx-lm's server surfaces this as the
reasoning channel.

## Conversion provenance

- Source: `Qwen/Qwen3.8-27B` (bf16 safetensors, released 2026-08-14)
- Converted with `mlx_lm.convert` (no `-q`); the vision encoder (333 tensors)
  and MTP drafter (15 tensors) are dropped, the remaining 851 text tensors
  verified as described above
- Tokenizer re-serialized by transformers 5; verified id-identical to the
  source tokenizer on text, code, CJK, emoji, and chat-template inputs
- Toolchain: mlx `0.32.1.dev20260814+3d23f7d87` (built from ml-explore/mlx
  main), mlx-lm at ml-explore/mlx-lm main `254d153` (conversion code path
  verified identical to upstream main)
- Hardware: Apple silicon (macOS), Metal backend
- Measured on this build: wikitext-2 (test) perplexity **6.9352** — disjoint
  2048-token windows, no overlap, single token stream. Decode ~10 tok/s on an
  M5 Max MacBook Pro at ~54 GB peak memory. A 4-bit quantization measured in
  the identical harness is at
  [malekoo/Qwen3.8-27B-MLX-4bit](https://huggingface.co/malekoo/Qwen3.8-27B-MLX-4bit)
  (ppl 7.0871, ~15 GB, ~33 tok/s).

## License and attribution

Apache-2.0, inherited from the base model. Copyright the Qwen team; this
repository is a format conversion and claims no additional rights.

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
