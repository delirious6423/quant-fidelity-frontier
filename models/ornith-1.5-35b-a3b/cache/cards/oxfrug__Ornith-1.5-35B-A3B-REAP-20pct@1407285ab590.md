---
language:
- en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- moe
- reap
- pruning
- expert-pruning
- qwen3.5
- qwen3.5-moe
- coding
- conversational
- text-generation
- vllm
datasets:
- theblackcat102/evol-codealpaca-v1
- open-r1/Mixture-of-Thoughts
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
widget:
- text: Write a one-line Python lambda that squares a number.
  example_title: Coding
- text: What is the weather in Paris right now?
  example_title: Tool call
---

# Ornith-1.5-35B-A3B-REAP-20pct

Ornith-1.5 coding MoE with **20% of experts gone**. Same 8 active per token, **~56 GB BF16** instead of ~67 GB. [REAP](https://arxiv.org/abs/2510.13999) (Lasby et al., Cerebras, ICLR 2026). **Not lossless.** Not GGUF. Not a 24 GB card.

| You want | Use |
|----------|-----|
| Max quality, full 256 experts | [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| Same family, less disk, vLLM / transformers | **this repo** |
| One 24 GB GPU, ExLlamaV3 | [`oxfrug/Ornith-1.5-9B-exl3`](https://huggingface.co/oxfrug/Ornith-1.5-9B-exl3) |
| LM Studio / Ollama / llama.cpp | no GGUF of this prune yet |

Same recipe on Qwen3.5-35B: [`0xSero/Qwen3.5-28B`](https://huggingface.co/0xSero/Qwen3.5-28B).

## Model details

| | Base | This |
|---|---|---|
| Base model | [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | same hybrid GatedDeltaNet + MoE |
| Relation | — | REAP 20% expert prune |
| Experts / layer | 256 | **205** (51 removed) |
| Active / token | 8 | 8 |
| Layers | 40 | 40 |
| On-disk BF16 | ~67 G | **~56 G** |
| Library | transformers ≥ 5.8.1 | transformers ≥ 5.8.1 |
| License | MIT | MIT |

Shared expert, vision tower, embeddings, and `lm_head` are untouched. Router weights renormalized after prune. `config.json` `text_config.num_experts` is **205**. Architecture: `Qwen3_5MoeForConditionalGeneration`.

## Quickstart

```bash
pip install -U "transformers>=5.8.1"
```

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer

model_id = "oxfrug/Ornith-1.5-35B-A3B-REAP-20pct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

messages = [{"role": "user", "content": "Write a one-line Python lambda that squares a number."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, temperature=0.6, top_p=0.95)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

vLLM (`>=0.19.1`), 2× 80GB recommended for long context:

```bash
vllm serve oxfrug/Ornith-1.5-35B-A3B-REAP-20pct \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

Sampling: general `temperature=0.6`, `top_p=0.95`, `top_k=20`. Ornith is a reasoning model — the assistant turn opens with `<think>…</think>` unless the server splits it into `reasoning_content`.

## Benchmarks

This checkpoint was **not** re-run on SWE-bench / Terminal-Bench / GPQA. Do not copy the base card’s 79 SWE-bench / 67.8 Terminal-Bench numbers onto this prune.

### vs base (this checkpoint)

| | Base | This |
|---|---:|---:|
| Experts / layer | 256 | 205 |
| Disk (BF16) | ~67 G | ~56 G |
| Active params / token | ~3B | ~3B |

### Proxy (same recipe, not this checkpoint)

The same REAP 20% recipe on [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) ([0xSero/Qwen3.5-28B](https://huggingface.co/0xSero/Qwen3.5-28B), greedy, 0-shot) dropped coding a few points and raised WikiText perplexity. Ornith is a Qwen3.5 hybrid (GatedDeltaNet + MoE); expect a **comparable** tax, not these exact numbers.

| Benchmark | Original Qwen3.5-35B | REAP 20% | Delta |
|-----------|---------------------:|---------:|------:|
| HumanEval pass@1 | 76.2% | 73.2% | −3.0 |
| HumanEval+ pass@1 | 72.0% | 70.1% | −1.9 |
| WikiText-2 PPL (10k tok) | 6.83 | 9.51 | +2.68 |

Source: 0xSero’s card, not a run of `oxfrug/Ornith-1.5-35B-A3B-REAP-20pct`.

### Smoke (this checkpoint)

Greedy, thinking off, 3090 offload. Not a published coding bench.

| Prompt | Base | This |
|--------|------|------|
| square lambda / `is_even` | match | match |
| `get_weather` with the [base card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) `tools=` schema | well-formed `<tool_call><function=get_weather>` / `Paris` | same |

## How it was made

Same mix as [0xSero/Qwen3.5-28B](https://huggingface.co/0xSero/Qwen3.5-28B), **not** Cerebras’ 24k×16k agentic mix:

- `theblackcat102/evol-codealpaca-v1` × 250
- `open-r1/Mixture-of-Thoughts` code / math / science × 250 each
- **1000** sequences, max length **1024** (3090 VRAM; 0xSero used 4096)
- seed 42, `prune_method=reap`, `renormalize_router_weights=true`
- Layerwise observer (model does not fit in 60 G RAM)

No xlam / SWE-smith traces — extra risk for Ornith’s tool-calling specialists.

## Intended use

Local / self-hosted coding and tool-calling with Ornith-1.5 quality, at ~11 GB less disk than the unpruned BF16. Accept a quality tax vs the 35B-A3B base.

## Limits

- Expert prune, not a quant and not a fine-tune.
- Shorter calibration sequences (1024 vs 4096) than the Qwen3.5-35B sibling run.
- Hybrid GatedDeltaNet + MoE: REAP only touches MoE experts.
- Do not claim base Ornith-1.5 SWE-bench / Terminal-Bench / GPQA scores for this file.

## License

MIT, same as the base. Prune by [oxfrug](https://huggingface.co/oxfrug).

## Citation

```bibtex
@misc{lasby2025reap,
  title  = {REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression},
  author = {Mike Lasby and Ivan Lazarevich and Nish Sinnadurai and Sean Lie and Yani Ioannou and Vithursan Thangarasa},
  year   = {2025},
  eprint = {2510.13999},
  archivePrefix = {arXiv}
}

@misc{ornith_1_5,
  title  = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
  url    = {https://ornith.ai/ornith_1_5.html},
  author = {{Ornith Team}},
  year   = {2026}
}
```
