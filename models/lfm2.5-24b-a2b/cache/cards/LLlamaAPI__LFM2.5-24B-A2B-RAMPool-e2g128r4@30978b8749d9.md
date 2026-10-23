---
library_name: mlx
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
tags:
- mlx
- liquid
- lfm2.5
- moe
- quantized
- 2-bit
- 4-bit
- mixed-precision
- rampool
---

# LFM2.5-24B-A2B-RAMPool-e2g128r4

A mixed-precision MLX quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) built for **RAMPool**, an app that runs LLMs too big for one device by pooling the memory of a Mac and an iPhone, iPad or second Mac. The 24B-parameter MoE is squeezed into **~7.1 GB**, so it can be split across devices (or run on a single Mac), while everything outside the routed experts stays at 4-bit.

The routed experts hold the bulk of the parameters, so they get the most aggressive treatment (2-bit). The small, quality-critical parts of the network (attention, convolution blocks, dense layers, router and embeddings) stay at 4-bit.

## Quantization recipe

All quantization is affine, applied per-layer as recorded in `config.json`.

| Component | Bits | Group size |
| --- | :---: | :---: |
| Routed experts (`switch_mlp.gate_proj` / `up_proj` / `down_proj`, layers 2-39) | **2** | **128** |
| Attention (`q_proj`, `k_proj`, `v_proj`, `out_proj`) | 4 | 64 |
| Double-gated short-conv blocks (`in_proj`, `out_proj`) | 4 | 64 |
| Dense MLPs (layers 0-1) | 4 | 64 |
| MoE router (`feed_forward.gate`) | 4 | 64 |
| Token embeddings (tied with output head) | 4 | 64 |

**Reading the name:** `e2g128` = 2-bit experts, group size 128; `r4` = <!-- TODO: confirm what r4 stands for, e.g. "rest at 4-bit" or "router at 4-bit" -->. `RAMPool` = the pooling app this checkpoint targets (see below).

On disk: two safetensors shards (5.4 GB + 1.7 GB). Quantization is stored as a per-layer `quantization` map in `config.json` (MLX mixed-precision format), and experts are stacked (`switch_mlp`), which is the layout RAMPool's `lfm2_moe` loader expects.

## Use with RAMPool

RAMPool does pipeline-parallel inference over MLX. The host device (where you chat) keeps the token embedding, the first and last transformer blocks and the output head; devices that lend memory each hold a contiguous block of middle layers plus their KV cache. Per token, the host sends a small hidden-state tensor to each worker and gets the result back. Each device downloads only the tensors for its own layers via HTTP range requests, so no device needs the whole 7.1 GB.

1. On the lending device (e.g. iPhone): **Pool → Offer memory**, note the 6-digit PIN and keep the app open.
2. On the host (e.g. Mac): **Pool → Connect**, enter the PIN.
3. **Models**: enter `LLlamaAPI/LFM2.5-24B-A2B-RAMPool-e2g128r4`. The planner shows whether it fits on one device or how it splits. Then **Download & load** and **Chat**.

Once a model is loaded, RAMPool also serves an OpenAI-compatible `POST /v1/chat/completions` and an Anthropic-compatible `POST /v1/messages` endpoint (streaming and tool calling). <!-- TODO: link to the RAMPool repo/release here. -->

Pooling adds memory capacity, not speed: each token pays a network round trip per worker, and a request runs on one device at a time.

## Base model

LFM2.5-24B-A2B is Liquid AI's reasoning model for on-device agents. Architecture details (unchanged by quantization):

- 24.0B total parameters, ~2.45B active per token
- 40 layers: 30 double-gated short-convolution blocks + 10 GQA attention layers
- 64 experts per MoE block, top-4 routing (first 2 layers are dense)
- 131,072-token context, 128K vocabulary
- Always thinks before answering (the chat template opens a `<think>` block)
- Languages: English, Arabic, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Spanish, Vietnamese, Thai, Indonesian, Hindi, Russian, Polish

See the [original model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for capabilities, benchmarks and training details.

## Use with mlx-lm

The checkpoint is a standard MLX mixed-precision model, so it also loads with `mlx-lm` outside RAMPool. Requires `mlx-lm>=0.32.0` (per the upstream MLX release; earlier versions route the experts incorrectly).

```bash
pip install -U "mlx-lm>=0.32.0"
```

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

model, tokenizer = load("LLlamaAPI/LFM2.5-24B-A2B-RAMPool-e2g128r4")

messages = [{"role": "user", "content": "What is C. elegans?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

sampler = make_sampler(temp=0.1, top_k=50)
logits_processors = make_logits_processors(repetition_penalty=1.05)

response = generate(
    model, tokenizer, prompt=prompt, max_tokens=4096,
    sampler=sampler, logits_processors=logits_processors,
    verbose=True,
)
```

Or from the command line:

```bash
mlx_lm.generate --model LLlamaAPI/LFM2.5-24B-A2B-RAMPool-e2g128r4 \
  --prompt "What is C. elegans?" --temp 0.1 --top-k 50 --max-tokens 4096
```

### Recommended generation settings

These match the upstream recommendation and the bundled `generation_config.json`: `temperature=0.1`, `top_k=50`, `repetition_penalty=1.05`.

### Chat template and tool use

The chat template is ChatML-like and unchanged from the base model. Tool calling works as upstream: pass `tools=` to `tokenizer.apply_chat_template`, and the model emits Pythonic calls between `<|tool_call_start|>` and `<|tool_call_end|>`. See the [Liquid AI tool-use docs](https://docs.liquid.ai/lfm/key-concepts/tool-use).

## Limitations

- **Quality loss from 2-bit experts.** Quantizing routed experts to 2 bits is lossy. Expect some degradation versus the BF16 original and versus the standard 4-bit MLX build ([LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit)), most visibly on long reasoning chains, precise code and exact-recall tasks. <!-- TODO: add measured results (perplexity / benchmark deltas) if you have them. -->
- **Router and embeddings are quantized.** The upstream 4-bit build keeps MoE routing gates unquantized and embeddings at 6-bit; here both are 4-bit (group size 64), which can shift expert selection slightly.
- **Pooling trade-offs.** With RAMPool, throughput is bounded by link round trips, and the lending iOS app must stay in the foreground. Without expert paging (`expertSlots`), all experts of a device's slice must fit in that device's memory. <!-- TODO: say whether this checkpoint was prepared/tested for expert paging; RAMPool's docs note paging works well only for models trained for it. -->
- Inherits the base model's limits: it is not recommended for knowledge-heavy tasks on its own, so pair it with retrieval or web search.
- Text-only; Apple Silicon / MLX only.

## License

Released under the LFM Open License v1.0, the same license as the base model. See the [upstream license](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE) for terms. This repository is an unofficial community quantization and is not affiliated with or endorsed by Liquid AI.

## Credits and citation

All credit for the base model goes to [Liquid AI](https://www.liquid.ai). If you use this model, please cite the original work:

```bibtex
@article{liquidAI202624BA2B,
  author  = {Liquid AI},
  title   = {LFM2.5-24B-A2B: Built for Fast On-Device Agents},
  journal = {Liquid AI Blog},
  year    = {2026},
  note    = {www.liquid.ai/blog/lfm2-5-24b-a2b},
}
```

```bibtex
@article{liquidai2025lfm2,
  title   = {LFM2 Technical Report},
  author  = {Liquid AI},
  journal = {arXiv preprint arXiv:2511.23404},
  year    = {2025}
}
```
