---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model:
- LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
quantized_by: AtomicChat
pipeline_tag: text-generation
library_name: gguf
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
tags:
- atomic-chat
- liquid
- lfm2.5
- lfm2moe
- moe
- gguf
- imatrix
- llama.cpp
---

# How to Run LFM2.5-24B-A2B Locally
<p style="margin-top: 0; margin-bottom: 0;">
  <em>Self-quantized from Liquid's original weights with our own importance matrix. The <a href="https://huggingface.co/datasets/AtomicChat/LFM2.5-24B-A2B-GGUF-metrics">measurements</a> behind this card are public.</em>
</p>
<div style="display: flex; gap: 8px; align-items: center; margin-top: 10px; margin-bottom: 10px;">
  <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_lfm2_5_24b_a2b&utm_content=btn_atomic"><img src="https://huggingface.co/AtomicChat/LFM2.5-24B-A2B-GGUF/resolve/main/btn_atomic.png" width="162" alt="Atomic Chat"></a>
  <a href="https://discord.gg/8wGSsvmg4V"><img src="https://huggingface.co/AtomicChat/LFM2.5-24B-A2B-GGUF/resolve/main/btn_discord.png" width="119" alt="Discord"></a>
  <a href="https://github.com/AtomicBot-ai/Atomic-Chat"><img src="https://huggingface.co/AtomicChat/LFM2.5-24B-A2B-GGUF/resolve/main/btn_github.png" width="115" alt="GitHub"></a>
  <a href="https://x.com/atomic_chat_hq"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/btn_x.png" width="108" alt="X"></a>
</div>
<ul style="margin: 0 0 12px 0;">
  <li>24B mixture of experts with 2B active per token: the 15.4 GB build fits a 24 GB machine with room for context.</li>
  <li>You can now run LFM2.5-24B-A2B in <a href="https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_lfm2_5_24b_a2b&utm_content=bullet_app">Atomic Chat</a>: 24B total parameters, 2B active per token.</li>
  <li>We checked llama.cpp against Liquid's PyTorch code on the preview's first revision: llama.cpp's BF16 is closer to the exact fp32 model than PyTorch's own bf16 is.</li>
  <li>This model is unusually sensitive to numerical precision. Read the KLD below against its floor, not against other models'.</li>
  <li>Need it without refusals? <a href="https://huggingface.co/AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF">LFM2.5-24B-A2B-Uncensored-GGUF</a> has baked builds and a LoRA adapter for the quants here.</li>
</ul>
<hr style="margin: 0 0 16px 0;">

## The builds

| Build | On disk | Mean KLD | Same top-1 |
|---|---|---|---|
| `BF16` | 48.0 GB | reference | reference |
| `AD-Q8_0` | 25.7 GB | 0.063 | 90.1% |
| `AD-Q6_K` | 20.9 GB | 0.075 | 89.1% |
| `AD-Q5_K_M` | 17.3 GB | 0.108 | 87.4% |
| **`AD-Q4_K_M`** | **15.4 GB** | **0.126** | **85.5%** |
| `AD-IQ4_XS` | 14.5 GB | 0.135 | 85.2% |
| `AD-IQ3_M` | 12.1 GB | 0.223 | 80.2% |
| `AD-IQ3_XXS` | 11.1 GB | 0.286 | 78.2% |

**`AD-Q4_K_M` is the one to take** on a 24 GB machine. With 32 GB, `AD-Q6_K` is
nearly as close to the original as `AD-Q8_0` at 5 GB less. Below 16 GB, the
3-bit builds lose noticeably more; on this model the step from 4 to 3 bits
nearly doubles the divergence.

Every number is against the BF16 model's own logits, measured on one machine.
"Same top-1" is how often the build picks the same next token as the BF16 model.

## Why the numbers look high

This model's output moves when the arithmetic changes, even when the weights do
not. We ran the same BF16 weights several ways, on the tokens and positions of
the reference above. This table was measured on the preview's first revision
(`c715ae25`); the current weights behave the same way at Q8_0 (0.063 against
0.064 then).

| Comparison | Mean KLD | Median KLD | Same top-1 |
|---|---|---|---|
| llama.cpp BF16, the same GPU twice | 0.000 | 0.000 | 100.0% |
| llama.cpp BF16, CPU against GPU | 0.019 | 0.002 | 95.2% |
| llama.cpp BF16 against exact fp32 (PyTorch) | 0.024 | 0.006 | 92.1% |
| Liquid's PyTorch bf16 against exact fp32 | 0.053 | 0.018 | 89.7% |
| our `AD-Q8_0` against llama.cpp BF16 | 0.064 | 0.016 | 89.7% |

Rounding the activations to bf16 alone moves this model by 0.02-0.05 KLD, where
most models move by around a thousandth. A Q8_0 at 0.064 is therefore the floor
here, not a defect of the build. It also means llama.cpp runs this architecture
correctly: its BF16 sits closer to the exact model than Liquid's own bf16 code
path. Compare the builds with each other and with `AD-Q8_0`; comparing these KLD
values with those of other models says more about the models than the quants.

## How these were made

1. **BF16** from Liquid's original safetensors (revision `f772950c`, the
   "agentic checkpoint" of 12 October), converted with llama.cpp `980aef8c`. The source is BF16, so nothing is requantized; the chat template
   is embedded.
2. **Importance matrix** on the BF16 model, never on a quantized proxy: 8,994
   chunks of 512 tokens from our calibration corpus, rendered through this
   model's own chat template, with `--parse-special`. Agentic sessions and tool
   calls 24%, code 18%, multilingual text 20%, reasoning 13%, long documents
   11%, structured data 4%, graphics code 4%, vocabulary sweep 6%. Every one of
   the 64 experts in all 38 MoE blocks saw at least 12.3k tokens, so no expert
   is quantized blind.
3. **Layout.** The rungs differ in the experts, which are nearly all of the
   weights. Attention, the short-convolution mixers and the two dense blocks
   stay at Q8_0 in every build, and the router is not quantized at all. The
   embeddings are tied to the
   output, so they stay at Q8_0 too (F16 in `AD-Q8_0` and `AD-Q6_K`). Blocks 34
   and 36-39 get a step up, chosen by where the importance matrix saw the most
   activation energy, not by a fixed fraction of the depth.

The importance matrix matters: on the first revision, the same `AD-Q4_K_M`
layout without it measured 0.134 instead of 0.125.

## Running them

```bash
llama-server -m LFM2.5-24B-A2B-AD-Q4_K_M.gguf -ngl 99 -c 32768 --jinja
```

Pass `--jinja` so the embedded chat template is used; without it llama.cpp falls
back to a generic one. Liquid's recommended sampling: temperature 0.1, top_k 50,
repetition penalty 1.05. Context goes up to 131,072 tokens. Needs a llama.cpp
build with `lfm2moe` and the LFM2.5 tokenizer, which upstream has.

In [Atomic Chat](https://atomic.chat/?utm_source=huggingface&utm_medium=referral&utm_campaign=hf_lfm2_5_24b_a2b&utm_content=get_started):
search `AtomicChat/LFM2.5-24B-A2B-GGUF`, pick a build, hit **Use this model**.

## Measurement

The reference is the BF16 model's logits over our held-out neutral set: 93
chunks of 4,096 tokens in 30 languages, BF16 PPL 15.67 ± 0.12. The calibration
corpus shares no 13-gram with it. One machine for everything: 1× H100 PCIe,
llama.cpp `980aef8c`. All logs, the importance matrix and the comparison above
are in the
[metrics repository](https://huggingface.co/datasets/AtomicChat/LFM2.5-24B-A2B-GGUF-metrics).

## Limitations

- Built from the preview's revision `f772950c` (12 October); the public release
  has the same weights, checked file by file.
- Measured on general text only. Tool calling, long context and the thinking
  channel were not tested on the quantized builds.
- Not run yet on a Mac, in LM Studio or in Ollama.

## License

LFM Open License v1.0, see `LICENSE`. Released by Liquid AI; quantized by Atomic Chat.
