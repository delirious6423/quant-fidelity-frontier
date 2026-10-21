---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - quantized
  - apex
  - moe
  - qwen
---

<!-- apex-banner-v2 -->
<div style="background-color: #f59e0b; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
<h2 style="color: white; margin: 0 0 10px 0;">⚡ Each donation = another big model quantized</h2>
<p style="font-size: 18px; margin: 0 0 15px 0;">I host <b>30+ free APEX quantizations</b> as independent research. My only local hardware is an <b>NVIDIA DGX Spark</b> (122 GB unified memory), enough for ~30-50B-class models, but <b>bigger ones (200B+) require rented compute</b> on H100/H200/Blackwell, typically $20-100 per quant.<br>If APEX quants are useful to you, your support directly funds those bigger runs.</p>
<p style="font-size: 20px; margin: 0;">
<a href="https://www.patreon.com/cw/mudler" style="color: white; text-decoration: underline;">🎉 Patreon (Monthly)</a> &nbsp;|&nbsp;
<a href="https://www.buymeacoffee.com/mudler" style="color: white; text-decoration: underline;">☕ Buy Me a Coffee</a> &nbsp;|&nbsp;
<a href="https://github.com/sponsors/mudler" style="color: white; text-decoration: underline;">⭐ GitHub Sponsors</a>
</p>
</div>

# Qwen3.8-Flash-Next APEX GGUF

Three quantizations of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), a 177 B parameter MoE with 512 experts.

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team** | [APEX Project](https://github.com/mudler/apex-quant)

## You need less VRAM than the file size suggests

This is the important thing to know about running this model, and every listing that quotes a single number gets it wrong.

29% of this model is one tensor: `per_layer_token_embd`, 51.2 B parameters. It is a hashed n-gram table, and llama.cpp always keeps it in **system RAM, never VRAM**, whatever you pass to `-ngl`. It is a lookup, so there is nothing for a GPU to do with it.

So the honest requirement is a pair of numbers, measured from the finished files:

| Tier | Size | VRAM | System RAM | Fits |
|------|------|------|------------|------|
| Qwen3.8-Flash-Next-APEX-I-Nano | 73.0 GB | **43.9 GB** | 29.1 GB | one 48 GB card |
| Qwen3.8-Flash-Next-APEX-I-Mini | 78.7 GB | **49.6 GB** | 29.1 GB | 64 GB or more |
| Qwen3.8-Flash-Next-APEX-I-Compact | 85.1 GB | **55.9 GB** | 29.1 GB | 64 GB or more |

Each tier ships as several shards, because HuggingFace caps a single file at 50 GB and every tier here is larger than that. Nano is 5 files, Mini and Compact are 6. Download a tier's whole set into one directory and point llama.cpp at the first file; it finds the rest itself.

One shard per tier is 28.8 GB and cannot be made smaller: the n-gram table is a single tensor, and GGUF splitting cuts at tensor boundaries.

A 73 GB file that needs 44 GB of VRAM is a different proposition from one that needs 73 GB.

Nano fits a single 48 GB card (A6000, L40S, RTX 6000 Ada) with room for context. On two 24 GB cards it is tight rather than comfortable: 43.9 GB of weights against 48 GB total leaves about 4 GB for KV cache and compute buffers, so expect to keep the context modest. Mini needs more than 48 GB, so plan on 2x32 GB or an 80 GB card.

```bash
llama-cli -m Qwen3.8-Flash-Next-APEX-I-Nano-00001-of-00005.gguf -p "Your prompt" -ngl 99
```

Needs a recent llama.cpp with `qwen4exp` support. The architecture landed after the model shipped, so older builds will not load these files.

## Why nothing here is smaller than 70 GB

Not a recipe choice. It is arithmetic on the tensor shapes, and it constrains everyone equally.

llama.cpp cannot use a quantization type whose block size does not divide a tensor's row length. Two families in this model fail that test:

| tensor | parameters | row length | divisible by 256 |
|---|---|---|---|
| `per_layer_token_embd` | 51.2 B (28.9%) | 160 | no |
| `ffn_down_exps` | 40.3 B (22.8%) | 640 | no |

Every K-quant and every 256-block IQ type is therefore illegal on **51.7% of the model**. Only 32-block types remain, and the cheapest usable one is 4.5 bits per weight. That pins 51.5 GB before anyone chooses anything.

Add the smallest sensible expert allocation on top and you land near 70 GB. That is why the smallest quantization anyone ships of this model, from any provider, sits around 72 GB.

It also means asking for a *cheaper* type on those tensors can cost you *more*. llama.cpp does not reject an illegal type, it promotes it: Q5_K becomes Q5_1, which is 5.5 bits going to 6.0. Configs here state the 32-block types explicitly so that never happens silently.

## Where the bits actually went

With half the model locked, the only budget left to allocate is `ffn_gate_exps` and `ffn_up_exps`, 80.5 B parameters between them. That is where these files differ from a stock recipe, in two ways:

**Layer position.** Earlier measurement on Qwen3.8-27B found FFN layers at the edges of the stack 2.63x more sensitive per byte than those in the middle, so the first and last few layers keep more bits and the middle is pushed hardest.

**Gate and up are not the same.** The same sweep measured `ffn_up` at 0.00806 dKL/GB against `ffn_gate` at 0.00596, so up is about 1.35x dearer. It is held one step above gate throughout. Shelf recipes give the two identical types.

Everything else, the attention tensors, output, shared experts and the hyper-connection heads, is 2.8% of the model combined. Pinning all of it high costs about 2 GB, so it is pinned high rather than economised on.

## What the quantization costs

Perplexity against the BF16 weights, same text and same 4096 token context, six chunks each:

| | perplexity | vs BF16 |
|---|---|---|
| BF16 | 4.3113 | |
| Mini | 4.4354 +/- 0.085 | +2.9% |
| Nano | 4.6659 +/- 0.093 | +8.2% |

For 4.5x compression against a 354 GB original, with 45% of the model at roughly 1.6 bits per weight in Nano, that is a good result.

Two caveats, because they matter. The text is the same corpus the importance matrix was built from, which flatters both quants; a held-out set would be the fair test. And perplexity is a blunt instrument, which is why the sensitivity work behind these allocations used KL divergence instead. Read these as evidence the files are sound, not as a claim that they beat any particular alternative.

A KL comparison against BF16 is not cheap here for a specific reason: the logits file stores about 497 KB per token at this 248320 token vocabulary, so it runs to terabytes at long context.

## Limitations

Throughput is not measured. Different quant types select different kernels, so speed is a separate question from quality per byte.

The three sizes are close together because the floor forces it. There is no Q4 or Q6 tier here: above roughly 80 GB the movable budget relaxes toward Q4, where a flat recipe is already near optimal, so a larger tier from us would be the same file you can get anywhere.

## One measurement worth sharing

The n-gram table is the cheapest place in this model to take bits away, and that is not obvious.

Quantizing it at IQ4_NL instead of Q8_0 saves 25.6 GB and costs a KL divergence of 0.0324, a perplexity rise of 0.71%, and 6% of top-1 token predictions. That works out to **0.00126 KL per GB**, which is about half the price of the cheapest tensor group measured on Qwen3.8-27B and roughly 28 times cheaper than the dearest. Measured on two builds differing in exactly that one tensor, scored over 100 chunks of held-out wikitext-2.

So every tier here keeps the table at IQ4_NL. Other providers move it to Q8_0 at their Q5 and Q6 tiers, which spends 25.6 GB on the one tensor that gives back the least.

## Method

Importance matrix from diverse calibration data (chat, code, reasoning, tool calling, agentic traces, Wikipedia), computed on the BF16 weights at 4096 token context. `per_layer_token_embd` gets no imatrix coverage because it is an embedding lookup rather than a matmul, which is moot here since its 4.5 bpw floor is forced anyway.
