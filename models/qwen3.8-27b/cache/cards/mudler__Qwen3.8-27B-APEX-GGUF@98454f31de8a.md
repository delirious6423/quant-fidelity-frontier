---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - quantized
  - apex
  - dense
  - hybrid-attention
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

# Qwen3.8-27B APEX GGUF

Two quantizations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), plus the vision projector.

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team** | [APEX Project](https://github.com/mudler/apex-quant)

| File | Size | Pick this if |
|------|------|--------------|
| Qwen3.8-27B-APEX-I-Mini.gguf | 13.94 GB | you have 16 GB of VRAM |
| Qwen3.8-27B-APEX-I-Nano.gguf | 11.24 GB | you have 12 GB |
| mmproj-Qwen3.8-27B-F16.gguf | 0.93 GB | you want vision, pair with either |

Both include the MTP / NextN head as `blk.64`, so speculative decoding runs against the file itself. The head is kept at Q8_0: a drafter that mispredicts the target wastes the speculation, so it does not get compressed with the rest.

```bash
# text
llama-cli -m Qwen3.8-27B-APEX-I-Mini.gguf -p "Your prompt" -ngl 99

# vision
llama-mtmd-cli -m Qwen3.8-27B-APEX-I-Mini.gguf \
               --mmproj mmproj-Qwen3.8-27B-F16.gguf -ngl 99

# speculative decoding
llama-cli -m Qwen3.8-27B-APEX-I-Mini.gguf --spec-type draft-mtp -ngl 99
```

Needs a recent llama.cpp with `qwen3_5` support. The architecture landed after the model shipped on 2026-08-13, and older builds segfault on load.

## Why only two sizes

The bit allocation in these files comes from a measurement rather than a hand-picked recipe, and the measurement said the approach only helps below about 15.2 GB.

The same allocation was built at four sizes and compared against a flat per-role allocation, the kind unsloth, bartowski and ggml-org ship. Each of ours came out slightly smaller than its control, so the comparison interpolates the flat curve to each file's exact size:

| band | size | KL vs BF16 | flat, same size | difference |
|---|---|---|---|---|
| Balanced | 17.65 GB | 0.011211 | 0.009832 | 14.0% worse |
| Compact | 15.17 GB | 0.030468 | 0.030705 | tie |
| Mini | 13.49 GB | 0.049020 | 0.065369 | 25.0% better |
| Nano | 10.79 GB | 0.121640 | 0.156226 | 22.1% better |

It depends on how much damage there is to move around. A flat allocation's KL against BF16 runs 0.0021 at Q6_K, 0.0099 at 17.6 GB, 0.0617 at 13.7 GB, and 0.1448 at 11.0 GB. At Q4 and above the model is already close to lossless, so there is little for a smarter allocation to redistribute. By Mini there is roughly six times as much.

A Quality or Balanced tier from us would be the same flat recipe you can already download elsewhere, so it is not here.

(Those sizes are the trunk-only files used for the comparison. The published ones are about 0.45 GB larger because they carry the MTP head.)

## What the measurement found

Qwen3.8-27B has no experts, so the usual APEX trick does not apply: its FFN is 62.6% of the parameters and every weight runs for every token. It also interleaves three linear-attention layers per full-attention layer (64 layers, full attention at i%4==3), which splits attention into two groups that behave differently.

Dropping one tensor group at a time from Q6_K to Q3_K and dividing the KL increase by the gigabytes saved gives:

| group | dKL per GB |
|---|---|
| token_embd | 0.00231 |
| FFN middle layers | 0.00565 |
| ffn_gate | 0.00596 |
| ffn_up | 0.00806 |
| linattn_gate | 0.00880 |
| ffn_down | 0.00920 |
| linattn_qkv | 0.01077 |
| linattn_out | 0.01474 |
| FFN edge layers | 0.01483 |
| full-attention | 0.01583 |
| output | 0.03526 |

A 15.3x spread. Three things stand out. `output` and `token_embd` have identical shape, 5120 x 248320, yet `output` is 15.3x more sensitive, where most recipes place them one step apart. FFN edge layers cost 2.63x more per byte than middle layers. And the three linear-attention tensors range from 0.00880 to 0.01474, so compressing them uniformly loses to a flat baseline.

## Limitations

Throughput was not measured. Different quant types select different kernels, so speed is a separate question. The +35-52% tok/s figure quoted for `draft-mtp` on this model comes from a third-party report.

These numbers are specific to this architecture. The linear and full attention split is unusual, so they should not be assumed to carry over to other dense models.

`token_embd` has a floor. It is an embedding lookup rather than a matmul, so `llama-imatrix` collects nothing for it, and very-low-bit types that need imatrix data fail at quantize time. IQ2_S works, IQ2_XXS does not.

## Method

Imatrix from diverse calibration data (chat, code, reasoning, tool-calling, agentic traces, Wikipedia), 200 chunks. KL measured against BF16 reference logits on wikitext-2-raw, n_ctx 512, 200 chunks. Every arm shares one imatrix and one set of reference logits, so the numbers above are directly comparable. 19 arms in total.

Perplexity is not quoted per file because it does not separate them: all four land within about 0.07 of each other, well inside the error bars.
