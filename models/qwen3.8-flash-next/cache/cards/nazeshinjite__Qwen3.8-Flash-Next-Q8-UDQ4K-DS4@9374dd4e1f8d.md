---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen4exp
- dwarfstar
- text-generation
---
# Qwen3.8-Flash-Next Q8-UDQ4K for DwarfStar

This repo ships a self-contained Qwen3.8-Flash GGUF for [DwarfStar (ds4)](https://github.com/antirez/ds4) that definitively upgrades the precision of antirez's original Q4 release for 128GB systems to be competitive with Unsloth's releases, in a format DS4 will load. I upgraded `ffn_down_exps` routed experts to Q8_0, along with targeted `ffn_gate_exps` / `ffn_up_exps` tensors in layers 0, 1, 2 and 45. The other 44 layers' gate/up expert pairs are Unsloth's imatrix-calibrated Q4_K instead of the base file's. Everything else, including the original BF16 n-gram table, is byte-identical. This approach is demonstrated to provide a quality uplift nearly comparable to Unsloth's Q5_K_XL. It even decodes 10 tps faster than the base Q4. 

Why run this model? 
- **50% faster decode than llama.cpp** (~65 tps w/ MTP)
- **Equivalent quality to Unsloth's UD-Q5_K_XL**
- **MTP and Vision included**


| Filename | File size | Resident |
| --- | ---: | ---: |
| Qwen3.8-Flash-Next-Q8-UDQ4K.gguf | 188.57 GiB | 93.19 GiB |
| mmproj-Qwen3.8-Flash-Next-Q8_0.gguf | 588 MiB | |

The remaining 95.37 GiB of the file is Qwen's original BF16 n-gram table, which ds4 reads row-by-row from disk (nothing resident). Keep the file on a local SSD.

## The base quant

The starting point is `Qwen3.8-Flash-Next-Q4.gguf` from
[antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf)
(165.11 GiB file, 69.74 GiB resident): imatrix-calibrated Q4_K `gate`/`up` routed experts (21.53 GiB each), uncalibrated MXFP4 `down` routed experts (20.34 GiB), Q8_0 attention, shared expert and output, BF16 embeddings. 

## Why Q8_0 and not Q5_K or Q6_K

ds4's Metal graph only decodes routed experts as `q8_0`, `mxfp4`, `q4_K`, `q2_K`, `iq2_xxs`, `f16` or `f32`. On this engine the precision ladder for experts is Q4_K -> Q8_0, so "more quality" means choosing which tensors get Q8_0 within a given memory budget. My target was matching the performance and footprint of Unsloth's UD-Q5_K_XL (96.7 GiB), the largest quant that fitsin 128GiB with 262k context. The end result came in 3.5 GiB under Unsloth, leaving room for a couple open apps.

Two facts about the base file decided the first step. `down` is MXFP4, and ggml's MXFP4 quantizer discards the imatrix outright, so a quarter of the model was quantized blind. Furthermore, `down` is usually more sensitive than `up`/`gate`, so all 49 `down` tensors therefore go to Q8_0 first (+20.5 GiB). Each `gate`/`up` pair costs a
further 0.78 GiB per layer (850 MiB per Q8_0 tensor against 450 MiB for Q4_K), so the rest of the work was deciding which layers deserve it.

## Targeting the layers

### Unsloth Dynamic v3 tiers

Unsloth's dynamic quants encode a per-tensor sensitivity study in their type choices. Their choices for this model:

| Tier | gate / up / down (default) | Exceptions |
| --- | --- | --- |
| UD-Q3_K_XL | IQ3_XXS / IQ3_XXS / IQ4_NL | layer 2 gate/up IQ4_XS + down Q8_0; layers 4, 30, 46, 47 down Q8_0 |
| UD-Q4_K_XL | Q4_K / Q4_K / Q5_1 | layer 2 gate/up Q5_K + down Q8_0; layers 4, 30, 46, 47 down Q8_0 |
| UD-Q5_K_XL | Q5_K / Q5_K / Q8_0 | layer 2 gate/up Q6_K |
| UD-Q6_K_XL | Q6_K / Q6_K / Q8_0 | layer 2 gate/up Q8_0 |

Layer 2 is the only layer Unsloth bumps for `gate`/`up`, at every tier. It is also the layer where the n-gram embedding is injected. Not coincidental.

### Two predictions that didn't survive measurement

Before spending memory on more gate/up layers, two ways of predicting which layers deserve it were tried, and both were tested directly. An imatrix-weighted ranking of Q4_K rounding error nominated seven mid-trunk layers; bumping all seven to Q8_0 changed the short-set score by +0.0004 NLL (insignificant). A calibration coverage-based approach suggested edge layers 0, 1, 45, 46 and 47, where Unsloth's imatrix shows experts with zero or near-zero routing counts, and Q8_0 would need no calibration; bumped individually and together, none moved the composite score by more than noise. Q4_K rounding error is close to uniform across layers, and under-calibrated experts are apparently too rarely routed to matter. 

### What did move the needle

- **Layer 2 itself.** A downs-only build with layer 2 reverted to Q4_K is the only variant below the reference on the short set (+0.0008 NLL, composite −1.28, four of five metrics agreeing), proving the importance of the n-gram embedding injection layer.
- **Unsloth's calibration for the Q4_K layers.** Swapping the 42 interior Q4_K gate/up pairs to Unsloth's, at identical type and size, was the one change that leaned the same way on every metric on both sets (long set −0.0029 NLL, composite +0.92). It costs zero bytes.

### Recipe

| Tensor family | Layers | Type | Source of the bytes |
| --- | --- | --- | --- |
| ffn_down_exps | all 49 (0-47 + MTP 48) | Q8_0 | UD-Q5_K_XL shards (0-47); `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (48) |
| ffn_gate_exps, ffn_up_exps | 0, 1, 2, 45 | Q8_0 | Unsloth Q8_0 shards |
| ffn_gate_exps, ffn_up_exps | 3-44, 46-47 | Q4_K (Unsloth imatrix) | UD-Q4_K_XL shards |
| MTP layer 48: ffn_down_exps / gate+up / dense | 48 | Q8_0 / Q4_K (base, Ivan imatrix) / Q8_0 | UD MTP file (down); base file (rest) |
| everything else, n-grams | | unchanged | base file |

Layers 0, 1 and 45 are carried at Q8_0: their singles were the three best
short-set rows of the screen at 2.34 GiB total. I tried aother mix with the top five layers, but saw no improvement for the extra memory squeeze. The file is 3.5 GiB under the Unsloth Q5 this way.

## Quality confirmation

Quality was measured with ds4's `gguf-tools/quality-testing/score_official` against the official API continuations that ship with the tool: `qwen38-flash-alibaba-100` (100 non-thinking prompts, 5,696 target tokens, top-five logprobs from the Alibaba endpoint, scored at context 4096) and `qwen38-flash-alibaba-long` (12 archive/code continuations with prefixes up to 24k tokens, 766 target tokens, context 32768). This gave a balance of statistical significance (short n=100) and a focus on protecting long-context agentic coding capabilities. The score is the model's negative log-likelihood of the official continuation plus agreement statistics with the API's top-five distribution. Ordinary decoding, no MTP.

The base Q4 scored on this Metal build reproduces the numbers antirez published from a DGX Spark for the same file (section 7). The
scorer is exactly deterministic: the same file scored twice gives 100 of 100 cases bit-identical, so every difference in the tables is sampling of the fixed token set, not measurement jitter. And a
significance rule was fixed before the screen: a variant counts as different only if |dNLL| >= 0.002
on the short set (a third of the base-to-custom gap), its 99% case-bootstrap interval clears zero, and top-1 does not move against it; the long set corroborates, not decides. A composite uplift score (joint case bootstrap over NLL, top-1, top-n recall, pair-order agreement and top-5 MAE, each oriented and standardized, averaged) was added for sensitivity; on it the base file reads −4.7 standard units below the final reference.

### Headline

| Quant | avg NLL | API top-1 | top-n recall | pair-order agree | top-5 MAE | greedy first-token match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base antirez Q4 (this box) | 0.2906 | 92.08% | 88.62% | 91.58% | 0.0985 | 84/100 |
| **Q8-UDQ4K** | **0.2842** | **92.96%** | **89.49%** | **92.13%** | **0.0917** | **85/100** |
| base Q4, upstream Spark reference | 0.2903 | 92.08% | | | | |

Paired per-case dNLL (final minus base), token-weighted, short set: **−0.0064**, 95% CI
[−0.0095, −0.0026]; composite +4.7 (99% interval clear of zero). Long set: 0.1253 → 0.1207, top-1
96.87% → 97.39%. For scale, the Q2 to Q4 step on this set is about 0.063 NLL.

### Versus Unsloth's own quants

Unsloth's UD-Q4_K_XL and UD-Q5_K_XL were scored on the identical fixtures and the identical
prompts with ds4's llama.cpp control scorer (`score_llama`, rebuilt against llama.cpp 0.4.1, plus a `qwen-ds4` prompt mode that reproduces ds4's no-thinking rendering.

| file | runtime | resident | short NLL | greedy 1st-tok | long NLL | short dNLL vs this file (95% CI) | better/worse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unsloth UD-Q5_K_XL | llama.cpp | 96.7 GiB | 0.2831 | 88/100 | 0.1209 | −0.0011 (−0.0032..+0.0010) | 52/48 |
| **this file (Q8-UDQ4K)** | ds4 | 93.2 GiB | **0.2842** | 85/100 | 0.1207 | | |
| Unsloth UD-Q4_K_XL | llama.cpp | 76.9 GiB | 0.2861 | 86/100 | 0.1229 | +0.0019 (+0.0005..+0.0034) | 37/63 |
| base antirez Q4 | ds4 | 69.7 GiB | 0.2906 | 84/100 | 0.1253 | +0.0064 (+0.0026..+0.0095) | 23/77 |

So this file beats UD-Q4_K_XL by a margin the interval supports on both sets, at 16 GiB more
resident memory, and ties UD-Q5_K_XL (a 0.0011 NLL edge for Q5 on the short set, a 52/48 split, and a dead heat on the long set) at 3.5 GiB less. The Q5 tier's
Q5_K gate/up cannot run on ds4's Metal graph at all, which is why this recipe exists.

Unsloth's published figures for the same tiers (UD-Q4_K_XL top-1 92.26%, KLD 0.047; UD-Q5_K_XL top-1
93.68%, KLD 0.030; Q8_0 top-1 94.12%) are **not comparable** to any column above: they are agreement
with the BF16 model's own argmax over Unsloth's corpus via `llama-perplexity --kl-divergence`, while
the top-1 here is agreement with the Alibaba API over the official fixtures. Same name, different
reference and different text. A KL divergence against BF16 is not obtainable for this file: no BF16
reference logits are published and
llama.cpp's KLD tool cannot load a ds4 GGUF.

## Decode speed and MTP

Same box, ctx 196608, vision encoder loaded, temperature 1.0 / top_p 0.95 / top_k 20, five prompts,
MTP arms interleaved on / off / on so box drift cannot pose as a speculative gain:

| prompt | MTP exact on, arm 1 | MTP off | MTP exact on, arm 3 | gain |
| --- | ---: | ---: | ---: | ---: |
| 500-word essay, thinking off | 57.0 t/s | 50.7 | 55.6 | +11% to +12% |
| Python LRU cache + tests, thinking off | 67.5 | 50.3 | 65.5 | +30% to +34% |
| 25-record JSON, thinking off | 69.6 | 49.6 | 67.5 | +36% to +40% |
| geometry problem, thinking xhigh | 66.5 | 48.5 | 65.8 | +36% to +37% |
| Rust borrow error, thinking xhigh | 61.4 | 48.9 | 56.7 | +16% to +26% |

Cumulative draft acceptance: 86.5% of 1251 verify cycles. Plain
decode on this file is 49 to 51 t/s against 39 to 40 t/s on the base Q4: Q8_0 experts decode faster
than Q4_K plus MXFP4 on this ds4 build, so the quality upgrade also speeds things up. Bumping the
MTP layer's own gate/up to Q8_0 was considered and rejected as it didn't improve acceptance.

## Memory on a 128 GB Apple Silicon box

ds4 plans resident weights/KV/compute buffers up front. With the built-in MTP head on and the vision encoder loaded:

| Context | KV | buffers | resident model | total |
| ---: | ---: | ---: | ---: | ---: |
| 131072 | 4.17 GiB | 9.24 GiB | 93.19 GiB | 106.6 GiB |
| 196608 | 6.25 GiB | 10.24 GiB | 93.19 GiB | 109.68 GiB (default) |
| 262144 | 8.33 GiB | 11.24 GiB | 93.19 GiB | 112.8 GiB |

macOS caps GPU-wired memory well below the 128 GB, so this file needs the limit raised. One line, takes effect immediately, resets at reboot:

    sudo sysctl -w iogpu.wired_limit_mb=118811


## Running it

    ds4-server -m Qwen3.8-Flash-Next-Q8-UDQ4K.gguf --metal -c 196608 \
        --mtp --mtp-exact-sampling \
        --vision mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
        --kv-disk-dir kv-cache-qwen38f --kv-disk-space-mb 65536

ds4-server takes no sampling flags; the request decides. Qwen's thinking profile is temperature
1.0 / top_p 0.95 / top_k 20 / min_p 0. `--mtp` uses the nextn head inside the file; exact
sampling keeps the target distribution while speculating (opportunistic mode is greedier).
Give this file its own KV checkpoint directory: ds4 keys checkpoints by prompt hash, quant bits
and model id, so two quants of the same model collide in a shared directory. The vision encoder
is ggml-org's `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` (616 MB).

## Checksums

    397c87adb7345d66d679da1946e380b12b07edb44facd4c3e16e463cf4c83439  Qwen3.8-Flash-Next-Q8-UDQ4K.gguf
    b2e9b5e4a44c107f8867e67dbf09b607fd99ae33c1a97a60a6720aeb252a9dad  mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
