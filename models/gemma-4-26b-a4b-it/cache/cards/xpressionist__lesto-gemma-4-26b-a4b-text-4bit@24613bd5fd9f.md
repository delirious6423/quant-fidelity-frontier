---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- gemma4
- 4-bit
- apple-silicon
---

# Lesto 26B-A4B, 4-bit

A 13 GB 4-bit quantization based on Gemma 4 26B-A4B (`google/gemma-4-26B-A4B-it`), made for [Lesto](https://github.com/fedeandri/lesto), a personal experiment in fast Gemma 4 inference on Apple silicon Macs. It is an independent project, not affiliated with or endorsed by Google. Gemma is a trademark of Google LLC.

## What is different

- **Attention rounded with oQe.** The 115 attention projections are 4-bit affine, group size 64, rounded with oMLX's oQe method (importance-weighted clipping and a least-squares refit), calibrated on oMLX's public multilingual text set. Almost everything else is 4-bit too; the routers and the other tensors the config lists at 8 bits stay at 8. It is standard MLX affine format, so MLX engines that support Gemma 4 can load the weights.
- **A Latin-script output head.** An extra file lists the 162,489 of 262,144 output tokens whose text is Latin script, plus the special and byte-fallback tokens. Lesto scores only those, which reads less memory per token and gave about 5% faster decoding in our measurements. Nothing is deleted: the full table stays, and it is still used for reading input.

## Which languages the head suits

The head is on by default in Lesto with this checkpoint. Set `MLX_SERVE_LATIN_HEAD_OFF=1` to turn it off. Other engines do not use the list.

We translated one short text into 28 languages with the head on and off:

| Result | Languages |
|---|---|
| Identical replies | Italian, French, Spanish, Portuguese, Catalan, Romanian, German, Dutch, Swedish, Danish, Norwegian, Finnish, Polish, Czech, Slovak, Slovenian, Croatian, Estonian, Latvian, Lithuanian, Irish, Welsh, Icelandic, Albanian, Turkish |
| Different: the head removed a stray token from the reply | Hungarian, Maltese |
| Damaged by the head: turn it off | Vietnamese |

English and Italian were also identical on 80 longer bench replies. That is a check of the head, not of how well the model writes each language.

**Other scripts need the head off.** With it on, a reply in another script comes out in Latin letters, and some of it can be wrong:

| Asked for | Head on | Head off |
|---|---|---|
| Russian | `Dobroye utro, kak vy segodnya?` | `Доброе утро, как у вас сегодня дела?` |
| Chinese | `Zǎoān, nǐ jīntiān zěnmeyàng?` | `早上好，你今天怎么样？` |
| Arabic | `de sabah al-khair, kayfa haluka al-yawm?` | `صباح الخير، كيف حالك اليوم؟` |
| Greek | `Kalimera, pos eiste simera;` | `Καλημέρα, πώς είστε σήμερα;` |
| Three sea emoji | `waves 🪸 🪼 🩵` | `🌊🐚🐟` |

## Quality

We read pairs of replies blind against `mlx-community`'s 8-bit checkpoint, which is twice the size and close to the source model:

| Read | Pairs | This one preferred | 8-bit preferred | Tie | Fact conflicts | Defects, this one / 8-bit |
|---|---:|---:|---:|---:|---:|---:|
| Italian, 8k prompts | 20 | 2 | 1 | 17 | 0 | 1 / 4 |
| English, 8k prompts | 20 | 2 | 1 | 17 | 0 | 3 / 3 |
| English, 1k and 16k prompts | 40 | 4 | 2 | 34 | 0 | 3 / 4 |

A native Italian speaker, also blind, read the same 20 Italian pairs in a separate order and marked all 20 as ties, without the source text. Replies from this checkpoint ran a little shorter: a median of 117 tokens against 131.5 on those pairs.

These replies come from the checkpoint without the head; the head's replies on the 8k and on the 1k and 16k prompts are identical to them. The reader was an AI model without the source text, so an error both replies share goes unseen. Italian prompts exist only at 8k.

It is still a lossy 4-bit quantization: replies are not word for word the source model's. It is closer to the source model than the standard 4-bit, measured as KL divergence of the next-token distribution, top 10 tokens, on 200 prompts through one engine (lower is closer):

| Checkpoint | KL to the source model |
|---|---:|
| `mlx-community` 4-bit | 0.308 |
| This one | 0.171 |
| `mlx-community` 8-bit | 0.023 |

This one is lower than the standard 4-bit on 160 of the 200 prompts. KL does not say how replies read, and we did not read this one against the standard 4-bit.

## Speed

Gemma 4 26B-A4B through [Lesto](https://github.com/fedeandri/lesto) 0.17.1 and the latest releases of other Mac engines, on 2026-11-06. M4 Pro, 20 GPU cores, 64 GB, macOS 26.7.2; default flags, one user at a time, 15 cold requests per prompt size for a 200-token reply. Reproduce with Lesto's `scripts/bench.py`.

| Engine and checkpoint | Decode tok/s, 1k / 8k | First text s, 1k / 8k | 200-token reply s, 1k / 8k |
|---|---:|---:|---:|
| Lesto, this checkpoint | 89.3 / 83.3 | 1.38 / 10.25 | 3.61 / 12.64 |
| Lesto, this checkpoint, head off | 84.5 / 79.2 | 1.38 / 10.23 | 3.73 / 12.74 |
| Lesto, `mlx-community` 4-bit | 84.5 / 79.1 | 1.38 / 10.34 | 3.74 / 12.86 |
| mlx-serve 26.11.2, this checkpoint | 71.8 / 51.6 | 1.37 / 9.44 | 4.14 / 13.30 |
| mlx-serve 26.11.2, `mlx-community` 4-bit | 71.9 / 51.7 | 1.37 / 9.37 | 4.13 / 13.22 |
| mlx-lm 0.32.2, this checkpoint | 67.3 / 62.7 | 1.44 / 10.79 | 4.39 / 13.97 |
| mlx-lm 0.32.2, `mlx-community` 4-bit | 67.3 / 62.6 | 1.44 / 10.90 | 4.39 / 14.10 |

Without the head, this checkpoint runs at the same speed as the standard 4-bit on all six engines we measured, including oMLX, vllm-mlx and mlx-vlm, which the table leaves out. Its gain there is closeness to the source model, not speed. The "head off" row is the same weights without the token list, a stand-in for `MLX_SERVE_LATIN_HEAD_OFF=1`. Several users at once were not measured.

## Using it

Download it into a folder of models, then point [Lesto](https://github.com/fedeandri/lesto) at that folder:

```bash
hf download xpressionist/lesto-gemma-4-26b-a4b-text-4bit --local-dir ~/models/lesto-gemma-4-26b-a4b-text-4bit
lesto serve --model-dir ~/models --host 127.0.0.1
```

## Licence

Apache License 2.0, the licence of the [source model](https://ai.google.dev/gemma/docs/gemma_4_license); see `LICENSE`. These files are modified from the source model: its weights are quantized as described above, and the output-token list is added. Quantization calibration used oMLX's public text set; no private data went into this checkpoint.
