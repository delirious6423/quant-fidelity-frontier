---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - gguf
  - speculative-decoding
  - dspark
  - deepseek
  - lucebox
---

# DeepSeek V4.1 Flash DSpark drafter (GGUF)

The speculative-decoding drafter for DeepSeek V4.1 Flash in the
[Lucebox engine](https://github.com/Luce-Org/lucebox). It is the draft block
DeepSeek ships inside the checkpoint (the `mtp.*` tensors: three stages, 128
routed experts top-3, a block of five tokens), converted to GGUF. The target
model drafts nothing itself; this file proposes tokens and the target verifies
them in one pass, and every verified token is bit-identical to plain decoding.

| file | size | sha256 |
|---|---:|---|
| `DeepSeek-V4.1-Flash-DSpark-draft-MXFP4-Q8.gguf` | 7.4 GiB | `3c96c3ee7801e5f166428b946e7590dbcf259b2ffe1fb09dd5fbed7d96bfe894` |
| `DeepSeek-V4.1-Flash-DSpark-draft-Q2K-Q4K.gguf` | 4.6 GiB | `5efffdb5ab4769bac7192fd2ce7c08cfa37945cad37be0825526c83139ef200a` |

Routed experts are repacked bit-exactly to MXFP4, dense projections are Q8_0.
The Q2K-Q4K build requantizes the same drafter (routed experts Q2_K, dense
projections Q4_K) for the 192 GB setup, where the drafter shares the R9700 with
12 GB of hot experts: it leaves room for larger prompt chunks and drafts faster,
with about the same acceptance.
Converted with `server/scripts/convert_ds4_dspark_draft_to_gguf.py` from the
[source checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

## Run

With the target from
[Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF)
on a Lucebox with 128 GB (AMD Radeon AI PRO R9700 + Ryzen AI MAX+ 395):

```bash
luce_server DeepSeek-V4.1-Flash-ROCMFP2S.gguf \
  --profile ds41-lucebox \
  --draft DeepSeek-V4.1-Flash-DSpark-draft-MXFP4-Q8.gguf
```

With 192 GB (Ryzen AI MAX+ PRO 495) every expert stays in memory, with the
smaller drafter and an expert placement and router bias fitted to the box:

```bash
luce_server DeepSeek-V4.1-Flash-ROCMFP2S.gguf \
  --profile ds41-gorgon \
  --draft DeepSeek-V4.1-Flash-DSpark-draft-Q2K-Q4K.gguf \
  --ds4-expert-placement placement.json \
  --ds4-router-bias router_bias.bin
```

Lucebox Manage installs the files and the fitted layout, and picks the launch
for the box's memory from its Engine page.

## Measured

`--profile ds41-gorgon` with the Q2K-Q4K drafter, 192 GB, fresh server:

| prompt | prompt reading | decode |
|---|---|---|
| code, 36 tokens, 256 out | - | 49.4-49.5 tok/s |
| 2,027 tokens, 256 out | 811 tok/s | 32.0-32.6 tok/s |
| 30,752 tokens, 256 out | 1,039 tok/s | 31.8 tok/s |

`--profile ds41-lucebox` at its 128K context, first request after start
([DS41 guide](https://github.com/Luce-Org/lucebox/blob/main/server/docs/DS41.md)):

| prompt | decode |
|---|---|
| code, 29 tokens, 256 out | 25.4-25.9 tok/s |
| 3,981 tokens, 128 out | 16.8-17.6 tok/s |
| 21,420 tokens, 32 out | 13.7 tok/s |

The engine verifies three drafted tokens per step by default (`LUCE_DS4_SPEC_Q`,
2 to 6), the fastest on the Lucebox.

## License

MIT, as the source checkpoint.
