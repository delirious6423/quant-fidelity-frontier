---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - mtp
  - speculative-decoding
---

# tobu MTP head P for Qwen3.8-Flash-Next (UD-IQ4_XS)

Head P is a multi-token-prediction (MTP) draft head for Qwen3.8-Flash-Next. The [tobu](https://github.com/hakobu/tobu-dist)
engine uses it to draft tokens that the main model then verifies. It changes decode speed only, never the output:
with this head, tobu's greedy outputs are bit-identical to its outputs without any MTP head.

| file | sha256 |
|---|---|
| `mtp-tobu-p-iq4xs.gguf` (2.79 GB) | `c0d60e1c2374f292caed86f94018ec1c2fda82556a5b4ee04867bf4f47ec29f0` |

It is fine-tuned from unsloth's `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (repository
`unsloth/Qwen3.8-Flash-Next-GGUF`, revision `766911a6`), which is derived from Qwen3.8-Flash-Next by Qwen. Use it
with the UD-IQ4_XS quantization from that same revision; the head was trained against its hidden states.

## Training data: public and synthetic only

The targets are the model's own greedy outputs (self-distillation). No customer data, no private conversations and no
internal documents were used:

1. **Prose:** 250 answers of the model to synthetic prompts ("write a short essay / story / email about
   <generic topic>"), in German and English; 30 % with thinking on.
2. **Agent sessions:** 3 coding-agent sessions driven by the model itself on neutral public tasks: a key-value store
   with compaction, a Markdown table formatter and an HTTP rate limiter. The task texts are in `training-tasks/`.

Method: stage-B self-distillation behind the attention. Trained: `hc_ffn`, the shared expert and `hc_head`
(about 25 M parameters); router, routed experts, attention and LM head frozen. 1 epoch, 1,450 steps, batch 256,
learning rate 5e-5, 430,000 training rows. Details in `NOTICE.md` and `training-report.json`.

## Results

**Offline top-1 draft accuracy** on held-out data (after Q8_0 rounding):

| data set | unsloth head | head P |
|---|---:|---:|
| prose | 72.6 % | 76.9 % |
| agent: key-value store | 87.4 % | 88.6 % |
| agent: Markdown tables | 86.9 % | 88.2 % |
| agent: rate limiter | 91.0 % | 91.6 % |

**Online decode** with tobu-dist v0.8.0 on a Bosgame M5 (Ryzen AI Max+ 395, Radeon 8060S / gfx1151, 128 GB) at the
85 W default power profile, `./tobu bench`, greedy. Each figure is the median of 4 back-to-back runs after one
discarded warmup.

| tok/s | no MTP | unsloth head | head P |
|---|---:|---:|---:|
| decode: code | 31.1 | 73.5 | **74.3** |
| decode: code refactor | 31.2 | **86.9** | 86.2 |
| decode: JSON extraction | 31.1 | 80.2 | **83.7** |
| decode: German prose | 31.0 | 40.4 | **42.4** |
| decode: short German answer | 31.2 | 39.9 | **42.6** |
| cold prefill 8.7k / 36k tokens | 1,706 / 1,669 | 1,635 / 1,596 | 1,637 / 1,601 |

**Bit-identical outputs:** all three configurations produced exactly the same 7 output hashes on the bench prompts.

## Use with tobu

With [tobu-dist](https://github.com/hakobu/tobu-dist):

```bash
./install.sh --head-p          # downloads the head pinned by revision, checks the sha256, starts the server with it
```

Or by hand: download `mtp-tobu-p-iq4xs.gguf`, check it with `sha256sum -c mtp-tobu-p-iq4xs.gguf.sha256`, then

```bash
TOBU_MTP=/path/to/mtp-tobu-p-iq4xs.gguf scripts/serve.sh start
```

## License

The weights are a derivative of Qwen3.8-Flash-Next and are distributed under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/LICENSE);
the full text is in `LICENSE`. Copyright (c) 2026 Qwen. Fine-tuning by hakobu GmbH.

## Contact

David Müller, hakobu GmbH, david.mueller@hakobu.de
