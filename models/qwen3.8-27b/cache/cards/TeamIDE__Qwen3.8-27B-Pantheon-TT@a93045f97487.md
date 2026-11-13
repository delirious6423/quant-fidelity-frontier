---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
tags: [pantheon, pantheon-local, tenstorrent, blackhole, p150]
---

# Qwen3.8-27B — Pantheon Tenstorrent build (Q8_0 source)

The files Pantheon Local installs for **Tenstorrent Blackhole p150** boxes (one card). They are
for Pantheon only: the `.fttw` files are Pantheon/Foundry's card tile format, not GGUF.

All three are made from unsloth's Q8_0 GGUF
([unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
`Qwen3.8-27B-Q8_0.gguf`, sha256 `a680f44a…b67e348`) of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team. Text only (no vision).

| file | bytes | what |
|---|---|---|
| `Qwen3.8-27B-Q8_0.meta.gguf` | 21,652,032 | the GGUF's keys (tokenizer, config) and small tensors (norms, conv taps); the matmul weights and embedding are left out. **Modified** (subset). |
| `Qwen3.8-27B-Q8_0.meta.bfp8.fttw` | 28,602,143,000 | every matmul weight and the embedding in the card's Bfp8 tile format. **Modified** (format conversion). |
| `Qwen3.8-27B-Q8_0.meta.bfp8.mtp.fttw` | 451,323,072 | the model's MTP layer, converted the same way, for Pantheon's speculative decoding. **Modified.** |

Measured on Pantheon, one p150, context cap 30,720 with the prefix checkpoint, one user, MTP off
(greedy): decode 6.8 tok/s on short prompts, 6.6 / 6.4 / 5.6 tok/s at 6k / 11k / 29k context; time
to first token 15.4 s for a 2,844-token prompt, 177 s for 28,882; a second turn resumes from the
checkpoint in about 1.4 s. With MTP on (1 draft, context 8,192): 10.6 tok/s on short prompts, same
output.

**License:** Apache License 2.0 (see `LICENSE`), the license of the original model:
https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE. Model © the Qwen team, Alibaba Cloud.
Q8_0 quantization by unsloth. These files are modified (format-converted) forms of the original
weights.

Pantheon: https://git.teamide.dev/David/Pantheon
