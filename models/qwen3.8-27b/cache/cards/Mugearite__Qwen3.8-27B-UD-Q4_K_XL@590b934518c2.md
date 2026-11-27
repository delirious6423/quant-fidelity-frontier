---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-GGUF
base_model_relation: quantized
tags:
  - qwen3.8
  - gguf-derived
  - rtx-5090
  - windows
---

# Qwen3.8-27B UD-Q4_K_XL for the q35 engine

This is unsloth's UD-Q4_K_XL GGUF of Qwen3.8-27B, repacked for the q35 engine of the Mugearite fork of Basalt
(Windows, one RTX 5090). The weights are the GGUF's bytes, unchanged. The pack adds a draft vocabulary for speculative
decoding and unsloth's vision projector for image input.

Experimental. The file format may change.

## Files

| File | Size | SHA-256 |
|---|---|---|
| `Qwen3.8-27B-UD-Q4_K_XL.basalt` | 17.3 GiB | `0c214ecca7d931c2e2266f8ac381bf2212f4a05ad130a7f8cca9e43dda04766a` |
| `basalt-q27-1slot.json` | | example server config |

## Where it comes from

- Source: `Qwen3.8-27B-UD-Q4_K_XL.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
  a quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Both are Apache-2.0.
- Made with `tools/convert/convert.py` at the fork's commit b31c8283.
- What the conversion changes: the tensors are copied as stored. The pack adds `mtp/draft_vocab`, a 131,072-token
  shortlist for the draft head (`data/q35-draft-vocab-128k.bin`). It is ranked from NInfer's token-frequency counts
  (Apache-2.0, github.com/Neroued/ninfer), weighted toward English chat and code.
- Vision: `mmproj-BF16.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (revision 4ca72078), stored whole
  as `vision/mmproj`, plus `vision/text_vocab` (the text model's tokenizer metadata, for the image markers).
  The earlier version of this pack had no vision sections; its text weights and draft vocabulary are the
  same bytes as here.

## Quality

Wikitext-2, 100 chunks of 512 tokens, against a llama.cpp Q8_0 KL-divergence base (base PPL 6.7466).

| | PPL | KLD mean | same top-1 |
|---|---:|---:|---:|
| this pack, q35 prompt path | 6.7442 | 0.00550 | 96.75% |
| this pack, q35 decode path | 6.7436 | 0.00554 | 96.73% |
| the same GGUF on llama.cpp | 6.7410 | 0.006849 | 96.31% |

Speculative decoding is exact: output with drafts equals output without them, and batched slots equal solo runs.

## Speed

Development measurements on one RTX 5090 power-limited to 450 W, without the image encoder, not benchmark runs.

| | tok/s |
|---|---:|
| decode, no drafts | about 86 |
| decode, 3-4 MTP drafts (varies with the text) | about 150-240 |
| prompt read, 8K / 32K tokens | about 5,200 / 4,800 |
| 16 slots together | about 600 |

## Conversation cache

The server keeps each slot's conversation on the GPU between requests: its KV cache and its recurrent (GDN) state. The
next turn of that conversation reads only the new part of its prompt.

When another conversation needs the slot, the old one is parked in host RAM. The example config allows 64 GiB for up
to 32 parked conversations (`conversation-cache-mib`, `conversation-cache-slots`). When its next turn arrives it comes
back without re-reading what it had.

The engine also saves checkpoints of the recurrent state in pinned host memory, at the points where a chat's next
request most often departs, such as before the last turn. A request that edits earlier history restarts from the
nearest checkpoint, not from the start.

Parked conversations can also go to disk (`--conversation-cache-dir` with `--conversation-cache-disk-mib`).

## Running it

Needs the q35 engine build (Windows, CUDA 13, sm_120). With one slot and an int8 KV cache the whole 262,144-token
context fits on a 32 GB card. The example config turns on image input (`"vision": true`): the encoder
then runs on the same card and the server keeps 1.5 GB of VRAM free for it, so the context is 252,672
tokens. Drop `vision` for the full context. Start from the example config:

```
basalt serve --config basalt-q27-1slot.json --api-key <your key>
```

Keep the server on 127.0.0.1, or set a real API key before you expose it.
