---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - qwen3.8
  - nvfp4
  - fp8
  - rtx-5090
  - windows
---

# Qwen3.8-27B NVFP4 for the q35 engine

This is unsloth's NVFP4 checkpoint of Qwen3.8-27B, repacked for the q35 engine of the Mugearite fork of Basalt
(Windows, one RTX 5090). The NVFP4 and FP8 words are copied, not re-quantized. The pack also carries unsloth's vision projector
for image input.

Experimental. The file format may change.

## Files

| File | Size | SHA-256 |
|---|---|---|
| `Qwen3.8-27B-NVFP4.basalt` | 21.6 GiB | `dcddade5c2f2264d1fcc347474970b0950940ad75a3e3756269ffab3ca2e71b6` |
| `basalt-q27-nvfp4-1slot.json` | | example server config |

## Where it comes from

- Source: [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4), a quantization of
  [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Both are Apache-2.0. The tokenizer and metadata come from
  unsloth's Q8_0 GGUF of the same model.
- Made with `tools/convert/q35_nvfp4.py --meta-from` and `tools/convert/convert.py` on the fork's `q35-dense` branch
  (fa8daa59). The vision sections were added by packing the same GGUF again with `convert.py --mmproj` at
  the fork's commit a7e4e096.
- What the conversion changes:
  - MLP of layers 0-55: NVFP4 rows (e2m1 codes, e4m3 scales per 16 values, one global scale), as in the checkpoint.
  - Attention, GDN projections, MLP of layers 56-63 and the output head: FP8 rows (e4m3, one scale per row), as in
    the checkpoint.
  - MTP layer: Q8_0. Token embedding: bf16.
  - The pack adds `mtp/draft_vocab`, a 131,072-token shortlist for the draft head (`data/q35-draft-vocab-128k.bin`).
    It is ranked from NInfer's token-frequency counts (Apache-2.0, github.com/Neroued/ninfer), weighted toward English
    chat and code.
- Vision: `mmproj-BF16.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (revision 4ca72078), stored whole
  as `vision/mmproj`, plus `vision/text_vocab` (the text model's tokenizer metadata, for the image markers).
  The earlier version of this pack had no vision sections; its text weights and draft vocabulary are the
  same bytes as here.

## Quality

Wikitext-2, 100 chunks of 512 tokens, against a llama.cpp Q8_0 KL-divergence base (base PPL 6.7466).

| | PPL | KLD mean | same top-1 |
|---|---:|---:|---:|
| q35 prompt path (4-bit activations in the NVFP4 GEMMs) | 6.8488 | 0.02851 | 92.46% |
| q35 decode path (8-bit activations) | | 0.01616 | |

Speculative decoding is exact: output with drafts equals output without them, and batched slots equal solo runs.

## Speed

Development measurements on one RTX 5090 power-limited to 450 W, without the image encoder, not benchmark runs.

| | tok/s |
|---|---:|
| decode, no drafts | about 75 |
| decode, 3-4 MTP drafts (varies with the text) | about 140-245 |
| prompt read, 8K / 32K / 100K tokens | about 12,000 / 9,700 / 6,300 |

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

Needs the q35 engine build (Windows, CUDA 13, sm_120; NVFP4 needs Blackwell). With one slot and an int8 KV cache,
195,072 tokens of context fit on a 32 GB card. The example config turns on image input (`"vision":
true`): the encoder then runs on the same card and the server keeps 1.5 GB of VRAM free for it, so the
context is 179,712 tokens. Drop `vision` for the full 195,072. Start from the example config:

```
basalt serve --config basalt-q27-nvfp4-1slot.json --api-key <your key>
```

Keep the server on 127.0.0.1, or set a real API key before you expose it.
