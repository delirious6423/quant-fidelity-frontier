---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- ds4
- apple-metal
- qwen
- mtp
---

# Qwen3.8-Flash-Next MTP block (1.4 GB)

The MTP (multi token prediction) block of Qwen3.8-Flash-Next as a standalone
GGUF: the 32 `blk.48.*` tensors and the model metadata, nothing else. It is
meant for GGUF releases that ship without the block, such as
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
so that speculative decoding works without downloading a second full model.

| File | Size | SHA-256 |
|---|---|---|
| `Qwen3.8-Flash-Next-MTP-block.gguf` | 1.52 GB (1.41 GiB) | `bbb435d7ba548ddd5b2d6f660863bf56836e76f2b7d7ef98fa2c983fd0fefae1` |

Quantization: Q4_K gate/up experts, MXFP4 down experts, Q8_0 dense
projections, F16 hyper-connection mixers.

## Where it comes from

The tensors are copied byte for byte from
[ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2),
which was quantized from the original
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
weights. Nothing was requantized or retrained here.

## Usage

With the [paperniuk/ds4](https://github.com/paperniuk/ds4/tree/m1-flash-next)
fork of DwarfStar (branch `m1-flash-next`):

```sh
hf download paperniuk/Qwen3.8-Flash-Next-MTP-block --local-dir ~/models/flash-next/mtp

M=~/models/flash-next/Q2_0
DS4_QWEN_NGRAM_GGUF=$M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
DS4_QWEN_MTP_GGUF=~/models/flash-next/mtp/Qwen3.8-Flash-Next-MTP-block.gguf \
./ds4-server -m $M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ctx 262144 --prefill-chunk 2048 --mtp
```

## Measurements

M1 Max 64 GB, ISTA Q2_0 main model, 4K context, greedy, 400 tokens. Output
is identical to plain decoding in every run.

| Prompt | No MTP | This block | ggml-org `mtp-...-Q8_0.gguf` (3.9 GB) |
|---|---|---|---|
| Code | 34.6 tok/s | 42.7 to 43.8 tok/s | 43.2 tok/s |
| Prose | 33.6 tok/s | 37.9 tok/s | 37.7 tok/s |

First draft acceptance: 91% on code, 70% on prose. The larger Q8_0 block
from [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF)
also works with the fork and is as fast; this one needs 1.2 GiB less memory.

## License

Qwen Community License 1.0, the same as the original model. The full terms
are in [LICENSE](LICENSE). Read it before any commercial use.
