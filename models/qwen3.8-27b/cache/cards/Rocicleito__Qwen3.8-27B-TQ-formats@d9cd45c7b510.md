---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: tqllm
tags:
  - qwen3.8
  - tqllm
  - quantized
  - trellis
  - gguf
---

# Qwen3.8-27B in tqLLM's TQT and TQV-12 formats

Two 4.125-bit-per-weight quantisations of the **original** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(unmodified weights), in weight formats added to [tqLLM](https://github.com/turbo-tan/tqllm):

| File | Format | Size | KLD vs BF16 | Same top token | PPL (BF16 4.880) | GSM8K 250 |
|---|---|---|---:|---:|---:|---:|
| `Qwen3.8-27B-TQT.gguf` | **TQT** (trellis code, GGUF type 306) | 14.72 GB | **0.0134** | 95.60 % | 4.917 | 243 |
| `Qwen3.8-27B-TQV12.gguf` | TQV-12 (4-D vector codebook, GGUF type 305) | 14.72 GB | 0.0184 | 94.93 % | 4.956 | 242 |
| stock `Qwen3.8-27B-TQ_L.gguf` (for reference, [YTan2000/Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ)) | TQ4 | 14.72 GB | 0.0245 | 94.17 % | 4.959 | 245 |

Same size as the stock TQ_L, about 45 % (TQT) and 25 % (TQV-12) lower KL divergence from the BF16 model.

**These files only run in tqLLM** (CUDA), from the release that includes
[PR #56](https://github.com/turbo-tan/tqllm/pull/56) (TQV-12) and [PR #59](https://github.com/turbo-tan/tqllm/pull/59) (TQT).
They do **not** load in llama.cpp, Ollama or LM Studio: the tensor types 305 / 306 are tqLLM's own.

## Run

```sh
tqserve --model Qwen3.8-27B-TQT.gguf --port 8086 \
  -md drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --draft-vocab mtp-vocab/atx_65536.txt
```

The DFlash2 drafter and vocabulary are the ones from [YTan2000/Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ);
all tqserve options for TQ_L apply.

## Measurements

RTX 3090, CUDA 13.0, one session.

- **KLD:** `llama-perplexity --kl-divergence-base` from the BF16 GGUF (llama.cpp b12198), WikiText-2 test, 8 chunks of 2048
  tokens; then `tqkld --chunks 8 --llama-prefill`. TQT 0.013443 ± 0.00058, TQV-12 0.018432 ± 0.0011, TQ_L 0.024462 ± 0.0012.
- **GSM8K:** first 250 test questions, low reasoning effort, greedy, with the DFlash2 drafter.
- **Speed** (`bench/relbench.py`, greedy, 400 tokens, `--kv tq3`, 40K context, same session), tok/s at 2K / 8K / 32K prompt:

  | | TQ_L | TQT |
  |---|---|---|
  | decode with the DFlash2 drafter | 140.2 / 125.9 / 119.3 | 145.3 / 116.3 / 116.5 |
  | decode without a drafter | 51.6 / 50.5 / 47.0 | 50.6 / 49.6 / 46.2 |
  | prefill | 1677 / 1861 / 1700 | 1658 / 1853 / 1696 |

  Without a drafter TQT costs about 2 %; with the drafter the differences follow how many drafted tokens are accepted
  on the text each file generates, not the kernel. Served with the drafter at 144K context on the same card, the TQT
  format runs at about 140-170 tok/s.

Not exact: like any quantisation, these files generate slightly different text from TQ_L and from BF16.

## How they were made

Both with `tools/convert/qwen35/tqv12_quant.py` from the tqLLM repository: vector GPTQ (TQV-12) or BlockLDLQ with a
Viterbi trellis encoder (TQT) in TQ4's rotated basis, calibrated on ~110K tokens, with the stock TQ_L as the layout
template. Step-by-step guides: [docs/guides/tqv12.md](https://github.com/turbo-tan/tqllm/blob/main/docs/guides/tqv12.md)
and [docs/guides/tqt.md](https://github.com/turbo-tan/tqllm/blob/main/docs/guides/tqt.md). A rebuild from the guides
lands at the same KLD but is not byte-identical (GPU GPTQ is not bit-reproducible).

## Checksums

| File | sha256 |
|---|---|
| Qwen3.8-27B-TQT.gguf | `69ff155303c7d31c227e82ca4ce9eb4d9229339959f8c4103026d62d1802f6fa` |
| Qwen3.8-27B-TQV12.gguf | `a4aab189ec1ad7661c202bb6d9e1209939091b7bbfd212c150ed54bde7a98d27` |

## Credits and licence

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team, Apache-2.0 (licence included).
- TQT follows QTIP (Tseng, Sun, Hou, De Sa, 2024, [arXiv 2406.11235](https://arxiv.org/abs/2406.11235)): bitshift trellis
  codes with incoherence processing. Rounding builds on GPTQ (Frantar et al.) and QuIP# (Tseng et al.).
- Engine and stock TQ formats: [tqLLM](https://github.com/turbo-tan/tqllm) by turbo-tan.
