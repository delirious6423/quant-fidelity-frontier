---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- gguf
- ds4
- quantized
- imatrix
---

# DeepSeek V4.1 Flash, Q2 with an imatrix, for ds4

A 2-bit quantization of [DeepSeek V4.1 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(source revision `1bae0e1088993b9b82de08f59de5be5a34e42170`) for antirez's
[ds4](https://github.com/antirez/ds4) inference engine. It is a ds4 GGUF, so load it with ds4, not llama.cpp.

## Files

| File | Size | SHA-256 |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-Q2-imatrix.gguf` | 365 713 686 528 B | `c2178c08ba779ddb6f337fa7ed8ce787c5e4974be77a8e8d4187eba17cca5481` |
| `ds1017-imatrix-mixed-rust.dat` (the imatrix used) | 770 707 990 B | `6f92349b6446c27df1ff00cac7eae5e718f341e5883bd6af52b48d5c87325138` |

## How it was made

ds4's own quantizer (`gguf-tools/deepseek41_quantize.py` from [antirez/ds4](https://github.com/antirez/ds4) at commit
`fc80bd6`), at the same Q2 recipe and the same size as ds4's plain Q2 (routed experts IQ2_XXS gate/up,
Q2_K down), with an importance matrix added:

```
python3 gguf-tools/deepseek41_quantize.py --hf <DeepSeek-V4.1-Flash checkout> \
  --source-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --quant q2 --imatrix ds1017-imatrix-mixed-rust.dat --out DeepSeek-V4.1-Flash-1017-Q2-imatrix.gguf
```

To rebuild the file yourself instead of downloading it: get the DeepSeek V4.1 Flash checkpoint at the revision above
(~476 GB), `make -C gguf-tools` in ds4, and run the command with the imatrix from this repo. It took 8.3 h with `--threads 12` on an NVIDIA GB10
(20-core ARM). Compare the result with the SHA-256 in the table.

The imatrix was collected with ds4 `--imatrix-out` (CUDA, which needed a 2-line patch) over 464 prompts /
305 196 tokens: ds4's own calibration prompt set plus 34 Rust programming prompts. It covers 99.72% of routed experts.

## Results

All three files are the same size (365.7 GB) and were measured on the same GPU with the same commands. The plain-Q2
row used a ds4 build from one day earlier; the commits in between only add IQ1 code paths that these files don't use.

| | Perplexity (lower is better) | ds4-eval hard-smoke | Prefill / generation |
|---|---|---|---|
| **This file** | **2.1802** | **11/12** | ~38 / ~7 tok/s |
| antirez's IQ2_XXS-Q2_K imatrix Q2 | 2.2025 | 9/12 | same |
| ds4 plain Q2, no imatrix | 2.2369 | 7/12 | same |

- Perplexity: 15 044 held-out wikitext tokens at `--ctx 16384`. None of 160 sampled windows occur in the
  calibration text.
- hard-smoke is ds4-eval's 12-question suite (MMLU-Pro, OlympiadBench, LiveBench, NIST Juliet), greedy decoding.
  The misses are answers that ran into the token cap. **12 questions is a small sample**: treat the eval column as
  a hint, and the perplexity as the main number.

## Made by

ayourtch, with Claude (Anthropic) agents doing the quantization runs and measurements.
