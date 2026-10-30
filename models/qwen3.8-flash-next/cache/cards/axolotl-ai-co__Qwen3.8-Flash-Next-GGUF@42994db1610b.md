---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
tags:
- gguf
- llama.cpp
- quantized
---

# Qwen3.8-Flash-Next XQD GGUF

XQD quantizations of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at six sizes. How XQD works is described in our write-up on Qwen3.8-27B, [Old ideas, standard formats, better quants](https://axolotlai.substack.com/p/old-ideas-standard-formats-better).

## Results

KL divergence from the BF16 model on assistant-generated tokens, over agent traces that share no text with our calibration data (`lambda/hermes-agent-reasoning-traces`, 201,812 scored positions). Lower is better. The KLD column is comparable across rows; each margin is only meaningful against the release named on its row.

| files | tensor payload | KLD vs BF16 | change in KLD vs a same-size release |
|---|---|---|---|
| `Qwen3.8-Flash-Next-XQD-IQ1_M-*.gguf` | 74.5 GB | 0.038461 | -23.4% vs UD-IQ1_M |
| `Qwen3.8-Flash-Next-XQD-Q2_K_XL-*.gguf` | 78.8 GB | 0.026465 | -21.1% vs UD-Q2_K_XL |
| `Qwen3.8-Flash-Next-XQD-IQ3_XXS-*.gguf` | 81.8 GB | 0.019658 | -23.7% vs UD-IQ3_XXS |
| `Qwen3.8-Flash-Next-XQD-Q3_K_XL-*.gguf` | 89.8 GB | 0.013026 | -15.5% vs UD-Q3_K_XL |
| `Qwen3.8-Flash-Next-XQD-IQ4_XS-*.gguf` | 93.7 GB | 0.010983 | -12.5% vs UD-IQ4_XS |
| `Qwen3.8-Flash-Next-XQD-Q4_K_XL-*.gguf` | 111.3 GB | 0.006672 | see below |

Sizes are the sum of tensor bytes; the files are about 11 MB larger because of the GGUF header in each shard. Each comparison release is the same size or larger. At 89.8 GB the head and embeddings use different formats from UD-Q3_K_XL, so part of that difference may come from where the bytes go rather than from the rounding.

At 111.3 GB the mean difference is not stable across parts of the evaluation set (two parts favour XQD, one favours UD-Q4_K_XL), so we do not quote one. Across all positions, XQD's KLD is 0.88x UD-Q4_K_XL's at the median, 90th and 95th percentiles, and 0.94x at the 99th.

At the two sizes where we also scored prompt and tool-output tokens, XQD was lower on every kind of position: at 81.8 GB, -23.7% on assistant tokens, -17.3% on context tokens and -17.4% over all positions; at 93.7 GB, -12.5%, -13.3% and -13.3%.

On wikitext, all positions, the 74.5 GB build has 8.0% lower KLD than UD-IQ1_M. Wikitext is one of our calibration sources, so this is not a held-out comparison.

## How these quants were made

They are post-training quantizations. We did no fine-tuning: the weights are the original model's weights rounded to fewer bits, and they load as ordinary GGUF files.

Each build quantizes with our own importance matrix, keeping every tensor `llama-quantize` leaves unquantized at full precision, and applies GPTQ-style error compensation: as each block of weights is rounded, its rounding error is pushed onto the weights not yet rounded, using one shared input Hessian per layer. Compensation covers the expert gate and up projections at every size, and in the two smallest builds also the full-attention projections and the output head. Every tensor we did not change is byte-identical to a plain `llama-quantize` build.

The small tensors left at full precision matter. 169 of them (the hyper-connection injections, the SSM `alpha`/`beta` and the n-gram conv) take 0.064% of the file; quantizing them to IQ2_XXS made the model 7.2% worse, so we keep them at full precision.

Caveat: lower KLD means closer to the original model's output distribution. We have not shown that it moves downstream benchmarks, so do not read these margins as benchmark margins.

Text only: no `mmproj` is included, so the base model's vision path is not available in these files.

## Usage

Files are split into shards under the Hugging Face 50 GB limit; point llama.cpp at the first shard and it finds the rest.

```bash
hf download axolotl-ai-co/Qwen3.8-Flash-Next-GGUF --include "Qwen3.8-Flash-Next-XQD-IQ1_M-*" --local-dir .
llama-server --model Qwen3.8-Flash-Next-XQD-IQ1_M-00001-of-00002.gguf --n-gpu-layers 99 --ctx-size 32768
```

### Low-VRAM tip

The 28.8 GB n-gram table (`per_layer_token_embd.weight`) is only ever indexed, never multiplied, so it can live in host RAM instead of VRAM:

```
-ot "per_layer_token_embd\.weight=CPU"
```

On the 74.5 GB build at `-c 4096` this needs 43,898 MiB of VRAM, with the 28.8 GB table in host RAM. The KV cache and compute buffers grow with context length and come on top of that. Because the table is a lookup, moving it does not change the output.

On a single B300 with fast host memory the override cost nothing measurable (71.2 tok/s with it, 70.0 tok/s with everything on the GPU). That is one short run on server hardware; on a machine whose host memory is much slower than its VRAM, expect a real cost.

The importance matrix used for these builds is included as `imatrix-calib-v3.gguf`.

## License

The [Qwen community license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), inherited from the base model.
