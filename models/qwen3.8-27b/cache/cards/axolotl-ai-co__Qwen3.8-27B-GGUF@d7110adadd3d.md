---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
tags:
- gguf
- llama.cpp
- quantized
---

# Qwen3.8-27B XQD GGUF

XQD quantizations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at 2, 3 and 4 bits. Each quant comes in two versions: one with the model's multi-token-prediction (MTP) block, and one without it (`-nomtp`) for runtimes that don't use it. The 4-bit quant is also at [axolotl-ai-co/Qwen3.8-27B-Q4-GGUF](https://huggingface.co/axolotl-ai-co/Qwen3.8-27B-Q4-GGUF), and the 5-bit quant is at [axolotl-ai-co/Qwen3.8-27B-Q5-GGUF](https://huggingface.co/axolotl-ai-co/Qwen3.8-27B-Q5-GGUF).

How these quants are made and measured is in our write-up, [Old ideas, standard formats, better quants](https://axolotlai.substack.com/p/old-ideas-standard-formats-better).

| file | size |
|---|---:|
| `Qwen3.8-27B-XQD-Q2.gguf` | 9.82 GB |
| `Qwen3.8-27B-XQD-Q2-nomtp.gguf` | 9.47 GB |
| `Qwen3.8-27B-XQD-Q3.gguf` | 13.14 GB |
| `Qwen3.8-27B-XQD-Q3-nomtp.gguf` | 12.79 GB |
| `Qwen3.8-27B-XQD-Q4.gguf` | 17.55 GB |
| `Qwen3.8-27B-XQD-Q4-nomtp.gguf` | 17.20 GB |

## Results

![Assistant-token KLD vs file size for XQD and other Qwen3.8-27B GGUF quants](https://huggingface.co/axolotl-ai-co/Qwen3.8-27B-GGUF/resolve/main/assets/kld-vs-size.svg)

KL divergence from the full-precision model, on assistant-generated tokens only, over a held-out set of agentic and chat transcripts (1.2M scored positions). Each quant is compared with another release of about the same size. Lower is better.

| Comparison | Size | Change in mean KLD | Change in worst 0.5% |
|---|---|---|---|
| XQD Q2 vs unsloth UD-Q2_K_XL | 9.82 GB vs 9.83 GB | -22.2% (95% CI [-23.5, -20.9]) | -9.0% |
| XQD Q3 vs unsloth UD-Q3_K_XL | 13.14 GB vs 13.15 GB | -32.0% (95% CI [-33.1, -30.9]) | -29.4% |
| XQD Q4 vs unsloth UD-Q4_K_XL | 17.55 GB vs 17.56 GB | -27.3% (95% CI [-33.0, -21.7]) | -33.3% |
| XQD Q5 vs kaitchup SLQ-EAR98 | 18.85 GB vs 18.88 GB | -23.5% (95% CI [-26.3, -20.6]) | -29.9% |

The Q4 also ties two larger quants, kaitchup SLQ-EAR98 and unsloth UD-Q5_K_S, while being 1.1-1.3 GB smaller than either.

## How these quants were made

These GGUF quants are built with XQD (aXolotl Quantization for Decoding), our quantization recipe. They are tuned for chat and agentic use, meaning the tokens the model itself writes in conversations and tool-use sessions. Calibration, allocation and evaluation all target assistant-generated tokens. We do not tune for prose or pretraining-style text, such as wikitext perplexity or web-document fidelity, and on that kind of text other quantizations may score better.

They are post-training quantizations of Qwen3.8-27B. We did no fine-tuning: the weights are the original model's weights, rounded to fewer bits, and they load as ordinary GGUF files that need nothing special to run.

We capture activation statistics (a Hessian, which records which input directions matter most) from our own calibration mix, mostly long agentic and reasoning transcripts, and build the importance matrix (imatrix) from the same data. On top of the standard llama.cpp formats we apply GPTQ-style error compensation: as each block of weights is rounded, its rounding error is pushed onto the weights not yet rounded, so the layer's output stays close to the original. A bit allocator then chooses a format for each tensor under a fixed byte budget, using the KLD each group of tensors adds when quantized, measured on this model.

Because the output head gets its own importance data (a default `llama-imatrix` run leaves it out), a guided head needs fewer bits. Other releases keep the head at Q8_0, a format that cannot use importance data at all, and we spend the difference on the rest of the model.

The MTP block is kept in the files without the `-nomtp` suffix but is not counted in the size budget.

Credits: built on llama.cpp's formats and kernels, with error compensation from GPTQ and its predecessors. Our comparisons and several findings owe a lot to unsloth's dynamic quants, APEX, and SLQ with kaitchup's release. The full method and measurements are in the write-up, [Old ideas, standard formats, better quants](https://axolotlai.substack.com/p/old-ideas-standard-formats-better).

Caveat: lower KLD means closer to the original model's output distribution. We have not shown that it moves downstream benchmarks, so do not read these margins as benchmark margins.

Text only: no vision tensors or multimodal projector are included.

## Experimental builds

These files are earlier experiments, kept for reference. They were measured only on an older, smaller held-out set, not the one above, and are not XQD releases.

| file | size |
|---|---:|
| `qwen38-masked199-iq3shead.gguf` | 6.67 GB |
| `qwen38-2.02bpw-quality-iq3shead.gguf` | 6.80 GB |
| `qwen38-v3capped-iq3shead.gguf` | 7.27 GB |
| `qwen38-iq3xxs-c497.gguf` | 10.93 GB |
| `qwen38-iq3s-c497.gguf` | 12.04 GB |
| `qwen38-q3kxl-c497.gguf` | 13.15 GB |

## Usage

Use a recent llama.cpp build with Qwen3.8 support.

```bash
hf download axolotl-ai-co/Qwen3.8-27B-GGUF Qwen3.8-27B-XQD-Q3.gguf --local-dir .
llama-server --model Qwen3.8-27B-XQD-Q3.gguf --n-gpu-layers 99 --ctx-size 32768
```

## License

Apache 2.0, inherited from the base model. See [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for the original license and terms.
