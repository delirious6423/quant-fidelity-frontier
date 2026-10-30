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

# Qwen3.8-27B-Q4-GGUF

A 4-bit-class XQD quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in GGUF. How it is made and measured is in our write-up, [Old ideas, standard formats, better quants](https://axolotlai.substack.com/p/old-ideas-standard-formats-better).

| file | size |
|---|---:|
| `Qwen3.8-27B-XQD-Q4.gguf` | 17.55 GB |

### Tensor formats

The format chosen for each weight matrix, by layer:

![Per-tensor GGML formats in XQD Q4](https://huggingface.co/axolotl-ai-co/Qwen3.8-27B-Q4-GGUF/resolve/main/assets/precision-axolotl-q4.svg)

Compared with kaitchup's SLQ-EAR98 (upper-left triangle is ours, lower-right is SLQ-EAR98; a solid cell means both use the same format):

![XQD Q4 vs SLQ-EAR98 per-tensor formats](https://huggingface.co/axolotl-ai-co/Qwen3.8-27B-Q4-GGUF/resolve/main/assets/precision-q4-vs-slq-ear98.svg)

### How these quants were made

These GGUF quants are built with XQD (aXolotl Quantization for Decoding), our quantization recipe. They are tuned for chat and agentic use, meaning the tokens the model itself writes in conversations and tool-use sessions. Calibration, allocation and evaluation all target assistant-generated tokens. We do not tune for prose or pretraining-style text, such as wikitext perplexity or web-document fidelity, and on that kind of text other quantizations may score better.

They are post-training quantizations of Qwen3.8-27B. We did no fine-tuning: the weights are the original model's weights, rounded to fewer bits, and they load as ordinary GGUF files that need nothing special to run.

We capture activation statistics (a Hessian, which records which input directions matter most) from our own calibration mix, mostly long agentic and reasoning transcripts, and build the importance matrix (imatrix) from the same data. On top of the standard llama.cpp formats we apply GPTQ-style error compensation: as each block of weights is rounded, its rounding error is pushed onto the weights not yet rounded, so the layer's output stays close to the original. A bit allocator then chooses a format for each tensor under a fixed byte budget, using the KLD each group of tensors adds when quantized, measured on this model.

Because the output head gets its own importance data (a default `llama-imatrix` run leaves it out), a guided Q6_K head is enough. Other releases keep the head at Q8_0, a format that cannot use importance data at all, and we spend the difference on the rest of the model.

The multi-token-prediction (MTP) block is kept but is not counted in the size budget.

We judge quality by KL divergence from the full-precision model, measured only on tokens the assistant generated, over agentic and chat transcripts, rather than by perplexity. On our held-out set the Q4 (17.55 GB) has 27.3% lower KLD than unsloth's UD-Q4_K_XL and statistically ties two larger quants (kaitchup SLQ-EAR98 and unsloth UD-Q5_K_S). A Q5 sized to match SLQ-EAR98 has about 23% lower KLD than it. The worst 0.5% of positions, where the damage concentrates, improve by about 30% against both. On SLQ's own EAR@10 metric the ranking is the same on agentic text, and on web prose SLQ-EAR98 scores higher.

Credits: built on llama.cpp's formats and kernels, with error compensation from GPTQ and its predecessors. Our comparisons and several findings owe a lot to unsloth's dynamic quants, APEX, and SLQ with kaitchup's release. The full method and measurements are in the write-up, [Old ideas, standard formats, better quants](https://axolotlai.substack.com/p/old-ideas-standard-formats-better).

Caveat: lower KLD means closer to the original model's output distribution. We have not shown that it moves downstream benchmarks, so do not read these margins as benchmark margins.

Text only: no vision tensors or multimodal projector are included. The file keeps the model's multi-token-prediction (MTP) block.

## Usage

Use a recent llama.cpp build with Qwen3.8 support.

```bash
hf download axolotl-ai-co/Qwen3.8-27B-Q4-GGUF Qwen3.8-27B-XQD-Q4.gguf --local-dir .
llama-server --model Qwen3.8-27B-XQD-Q4.gguf --n-gpu-layers 99 --ctx-size 32768
```

## License

Apache 2.0, inherited from the base model. See [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for the original license and terms.
