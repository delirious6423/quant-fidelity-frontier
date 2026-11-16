---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: qvb8
tags: [qvb8, quantization, compression, certified-compression, linf-bounded, certified-error, qwen]
---
# Qwen3.8-27B_qvb8r_2e-3_fp16

> ## ⚡ Runs on the QVB8 Inference Engine
> This pack is made for the **[QVB8 Inference Engine](https://github.com/addiesam/qvb8)**. It does not load in `transformers` or other engines.
> The engine decodes the compressed weights on the GPU, inside its own CUDA kernels, so a large model runs on small hardware.
> **Get the engine: [github.com/addiesam/qvb8](https://github.com/addiesam/qvb8)** -- use **v1.2.0 or later**.

**Certified L∞-bounded compression:** for every restored weight, the error against the original is less than the error bound tau.

The Qwen3.8-27B language model (hybrid Gated-DeltaNet / attention, with vision input), in the smaller entropy-coded QVB8R format, compressed with **QVB8R** at **tau = 0.002**. This is a weight pack for the
[QVB8 inference engine](https://github.com/addiesam/qvb8); it is not loadable by `transformers` directly.

| | |
|---|---|
| Source model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (bf16) |
| Format | QVB8R |
| Certified for | fp16 (`restore: fp16` in `manifest.json`); run it with `--dtype fp16` |
| tau | 0.002 (for every restored weight, the error against the original bf16 value is less than tau; largest error found: 0.001999) |
| Compressed tensors | 618, average 3.91 bits per weight over them |
| Size on disk | 13.6 GB |

## Download
All Qwen packs of this release. Use one target pack and its matching draft pack with the [QVB8 Inference Engine](https://github.com/addiesam/qvb8).

| Pack | Format | tau | Size | Download |
|---|---|---|---|---|
| Qwen3.8-27B | qvb8 | 1e-3 | 21.6 GB | [Qwen3.8-27B_qvb8_1e-3_fp16](https://huggingface.co/addiesam/Qwen3.8-27B_qvb8_1e-3_fp16) |
| Qwen3.8-27B | qvb8r | 2e-3 | 13.6 GB | [Qwen3.8-27B_qvb8r_2e-3_fp16](https://huggingface.co/addiesam/Qwen3.8-27B_qvb8r_2e-3_fp16) |
| Qwen3.8-27B DFlash2 draft model | qvb8r | 5e-3 | 1.5 GB | [Qwen3.8-27B-DFlash2_qvb8r_5e-3_fp16](https://huggingface.co/addiesam/Qwen3.8-27B-DFlash2_qvb8r_5e-3_fp16) |
| Qwen3.5-9B | qvb8 | 1e-3 | 7.6 GB | [Qwen3.5-9B_qvb8_1e-3_fp16](https://huggingface.co/addiesam/Qwen3.5-9B_qvb8_1e-3_fp16) |
| Qwen3.5-9B DFlash draft model | qvb8 | 1e-3 | 1.4 GB | [Qwen3.5-9B-DFlash_qvb8_1e-3](https://huggingface.co/addiesam/Qwen3.5-9B-DFlash_qvb8_1e-3) |

## Use

```bash
git clone https://github.com/addiesam/qvb8 && cd qvb8 && ./install.sh
./qvb8 get qwen-27b      # downloads this pack and its draft model
./qvb8 start qwen-27b    # server on port 8080
./qvb8 chat
```

By hand, with the launcher: `MODEL=./models/Qwen3.8-27B_qvb8r_2e-3_fp16 DRAFT=./models/<draft pack> DRAFT_K=7 scripts/qwen_server.sh start`.

## Why certified L∞-bounded compression?
Most quantization is judged by an average: a perplexity or benchmark score. An average hides the worst case. A format can
look good on average and still leave a few weights with a large error, and a large error in the wrong weight can flip a
single token. The result is a sudden wrong word or a stray character from another language in the middle of a normal
answer, which an average score may not show.

L∞-bounded means the **largest** error is controlled, not the average. For this pack:
- every restored weight differs from the original by less than **tau** (0.002), and no tensor is allowed to exceed it;
- the check is made after packing, tensor by tensor, and the largest error found is stored in `manifest.json` (`max_err`);
- tau is your choice: a smaller tau is a tighter bound, a larger file and a perplexity closer to the original; a larger
  tau is a smaller file and a higher perplexity. You choose the trade-off knowing the worst case.

What the bound does **not** do: it covers the stored weights only. It does not promise that the model's answers are
identical to the original's, or correct. Even with the exact same unquantized FP16 weights, different inference
engines, batch sizes, or floating-point operation orders will still produce slightly different outputs due to
non-associative GPU math. See the engine README for the full disclaimer.

## How it was made
Each weight matrix is packed with a per-row bit width and checked after packing: the largest error between the
restored value and the original bf16 weight is recorded per tensor in `manifest.json` (`max_err`), and no tensor
exceeds tau. The packing uses no calibration data; it depends only on the weights.

## Update history

**2026-11-15: first release of this pack** (QVB8 Inference Engine v1.2.0)
- Qwen3.8-27B (QVB8R, tau 2e-3), certified for fp16; `max_err` per tensor is stored in `manifest.json`.

## License and credits
The weights are derived from Qwen/Qwen3.8-27B by its authors (Apache-2.0); this pack keeps that license.
The engine code is Apache-2.0. No warranty is given: see the engine README for the full disclaimer.

## Support this work
I am an independent researcher, and I publish these packs and the engine openly. I need machines to run more tests:
other GPU generations, and more models. If this is useful to you, please support the work:

**[Sponsor Dr. Addie Sam on GitHub](https://github.com/sponsors/addiesam)**

## Papers
Sam, Chi Ngai. *Certified L∞-Bounded Compression for AI Models, I: The Contract and the Virtual-Band Index.* [doi:10.5281/zenodo.23815007](https://doi.org/10.5281/zenodo.23815007)

Sam, Chi Ngai. *Certified L∞-Bounded Compression for AI Models, II: Compressing a Language Model's Weights, Key–Value Cache and Activations.* [doi:10.5281/zenodo.23816044](https://doi.org/10.5281/zenodo.23816044)

Sam, Chi Ngai. *Applying Certified L∞-Bounded Compression to LLM Serving: Portable Saved State, Shared Pools and Mixed-Generation GPUs.* [doi:10.5281/zenodo.24267208](https://doi.org/10.5281/zenodo.24267208)
