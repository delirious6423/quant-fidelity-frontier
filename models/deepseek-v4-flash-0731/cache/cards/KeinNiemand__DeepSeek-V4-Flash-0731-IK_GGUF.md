---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
- gguf
- quantization
- ik_llama.cpp
- deepseek
- deepseek-v4
- deepseek_v4
- deepseek4
- moe
- mixture-of-experts
- imatrix
- mxfp4
- qat
- iq1_kt
- iq2_kt
- iq2_ks
- iq2_kl
- iq3_ks
- iq3_k
- iq4_kss
pipeline_tag: text-generation
license: mit
---

# DeepSeek V4 Flash 0731 IK GGUF

Mixed-precision GGUF quantizations of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), made with [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and an importance matrix. The ladder ranges from a sub-64-GiB build to a near-native 4-bit build.

The routed MXFP4 tensors were requantized directly from their native representation. FP8-sourced tensors were expanded to BF16 before quantization.

## September 2026 file correction

`XS_IQ1_KT`, `IQ1_KT`, `IQ2_KL`, `IQ3_KS`, `IQ3_K`, and `IQ4_KSS` have been rebuilt. A Python metadata rewrite had truncated tensors with per-row headers in the previous files. Those files must be replaced. `IQ2_KS` and `IQ2_KT` were unaffected and are unchanged.

The rebuilt files embed the template during native quantization. Their tensor-region sizes and template were checked, and sampled first/last rows were dequantized and checked for finite values before transfer. Selected expert groups previously using `IQ4_KS` or `IQ4_K` now retain native MXFP4. This change does not apply to higher-precision non-expert tensors.

## Compatibility warning

> These files are intended for a recent build of `ik_llama.cpp`. They contain IK-specific IQK and IQT tensor types. Mainline `llama.cpp` and applications built against it, including many releases of LM Studio, KoboldCpp, and text-generation-webui, may not load them. Check that your backend explicitly supports the tensor types in the chosen file.

DeepSeek V4 Flash advertises a very large context window. Set `-c` explicitly unless you want the runtime to reserve a correspondingly large KV cache. `IQ2_KT` is slower than `IQ2_KS` on CPU and CPU/GPU hybrid inference despite its similar size.

## Available files

Sizes below are the final release-file sizes. Leave additional memory for the KV cache, runtime buffers, the operating system, and any layers that do not fit cleanly on a GPU.

| Quant | Size (GB) | Size (GiB) | Average bpw | Notes |
|---|---:|---:|---:|---|
| `XS_IQ1_KT` | 65.435 | 60.941 | 1.841 | Smallest build; intended for tight 64-GiB-class systems |
| `IQ1_KT` | 73.178 | 68.152 | 2.059 | Higher-quality IQ1 tier |
| `IQ2_KT` | 93.028 | 86.639 | 2.618 | IQT alternative; slower on CPU and CPU/GPU hybrid inference |
| `IQ2_KS` | 93.443 | 87.026 | 2.629 | Recommended starting point near 93 GB |
| `IQ2_KL` | 106.319 | 99.018 | 2.992 | Clear quality step above the IQ2_KS/KT pair |
| `IQ3_KS` | 122.017 | 113.637 | 3.434 | Untested: too large for the available KLD-test hardware |
| `IQ3_K` | 132.757 | 123.640 | 3.736 | Untested: too large for the available KLD-test hardware |
| `IQ4_KSS` | 149.239 | 138.990 | 4.200 | Untested: too large for the available KLD-test hardware |

Average bpw is file-size-derived relative to the 4.555-bpw BF16/MXFP4 source. It covers the full GGUF, including non-expert tensors and format overhead.

Pick the largest file that leaves enough headroom for your intended context and offload split. File size alone is not a complete memory estimate.

## Quality measurements

The chart compares the locally evaluated IK ladder with the two valid Atomic reference runs. Every plotted quant was tested against the same lossless BF16 reference distributions.

![Mean KL divergence versus file size](DeepSeek-V4-Flash-0731-kld-ladder.png)

| Quant | Evaluated size (GB) | PPL | Mean KLD | RMS delta-p | Same top-1 |
|---|---:|---:|---:|---:|---:|
| `XS_IQ1_KT` | 65.435 | 6.028959 | 0.497826 | 23.005% | 76.077% |
| Atomic reference `AD-IQ1_M_XL` | ~72.800 | 5.867262 | 0.485612 | 23.218% | 76.410% |
| `IQ1_KT` | 73.178 | 5.463794 | 0.393212 | 20.338% | 79.076% |
| `IQ2_KT` | 93.028 | 4.931026 | 0.269956 | 16.558% | 83.000% |
| Atomic reference `AD-IQ2_S` | 93.396 | 5.014600 | 0.293420 | 17.659% | 82.109% |
| `IQ2_KS` | 93.443 | 4.914112 | 0.268909 | 16.497% | 83.029% |
| `IQ2_KL` | 106.319 | 4.664766 | 0.205305 | 14.368% | 85.427% |

Lower PPL, KLD, and RMS delta-p are better. Higher top-1 agreement is better.

At almost the same size as Atomic `AD-IQ2_S`, this release's `IQ2_KS` has 8.35% lower mean KLD, 0.100488 lower PPL, 1.162 percentage points lower RMS delta-p, and 0.920 percentage points higher top-1 agreement. The measured `IQ1_KT` also has 19.03% lower mean KLD than Atomic `AD-IQ1_M_XL` at a similar size.

`XS_IQ1_KT`, `IQ1_KT`, and `IQ2_KL` were retested on the rebuilt release artifacts. The unchanged `IQ2_KT` and `IQ2_KS` retain their previous release-artifact measurements.

### Test method

The KLD evaluation follows AtomicChat's published setup and uses its [DeepSeek V4 evaluation artifacts](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts):

- lossless `AD-BF16` logits from Atomic's `ref5632.kld` as the reference distribution;
- the exact corresponding WikiText-2 tokens at context 5632;
- the first 31 of Atomic's 51 evaluation chunks;
- the same batching configuration and tensor split, with batch and microbatch sizes of 512 and seed 1337;
- perplexity, full-distribution KL divergence, RMS probability change, and reference top-1 agreement.

The rebuilt `XS_IQ1_KT`, `IQ1_KT`, and `IQ2_KL` were tested with IK build 4882, commit `3e416d7f5`. The retained `IQ2_KS`, `IQ2_KT`, and two Atomic reference runs used build 4838, commit `7cd62a3eb`. The reference tokens, BF16 logits, chunk selection, hardware, and requested test settings are unchanged, but this is not a single-executable comparison. Small differences may include runtime changes.

The two Atomic GGUFs in the table were rerun locally through the same 31-chunk harness. Do not compare these absolute values directly with Atomic's published 51-chunk table.

`IQ3_KS`, `IQ3_K`, and `IQ4_KSS` were not KLD-tested. They exceed the available local hardware limit for this evaluation, so no quality claim is made for them here.

## Running the model

Use a current `ik_llama.cpp` build. For example:

```bash
./llama-server \
  -m DeepSeek-V4-Flash-0731-IQ2_KS.gguf \
  -c 8192 \
  -ngl 99 \
  --jinja \
  --chat-template-kwargs '{"reasoning_effort":"max"}' \
  --reasoning-format deepseek
```

Adjust context, GPU layers, tensor split, and batch settings for your hardware. The GGUF metadata embeds the corrected Jinja template from [Unsloth's safetensors repository](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731), including its reasoning-effort handling.

## Credits and provenance

- Base model: [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- Quantization backend: [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- Importance matrix and reference methodology: [AtomicChat/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)
- BF16 KLD reference dump and evaluation data: [AtomicChat/dsv4-eval-artifacts](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)
- Embedded Jinja chat template: [unsloth/DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731)

Thanks to AtomicChat for publishing the imatrix, reference logits, comparison quants, and enough evaluation detail to reproduce a paired local test.

## License

This release is available under the [MIT License](LICENSE). The base model is also MIT-licensed; review the [upstream model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) before redistribution or deployment.
