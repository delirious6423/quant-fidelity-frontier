---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- deepseek
- deepseek-v4.1
- quantized
- text-generation
- long-context
---

# DeepSeek-V4.1-Flash GGUF

Community GGUF quantizations of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

<div align="center" style="background-color:#f59e0b;color:#ffffff;padding:16px 20px;border-radius:10px;line-height:1.7;">
☕ If this GGUF made your day easier, a coffee would make mine.<br>
<a href="https://ko-fi.com/ngquocvinh" style="color:#ffffff;"><strong style="color:#ffffff;">Send a coffee ☕</strong></a><br>
I build and test these releases myself. Your coffee helps keep me going.<br>
Thank you for supporting this work.
</div>

## Release status

This model card is ready for the release package. Quantized files will be added
only after conversion, GGUF loading, and text generation have passed validation.
No GGUF artifact is published in this initial card revision.

The planned text-only ladder is:

| Priority | Planned file | Status |
|---:|---|---|
| 1 | `DeepSeek-V4.1-Flash-Q1_0.gguf` | pending conversion and validation |
| 2 | `DeepSeek-V4.1-Flash-Q2_K.gguf` | pending conversion and validation |
| 3 | `DeepSeek-V4.1-Flash-Q3_K_M.gguf` | pending conversion and validation |
| 4 | `DeepSeek-V4.1-Flash-Q4_K_M.gguf` | pending conversion and validation |
| 5 | `DeepSeek-V4.1-Flash-Q5_K_M.gguf` | pending conversion and validation |

## About DeepSeek-V4.1-Flash

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
is a multimodal sparse Mixture-of-Experts model. The upstream card describes a
763B-parameter model with a 552B backbone, a 1M-token maximum context, and
approximately 8B active parameters during prefill and 16B during decode. It
uses a 40-layer causal encoder-decoder design, Engram n-gram lookup tables,
hyper-connections, compressed sparse attention, and a vision encoder. See the
[official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
for the original architecture, reference implementation, and usage limits.

![DeepSeek-V4.1-Flash upstream agentic performance benchmark](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/assets/dsv41_agentic_performance.png)

*Upstream DeepSeek-V4.1-Flash agentic performance benchmark; this image and
its scores are from the [official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash),
not from these GGUF files.*

The first GGUF package targets text inference. Vision, multimodal projector,
and speculative MTP components will not be claimed unless they are separately
converted and validated with the actual runtime.

This is a quantization-only release. No training, fine-tuning, merging, or
model capability modification other than GGUF conversion and quantization is
performed here.

## Fidelity measurements

This section will be added after a fixed hold-out comparison against a valid
reference has completed. No fidelity percentages are claimed yet.

## Quick start

Commands will be added here with the exact published filename after the first
GGUF passes load and generate smoke tests.

## Reproducibility and validation

The upstream checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash` at revision
`dba1be0a40aa45a94ad051997016db3960a90277`. Its index reports 48 safetensors
shards and a total source size of 510,286,023,000 bytes. The checkpoint uses
the official mixed FP8/FP4 representation declared by its `quantization_config`;
it is not a BF16 GGUF input and no community GGUF is used as a quantization
source.

Conversion and runtime support use the DeepSeek-V4.1 work in
[`vcruz305/llama.cpp`](https://github.com/vcruz305/llama.cpp), commit
`5210c7c5ed61dddaee6ed476623abf4b63093d16`, with the V4.1 converter/runtime
path documented in the local reproducibility manifest. Each published file
will include a SHA-256 checksum and will be listed only after a real load and
generation smoke test.

Raw conversion, imatrix, quantization, smoke-test, fidelity, and benchmark
logs remain local under `reports/` and are not uploaded to the Hub.

## License and attribution

The upstream model is released under the MIT License. Preserve upstream
attribution and the included [`LICENSE`](LICENSE) when redistributing these
derivative artifacts.

These are community GGUF quantizations, not an official DeepSeek release or
endorsement.
