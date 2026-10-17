---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: transformers
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- qwen
- qwen3.8
- auto-round
- auto-gptq
- mixed-precision
- mtp
- ampere
- vllm
- ssd-offload
---

# Qwen3.8 Flash Next — AutoRound 3 bpw + MTP

**Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(official Qwen release, BF16). This checkpoint is a mixed-precision quantization
of that base: 2/3-bit routed experts (~2.998 effective bpw), 8-bit main
projections, BF16 sensitive paths, and 4-bit MTP experts. All 13 safetensors
shards are required. The chat template is inherited unchanged from the Qwen
base model.

**KLD vs original weights: 0.1208.**

**Checkpoints:**
· [This model](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP)
· [Uncensored variant](https://huggingface.co/klee100/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP)
· [GitHub: full vLLM patch and setup guide](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx)

## SSD offloading with vLLM

The **95.37 GiB BF16 PLE table stays on SSD**, using asynchronous Linux AIO;
loaded model weights use approximately **47.3 GiB VRAM**. Tested on a 64 GiB
SM80 GPU at 180 W with 15 GiB RAM, 31 GiB swap, and enterprise SATA storage.

Apply the [complete patch](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/blob/main/patches/qwen38-ple-ssd.patch)
to vLLM **`a5a30471ff2bb7f0824f2da10e358af98d304472`** and build `ple_ssd_io.so`.
Required additions: metadata-only PLE loading, async row reads/read-ahead,
CUDA graph host breaks, mixed-bit expert overrides, and MTP layer remapping.
[Installation and screen guide](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/blob/main/docs/GUIDE.md)
uses `~/vllm/.venv`.

## Performance

| Workload | Throughput |
| --- | ---: |
| Fresh 8K prefill | **2,636 prompt tok/s** |
| Single-request decode | **111 tok/s** |
| Aggregate output, 16 requests | **667 tok/s** |

MTP=1, 2K prefill chunks. Three-round medians; chat uses 128 output tokens,
temperature zero, thinking disabled, and EOS ignored. Aggregate throughput
includes prefill; decode excludes first-token latency.
[Full measurements](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx/blob/main/docs/RESULTS.md)

Experimental runtime: an arithmetic consistency issue remains. Context is
advertised as 262K; varied-token timing was tested through 32K.

## License

[Qwen Community License 1.0](LICENSE), inherited from the base model.
