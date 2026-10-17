---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - mlx
  - mlx-vlm
  - omlx
  - oq
  - mtp
  - speculative-decoding
  - qwen
  - qwen3.8
  - qwen4-exp
  - mixture-of-experts
  - vision-language
  - quantized
  - apple-silicon
  - 3-bit
---

> [!WARNING]
> ## M1 Ultra Speed Benchmark and Anecdotal Intelligence Review Figures Below ##

<p align="center">
  <a href="https://qwen.ai/"><img src="qwen-logo.png" width="96" height="95" alt="Qwen"></a>
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Qwen-Qwen3.8-615CED?style=for-the-badge&logo=qwen&logoColor=white" alt="Qwen Qwen3.8">
  <img src="https://img.shields.io/badge/Apple_Silicon-MLX-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Apple silicon MLX">
  <img src="https://img.shields.io/badge/Native_MTP-Included-22C55E?style=for-the-badge" alt="Native MTP included">
  <img src="https://img.shields.io/badge/Vontra-oQ3-6E56CF?style=for-the-badge&logo=huggingface&logoColor=white" alt="Vontra oQ3">
</p>


<h1 align="center">Qwen3.8 Flash Next — MLX oQ3 with native MTP</h1>


<p align="center">
  A sensitivity-guided, mixed-precision MLX conversion of <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen/Qwen3.8-Flash-Next</a>, rebuilt directly from the official BF16 checkpoint with the model's matching native MTP block preserved.
</p>


<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Original model</a> ·
  <a href="https://qwen.ai/blog?id=qwen3.8-flash-next">Qwen overview</a> ·
  <a href="https://github.com/ml-explore/mlx-vlm">MLX-VLM</a> ·
  <a href="LICENSE">Qwen Community License 1.0</a>
</p>


## About this conversion


oQ3 uses a 3-bit affine base and spends additional precision on sensitive modules. Layer sensitivity was measured with a validated quantized calibration proxy, while every released weight was quantized from the official BF16 checkpoint. The result is a compact model with 746 higher-precision module overrides rather than a uniform 3-bit layout.


| Item | Value |
| --- | --- |
| Repository | `Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP` |
| Base model | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Source weights | Official BF16 checkpoint |
| Format | MLX safetensors |
| Quantisation | oQ3 mixed precision, 3-bit affine base |
| Base group size | 32 |
| Protected modules | 314 at 4-bit, 82 at 5-bit, 127 at 6-bit, and 223 at 8-bit |
| Native MTP | Included, one Qwen4Exp draft block |
| Indexed tensors | 3,747 total, including 76 MTP entries |
| Weight shards | 19 |
| Weight size | 92.505 GB / 86.152 GiB |
| Configured context | 262,144 tokens |
| Architecture | `qwen4_exp` vision-language sparse MoE |


The upstream tokenizer, current chat template, vision processor, generation configuration, licence, and native MTP configuration are retained.


> [!IMPORTANT]
> Use a runtime with explicit `qwen4_exp` and native-MTP support. A runtime that does not construct the Qwen4Exp MTP module may reject the 76 MTP tensor entries during strict loading.


## Download and use


```bash
python -m pip install --upgrade huggingface_hub


hf download Vontra/Qwen3.8-Flash-Next-MLX-oQ3-MTP \
  --local-dir ./Qwen3.8-Flash-Next-MLX-oQ3-MTP
```


In a compatible oMLX build, add the downloaded directory to the model directories, refresh the registry, select the model, and enable native MTP. MTP can be disabled for baseline comparisons and troubleshooting.


## Apple M3 Studio performance


Benchmark environment: oMLX 0.6.3rc3 (build 2475), MLX-VLM 0.6.3, and MLX 0.32.0 on an Apple M3 Studio. Measurements use greedy decoding, seed 6330, identical prompts, a separate warm-up, and three measured 512-token runs per mode.


| Runtime mode | Runs | Output per run | Median generation speed | Drafted | Accepted | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native MTP disabled | 3 | 512 tokens | **26.5352 tokens/s** | Not applicable | Not applicable | Not applicable |
| Native MTP enabled | 3 | 512 tokens | **29.0822 tokens/s** | 847 | 583 | 68.83% |


Native MTP improved median decode throughput by **9.60%** in this test. The MTP-off and MTP-on runs produced exactly matching output hashes. All six sustained runs reached 512 generated tokens; exact instruction, factual, arithmetic, and coherent long-generation gates passed in both modes.


Results vary with prompt length, context growth, cache state, runtime version, memory pressure, and thermal conditions. The first request after loading includes model and kernel warm-up and is excluded from the steady-state result.

> [!IMPORTANT]
>## Apple M1 Ultra (128 GB) performance
>Benchmark environment: oMLX 0.6.4, engine auto, context Code (Python), on an Apple M1 Ultra 128 GB. As of this writing, this is the preferred oMLX pick for Qwen3.8 Flash Next on this system.
>### Single request
>| Test | TTFT (ms) | TPOT (ms) | Prefill TPS | Decode TPS | E2E (s) | Throughput | Peak Mem |
>| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
>| pp1024/tg128 | 8157.6 | 49.00 | 125.5 | 20.6 | 14.394 | 80.0 | 64.60 GB |
>| pp4096/tg128 | 22503.5 | 50.09 | 182.0 | 20.1 | 28.878 | 146.3 | 67.25 GB |
>| pp8192/tg128 | 35326.7 | 58.74 | 231.9 | 18.0 | 36.342 | 225.9 | 67.39 GB |
>| pp16384/tg128 | 62766.1 | 160.83 | 261.0 | 12.4 | 63.079 | 259.8 | 67.68 GB |
>| pp32768/tg128 | 118533.7 | 69.03 | 276.4 | 15.5 | 119.704 | 273.9 | 68.26 GB |
>| pp65536/tg128 | 231924.8 | 48.68 | 282.6 | 20.7 | 238.127 | 275.8 | 69.42 GB |
>### Continuous batching — pp1024/tg128
>| Batch | Decode TPS | Speedup | Prefill TPS | Prefill TPS/req | TTFT (ms) | E2E (s) |
>| --- | ---: | ---: | ---: | ---: | ---: | ---: |
>| 1× | 20.6 | 1.00× | 125.5 | 125.5 | 8157.6 | 14.394 |
>| 2× | 30.7 | 1.49× | 187.9 | 94.0 | 7979.6 | 19.241 |
>| 4× | 52.8 | 2.56× | 155.8 | 39.0 | 15141.8 | 35.992 |
>| 8× | 102.0 | **4.95×** | 144.2 | 18.0 | 30174.5 | 65.264 |
>#### Peak memory stays below 70 GB across all tested context lengths, leaving comfortable headroom on a 128 GB system. ####
>### Intelligence benchmarks (sample 30, no-think mode)
>| Benchmark | Accuracy | Correct | Total | Time (s) |
>| --- | ---: | ---: | ---: | ---: |
>| MMLU | 80.0% | 24 | 30 | 125.4 |
>| TruthfulQA | 100.0% | 30 | 30 | 37.7 |
>| HumanEval | 90.0% | 27 | 30 | 272.0 |
>| MBPP | 90.0% | 27 | 30 | 144.5 |


## Architecture


Qwen3.8 Flash Next combines Gated DeltaNet, Qwen Sparse Attention, sparse mixture-of-experts layers, widened gated residual streams, hashed bigram and trigram embeddings, and a native next-token-prediction block for speculative decoding.


| Architecture detail | Upstream value |
| --- | ---: |
| Language-model parameters | 125B total / 6B active |
| N-gram embedding | 51B parameters, 20,000,000 entries |
| Native MTP | 4B parameters, one draft layer |
| Hidden size | 2,560 |
| Layers | 48 |
| Routed / active experts | 512 / 10, plus 1 shared expert |
| Native context | 262,144 tokens, extensible upstream to 1,000,000 |


For upstream evaluations, intended use, limitations, safety guidance, and the full architecture discussion, see the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).


## Conversion and validation


- The converter read the official BF16 checkpoint directly.
- A 3-bit affine base at group size 32 was combined with 746 sensitivity-guided 4/5/6/8-bit overrides.
- Structural validation passed for all 3,747 indexed tensors, all 19 weight shards, and all 76 native-MTP tensor entries.
- Three deterministic 512-token runs passed in each MTP mode with exact output parity.
- Native-MTP telemetry, exact-answer gates, and coherent long-generation checks passed.


This is a community conversion, not an official Qwen release.


## Limitations


- Quantisation can reduce quality relative to BF16; 3-bit models should be evaluated on the intended workload.
- The native 262,144-token context does not guarantee that every Apple-silicon system can run that length within available unified memory.
- Native MTP needs a compatible runtime and may not improve every prompt or context length.
- This is an MLX release for Apple silicon, not a GGUF, CUDA, TensorRT-LLM, or vLLM checkpoint.
- Upstream model limitations and safety considerations still apply.


## Licence and attribution


The upstream model is released under the **Qwen Community License 1.0**. The required licence text is included in this repository and should be reviewed before use or redistribution.


Model design, training, evaluations, and upstream documentation belong to Qwen and the original contributors. The MLX conversion, native-MTP preservation, Apple-silicon validation, and packaging are provided by [Vontra](https://huggingface.co/Vontra).
