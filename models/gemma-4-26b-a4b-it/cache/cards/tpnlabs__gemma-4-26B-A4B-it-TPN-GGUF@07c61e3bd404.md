---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
- gguf
- tpn-workbench
- llama.cpp
- gratisai
---

# gemma-4-26B-A4B-it · TPN-optimised GGUF

Quantised and measured by [TPN Workbench](https://bench.trueperformancenetwork.com) from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) @ `4d7ae49`. Every format is scored against the BF16 reference on next-token KL divergence, perplexity and top-1 agreement (llama.cpp b11528).

Full report: https://bench.trueperformancenetwork.com/r/2Ul4jX0DoRqqxHJxuyULkQ

**No valid fidelity scores**: TPN's BF16 reference run for this architecture is broken (reference perplexity 7552 on wiki text, where a working model scores about 10; Q8_0 measured at KLD 0.61 against a predicted 0.004), so the report's closeness, perplexity and top-1 figures are not meaningful. The files themselves load and answer correctly (text and image) in llama.cpp b11818; only the measurement is void, so this card lists no scores.

**TPN recommendation: IQ4_XS**: largest format under the size goal.

## Files

| file | format | size | mean KLD | KL similarity | PPL Δ | top-1 | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`gemma-4-26B-A4B-it-IQ4_XS.gguf`](./gemma-4-26B-A4B-it-IQ4_XS.gguf) | IQ4_XS | 13.92 GB | – | – | – | – | TPN recommendation |
| [`gemma-4-26B-A4B-it-Q4_K_M.gguf`](./gemma-4-26B-A4B-it-Q4_K_M.gguf) | Q4_K_M | 16.80 GB | – | – | – | – |  |
| [`gemma-4-26B-A4B-it-Q8_0.gguf`](./gemma-4-26B-A4B-it-Q8_0.gguf) | Q8_0 | 26.86 GB | – | – | – | – |  |
| [`mmproj-gemma-4-26B-A4B-it-F16.gguf`](./mmproj-gemma-4-26B-A4B-it-F16.gguf) | vision projector (F16) | 1.19 GB | | | | | load with any file above for image input |

Lower KLD = closer to the original. K-quants (`Q4_K_M`, `Q8_0`) have SIMD kernels in llama.cpp WebAssembly, so they are the faster pick in the browser; `IQ*` formats are smallest-for-quality on native CPU/GPU builds.


## Use

```bash
llama-server -hf tpnlabs/gemma-4-26B-A4B-it-TPN-GGUF --hf-file gemma-4-26B-A4B-it-IQ4_XS.gguf
```

Runs fully on-device in [Gratis](https://gratis.trueperformancenetwork.com/), in the browser or desktop app.

Licence: inherited from the base model (apache-2.0).
