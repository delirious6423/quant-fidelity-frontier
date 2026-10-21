---
base_model: zai-org/GLM-5.3
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3
tags:
- exl3
- exllamav3
- sage
- mixed-k
- moe
- glm
- tensorfold
- dgx-spark
- dflash
---

# GLM-5.3 MixedK EXL3 3.38 bpw

An EXL3 quantization of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), made with **SAGE**.

SAGE dynamically and intelligently assigns bit widths across the model, making this a MixedK EXL3. It averages 3.38 bits per weight, and every one of the 19,200 routed experts is kept: no pruning and no expert merging.

The pack is sized so the weights plus a full 1,048,576-token KV cache at 4 bits fit in the memory of four NVIDIA DGX Sparks.

**Serving: [DEPLOY.md](DEPLOY.md). An agent should follow that file.** It runs this pack on four NVIDIA DGX Sparks with TensorFold's GLM-5.3 tensor-parallel engine and DFlash2 speculative decoding, through the [GLM-5.3 EXL3 DGX Spark recipe](https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe): up to 69 tok/s, 56.5 tok/s on math and about 41 tok/s on average across real prompts, with every drafted reply token-identical to decoding without the drafter. Do not serve it with stock ExLlamaV3: it cannot load GLM-5.3.

## Summary

| | |
|---|---|
| Base model | [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (78 layers, 256 routed experts per MoE layer, 8 active) |
| Format | EXL3 |
| Quantization | SAGE MixedK |
| Body bitrate | 3.38 bpw nominal, 3.39 bpw including scales |
| Output head | 8-bit |
| Routed experts | all 256 per layer kept (19,200 total) |
| MTP / draft layer | not included |
| Max context | 1,048,576 tokens (unchanged from the base model) |
| Total size | 319.0 GB (297.1 GiB), 58 weight shards |

## Performance on four DGX Sparks

TensorFold TP=4 through the recipe's OpenAI-compatible server, greedy, 512 new tokens, one request at a time. Details, settings and the served-quality check: [DEPLOY.md](DEPLOY.md#what-was-measured).

| Measurement | Result |
|---|---:|
| Decode with DFlash2, peak (SixCat decode benchmark) | **69 tok/s** |
| Decode with DFlash2, math prompt | **56.5 tok/s** |
| Decode with DFlash2, mean of 6 real prompts (163,840-token context) | **41.1 tok/s** |
| Decode with DFlash2, mean of 6 real prompts (262,144-token context, int4 KV cache) | 40.6 tok/s |
| Decode without drafting, mean of 6 real prompts | 26.5 tok/s |
| 127,544-token prompt in context, decode with DFlash2 | 33.3 tok/s |
| Same four Sparks on ExLlamaV3, no drafter | 12.2 to 15.7 tok/s |

## Quality

Measured against the original BF16 model on a held-out evaluation set: text the quantizer never saw.

**The held-out set:** 10 sequences of 1,024 tokens each (10,240 scored positions), built from the test splits of public benchmarks:
- Web text: WikiText-103 (4 sequences)
- Code: HumanEval (2)
- Math: GSM8K (2)
- Chat: UltraChat-200k (2)

Each sequence is whole documents packed end to end. Math and chat examples are formatted with GLM-5.3's own chat template. None of this text was used while quantizing the model.

**Scoring:** both models read the same tokens. At every position their next-token predictions are compared over the full 154,880-token vocabulary, in float64.

| Metric | Result |
|---|---|
| Top-1 agreement with BF16 | **92.98%** (9,521 / 10,240) |
| BF16 top-1 token within the quant's top 5 | 99.38% |
| Mean KL divergence (BF16 ‖ quant) | 0.0948 |
| Median KL divergence | 0.0018 |
| 99th-percentile KL divergence | 1.61 |
| Top-5 set overlap | 0.827 |
| Mean NLL, BF16 → quant | 1.0092 → 1.0300 |
| Perplexity increase | +2.1% |

**Scope of these numbers:**
- The BF16 reference is the original weights run through the same ExLlamaV3 runtime, not the vendor's own implementation.
- Scores come from full-sequence forward passes at 1,024 tokens. Long-context and cached generation were not part of this evaluation.

## Memory budget: four DGX Sparks at 1M context

| Item | Size |
|---|---|
| Weights | 319.0 GB |
| KV cache, 1,048,576 tokens at Q4 (MLA latent plus indexer keys) | ~39.7 GB |
| Total | ~358.7 GB of 512 GB (4 × 128 GB) |

The rest is left for activations, the runtime and the OS. Four-node serving is validated with TensorFold at 163,840 tokens (bf16 KV cache) and 262,144 tokens (int4 KV cache, whole on every Spark); full 1M-token inference has not been run.

## Runtime

Serve this pack with TensorFold, following [DEPLOY.md](DEPLOY.md). TensorFold's full GLM-5.3 tensor-parallel engine is [ashhart/TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159) by [@drowzeys](https://github.com/drowzeys). The fixes this pack needs (loading on GB10, its fp16 tensors, the cache guard, an int4/int8 KV cache, RoCE robustness) are on the [vcruz305/TensorFold](https://github.com/vcruz305/TensorFold) fork and submitted to that PR as [drowzeys/TensorFold #1 to #5](https://github.com/drowzeys/TensorFold/pulls). The [recipe](https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe) pins them, so it works before they land.

Stock ExLlamaV3 cannot load GLM-5.3. The quality scores above were produced with an ExLlamaV3 fork used for evaluation (commit `affc194d5476710f167f30729e2508e577759612`); it is not the serving path.

## Download

```bash
hf download vcruz305/GLM-5.3-EXL3-3.38bpw --local-dir GLM-5.3-EXL3-3.38bpw
```

The recipe's `./glm53 setup --download-once` downloads it for you, once, and copies it to all four Sparks.

## License

Same license as the base model; see [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3).
