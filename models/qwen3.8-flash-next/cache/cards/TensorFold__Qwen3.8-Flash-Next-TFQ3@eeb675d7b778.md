---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- tensorfold
- tfq3
- qwen
- quantized
- apple-silicon
- metal
- mtp
---

# Qwen3.8-Flash-Next-TFQ3

TensorFold’s TFQ3 checkpoint of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). The weight files total **84.82 GB (78.99 GiB)**, including the MTP head.

<p align="center">
  <img src="qwen-logo.png" alt="Qwen" width="200" />
</p>

## Before you download

This checkpoint requires a TensorFold native build with TFQ3 support for Qwen3.8 Flash Next on Apple silicon (Metal). The runtime reads the `.tffpw` files and Safetensors together. The measurements below cover text generation on an Apple M5 Ultra.

## Checkpoint and quality comparison

| Metric | 8-bit source | MLX 4-bit | TFQ3 |
| --- | ---: | ---: | ---: |
| Measured weight size | 189.04 GiB | 105.50 GiB | **78.99 GiB** |
| Main-set NLL / perplexity, 1,670 tokens | 1.7315 / 5.649 | 1.8324 / 6.249 | **1.7298 / 5.640** |
| Held-out NLL / perplexity, 334 tokens | 2.1003 / 8.169 | 2.2006 / 9.031 | **2.0978 / 8.148** |
| Combined perplexity, 2,004 tokens | 6.007 | 6.644 | **5.996** |

**TFQ3 recorded 9.8% lower combined perplexity than MLX 4-bit** and used **26.51 GiB less disk space**. Against the 8-bit source, it recorded 5.996 versus 6.007 combined perplexity while using **58.2% less disk space**. The evaluation covered seven domains, including three held-out domains.

## Apple M5 Ultra measurements

| TFQ3 test | Result |
| --- | ---: |
| Prefill, 2,048-token prompt | 3,855 tok/s |
| Single-row decode | 82.9 tok/s |
| Decode with MTP depth 3 | 154.5 tok/s |
| Live streaming with MTP and copy drafting | 126–139 tok/s |

The tested live server was configured for 262,112 tokens of context. In the reported benchmark, MTP depth 3 matched depth 0 token for token on this checkpoint.

## Licence and credit

The original model is by the Qwen team. TensorFold publishes this quantized checkpoint under the [Qwen Community License 1.0](LICENSE).
