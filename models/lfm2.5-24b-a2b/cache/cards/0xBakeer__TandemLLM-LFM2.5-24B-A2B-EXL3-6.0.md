---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags: [exl3, exllamav3, quantized, lfm2.5, lfm2_moe, tandemllm]
---
# LFM2.5-24B-A2B in EXL3, 6.0 bits per weight

These are EXL3 weights of Liquid AI's LFM2.5-24B-A2B at 6.0 bits per weight, 18.63 GB in total. We made them with exllamav3 1.6.0 from Liquid's BF16 release, [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), with exllamav3's built-in calibration. Every layer is at 6.0 bits; the output head, which the model ties to its embedding, is at 8 bits, and the embedding itself stays in BF16. The raw conversion log and quality output are in `quality/`.

We serve them with [TandemLLM](https://github.com/0xBakeer/TandemLLM), our inference engine for one DGX Spark.

## Quality of every size

We measured each size against the BF16 model with exllamav3's `model_diff`, on 20 rows of 2,048 tokens of two texts: wikitext-2, and chat (conversations from the oasst1 validation split, human-written answers, in the model's chat template). BF16's perplexity is 21.00 on the wikitext rows and 7.43 on the chat rows.

| size | GB | KL wikitext | top-1 wikitext | KL chat | top-1 chat |
|-|-|-|-|-|-|
| [2.5](https://huggingface.co/0xBakeer/TandemLLM-LFM2.5-24B-A2B-EXL3-2.5) | 8.20 | 0.463 | 70.4 % | 0.313 | 78.9 % |
| [3.0](https://huggingface.co/0xBakeer/TandemLLM-LFM2.5-24B-A2B-EXL3-3.0) | 9.68 | 0.335 | 75.2 % | 0.195 | 83.6 % |
| [3.5](https://huggingface.co/0xBakeer/TandemLLM-LFM2.5-24B-A2B-EXL3-3.5) | 11.16 | 0.226 | 79.0 % | 0.136 | 86.7 % |
| [4.0](https://huggingface.co/0xBakeer/TandemLLM-LFM2.5-24B-A2B-EXL3-4.0) | 12.64 | 0.201 | 80.7 % | 0.098 | 88.5 % |
| [4.5](https://huggingface.co/0xBakeer/TandemLLM-LFM2.5-24B-A2B-EXL3-4.5) | 14.12 | 0.113 | 85.4 % | 0.063 | 91.2 % |
| **6.0** | 18.63 | 0.080 | 88.2 % | 0.039 | 93.4 % |

On chat text, closer to how the model is used, the KL of every size is 0.49 to 0.68 times its wikitext KL.

## Speed

Measured in TandemLLM on one DGX Spark: one request, 512 greedy tokens. Chat is plain decoding. Edit and copy are asks whose answer repeats much of the prompt; there TandemLLM's lookup drafter proposes the repeated text and the model checks it, with the same output as plain decoding.

| size | chat tok/s | edit tok/s | copy tok/s |
|-|-|-|-|
| 2.5 | 185 | 351 | 271 |
| 3.0 | 171 | 328 | 414 |
| 3.5 | 151 | 269 | 287 |
| 4.0 | 144 | 291 | 326 |
| 4.5 | 128 | 281 | 339 |
| **6.0** | 105 | 210 | 280 |
| BF16 | 45 | 154 | 158 |

![Chat speed against quality for every size](assets/1-speed-vs-quality.png)

![Chat KL and same top token for every size](assets/3-quality-by-size.png)

![The lookup drafter on edit and copy asks](assets/2-lookup-edit-copy.png)

TandemLLM's default is 3.5: the fastest size whose top-1 agreement with BF16 on chat text is at least 85 %.

## How to run it

TandemLLM 0.9.0 runs every size ([release notes](https://github.com/0xBakeer/TandemLLM/releases/tag/v0.9.0)). On a DGX Spark, install it in setup mode, then switch to this size:

```bash
curl -fsSLO https://raw.githubusercontent.com/0xBakeer/TandemLLM/v0.9.0/install.sh
bash install.sh --setup --yes
~/TandemLLM/run/ops/use-profile.sh lfm25-exl3-6.0
```

`use-profile.sh` downloads the files of this repository at a pinned commit, checks each one against its sha256 and starts the engine on port 8000. It needs no Hugging Face token. All six sizes are in the [collection](https://huggingface.co/collections/0xBakeer/tandemllm-lfm25-24b-a2b-exl3-6acf468024dd5288750dd683).

The files are in the standard EXL3 format. exllamav3 1.6.0 loads them; we measured them with it. Other EXL3 loaders such as TabbyAPI should read them too. We have not tested those.

## License

These weights are a quantization of Liquid AI's model and carry its license, the LFM Open License v1.0, in `LICENSE` in this repository. Liquid AI's terms apply to every use of them. Everything else about the model is in [its model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).
