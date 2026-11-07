---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: pulsar
base_model:
- Qwen/Qwen3.8-Flash-Next
- incoai/Qwen3.8-Flash-Next-DFlash2
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- pulsar
- exl3
- qwen
- speculative-decoding
- dgx-spark
---

# Qwen 3.8 Flash Next: pulsar container, EXL3 K4/K5 + DFlash2

A ready-to-serve container for [pulsar](https://github.com/tylerwagler/pulsar), our inference engine for DGX Spark
(GB10). It holds [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized to EXL3, plus the
[incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) block drafter (z-lab's
DFlash2 design) for speculative decoding. This is the Qwen model we serve in production.

The download is about 169 GB: about 67 GB of weights and a 102 GB PLE n-gram row file.

**Parameters:** 180B upstream, about 51B of them in the PLE table. Hugging Face's auto-detected count is lower because it
counts EXL3's packed bytes as parameters.

## What's inside

| | |
|---|---|
| Experts | EXL3 K4 (about 4 bpw) |
| Other dense Linears | EXL3 K5 (about 5 bpw) |
| Output head | EXL3 K5, in `model-qhead.safetensors` |
| Remaining dense layers | MXFP8 |
| Norms and similar | BF16 |
| Draft model | DFlash2 block drafter, in the last shard |
| `ple-l1.rows` | PLE n-gram table (layer 1, BF16), served from disk beside the container |
| MTP head | Not included: DFlash2 replaces it |
| Vision | Included |

Recipe: `U-e4-d5-dflash2`. Source revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

## Using it

This container runs on pulsar only. Standard Transformers, vLLM or llama.cpp loaders won't read it.

Keep the whole folder together, on local NVMe: the row file must sit beside the container.

```bash
hf download ElytronSystems/Qwen-3.8-Flash-Next-pulsar-k4 --local-dir /path/to/qwen-container
./pulsar-server -m /path/to/qwen-container --rope-yarn 4
```

`--rope-yarn 4` gives the model's 1M-position context.

## License

The weights are under the [Qwen Community License 1.0](LICENSE). Read its terms before commercial use, especially
those covering Model as a Service and AI Work Assistant businesses, which need a separate license from Qwen. The DFlash2
drafter is Apache-2.0 ([incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)).
