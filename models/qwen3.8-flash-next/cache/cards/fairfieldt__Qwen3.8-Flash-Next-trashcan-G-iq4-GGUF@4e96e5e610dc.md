---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- trashcan
---

# Qwen3.8-Flash-Next G-iq4 GGUF for trashcan

The quant [trashcan](https://github.com/fairfieldt/trashcan) runs Qwen3.8-Flash-Next with, pipeline parallel over
six 2013 Mac Pros (12 FirePro D700s, Vulkan). Made with `llama-quantize --imatrix` (unsloth's importance matrix) from
the Q8_0 of [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):

- routed and shared experts: IQ4_XS gate/up, IQ4_NL down
- dense matrices: Q8_0, as released (at Q4_0 the 4.6B dense parameters cost five times the divergence of all 121B
  expert parameters at Q4_0)
- output head: Q6_K
- n-gram table: Q8_0, as released. Shard 3 is the release's shard 3, unchanged.

Against the Q8_0 model, over 39K tokens of mixed text (llama.cpp, CPU): mean KL divergence 0.0391, same top token
95.1%, perplexity ratio 1.002.

`mtp-G-iq4.gguf` is the multi-token-prediction block for speculative decoding (`--mtp`).

| File | Bytes | SHA-256 |
|---|---|---|
| `G-iq4-00001-of-00006.gguf` | 10,946,656 | `d531d17a3eb1a23793a6ef328f45efd90b2d3f4b8b7081ce6c1cce537955c5d8` |
| `G-iq4-00002-of-00006.gguf` | 528,476,512 | `de8328a8d88b3a4b5364e198efe62c3d697ca378bd131d02817b4a77a0eb3b04` |
| `G-iq4-00003-of-00006.gguf` | 54,400,261,312 | `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20` |
| `G-iq4-00004-of-00006.gguf` | 26,254,316,416 | `8ef48b441412023d1576ac12606888bdaef211f5f75840eda8d00d82065deb15` |
| `G-iq4-00005-of-00006.gguf` | 26,029,890,400 | `ae169ebd9ed97944417dcaa25d13668e2d39ad276bec808028130e2c8bdc9b9c` |
| `G-iq4-00006-of-00006.gguf` | 17,809,743,584 | `916805561ec70ea6ae6e0ef9f056477f225e355c56e44fcee90e8646376f168e` |
| `mtp-G-iq4.gguf` | 2,670,190,656 | `799571b8310c5c5e02cfb7165267971e07baba6c4787ae9bb73578e38c7696e5` |

Single-stream decode on trashcan: 22.6 tok/s (31.3 with MTP), prefill 518 tok/s on 512 tokens and 1,211 on 16K, up to
112K context. Recipe `6-flashnext`: [docs/deploy.md](https://github.com/fairfieldt/trashcan/blob/main/docs/deploy.md).
