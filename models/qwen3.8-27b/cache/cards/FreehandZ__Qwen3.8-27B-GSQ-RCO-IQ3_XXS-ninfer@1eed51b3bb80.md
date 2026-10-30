---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - ninfer
  - qwen3.8
  - gsq-rco
  - iq3_xxs
  - mtp
---

# Qwen3.8-27B GSQ-RCO IQ3_XXS for NInfer-Qwen

Qwen3.8-27B in the GSQ-RCO IQ3_XXS quantization, converted to a `.ninfer` artifact for
[NInfer-Qwen](https://github.com/2-com-org/ninfer-qwen/tree/qwen38-5060ti). It serves a
262,144-token context on one GeForce RTX 5060 Ti 16 GB.

This is the file `scripts/qwen38/prepare-model.sh` builds. Downloading it replaces that script's
10.4 GB download and 10-minute conversion. It runs only on NInfer-Qwen, not on llama.cpp,
vLLM or Transformers.

| File | Bytes | sha256 |
|---|---:|---|
| `qwen38-27b-gsq-iq3xxs.ninfer` | 10,846,918,144 | `e715439d19d0e156f43a128d724d50825ccd55a2b2c0f898d37716a7d269f819` |
| `qwen38-27b-gsq-iq3xxs.ninfer.conversion.json` | 323,911 | `be9aa3996f0bf1c46a4799cd11848feeabf33567e9d4b117a34665e95a81eba8` |

The conversion record lists every object in the artifact with its source tensors, format and method.

## Use

Build NInfer-Qwen ([Installation](https://github.com/2-com-org/ninfer-qwen/wiki/Installation)),
then download the artifact into the repository's `models/` directory, where the launch script
looks for it:

```bash
hf download FreehandZ/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-ninfer qwen38-27b-gsq-iq3xxs.ninfer --local-dir models
scripts/qwen38/serve.sh --port 8080
```

The server speaks the OpenAI, Anthropic and llama.cpp APIs at `http://HOST:8080/v1`. See
[Serving](https://github.com/2-com-org/ninfer-qwen/wiki/Serving) for several requests at once,
the conversation cache and the other options.

## Contents

- **Text model.** All 64 layers of ISTA-DASLab's GSQ-RCO IQ3_XXS GGUF, each tensor with its ggml
  blocks kept byte for byte (IQ1 to IQ4, Q2_K, Q4_K, BF16 and F32).
- **MTP head.** The multi-token-prediction head used for speculative decoding, its projections
  requantized from Q6_K to IQ4_XS.
- **Proposal head.** A shortlist of the 131,072 most frequent tokens for the drafts, with an
  IQ2_XXS copy that picks candidates before exact rescoring.
- **Frontend.** Qwen's config, tokenizer and processor files, and NInfer's Qwen3.8 chat template.
- **No vision.** The artifact has no image or video encoder.

[Model preparation](https://github.com/2-com-org/ninfer-qwen/wiki/Model-Preparation) describes each
step and its measured effect.

## Speed on an RTX 5060 Ti 16 GB

One request at a 262,144-token context, greedy replies of 512 tokens with thinking off:

| Workload | Result |
|---|---:|
| 30K-token prompt: prefill / decode | 3,644-3,648 / 88.7 tok/s |
| 160K-token prompt: prefill / decode | 2,143 / 66.0 tok/s |
| Short prompt, decode | 87.0-87.7 tok/s |
| 8 / 32 requests at once, total | 342 / ~625 tok/s |

Method and comparisons: [Performance](https://github.com/2-com-org/ninfer-qwen/wiki/Performance).
Capability evaluations are running:
[Evaluation](https://github.com/2-com-org/ninfer-qwen/wiki/Evaluation).

## Sources and license

| Source | Revision | Used | License |
|---|---|---|---|
| [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | `d562806d` | `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` | Apache-2.0 |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2` | config, tokenizer, chat template and processor files | Apache-2.0 |

The weights are Qwen's Qwen3.8-27B, quantized by ISTA-DASLab with GSQ-RCO
([arXiv:2604.18556](https://arxiv.org/abs/2604.18556),
[arXiv:2605.00649](https://arxiv.org/abs/2605.00649)). This artifact is distributed under the same
Apache-2.0 license.
