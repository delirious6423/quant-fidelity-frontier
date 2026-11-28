---
license: apache-2.0
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- dfly2
- speculative-decoding
- gguf
- jucee
- apple-silicon
- metal
---

# DeepSeek-V4.1-Flash DFly2 Drafter (MXFP4 + Q8_0 Attention GGUF for Jucee)

Quantized GGUF conversion of [AngelSlim/DS-V4.1-Flash-DFly2-B6](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6) for [Jucee](https://github.com/bitnautlabs/jucee).

## Model Details

- **Source checkpoint:** [AngelSlim/DS-V4.1-Flash-DFly2-B6](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6) (Tencent AngelSpec, [arXiv:2611.34776](https://arxiv.org/abs/2611.34776))
- **Target model:** [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- **File:** `DeepSeek-V4.1-Flash-DFly2-MXFP4-AttnQ8.gguf` (7.53 GiB / 8,089,815,872 bytes)
- **SHA-256:** `f00d475c93a5f1ca9f9e7df647c3c2d01e3d2eb488296f31fa019c908972b499`
- **Architecture:** 3 sparse MoE draft layers (128 routed experts, 3 active, 1 shared expert, mHC-4 residual streams), paired with target latent-MQA layers `[28, 32, 36]` (sliding-window + sparse top-512 compressed KV sharing), block size 6, `slot0` MTP hidden fusion + gated attentive hidden correction, zero drafter KV cache, zero drafter prefill.
- **Quantization:** `MXFP4` routed experts (losslessly repacked from upstream FP4 E2M1 + E8M0), `Q8_0` attention / shared-expert / correction projections, `F16` mHC projections, `F32` norms / routers / sinks / biases (`93` tensors).
- **Reproducible recipe:** [`recipes/dsv41-dfly2/dsv41-dfly2.sh`](https://github.com/bitnautlabs/jucee/blob/main/recipes/dsv41-dfly2/dsv41-dfly2.sh) (`ds4/gguf-tools/dsv41_dfly2_convert.py`).

DFly2 is lossless speculative decoding — the target model verifies every drafted token (exact greedy match and exact rejection sampling).

## License & Attribution

Original DFly2 checkpoint by [AngelSlim/DS-V4.1-Flash-DFly2-B6](https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6) (Tencent AngelSpec), licensed under **Apache 2.0**.
Target base model: [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (MIT).
Jucee is Apache 2.0. Jucee is a trademark of Bitnaut, LLC.
