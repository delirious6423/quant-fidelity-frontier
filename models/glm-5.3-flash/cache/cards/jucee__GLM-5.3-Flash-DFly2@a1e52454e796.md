---
license: apache-2.0
base_model: zai-org/GLM-5.3-Flash
tags:
- dfly2
- speculative-decoding
- gguf
- jucee
- apple-silicon
- metal
---

# GLM-5.3-Flash DFly2 Drafter (Q4_K + Q8_0 Attention GGUF for Jucee)

Quantized GGUF conversion of [AngelSlim/GLM-5.3-Flash-DFly2-B6](https://huggingface.co/AngelSlim/GLM-5.3-Flash-DFly2-B6) for [Jucee](https://github.com/bitnautlabs/jucee).

## Model Details

- **Source checkpoint:** [AngelSlim/GLM-5.3-Flash-DFly2-B6](https://huggingface.co/AngelSlim/GLM-5.3-Flash-DFly2-B6) (Tencent AngelSpec, [arXiv:2611.34776](https://arxiv.org/abs/2611.34776))
- **Target model:** [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
- **File:** `GLM-5.3-Flash-DFly2-Q4_K-AttnQ8.gguf` (5.74 GiB / 6,160,648,384 bytes)
- **SHA-256:** `c985b0de5b5bf5f22262a8e1b265a8510d7c4b9c8baf834ede3ed3fcc0b75015`
- **Architecture:** 3 sparse MoE draft layers (128 routed experts, 3 active, 1 shared expert), paired with target DSA layers `[35, 39, 43]` (KV sharing + top-2048 index sharing), block size 6, `slot0` hidden fusion + intra-block hidden/attention correction, zero drafter KV cache, zero drafter prefill.
- **Quantization:** `Q4_K` routed experts (`gate_up`/`down`), `Q8_0` attention / shared-expert / correction projections, `F32` norms / routers / gates (`93` tensors).
- **Reproducible recipe:** [`recipes/glm53-dfly2/glm53-dfly2.sh`](https://github.com/bitnautlabs/jucee/blob/main/recipes/glm53-dfly2/glm53-dfly2.sh) (`ds4/gguf-tools/glm53_dfly2_convert.py`).

DFly2 is lossless speculative decoding — the target model verifies every drafted token (exact greedy match and exact rejection sampling).

## License & Attribution

Original DFly2 checkpoint by [AngelSlim/GLM-5.3-Flash-DFly2-B6](https://huggingface.co/AngelSlim/GLM-5.3-Flash-DFly2-B6) (Tencent AngelSpec), licensed under **Apache 2.0**.
Target base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT).
Jucee is Apache 2.0. Jucee is a trademark of Bitnaut, LLC.
