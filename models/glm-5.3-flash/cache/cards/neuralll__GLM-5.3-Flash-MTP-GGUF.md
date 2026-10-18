---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
license: mit
library_name: gguf
tags:
  - gguf
  - glm5-next
  - mtp
  - speculative-decoding
---

# GLM-5.3-Flash MTP draft head (GGUF, Q4_K)

The GLM-5.3-Flash multi-token-prediction (MTP / NextN) block as a small standalone GGUF
(4.3 GiB), for speculative decoding next to a GLM-5.3-Flash model you already have. No need
to download a whole model again just to get MTP.

```sh
llama-server -m GLM-5.3-Flash-GSQ-RCO-3.0bit-q4kattn.gguf \
    -md GLM-5.3-Flash-MTP-Q4_K.gguf --spec-type draft-mtp
```

The draft borrows the token embeddings and output head of the main model, so only the MTP
block itself is stored here. The main model verifies every drafted token: the draft's
quantization changes how often drafts are accepted, never the output.

## Requirements

- **The [neurall/llama.cpp](https://github.com/neurall/llama.cpp) fork, release
  [b11327](https://github.com/neurall/llama.cpp/releases/tag/release-b11327-79e9090) or newer.** Stock llama.cpp does not run GLM-5.3-Flash MTP
  yet (PR [#27917](https://github.com/ggml-org/llama.cpp/pull/27917)), and loading an MTP-only
  file as `-md` is a fork addition.
- A main GLM-5.3-Flash GGUF with the `glm5-next` architecture name, e.g.
  [pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)
  (3.0-bit, 3.5-bit) or
  [neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF).
  unsloth's GLM-5.3-Flash GGUFs use a different architecture name (`glm5next`) and don't load
  in the fork.

## Will it be faster?

It depends on how much of the model the GPUs hold. Every drafted token has to be verified,
and each token picks its own experts; experts that run on the CPU cost about as much per
drafted token as per generated one.

Measured on 2x RTX 3090 (48 GB) + 125 GB DDR4 with the fork's expert cache, 1500-token chat
reply, GLM-5.3-Flash 3.0-bit Q4_K attention (106 GB, 2.3x the VRAM):

| | decode t/s |
|---|---|
| without MTP | 20.2 |
| with this MTP head (best draft depth, 1) | 17.6 to 18.4 |

Draft acceptance is ~70-75% at depth 1, but here ~30% of expert work runs on the CPU and the
draft's VRAM shrinks the expert cache, so MTP is slower on this setup. Expect a gain when
most of the model fits in VRAM (more or bigger GPUs, smaller quants). For comparison, on the
same box Qwen3.8-Flash-Next (1.9x the VRAM, 95% of experts cached) gets 1.21x from MTP.

## How it was made

The NextN block (`blk.45.*`, 29 tensors) is taken byte for byte from
[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) UD-Q4_K_XL
(attention Q8_0, experts Q4_K/Q5_K), fetched with HTTP range requests instead of downloading
the model. Metadata comes from the GSQ-RCO main model, with `block_count` 46,
`nextn_predict_layers` 1 and the MTP layer's per-layer values from unsloth's file. Scripts in
this repo (also in the fork under [`tools/moe-bench/`](https://github.com/neurall/llama.cpp/tree/release/tools/moe-bench), run from a fork checkout): `gguf_remote.py` (read a remote GGUF header) and `glm_splice_mtp.py`
(`--mtp-only` writes this file; without it, it writes a full model with the MTP block).

## Credits and license

- Model and MTP weights: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), MIT.
- Quantization of the MTP block: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), MIT.
- GLM-5.3-Flash MTP support in llama.cpp: PR [#27917](https://github.com/ggml-org/llama.cpp/pull/27917) (timkhronos).

MIT license, see `LICENSE`.
