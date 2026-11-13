---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- speculative-decoding
- mtp
- lamina
---

# Qwen3.8-Flash-Next draft layer and image projector for Lamina

The multi-token-prediction (MTP) layer of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in the
runtime format of [Lamina](https://github.com/eddoursul/Lamina), which drafts tokens with it for speculative decoding
(`--mtp <this folder>`). A draft is only a proposal: the model checks every one, so the output is the model's own. The
model's image projector in Q8_0, for Lamina's image encoder, is here too ([below](#the-image-projector)).

| File | Bytes | SHA-256 |
|---|---|---|
| `experts.bin` | 707,788,800 | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` |
| `dense.bin` | 116,099,072 | `a46561c5d3e76e78eebb9b54bbf4ae87ba1f99a62cad85562cbb516141cd5e47` |
| `dense.txt` | 1,880 | `8773c81ebb0986e37fe94a8a9933e87be48b1fabc6889a0106bcdab179d1c2ac` |
| `draft_vocab.bin` | 162,100 | `369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25` |

- `experts.bin`: the layer's 512 routed experts from Qwen's BF16 checkpoint in Q2_0 (64-value blocks, each block's scale
  chosen to minimize the squared error over the levels {-1, 0, 1, 2}), in Lamina's expert layout.
- `dense.bin`, indexed by `dense.txt`: the rest of the layer (attention, projections, hyper-connections, router, shared
  expert, norms), the large projections in Q8_0, the hyper-connection and router weights in BF16, the norms in F32. Its
  weights start from the self-distilled ones of [basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) and are
  trained further on Lamina's own output with ISTA-DASLab's GSQ-RCO IQ3_S and unsloth's UD-Q4_K_XL (987 requests,
  2.4M positions), the target each draft's chance to be accepted by the model's greedy or sampled choice.
- `draft_vocab.bin`: the 40,525 token ids (English and code) the draft head proposes from.

On held-out text the drafts carry 2.489 (IQ3_S) / 2.433 (UD-Q4_K_XL) tokens a verify pass, against 2.402 / 2.344 with
basalt's dense part; on one PC (RTX 3090 and RTX 5070 Ti) Lamina writes thinking text 1.3% / 4.5% faster with it.

## The image projector

| File | Bytes | SHA-256 |
|---|---|---|
| `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` | 616,703,328 | `2486a2e7bd834a0494fd1f5b324f66b78f7836cb0c6644b090be7ed1c58527c3` |

Qwen3.8-Flash-Next's vision projector for llama.cpp's mtmd, which Lamina's image encoder (`lamina-vision`) runs:
[ISTA-DASLab's BF16 file](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(`mmproj-Qwen3.8-Flash-Next-BF16.gguf`, SHA-256 `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0`)
converted by Lamina's `tools/mmproj_q8.py` with the gguf-py of llama.cpp 3cf0325: each BF16 matrix whose rows are whole
32-value blocks in Q8_0, the other BF16 tensors in F16, the F32 ones as they are, as llama-quantize converts it (its
tensors byte for byte; its embeddings of a picture identical). On the CPU it encodes a picture ~20% faster than the
BF16 file.

## License

Derived from Qwen3.8-Flash-Next's weights: the [Qwen Community License 1.0](LICENSE) applies.
