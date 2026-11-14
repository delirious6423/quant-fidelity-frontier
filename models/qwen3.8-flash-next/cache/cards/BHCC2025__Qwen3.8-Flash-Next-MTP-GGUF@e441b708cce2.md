---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
tags:
- gguf
- mtp
- speculative-decoding
- ferrite
---

# Qwen3.8-Flash-Next MTP draft (GGUF)

The multi-token prediction (MTP) head of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) as
one small GGUF file, for speculative decoding next to a quantized Flash-Next model. It holds only the MTP block
(`blk.48.*`, 32 tensors): no embeddings, no output head, no trunk layers. The draft reads the trunk's token
embeddings and output head, so it is not a model on its own.

It is made for [Ferrite](https://github.com/BHCC2025/ferrite) serving the
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
IQ1_M files, which carry no MTP block. Ferrite loads those files as published and this file beside them
(`scripts/fetch-flashnext.sh` downloads both and checks their sha256; see `docs/flash-next.md` in the Ferrite
repository). Drafting is lossless: greedy output is the same token stream with and without it, only faster.

| File | Size | sha256 |
|---|---|---|
| `Qwen3.8-Flash-Next-MTP.gguf` | 1,517,334,816 bytes (1.41 GiB) | `54ba284d7538c534e3d8e6231c1a8fa46715d27e56f0f31af66e2ac39ee7914b` |

## Contents

| Tensors | Type | Size |
|---|---|---|
| Routed experts, gate and up (`ffn_gate_exps`, `ffn_up_exps`: 512 experts x 640 x 2560) | Q4_K | 2 x 450 MiB |
| Routed experts, down (`ffn_down_exps`) | IQ4_NL | 450 MiB |
| Attention q/k/v/output, shared expert, hyper-connection mixers, `nextn.eh_proj` | Q8_0 | 88.7 MiB |
| QSA indexer q/k projections | BF16 | 3.1 MiB |
| Norms, router (`ffn_gate_inp`), shared-expert gate | F32 | 5.2 MiB |

Metadata: `general.architecture = qwen4exp`, `qwen4exp.block_count = 49`, `qwen4exp.nextn_predict_layers = 1`
and the rest of the `qwen4exp.*` hyper-parameters of the checkpoint (no tokenizer). Tensor names follow llama.cpp's
`qwen4exp` naming, with the hyper-connection down projection and inject weights as separate tensors
(`hc_attn_down` / `hc_attn_inject`, `hc_ffn_down` / `hc_ffn_inject`).

## How it was made

1. The 31 `mtp.*` tensors of Qwen/Qwen3.8-Flash-Next at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`
   (BF16, 4.86 GiB) were read with HTTP range requests from the checkpoint's safetensors files; nothing else of the
   checkpoint was downloaded. Their safetensors files are the same (same LFS sha256) at the later revision
   `812d96e10a0dd82b69d749e68693698cf6c36daa`.
2. llama.cpp's `convert_hf_to_gguf.py --mtp` (the `qwen4exp` converter, unmodified, llama.cpp commit `74b104c9`)
   converted them twice, with `--outtype q8_0` and `--outtype bf16`.
3. A short script assembled the file: the routed experts requantized from the Q8_0 tensors to Q4_K (gate, up) and
   IQ4_NL (down) with ggml's own quantizers (`ggml_quantize_chunk`, no importance matrix); the QSA indexer
   projections taken from the BF16 conversion; everything else as Q8_0 / F32 from the Q8_0 conversion; the
   converter's merged `hc_*_down_inject` [10240 x 324] split into its 320 down rows and 4 inject rows (exact for Q8_0,
   whose blocks never cross a row).

The tensor data of this file is byte for byte the MTP block Ferrite was developed and gated with.

## Measured with Ferrite

Qwen3.8-Flash-Next GSQ-RCO IQ1_M, two RTX 5060 Ti 16 GB in tensor parallel, AMD EPYC 7313 host, 128K context,
greedy decoding, 256 tokens after a short chat prompt, three warm runs:

| | Without the draft | With the draft (MTP) |
|---|---|---|
| Code | 74.5-74.8 tok/s | 135.2-136.4 tok/s |
| Prose | 74.8-75.3 tok/s | 110.0-111.0 tok/s |

About 92% of the drafted tokens are accepted on code. Greedy output with the draft is identical to greedy output
without it (checked token for token on four reference prompts of 39 to 20,000 tokens).

## Credits

- **Qwen** for Qwen3.8-Flash-Next and its MTP head (the weights in this file).
- **ISTA-DASLab** for the GSQ-RCO quantization of Flash-Next this draft is paired with.
- **llama.cpp / ggml** for the `qwen4exp` GGUF converter and the Q4_K / IQ4_NL / Q8_0 quantizers.

## License

The weights are a derivative of Qwen/Qwen3.8-Flash-Next and are distributed under the
[Qwen Community License 1.0](LICENSE), the license of the original checkpoint. Read
[LICENSE-NOTE.md](LICENSE-NOTE.md) before you offer a service built on it: the license restricts commercial use for
"Model as a Service" and "AI Work Assistant" businesses.
