---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - gguf
  - qwen4exp
  - q4_k
  - mxfp4
  - qwen
  - abliterated
  - uncensored
  - ds4
  - metal
---

# Qwen3.8-Flash-Next-abliterated — Q4 GGUF for DS4

**This GGUF was made for use with [DS4 / DwarfStar by antirez](https://github.com/antirez/ds4).**

It is antirez's [Qwen3.8-Flash-Next Q4 DS4 release](https://huggingface.co/antirez/qwen3.8-flash-next-gguf) with huihui-ai's [Qwen3.8-Flash-Next abliterated v2](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF) edit applied, based on [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). The file keeps DS4's own `qwen4exp` layout, including MTP weights and the original BF16 n-gram table, so it is a drop-in replacement for `Qwen3.8-Flash-Next-Q4.gguf`.

## Source ablation

huihui-ai's v2 abliteration edits only the matrices that write back into the residual stream outside the routed experts:

| Tensor family | Role | Layers edited |
| --- | --- | ---: |
| `ssm_out` | Linear-attention output projection | 36 |
| `attn_output` | Full-attention output projection | 12 |
| `ffn_down_shexp` | Shared-expert down projection | 48 |

huihui-ai offers three strengths. This release uses the strongest setting, `--ablate-experts 2`, which edits all three families across trunk layers 0–47; huihui-ai describes it as "no warnings". The routed experts, embeddings, n-grams, MTP layer, router, hyper-connections and tokenizer are not changed. Compared with the original model, each edited tensor changes by about 3.6% of its norm, almost entirely in a few directions, which is consistent with a projection-style ablation. See the [huihui-ai model card](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF) for the method; no refusal or quality evaluation is published there.

## This conversion

huihui-ai distributes v2 as a patch for unsloth's GGUFs: pre-quantized replacement tensors, XOR-obfuscated with a published key. DS4's Q4 pack stores these 96 tensors as Q8_0 with the same names and shapes, and their original bytes are identical to unsloth's Q8_0, the base the patch was built against. The patch's Q8_0 tensors were therefore decoded and written in place at the same offsets.

- The GGUF header and tensor layout are byte-identical to the source file. Only the payloads of the 96 edited tensors differ.
- Each written tensor was read back and checked against the decoded patch data. Sampled unedited tensors, including the MTP layer, were checked byte-for-byte against the source.
- No requantization, training or additional ablation was performed.
- Routed experts keep the source's calibrated Q4_K gate/up and MXFP4 down projections. Dense projections and shared experts are Q8_0, hyper-connections F16, and embeddings and the n-gram table BF16.

The patching script is [`ds4_abliterate.py`](ds4_abliterate.py). Run with `--ablate-experts 2`, it reproduces this file from the source revisions below.

| Source | Revision |
| --- | --- |
| [antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf) `Qwen3.8-Flash-Next-Q4.gguf` | `d600fe1a43d2e1cdcadb85144ce3142f66f9eefe` |
| [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF) `patch-unsloth-all` | `6e298f56319a5735f88ae952dd27222369f2f44a` |

The source GGUF's SHA-256 was verified against antirez's published checksum, `680944460a8cbe93ba8b6d7b6107213ffb7e22320bd913000e563ca0a0f25a8a`.

## Testing so far

Tested on a 128 GB Apple Silicon Mac with DS4 commit `b3f50180ec2c7324892be1f50bc83555ccb8d5e7`, using `--temp 0 --nothink` and 120-token outputs:

| Prompt | Original Q4 | This release |
| --- | --- | --- |
| Capital of France, in two sentences | Correct answer | Word-for-word identical answer |
| Step-by-step pin tumbler lock picking | Refuses, then explains only how locks work | Answers directly |

This is a smoke test, not an evaluation. Refusal rates, reasoning and coding quality, MTP acceptance and vision have not been measured for this file. To compare against the original, run `./ds4-eval -m MODEL.gguf --suite hard` on both.

## Release files

| File | Size in bytes |
| --- | ---: |
| `Qwen3.8-Flash-Next-Q4-abliterated.gguf` | 177,280,286,720 |

`SHA256SUMS` contains its checksum. The file holds 69.74 GiB of main/MTP weights, which DS4 keeps resident, and the 95.37 GiB BF16 n-gram table, which DS4 reads from disk.

## Download and run

Download `Qwen3.8-Flash-Next-Q4-abliterated.gguf` and `SHA256SUMS` from this repository's **Files and versions** tab into the `gguf/` directory of a DS4 checkout. Then, from that directory:

```sh
shasum -a 256 -c SHA256SUMS
```

On Linux, use `sha256sum -c` instead. Build DS4 with `make` and run from the checkout's root:

```sh
./ds4 -m gguf/Qwen3.8-Flash-Next-Q4-abliterated.gguf --ctx 8192 --prefill-chunk 1024
```

MTP speculative decoding uses the weights already in the file; add `--mtp`. For image input, download the unmodified vision encoder with `./download_model.sh qwen38-vision` and pass `--vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf`. Keep the model on a fast local SSD, since the n-gram table is read from disk during inference.

The Q4 pack needs more than 64 GB of RAM for resident weights. On smaller Macs, use `--ssd-streaming`. See DS4's [Qwen3.8 Flash Next guide](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md) for memory sizing, MTP and vision. Other GGUF engines have not been validated with this layout.

## Attribution and license

Credits to Qwen for the original model, huihui-ai for the abliteration, Ivan Fioravanti for the quantized Q4 main/MTP weights, and Salvatore Sanfilippo (antirez) for DS4 / DwarfStar and the native n-gram packaging. This community conversion does not imply endorsement by any of them.

The Qwen Community License 1.0 is included as `LICENSE`. Note its conditions for very large commercial deployments and for Model-as-a-Service or AI Work Assistant businesses. The abliterated model is intended to reduce refusals and can produce unsafe or incorrect output; review its outputs, and refer to huihui-ai's model card for its usage warnings.
