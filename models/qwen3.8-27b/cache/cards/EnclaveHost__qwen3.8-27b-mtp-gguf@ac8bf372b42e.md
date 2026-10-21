---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- qwen3.8
- mtp
- enclave
---

# qwen3.8-27b-mtp-gguf (Enclave model volume)

Curated model volume for [Enclave](https://enclave.host) confidential inference:
Qwen3.8-27B at unsloth Dynamic **Q6_K_XL** (~25.9 GB), a single-file GGUF bundled
with the official `tokenizer.json` so the volume is self-contained.

Q6 rather than the family's usual Q4_K_XL deliberately: this volume is for code
work, where a wrong token is a broken program rather than a clumsy sentence. The
dynamic mix keeps attention, `ffn_down` and `output.weight` at Q8_0, `ffn_gate` /
`ffn_up` at Q6_K and the SSM projections at Q5_K.

Upstream [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
ships 22 quantizations plus BF16, vision `mmproj-*` files and an imatrix, and no
`tokenizer.json`. This volume carries ONLY the one served quant plus the
tokenizer (same rationale as `qwen3.6-27b-gguf` / `qwen3.5-9b-gguf`): a
multi-gguf volume makes the host's preload pick ambiguous, and wrapping the whole
repo wastes hundreds of GB of attested volume on files that never serve.

## Self-drafting: the MTP head is in this file

Qwen3.8 ships its trained multi-token-prediction head as part of the base model
(`mtp_num_hidden_layers` 1), and the quant carries it: `block_count` 65,
`nextn_predict_layers` 1, and `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`
present alongside a full dense-attention block 64. So this is the SELF-DRAFTING
27b - set `"draft": "mtp"` on the catalog entry and speculation costs no second
model and no second volume. `blk.64.nextn.embed_tokens` / `.shared_head_head` are
absent by design (`mtp_use_dedicated_embeddings` false - they are tied to the
trunk's, and llama.cpp marks both NOT_REQUIRED).

Hosts that predate the batch4 MTP toolchain (shim `ell_mtp_*`) serve this as a
plain 27b instead, which is a speed difference and never a correctness one.

## Geometry (from the GGUF header)

| Key | Value |
|-----|-------|
| `general.architecture` | `qwen35` (hybrid: 48 gated-DeltaNet + 16 full-attention of 64, `full_attention_interval` 4) |
| blocks | 65 = 64 trunk + 1 MTP head |
| `attention.head_count` / `head_count_kv` | 24 / 4 |
| `attention.key_length` / `value_length` | 256 / 256 |
| `embedding_length` / `feed_forward_length` | 5120 / 17408 |
| `context_length` | 262144 |
| `rope.freq_base` / `rope.dimension_count` | 10000000 / 64 |
| `rope.dimension_sections` | `[11, 11, 10, 0]` (interleaved M-RoPE) |
| `ssm.{conv_kernel,state_size,group_count,time_step_rank,inner_size}` | 4 / 128 / 16 / 48 / 6144 |
| vocab | 248320 padded (248044 real entries + 33 added) |
| eos | 248046 `<|im_end|>`, 248044 `<|endoftext|>` |
| template | chatml, `<think>` force-opened by the trained template |

For an app catalog entry: `n_layers` 64, `n_kv_heads` 4, `head_dim` 256,
`kv_layers` **17** (the 16 full-attention layers, which are the only trunk layers
holding KV, plus the MTP head's own dense block).

## Provenance

| File | Upstream | Revision | sha256 |
|------|----------|----------|--------|
| `Qwen3.8-27B-UD-Q6_K_XL.gguf` | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | `4604b899a826000505a834e623272db5b7fd62f6` | `739202186fd9389bb58497c58b56c8a0d4253d99d20131e6a0427e363e678fc8` |
| `tokenizer.json` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |

`LICENSE` is the upstream Apache-2.0 text from `Qwen/Qwen3.8-27B`; both upstreams
are Apache-2.0.

### The tokenizer is NOT byte-identical to the Qwen3.5/3.6 family copy

It is a strict superset. The 248044-entry vocab and all 247587 merges are
byte-identical to `qwen3.6-27b-gguf`'s `tokenizer.json` (sha256 `5f9e4d49...`),
as are the normalizer, pre-tokenizer, post-processor and decoder; Qwen3.8 adds
seven special ids on top, 248070-248076
(`<|audio_start|>`, `<|audio_end|>`, `<tts_pad>`, `<tts_text_bos>`,
`<tts_text_eod>`, `<tts_text_bos_single>`, `<|audio_pad|>`).

Ordinary text therefore tokenizes identically to the rest of the family, but the
file hash differs, and cross-model speculative decoding gates on byte-equality:
`qwen3.5-0.8b-gguf` will be REFUSED as a draft for this model. That costs
nothing here, because the volume carries its own MTP head, which is the better
draft anyway.

## Text-only, on purpose

Upstream Qwen3.8-27B is `image-text-to-text` (it has a 27-layer SigLIP-style
vision tower) and the unsloth repo ships `mmproj-F16.gguf` / `mmproj-BF16.gguf`
beside the quants. This volume omits them, like `qwen3.5-0.8b-mtp-gguf`: no
projector, and its catalog entry sets no `"vision"`. This GGUF contains no vision
tensors at all.

To make a Qwen3.8 volume that reads images, add `mmproj-F16.gguf` (~0.93 GB) to
the volume, name the weights in the `MODEL_VOLUMES` third field (a two-gguf
volume needs it, as `qwen3-vl-8b-gguf` does), set `"vision": true` on the catalog
entry, and price the projector into the share - it loads lazily on the first
image, so a deployment that fits only the language model serves text happily and
then dies on the first picture.

## Sizing

At the fleet's 176K q8_0 node window (`ENCLAVE_GGML_N_CTX` 180224) this volume
costs about **34.1 GB** to serve: 25.9 weights + 6.7 KV (17 KV-holding layers x
4 heads x 256 head_dim x 2 x q8_0, including the lazily-created MTP head
context) + ~1.5 GB working set. Against the 140.4 GB an H200 holds that is a
0.24 share on paper, so **`gpu` 0.26-0.27** is the honest ask - a 0.25 share
(35.1 GB) leaves under a GB of margin, and the MTP head context is allocated on
the first speculative session rather than at boot, which is exactly how
deployment a025ed60 preloaded, answered `/warmup` in 100 ms and then stalled its
first real chat request.

The plain `Q6_K` quant (22.9 GB) is the cheaper alternative at ~31.1 GB to serve
(gpu 0.23-0.24) if that margin matters more than the last increment of accuracy.

## Usage on Enclave

Wrapped as a Tinfoil Modelwrap volume (dm-verity; the root hash is part of the
enclave measurement). Deployments attach it by name and the guest reads it at
`/models/<name>`; the host preloads the GGUF as the wasi-nn ggml graph. As the
volume's single gguf it needs no `MODEL_VOLUMES` third field and no `model_file`
override, and with the tokenizer bundled, the app's default `tokenizer.json`
lookup works with no cross-volume configuration.
