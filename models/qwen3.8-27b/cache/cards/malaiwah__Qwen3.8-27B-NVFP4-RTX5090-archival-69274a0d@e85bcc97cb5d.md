---
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
tags:
- nvfp4
- modelopt
- archival-mirror
---

# Archival mirror: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` at revision `69274a0d8dff5dd35bcee8290612f71e03b6e981`

**This mirror exists so that a published measurement of ours stays checkable if the upstream revision ever stops resolving; it is precautionary, not a recovery — upstream still serves this revision today.** Every model file here is byte-identical to that revision; nothing has been requantized, re-exported, or edited. Use the [upstream repository](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) for anything current.

The one deliberate difference in this repository's file layout: upstream's model card is preserved byte-identically as
[`README.upstream-gittensor.md`](README.upstream-gittensor.md) (sha256 `65dc0fff9a6f829884fb1b3ef2ecd9b13696d13aca340406f92e80c6dc0207b3`),
and this file replaces it as the repo's README so the mirror explains itself. `README.qwen-upstream.md`
(Qwen's original card, as shipped by upstream) and every other file keep their upstream names and bytes.

## Why this exists

We measured this exact revision — teacher-forced KL divergence on shard 0 of our v5 held-out suite, plus
resident-weight memory under our published protocol — and our receipts cite it by revision id. A Hugging Face
revision id is not a durable citation: an upstream history squash makes it unresolvable (this happened to
`unsloth/Qwen3.8-27B-NVFP4@9c73e2da` on 2026-08-15, one day after we measured it, which is why this policy is
now unconditional). Xet storage is content-addressed, so this mirror costs a small fraction of its nominal
size in actual transfer; what it preserves is the citation — a resolvable repo id, revision and digest table.

Attribution: all model content by [gittensor-model-hub](https://huggingface.co/gittensor-model-hub), a
ModelOpt NVFP4 (W4A4, group 16, FP8-KV-calibrated) export of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Licence inherited unchanged from upstream (Apache-2.0; `LICENSE` is upstream's byte-identical copy).

## What this checkpoint is (read from its own files, not restated from anyone's card)

* `config.json` `quantization_config.quant_method` = `"modelopt"`; `hf_quant_config.json` declares
  `quant_algo: "NVFP4"`, `kv_cache_quant_algo: "FP8"`, `group_size: 16`, producer `modelopt 0.0.1.dev1+gc4129b6e0`.
* 2,399 tensors in the index: 400 body Linear modules quantized to NVFP4 (packed `U8` weights +
  `F8_E4M3` `weight_scale` + `F32` `weight_scale_2`); 149 modules excluded from quantization
  (`lm_head`, `embed_tokens`, and every GDN `conv1d`/`in_proj_a`/`in_proj_b`).
* MTP draft intact: 15 `mtp.*` tensors, all BF16. Vision intact: 333 `model.visual.*` tensors, all BF16.
* `model-00003-of-00003.safetensors` (sha256 `9ce944d5…`) is bit-identical to the trailing shard of at least
  three other ModelOpt NVFP4 exports of this base model (`vroomfondel`, `PassingByPixels`, `kristianpaul`) —
  it is the untouched BF16 vision/embed/lm_head/MTP block that ModelOpt emits identically. The two quantized
  body shards are unique to this build; this is not a re-upload of any checkpoint we had measured before.

## Digests

Verified locally against the pinned snapshot before upload, and re-verified after upload
(sha256 over every file of revision `69274a0d`; `README.md` upstream bytes live at
`README.upstream-gittensor.md` here).

| file | size (B) | sha256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 9,972,777,720 | `cdd37b0e61eccc8a3d7d08f9d1a4f52856a9d88e4e8b42089bd18a970e3a01ec` |
| `model-00002-of-00003.safetensors` | 9,875,839,416 | `f2faf6c100edad7014810c989754408251dce5064f3623679ef6c33a5c393bc9` |
| `model-00003-of-00003.safetensors` | 744,532,384 | `9ce944d534eabdd493076a3a52c7ebd31f41c135b340a1ea95c5a695e6f1f6b2` |
| `model.safetensors.index.json` | 237,457 | `1cefff97291d43710cc071ffc2b981b80bcf821be1f69d813853d89368c83c2b` |
| `config.json` | 15,064 | `92d0c418b744f2d1518cbaefd11739744ff43ce9b952219b64adfd5b96f38145` |
| `hf_quant_config.json` | 10,034 | `eb294f05267ba5fd0a2f8402a74c904ea5f01e13547003bdbc1e6cbbdcaccbab` |
| `tokenizer.json` | 12,809,320 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` | 1,150 | `c873857aae349387312ff4cb76d4a17a8d5ed79f89523146200a1570739132db` |
| `chat_template.jinja` | 8,952 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| `generation_config.json` | 213 | `7ae9e193dbcef99733ccf647c95ef668c35d1a80a8aa88a51ee40a9bcacf5a74` |
| `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `processor_config.json` | 1,191 | `d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1` |
| `video_preprocessor_config.json` | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| `crc32.txt` | 238 | `b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83` |
| `LICENSE` | 11,544 | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |
| `README.upstream-gittensor.md` (upstream `README.md`) | 7,085 | `65dc0fff9a6f829884fb1b3ef2ecd9b13696d13aca340406f92e80c6dc0207b3` |
| `README.qwen-upstream.md` | 65,012 | `57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641` |
| `.gitattributes` | 1,630 | `9a7c7f59da8a2bad26352923e6e6d4c8952822735ea3b8de5c0d614d22af85a7` |
| `assets/rtx5090-accuracy.png` | 86,034 | `70013eee4415a5e4c55c36e83f165d39070772976649a5f247223c1730fa9b8c` |
| `assets/rtx5090-context.png` | 81,913 | `0a0df80b4c1e4276994bd7688ad49b66efb8c845c4364089edc284222a5cff73` |
| `assets/rtx5090-decode.png` | 74,574 | `66d11fe41563f3c091177f0b0e05b2b2094f8febe8ccf6c7918e3661d94b45e6` |
| `assets/rtx5090-hero.png` | 123,721 | `65fdb7d51c26626e3fb9effd8fdafce6046be70e944f182ba15d331c018d44a5` |
| `assets/rtx5090-ttft.png` | 71,910 | `ab25dd6883749efb301635dc2ab4bf36e9fee31a2874d39b8c760661ff826abd` |

## What we measured against this revision

Our measurement receipts live in the research repository
([`malaiwah/qwen38-27b-exl3`](https://github.com/malaiwah/qwen38-27b-exl3)):
`receipts/gittensor-nvfp4-rtx5090.json` is the measurement receipt; the shard-0 KLD row is in
`receipts/cross-engine-comparator.json`. The upstream card's own serving claims (resident GB, KV pool
tokens, decode tok/s, 262k context on an RTX 5090) are **upstream's claims, measured by upstream** — this
mirror preserves them; our receipts measure teacher-forced fidelity and resident weight memory only.
