---
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
tags:
- unsloth
- nvfp4
- compressed-tensors
- archival-mirror
---

# Archival mirror: `unsloth/Qwen3.8-27B-NVFP4` at revision `9c73e2daee1d0fd494ffbd1d8753f2174a953796`

**This is a byte-identical archival mirror of one revision of [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) that the upstream repository no longer serves.** Every file here is the same bytes as that revision; nothing has been requantized, re-exported, or edited. Use the upstream repository for anything current — this mirror exists only so that a published measurement stays checkable.

## Why this exists

On 2026-08-15 the upstream repository's history was super-squashed (commit `16b6615a` is titled *"Super-squash branch 'main' using huggingface_hub"*). Squashing a model repository is routine and entirely the maintainer's call — the point of this mirror is narrower and is about *us*, not about unsloth: **pinning a Hugging Face revision is not a durable citation.** After the squash, `https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4/revision/9c73e2daee1d0fd494ffbd1d8753f2174a953796` answers `{"error":"Invalid rev id"}`, so the revision string in our published receipts stopped resolving.

We had measured this exact revision, and we cite it. So the snapshot is mirrored here, digests and all, and our receipts now cite this mirror alongside the original revision id.

## Relationship to current upstream HEAD (`16b6615af3548b88e2d8e382457bc705b00479cf`)

**The weights are byte-identical.** `model.safetensors` and `model_mtp.safetensors` have the same sha256 here as at current upstream HEAD, so this is not a different quantization of the model and nothing about the served numerics differs.

Three files do differ between the mirrored revision and current HEAD:

| file | mirrored revision `9c73e2da` | current HEAD `16b6615a` |
|---|---|---|
| `config.json` | `quantization_config.config_groups.group_1.weights.observer` = `"imatrix_mse"` | same field = `null` |
| `tokenizer.json` | sha256 `f399b3cd12fa270d51457bb749fb30863521e8359b8a27059c71b6c2f7d6dd6c` (19,989,424 B, not stored as LFS) | sha256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` (19,989,325 B, LFS) |
| `README.md`, `tokenizer_config.json` | as mirrored here | edited upstream |

`config.json` is otherwise identical field for field — that one `observer` entry is the only difference in the entire file, including every `config_groups` target, every scale strategy, and the whole `ignore` list.

The `observer` field is **metadata about how the quantizer chose scales at build time**, recorded by `llm-compressor`; it is not read at inference time, and the weights it describes are the same bytes in both revisions. So it does not change what the model computes. It is worth recording anyway, because it is the only surviving evidence of how the reviewed revision described its own construction: the revision we measured declares an imatrix-MSE observer, and the current one declares none.

### The `tokenizer.json` difference, and why this mirror was already needed once

This mirror earned its keep within the hour. One of our own documents pinned `tokenizer.json` sha256 `06b9509352d2af…`, while our local snapshot hashed `f399b3cd12fa…`, and **nobody could tell which side had drifted, because the revision we had cited no longer resolved.** Holding the mirror settled it by measurement: `06b9509352d2af…` (19,989,325 B) is current HEAD's, `f399b3cd12fa…` (19,989,424 B) is the reviewed revision's — the one in this repository, and it hashes byte-identically to our snapshot. Without the mirror that row would have had to stay unresolved permanently. This is the entire argument for archival mirroring in one paragraph: the point is not storage, it is that a citation stays checkable.

What actually differs between the two tokenizers is worth knowing if you use this repository, because it is inert in one API and a trap in another. A full structural walk of both 20 MB files reports **exactly one differing node**: `truncation`. This revision carries `{"direction":"Right","max_length":2048,"strategy":"LongestFirst","stride":0}`; HEAD carries `null`. The `vocab` (248,044 entries), `merges` and `added_tokens` are identical.

- Through `transformers.AutoTokenizer`, the two are **equivalent** — measured: a 12,002-token input tokenizes to 12,002 tokens with either file, because `PreTrainedTokenizerFast` sets truncation per call and defaults to off.
- Through the raw `tokenizers.Tokenizer` API, they are **not** — the file in this repository silently truncates that same input to **2,048 tokens**, while HEAD's returns all 12,002.

So if you load this revision with the raw library, disable truncation explicitly or you will quietly lose everything past 2,048 tokens. None of this affects the fidelity measurement described below, which replays a pre-tokenized suite and consults no tokenizer at all.

## Digests

Verified locally against the pinned snapshot before upload, and re-verified after upload.

| file | size (B) | sha256 |
|---|---|---|
| `model.safetensors` | 22,568,192,096 | `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05` |
| `model_mtp.safetensors` | 849,400,392 | `1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe` |
| `tokenizer.json` | 19,989,424 | `f399b3cd12fa270d51457bb749fb30863521e8359b8a27059c71b6c2f7d6dd6c` |
| `config.json` | 22,601 | `fd29e5c031dfb051fe591bec1c6417e313be3d1d777e742a2b67de2da937a6b6` |
| `model.safetensors.index.json` | 164,371 | `429430e1b9e65b2cb98eff8cd10a06e70a09cee89c48487a3914684aeb6df57f` |

The ten non-LFS files (`config.json`, `model.safetensors.index.json`, `.gitattributes`, `README.md` as mirrored under `upstream-README.md`, `generation_config.json`, `tokenizer_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `vocab.json`, `chat_template.jinja`) additionally match their upstream git blob ids at revision `9c73e2da`, recomputed as `sha1("blob <size>\0" + bytes)`.

The upstream card for the mirrored revision is preserved verbatim as `upstream-README.md`; this file replaces it as the repository card so that the archival status is the first thing a reader sees.

## Attribution and licence

Quantization and packaging by [unsloth](https://huggingface.co/unsloth); base model [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). Licence `apache-2.0`, inherited unchanged from the upstream repository, as are the `base_model` and `unsloth` tags. All credit for the artifact belongs upstream; this repository contributes nothing but durability.

## What we measured with it

`unsloth/Qwen3.8-27B-NVFP4` at this revision, served under vLLM with `--quantization compressed-tensors`, is one candidate in a teacher-forced distribution-fidelity comparison of Qwen3.8-27B quantizations: KL(BF16 reference ‖ candidate) over a held-out suite, full 248,320-entry vocabulary, both operands read at the final norm and projected through one shared BF16 LM head so that no candidate's own head quantization is counted. Reported in the `receipts/` tree of the qwen38-27b project, including `receipts/nvfp4-v5-measurement.json`, which records this mirror.
