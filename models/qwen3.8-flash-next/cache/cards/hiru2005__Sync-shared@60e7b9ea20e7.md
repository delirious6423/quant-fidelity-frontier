---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- qwen3.8-flash-next
- moe
- sync-strata
- experimental
---

# Sync shared parts for Qwen3.8-Flash-Next (experimental)

The parts every Sync model (Sync-XS, Sync-S, Sync-M) needs besides its own pack. Download them once.
They are used only by **[Sync-Strata](https://github.com/hiruhiru2005/sync-strata)**; the Sync-Strata app downloads them for you.

## Files

### `native-gguf/`

Shared weights cut from agentionai's Gyro-S GGUF (embeddings, n-gram/PLE table and the other non-expert tensors the engine reads with `--native` / `--ple-gguf`).

The file list and sizes of this folder will be added when it is published. The whole Gyro-S file is 58.49 GB; this folder keeps only the parts the engine reads.

### `mtp-rt/`

The multi-token-prediction (draft) layer, converted from the original BF16 checkpoint for the engine; `draft_vocab.bin` comes from upstream Strata (MIT).

Total: 0.82 GB (0.77 GiB)

| File | Size |
|---|---:|
| `dense.bin` | 116,099,072 B |
| `dense.txt` | 1,880 B |
| `draft_vocab.bin` | 425,196 B |
| `experts.bin` | 707,788,800 B |

### `tokenizer/`

The tokenizer and chat template of Qwen3.8-Flash-Next.

Total: 0.01 GB (0.01 GiB)

| File | Size |
|---|---:|
| `chat_template.jinja` | 9,993 B |
| `merges.txt` | 3,600,844 B |
| `token_type.json` | 744,960 B |
| `tokenizer.json` | 554 B |
| `vocab.json` | 5,737,005 B |

### `mmproj/` (optional)

Optional vision encoder and projector (BF16). **Not hosted here**: the Sync-Strata app downloads it from its original repository, [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF). Its card says the files inherit the license of the base model (Qwen Community License 1.0); the repository metadata says Apache-2.0.

| File | Size |
|---|---:|
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (original repository) | 907,543,008 B |

## Where they come from

- `native-gguf` is cut from [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)
  (Gyro-S; the revision will be added). It keeps only the tensors the engine reads; the tensors are meant to be
  byte-for-byte those of the original file, but this has not been checked with a tool yet.
- `mtp-rt` is converted from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (BF16) with upstream Strata's tools.
- `tokenizer` is the base model's tokenizer.
- `mmproj` is not in this repository; it is downloaded from [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (converted there from the base checkpoint).

## Credits

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team.
- [agentionai](https://huggingface.co/agentionai) (Gyro-S), [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) (vision projector),
  [Strata](https://github.com/Niko1221/Strata) (MIT; tools and `draft_vocab.bin`).

## Support (optional)

Sync-Strata and the Sync models are free. A donation page is at [Ko-fi](https://ko-fi.com/hiruhiru2005): a thank-you only, with nothing in return (no perks, no priority support, no feature promises), and it does not change the license terms.

## License

Derived from Qwen3.8-Flash-Next: **Qwen Community License 1.0** (`LICENSE`, Copyright (c) 2026 Qwen). Keep `LICENSE`
with every copy. See the full text for the conditions on large commercial products and on Model-as-a-Service / AI work
assistant businesses.
