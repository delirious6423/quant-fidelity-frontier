---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- ijah
- exl3
- apple-silicon
- ngram-embedding
---

# Qwen3.8-Flash-Next n-gram embedding table

The per-layer n-gram embedding table of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`de4b8e4`), bf16 as released,
as one flat file. The base repo has no separate file for it (the table is spread over 128 shard tensors
`model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_*` inside its weights); this is the flat
extraction made by [Sushi](https://github.com/beamivalice/sushi) and published in
[beamster/Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) (`f25c0c4`), byte for byte, with thanks. The values are Qwen's own: we compared the file with the 128 shard
tensors in Qwen/Qwen3.8-Flash-Next on 12,288 sampled rows covering every shard (its first, last and one random
block) and found them bit-for-bit identical; Sushi only concatenated the shards in order behind a small header.
We host a copy so
our packs do not depend on another repo staying up. Values are unchanged (`base_model_relation: quantized` is the
closest Hub relation for a re-packaged part of the weights; nothing is quantized).

Both [Ijah](https://github.com/publicExcess/ijah) packs need it:
[SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah) and
[SiliconSpecies/Swift-1.5-Flash-Next-EXL3-Ijah](https://huggingface.co/SiliconSpecies/Swift-1.5-Flash-Next-EXL3-Ijah) (Swift-1.5 did not
change the table).

| file | size | sha256 |
|---|---|---|
| `ngram_table.bin` | 102,400,491,704 bytes (102.4 GB) | `387f988eef8d8d16adc08b46545601090b757ce17d5849975307234480609a14` |

Format: a safetensors-style file, an 8-byte header length, a JSON header
(`{"format": "mlx-serve-ngram", "bits": "16", "group_size": "0"}` metadata) and one tensor `weight`, BF16
[320001536, 160]. Ijah and Sushi read it from the SSD; it is never loaded into GPU memory.

```sh
hf download SiliconSpecies/Qwen3.8-Flash-Next-ngram-table ngram_table.bin --local-dir ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah
# second pack on the same volume: a hard link, not a second copy
ln ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah/ngram_table.bin ~/.cache/ijah/packs/Swift-1.5-Flash-Next-EXL3-Ijah/
shasum -a 256 ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah/ngram_table.bin   # optional, ~5 min
```

Licence: the base model's, Qwen Community License 1.0 (`LICENSE`).
