---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- ingot
- lossless
- compression
---

# Qwen3.8-27B, Ingot v4

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) BF16 weights as lossless `.ingot` files.
Ingot is a zip for model weights: you decode the files back to the original safetensors before loading.
There is no GPU decode yet, so this is a download-size saving, not a runtime format.

| | Original | Ingot v4 |
|---|---|---|
| Size | 55.6 GB | 36.2 GB (36,221,798,920 bytes) |
| Bits per weight | 16 | 10.4305 |

Every shard decodes byte for byte to the original: the decoded sha256 equals the one Hugging Face
lists for `Qwen/Qwen3.8-27B` (revision `37561816`). `SHA256SUMS` has the sha256 of every `.ingot`
file and of every original shard.

## Decode

Needs a C compiler and `make`. The tool is `ingot/` in [Foundry](https://github.com/dcherrera/Foundry).

```
git clone https://github.com/dcherrera/Foundry && cd Foundry
make -C ingot build/ingot
```

Later shards use tensors from earlier shards as match sources, so decode in order and pass the
shards already decoded as `--refs`:

```
INGOT=Foundry/ingot/build/ingot
refs=()
for n in $(seq -w 1 18); do
  f=model-000$n-of-00018.safetensors
  $INGOT decode "${refs[@]}" $f.ingot $f
  refs+=(--refs $f)
done
sha256sum -c --ignore-missing SHA256SUMS   # checks the .ingot files and the decoded shards
```

`ingot verify [--refs ...] FILE.ingot [ORIGINAL]` checks every tensor hash and the file's sha256
without writing anything. Decoding is CPU-only and slow on small machines (`-j N` sets threads).

Copy the other files (config, tokenizer, `model.safetensors.index.json`) from
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); they are not changed.

## Licence

Qwen3.8-27B's licence applies unchanged: Apache License 2.0, copied here as `LICENSE`. These files
are an exact encoding of Qwen's weights; they add no new rights or restrictions.
