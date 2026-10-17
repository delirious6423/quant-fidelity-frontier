---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- gguf
- llama.cpp
- qwen4exp
- ple
---

# Qwen3.8-Flash-Next PLE tables (GGUF)

The per-layer embedding (PLE) table of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) as
standalone GGUF files, in three precisions. Load one next to **any** Qwen3.8-Flash-Next GGUF quant with `--ple`, and the
model uses this table instead of the one it ships.

> [!IMPORTANT]
> **This is not a runnable model.** It only works together with a Qwen3.8-Flash-Next GGUF, and only with a llama.cpp build
> that supports PLE sidecar files (the `--ple` option). Builds without that support, including mainline llama.cpp,
> LM Studio and Ollama today, cannot use these files.

## Why

Qwen3.8-Flash-Next (arch `qwen4exp`) has a hashed n-gram embedding table, `per_layer_token_embd.weight`, with
320,001,536 rows of 160 values (16 heads of ~20 M hashed n-grams, used by layer 1). In BF16 it is 102.4 GB, more than
the rest of a 4-bit model. Most GGUF quants therefore ship it in IQ4_NL (28.8 GB), some in Q8_0 or Q5_x, whatever
precision the rest of the model uses. llama.cpp can keep the table on disk and read only the rows each batch needs, so its size costs disk
space, not RAM. These files let you choose its precision separately from the main quant.

## Files

| File | Type | Size | Row error vs BF16 (relative RMS) | Source |
|---|---|---|---|---|
| `Qwen3.8-Flash-Next-PLE-BF16.gguf` | BF16 | 102.4 GB | 0 (reference) | byte copy of the table in unsloth's BF16 GGUF, a lossless conversion of the original weights |
| `Qwen3.8-Flash-Next-PLE-Q8_0.gguf` | Q8_0 | 54.4 GB | 0.0053 | converted from the BF16 table; identical to the Q8_0 table in AesSedai's quants on 20,000 sampled rows |
| `Qwen3.8-Flash-Next-PLE-IQ4_NL.gguf` | IQ4_NL | 28.8 GB | 0.076 | byte copy of the table in unsloth's UD-Q4_K_XL (the table most quants ship) |

Each file holds one tensor (`per_layer_token_embd.weight`, shape `[160, 320001536]`) plus `general.architecture`,
`general.name` and the ten `qwen4exp.ple.*` / `qwen4exp.embedding_length_per_layer_input` keys. llama.cpp compares those
keys with the model before it uses the table and refuses a mismatch. Checksums are in `SHA256SUMS`.

## Usage

```sh
hf download Delgon/Qwen3.8-Flash-Next-PLE-GGUF Qwen3.8-Flash-Next-PLE-Q8_0.gguf --local-dir .

llama-server \
  -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --ple Qwen3.8-Flash-Next-PLE-Q8_0.gguf \
  --load-mode dio --lazy-mode on-direct
```

- `--lazy-mode on-direct` keeps the table on disk and reads only the rows each batch needs (through the page cache), so
  even the BF16 table does not have to fit in RAM. `--lazy-mode on` maps the file instead and lets the page cache hold the rows it touches.
- Keep the file on a local NVMe drive. Reading the rows over NFS was about 3.5x slower end to end in our test.
- The table embedded in the model file is not read at all when `--ple` is given.
- MTP draft heads do not use the PLE table themselves (they only see it through the target's hidden state), so
  `--spec-type draft-mtp` with a shared MTP head works as usual. A separate draft model never gets the `--ple` file.

## Which one to use

All numbers: Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, 128 GB), main weights `UD-Q4_K_XL`, reference = the same model
with the BF16 table. Quality protocol: `llama-perplexity` KL divergence and top-1 agreement against the reference, 150
chunks of 512 tokens, on a held-out corpus written after the model's training cutoff, and on wikitext-2 as a
memorized-text control.

| Table | Corpus | PPL ratio | Mean KLD | 99 % KLD | Same top-1 |
|---|---|---|---|---|---|
| Q8_0 | held-out | 1.0019 +- 0.0013 | 0.0162 +- 0.0004 | 0.217 | 96.20 +- 0.10 % |
| IQ4_NL | held-out | 1.0022 +- 0.0016 | 0.0255 +- 0.0005 | 0.352 | 95.02 +- 0.11 % |
| Q8_0 | wikitext-2 | 0.9987 +- 0.0010 | 0.0122 +- 0.0002 | 0.150 | 96.44 +- 0.10 % |
| IQ4_NL | wikitext-2 | 1.0044 +- 0.0014 | 0.0239 +- 0.0004 | 0.289 | 94.71 +- 0.11 % |

- **BF16**: the original table. Best output; the largest file.
- **Q8_0**: cuts the KLD of the IQ4_NL table by about a third and closes a quarter to a third of its top-1 gap, at
  half the size of BF16.
- **IQ4_NL**: the table most quants already ship. Useful if you strip the table out of your model files to save disk
  and keep one shared copy.

Perplexity barely moves between the tables; the difference shows in which token wins (top-1 agreement) and in KLD.

Speed and speculative decoding showed no measurable difference between the three tables:

| Table | pp2048 ROCm (t/s) | tg128 ROCm (t/s) | pp2048 Vulkan (t/s) | MTP draft acceptance (24 prompts) |
|---|---|---|---|---|
| IQ4_NL | 1467 - 1536 | 27.0 - 27.4 | 847 - 849 | 66.0 % |
| Q8_0 | 1478 - 1527 | 27.3 | 848 | 64.8 % |
| BF16 | 1462 - 1472 | 27.1 - 27.3 | 843 | 65.2 % |

Ranges are two passes in opposite order (run order moved prefill by about 4 %, more than the table did). The MTP
acceptance differences are within their 95 % confidence intervals.

## Compatibility

All Qwen3.8-Flash-Next GGUF releases checked carry the same `qwen4exp.ple.*` keys (which define the hashing) and a
quantization of the same table: unsloth (UD-IQ1_S to UD-Q4_K_XL), bartowski, AesSedai and ilintar quants. The finetunes
Signal-3.8-Flash-Next and Swift-1.5-Qwen3.8-Flash-Next ship the base table too (their tables match it within
quantization noise), so these files work with them.

llama.cpp checks the keys, not the table contents: a future finetune that retrains the PLE table needs its own file.

## License

Derived from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and distributed under the
**Qwen Community License 1.0**; a copy is in [`LICENSE`](LICENSE). Note its conditions on commercial use (Model as a
Service and AI Work Assistant businesses need a separate license from Qwen).

Credits: the Qwen team for the model; unsloth for the GGUF conversion and the IQ4_NL table; llama.cpp.
