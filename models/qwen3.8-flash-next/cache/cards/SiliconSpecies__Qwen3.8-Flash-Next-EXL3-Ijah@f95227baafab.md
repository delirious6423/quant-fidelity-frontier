---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- exl3
- ijah
- apple-silicon
- metal
- moe
---

# Qwen3.8-Flash-Next EXL3 for Ijah (T6)

Qwen3.8-Flash-Next packed for [Ijah](https://github.com/publicExcess/ijah), a raw-Metal inference engine for Apple M5 Macs: EXL3 trellis weights,
5-bit routed experts and a 6/8-bit trunk, encoded by our own converter. **This T6 pack is the preferred Ijah pack.** Its faster 4-bit-expert sibling,
[SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah-4bit](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah-4bit), has about 2x the KLD.

**Engine: Ijah only** (v0.2 or later). Sushi, exllamav3, MLX and transformers do not load this pack as is (Sushi users:
[SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Sushi](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Sushi)).

## Install and run

```sh
brew install publicExcess/ijah/ijah        # or from source: git clone https://github.com/publicExcess/ijah && cd ijah && make venv && make
ijah download base drafter ngram           # this pack, the DFlash2 drafter, the n-gram table (~/.cache/ijah)
ijah serve start
```

Inco's DFlash2 drafter ([incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2))
is used by default when it is downloaded (about +7-15% decode at short context); without it Ijah drafts with the
model's own MTP head. By hand: `hf download SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah --local-dir ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah` and
`hf download SiliconSpecies/Qwen3.8-Flash-Next-ngram-table ngram_table.bin --local-dir ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah` (102.4 GB, the table is not in this repo).

## What is in it ("T6")

| part | format |
|---|---|
| routed experts (48 layers x 512) | EXL3 K5 (5.0 bpw), MCG codebook; whole-model imatrix, LDLQ |
| trunk Linears (GDN, attention + indexer, shared experts, HC down-mixes) | EXL3 K6, full Hessians, LDLQ |
| HC up-mixes, PLE key/value projections | EXL3 K8 |
| final hyper-connection mixer | bf16, from the base |
| MTP head, embeddings, lm_head, routers, norms, vision tower, tokenizer | from [beamster/Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) (`f25c0c4`) |
| n-gram table (`ngram_table.bin`, 102.4 GB, bf16) | **not in this repo**: [SiliconSpecies/Qwen3.8-Flash-Next-ngram-table](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-ngram-table) |

## Closeness to bf16

Teacher-forced KL divergence from the bf16 model, full vocabulary (lower is closer). B: reasoning traces, C: 32K-token
code and prose, A: wikitext-2.

| engine + weights | KLD B | KLD C | KLD A | top-1 B |
|---|---|---|---|---|
| **Ijah T6 (this pack)** | 0.0092 | 0.0148 | 0.0332 | 97.5% |
| Ijah 4-bit (EXL3, 4-bit experts) | 0.0185 | 0.0236 | 0.0513 | 96.6% |
| UD-IQ4_XS GGUF (Unsloth) | 0.0361 | 0.0539 | 0.1032 | 94.7% |
| oMLX oQ4e (Jundot, 4.72 bpw) | 0.0360 | 0.0780 | 0.1277 | 93.8% |
| Q4_0 GGUF (bartowski) | 0.1148 | 0.1803 | 0.2614 | 89.0% |

**Speed:** Ijah T6 + DFlash2 decodes at **122 tok/s** on an M5 Max (128 GB), against 128-132 for Splash 1.4 (GGUF) and
92 for oMLX (oQ4e), on one client. Method, prefill numbers and limitations: [Ijah README](https://github.com/publicExcess/ijah#how-we-measured).

## Requirements

An Apple **M5** Mac on macOS 27, **~95 GB of free unified memory** for one model at the full 262K context (a 128 GB
Mac), and 82.5 GB + 102.4 GB (n-gram table) of disk.

## Provenance and licences

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`de4b8e4`), Qwen Community License 1.0 (`LICENSE`).
- n-gram table: the base model's, extracted to a flat file by Sushi (as in beamster/Qwen3.8-Flash-Next-Sushi-4bpw), hosted at SiliconSpecies/Qwen3.8-Flash-Next-ngram-table.
- MTP head and experts, embeddings, head, routers, norms, vision tower, tokenizer and chat template:
  [beamster/Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) (`f25c0c4`, Qwen Community License 1.0), with thanks.
- EXL3 trellis format: [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3) (MIT); MCG codebook and bit
  layout as in [Sushi](https://github.com/beamivalice/sushi) (MIT). Engine: [Ijah](https://github.com/publicExcess/ijah).
- Modified files: all `model-exl3-L*` expert shards (re-encoded), all `model-t6-*` files (trunk re-encoded, bf16
  mixer, the remaining trunk tensors), the Sushi-4bpw shards re-laid out for alignment, `config.json`,
  `model.safetensors.index.json`.
