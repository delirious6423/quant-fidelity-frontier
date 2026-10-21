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

Qwen3.8-Flash-Next packed for [Ijah](https://github.com/publicExcess/ijah), a raw-Metal LLM engine for Apple Silicon (M5): the routed experts **and**
the trunk Linears in turboderp's EXL3 trellis format, encoded by our own converter. It is about 4x closer to the bf16
model than oMLX's oQ4e (Jundot's 4.72 bpw pack), at 81 GB of weights.

**Engine: Ijah only.** The trunk Linears are EXL3 and the final hyper-connection mixer is bf16, so Sushi and exllamav3
do not load this pack as is (Sushi users: [SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Sushi](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Sushi)).
MLX / transformers do not load it either.

## What is in it ("T6")

| part | format |
|---|---|
| routed experts (48 layers x 512) | EXL3 K5 (5.0 bpw), MCG codebook, window 15, svh out scales; our converter: whole-model count imatrix (280K tokens), LDLQ on the rotated Hessian |
| trunk Linears: GDN in/out projections, attention q/k/v/o + indexer, shared experts, hyper-connection down-mixes | EXL3 K6; our converter: full Hessians captured layer by layer (279.5K tokens), LDLQ |
| hyper-connection up-mixes, PLE key/value projections | EXL3 K8 |
| final hyper-connection mixer | bf16, from the base |
| MTP head and its experts (K4), embeddings, lm_head, routers, norms, vision tower, tokenizer, chat template | from [beamster/Qwen3.8-Flash-Next-Sushi-4bpw](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-4bpw) (`f25c0c4`), 8-bit affine / bf16 as published there, tensors re-laid out on 16-byte offsets |
| n-gram embedding table (`ngram_table.bin`, 102.4 GB, bf16) | the base model's own, extracted to a flat file by Sushi; **not in this repo**: [SiliconSpecies/Qwen3.8-Flash-Next-ngram-table](https://huggingface.co/SiliconSpecies/Qwen3.8-Flash-Next-ngram-table) (below) |

## Closeness to bf16 (whole model, KLD vs the bf16 teacher; lower is closer)

| pack | A wikitext | B reasoning | C 32K | top-1 A / B / C |
|---|---|---|---|---|
| **this pack (Ijah T6)** | 0.0332 | 0.0092 | 0.0148 | 93.3 / 97.5 / 96.9% |
| oMLX, Jundot oQ4e (4.72 bpw mixed) | 0.128 | 0.036 | 0.078 | 86.5 / 93.8 / 93.0% |

Same teacher and text for both rows. A: wikitext-2, 16 windows of 512 tokens; B: 16 real reasoning traces (2,048
tokens each); C: 4 windows of 32,768 tokens (2 code, 2 prose). oQ4e is the base model only (there is no Swift oQ4e).

## Speed (M5 Max, 128 GB, one streaming client for both engines)

| engine + weights | decode tok/s | cold prefill 10K tok/s | cold prefill 32K tok/s |
|---|---|---|---|
| Ijah, this recipe (T6) | 98.9 ± 10.2 | ~2,120 | ~2,050 |
| oMLX, Jundot oQ4e | 105.2 ± 9.2 | 2,412 | 2,812 |

Ijah decodes ~6% slower than oMLX's oQ4e (paired ratio 0.94, 95% CI [0.92, 0.96]) and prefills ~12% (10K) to ~27%
(32K) slower; it is about 4x closer to bf16 on sets A and C.

## How we measured

- **Machine:** one Apple M5 Max (40-core GPU), 128 GB, macOS 27.0.1; one engine loaded at a time; 18 Oct 2026.
- **Compared against:** oMLX 0.7.1.dev1 with Jundot's Qwen3.8-Flash-Next-oQ4e (mixed 4-8-bit, 4.72 bpw), tuned
  settings (MTP depth 2, INT8-activation prefill); Ijah with this T6 pack, MTP depth 2. Each engine runs its own
  weight format, so the numbers compare **engine + weights**, not the engines alone.
- **Decode:** one streaming OpenAI-compatible client, 6 prompts (code, maths, prose) x 3 seeds, 1,024 tokens each,
  temperature 1.0 / top-p 0.95 / top-k 20, thinking at medium effort; mean ± sd over the 18 runs (95% CI of the mean
  ±4.7 / ±4.2), compared pair by pair on the same prompt and seed.
- **Prefill:** cold 10K / 32K-token prompts, no cache hit, 5 runs each, time to the first streamed token over HTTP;
  medians.
- **Quality:** KL divergence from the bf16 model's next-token distribution, full vocabulary, teacher-forced (Ijah: f32
  rounding, bf16 KV, QSA attention); 95% bootstrap intervals over windows.

## Limitations

- One machine: other Macs will differ; run-to-run drift is ~3-8%, differences under ~1% are not resolvable.
- Engine + weights, not engine alone: Ijah's quality lead comes largely from its EXL3 weights, oMLX's speed lead partly
  from its cheaper-to-unpack 4-bit weights; oMLX ran with settings we chose.
- Synthetic single-turn benchmarks: real agent sessions mix prefill and decode differently.
- Set C has only 4 windows, so its intervals are wide.
- KL divergence measures closeness to the bf16 model, not task accuracy.
- This pack runs only on the Ijah engine, on an Apple M5 with ~90 GB of unified memory. Research code, no support
  promise.

## Requirements

- An Apple **M5**-family Mac (Ijah's trunk and prefill kernels use the M5 GPU's tensor units; measured on an M5 Max).
- **~90 GB of unified memory**: 81 GB of weights resident, ~79 GB used at a 32K window and ~86 GB at 262K; a 128 GB
  Mac. The n-gram table is read from the SSD and never loaded into GPU memory.
- Disk: 82.5 GB for this repo + 102.4 GB for the n-gram table.

## Download

```sh
hf download SiliconSpecies/Qwen3.8-Flash-Next-EXL3-Ijah --local-dir ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah
# the n-gram table both Ijah packs need (102.4 GB, sha256 387f988eef8d8d16...)
hf download SiliconSpecies/Qwen3.8-Flash-Next-ngram-table ngram_table.bin --local-dir ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah
# instead of the second download, when the other Ijah pack is installed: a hard link (no second 102 GB copy)
# ln ~/.cache/ijah/packs/Swift-1.5-Flash-Next-EXL3-Ijah/ngram_table.bin ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah/
```

## Run

```sh
git clone https://github.com/publicExcess/ijah && cd ijah && make
build/engine-cli pack ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah                              # opens the pack, lists its tensors
tools/serve.sh start -- --pack ~/.cache/ijah/packs/Qwen3.8-Flash-Next-EXL3-Ijah --model-id Qwen3.8-Flash-Next-native
```

OpenAI- and Anthropic-compatible server on 127.0.0.1:8140; context 262,144 tokens, MTP on, reasoning effort xhigh
when a request names none. See the Ijah repo's `docs/server.md`.

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
