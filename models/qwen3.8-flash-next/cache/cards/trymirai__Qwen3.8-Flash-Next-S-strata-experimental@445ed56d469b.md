---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - strata
  - mirai
  - 2-bit
  - experimental
---

# Qwen3.8-Flash-Next-S in Strata (experimental)

This is our 2-bit trellis build of Qwen3.8-Flash-Next (Qwen Next S, v11p), packed for
[Strata](https://github.com/Niko1221/Strata). It runs on one consumer NVIDIA card with the experts in system RAM,
like the GSQ-RCO GGUFs. The busiest experts are cached on the GPU, and the CPU computes the rest (AVX2).

- **Quality:** ROMA KL 0.040 (top-1 93.4%), against 0.303 for GSQ-RCO IQ2_XS. Both were measured outside Strata, on
  the same 37,620 positions against the bf16 model.
  - Inside Strata our build reproduces our vLLM package. Teacher-forced on the package's own answers, top-1 agreement
    is 99.28% over 1,536 positions with the Q8_0 backbone, and 99.74% on the two prompts rechecked with the trellis
    backbone and head. The package's own prefill-vs-decode noise floor is 99.48%. Greedy answers agree token for token
    for long stretches (all 192 tokens on 3 of 8 prompts).
- **Size:** 36.3 GB of experts in RAM, against IQ2_XS's 35.5 GB.
- **Backbone and head:** these stay the package's own k5 trellis on the GPU (`--qns-dense-gguf`), which frees 1.45 GiB
  of VRAM for experts compared with Q8_0.

## What you need

| | minimum | recommended |
|---|---|---|
| GPU | NVIDIA RTX 30/40/50 with 12 GB | 24 GB (RTX 3090 / 4090) |
| RAM | 64 GB | 96-128 GB |
| CPU | x86-64 with AVX2 | more cores help on 12 GB cards |
| Disk | 42 GB for the model, plus the n-gram table: 28.8 GB (ISTA's IQ4_NL shard 2) or 51.2 GB (Qwen's FP8 table) | NVMe |
| Software | Linux, NVIDIA driver 580+, CUDA toolkit 12.8+ (13.x for RTX 50), git, python3 | |

On a 12 GB card most expert reads miss the GPU, so decode there is bound by the CPU.

## Files

| file | what |
|---|---|
| `Qwen3.8-Flash-Next-S-qns.gguf` (42.3 GB) | the model: each layer's 512 experts as trellis blobs, plus backbone, head and embedding in Q8_0 |
| `Qwen3.8-Flash-Next-S-qns-dense.gguf` (2.1 GB) | the backbone (DeltaNet and attention projections) and the head as the package's own k5 trellis. The engine serves those from here instead of the Q8_0 copies. Leave out `--qns-dense-gguf` to fall back to Q8_0 |
| `ple-fp8.gguf` (51.2 GB, optional) | Qwen's FP8 n-gram table, byte for byte, as Strata's `tools/ple_fp8_pack.py` writes it. Alternatively use ISTA's `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` (28.8 GB) if you already have it |
| `strata-qns.patch` | the Strata changes, against v0.1.43 (`7695859a`) |
| `build_strata_qns.sh` | clones Strata at that commit, applies the patch and builds the engine for your GPU |
| `converter/` | the converters that made these GGUFs from our uzu packages, for any Qwen Next S build (see below) |

## Changes

- **2026-10-15:** multi-GPU fix. The expert rotations live in each card's constant memory and were only uploaded to the
  first GPU, so `--layer-split` gave repeated garbage tokens on the second card. Every stage now uploads its own copy.
  Thanks to Alexey (alesha-pro) for finding it: with the fix, two 3090s decode at 104-113 tok/s.

## Converting another Qwen Next S package

`converter/` holds the scripts that made the two GGUFs, for example to convert the 32 GB build
([Qwen3.8-Flash-Next-S-small-experimental](https://huggingface.co/trymirai/Qwen3.8-Flash-Next-S-small-experimental),
its `int4-table/` folder):

```bash
# needs a built Strata (build_strata_qns.sh) for its pinned llama.cpp, plus numpy, torch and safetensors
export STRATA_LLAMACPP=$PWD/Strata-qns/build/_deps/strata_llamacpp-src
python converter/qns_convert.py /models/qns-small/int4-table /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf     Qwen3.8-Flash-Next-S-small-qns.gguf
python converter/qns_dense_convert.py /models/qns-small/int4-table Qwen3.8-Flash-Next-S-small-qns-dense.gguf
```

The template GGUF (ISTA's GSQ-RCO IQ2_XS, first shard) supplies llama.cpp's tensor layout, metadata and tokenizer.
Pass a layer list as a fourth argument to `qns_convert.py` (e.g. `0,1`) for a quick partial check.

## Quickstart (24 GB card, 64-128 GB RAM)

```bash
# 1. the engine (10-20 minutes): Strata 0.1.43 + the qns patch, compiled for this GPU
./build_strata_qns.sh Strata-qns
cd Strata-qns

# 2. the pack (seconds): the engine's index of the GGUF
STRATA_GGUF_PY=build/_deps/strata_llamacpp-src/gguf-py \
  .venv/bin/python tools/iq_pack.py --gguf /models/Qwen3.8-Flash-Next-S-qns.gguf --out packs/qns

# 3. the MTP draft layer, from Qwen's checkpoint (as Strata's setup makes it; skip if you have Strata-data/mtp/rt)
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

4. Save this as `strata-qns.json` in `Strata-qns/`, with your paths:

```json
{
 "exe": "build/strata",
 "args": [
  "--pack", "packs/qns",
  "--native", "/models/Qwen3.8-Flash-Next-S-qns.gguf",
  "--native-dense-gguf", "/models/Qwen3.8-Flash-Next-S-qns.gguf",
  "--qns-dense-gguf", "/models/Qwen3.8-Flash-Next-S-qns-dense.gguf",
  "--ple-gguf", "/models/ple-fp8.gguf",
  "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
  "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "mtp/rt",
  "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768"
 ],
 "cwd": ".",
 "tokenizer": "packs/qns/tokenizer",
 "model_name": "qwen3.8-flash-next-s",
 "log": "strata-qns.log",
 "host": "127.0.0.1",
 "port": 8080
}
```

5. Start it:

```bash
STRATA_NO_ARENA_THP=1 .venv/bin/python -m serve.server --engine strata --config strata-qns.json --port 8080
```

The web app is at `http://127.0.0.1:8080`, and the OpenAI-compatible API at `http://127.0.0.1:8080/v1`.

Notes:
- **ISTA's IQ4_NL table:** point `--ple-gguf` at `...-IQ2_XS-00002-of-00002.gguf`. Keep `--native-dense-gguf` as
  above: without it the engine treats that file as a second shard of our model and refuses it.
- **`STRATA_NO_ARENA_THP=1`:** this works around Strata issue #771. On a machine with long uptime, transparent huge
  page compaction can stretch the first load from about a minute to many minutes. On a freshly booted desktop you
  can leave it out.
- **Context:** 128K as above. Up to 262K works with the same KV streaming, at ~13.7 KB of RAM per context token.
- **12 GB cards:** use `--max-context 65536`. The expert cache sizes itself to the free VRAM.

## Measured on an RTX 3090 + EPYC 7H12 (Zen 2, 48 cores used)

Strata 0.1.43 with setup.sh's settings for IQ2_XS; our build uses the same flags. Prompts are 5.3K and 34K tokens,
256 tokens out, thinking off, greedy. Results for two runs each are separated by a slash.

| | decode, 5.3K (tok/s) | decode, 34K (tok/s) | prompt (tok/s) | decode hit rate | experts in VRAM |
|---|---|---|---|---|---|
| ours, 24 GB, MTP on | 72.5 / 84.5 | 72.5 / 73.7 | 1,388-1,731 | 97-99% | 11,035 |
| IQ2_XS, 24 GB, MTP on | 85.7 / 94.5 | 83.8 / 90.8 | 1,912-2,420 | 98-99% | 13,075 |
| ours, 24 GB, MTP off | 45.4 / 49.5 | 47.6 / 47.9 | 1,386-1,736 | 97-99% | 11,722 |
| IQ2_XS, 24 GB, MTP off | 55.2 / 61.0 | 56.1 / 60.3 | 1,924-2,433 | | 13,075 |
| ours, 12 GB, MTP on, 16 cores | 77.6 / 90.8 | 69.6 / 75.4 | 1,333-1,719 | 77-82% | 3,530 |
| IQ2_XS, 12 GB, MTP on, 16 cores | 78.3 / 90.6 | 83.2 | 1,894-2,320 | 84-89% | 4,529 |
| ours, 12 GB, MTP on, 48 cores | 52.1 / 56.1 | 51.9 / 48.1 | 1,362-1,705 | 80-84% | 3,530 |
| IQ2_XS, 12 GB, MTP on, 48 cores | 61.1 / 67.9 | 60.9 / 61.3 | 1,924-2,321 | 84-88% | 4,529 |

- **Context:** the model was also run at 262K on the 24 GB card, with Q8_0 backbone and head. A 215,426-token prompt
  read at 1,728 tok/s (125 s), then decoded at 54.4 tok/s and answered a question from deep inside the document. Peak
  VRAM was 23.8 GB and peak RAM 40.5 GiB.
- **Memory at 128K, 24 GB card:** peak VRAM 23.8 GB, peak engine RAM 38.5 GiB.
- **Measurement caveat:** the test host is shared, so the CPU-side numbers are noisy. 16 cores ran faster than 48
  there.

## How it works

- **Experts:** each one is stored as trellis row tapes in 32-row groups: the vLLM plugin's GPU layout, with a
  64-byte header and the row scales, rounded up to 256 bytes. Each expert has its own bit width (1.5 to 5 bits per
  weight), so blobs differ in size.
  - Strata's RAM arena keeps them back to back (compact).
  - Every VRAM slot is sized to its own expert. The adaptive tier moves an expert only into a slot that holds it.
- **Rotation:** the weights live in a rotated basis, `kron(H, Q)` with signs. Each path's activation quantizer
  rotates its input once per token (2560 wide) and once per expert pair (640 wide), then quantizes it to two int8
  planes. The integer dot products are exact.
- **Kernels:** the GPU kernels are our vLLM plugin's trellis decode, ported to Strata's grouped-expert interface.
  The dot products run on the int8 tensor cores; the integer sums are exact, so a token's result does not depend on
  how many tokens share an expert. The CPU kernels are AVX2 (0.5 cycles per weight on Zen 2).
- **Backbone:** with `--qns-dense-gguf`, the DeltaNet and attention projections and the head stay the package's k5
  trellis, bit for bit. That includes the MTP drafter's draft-vocabulary rows, regrouped into 32-row sets. The shared
  experts and the token embedding are Q8_0; the embedding stays in mapped host memory, as for IQ2_XS.
