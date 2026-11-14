---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- gguf
- strata
- apple-silicon
- metal
- macos
- moe
- mtp
- speculative-decoding
- ssd-streaming
---

# Qwen3.8-Flash-Next IQ3_S on a 48 GB Mac (Strata Metal, tuned)

This is a tested setup for serving **Qwen3.8-Flash-Next** (ISTA-DASLab's GSQ-RCO IQ3_S quantization) on an Apple
Silicon Mac with **48 GB** of memory. It uses a tuned fork of [Strata](https://github.com/Niko1221/Strata)'s Metal engine.
The 54.8 GB model does not fit in memory, so the engine keeps the most-used routed experts in RAM and streams the rest
from the internal SSD, with native multi-token prediction (MTP) drafting four tokens per step.

On a MacBook Pro M5 Pro (48 GB), **decode went from 20.1 to 31.4 tok/s** in our benchmark suite with identical output
tokens, and the suite's prompt time from 55.5 to 44.5 s. See [results/RESULTS.md](results/RESULTS.md).

The weights are **not** re-uploaded here: `scripts/fetch-upstream.sh` downloads ISTA-DASLab's unmodified files at a
pinned revision and checks their sha256.

## What is in this repository

| Path | What | Size |
| --- | --- | --- |
| `mtp/mtp-Qwen3.8-Flash-Next-native-Q2_0.gguf` | The native MTP draft layer from Qwen/Qwen3.8-Flash-Next, packed for the Metal engine (Q2_0 experts). Only the drafter is quantized this far; the target model is ISTA-DASLab's IQ3_S, unchanged | 0.89 GB |
| `pack/iq3_s/` | Strata's pack for the IQ3_S GGUF: dense weights, tensor index, tokenizer, chat template | 1.5 GB |
| `scripts/` | download and verify upstream files, build the engine, write the expert sidecar, write the server config | |
| `config/` | `strata-iq3_s.example.json` (the served config with placeholder paths); a router service entry | |
| `results/RESULTS.md` | the measurements, what was tried and what was rejected | |
| `SHA256SUMS` | checksums of every file here | |

## Setup on a Mac

You need:

- an Apple Silicon Mac. Tested: M5 Pro, 48 GB. Less memory means a smaller expert arena and slower decoding.
- about **140 GB free on the internal SSD**: 84 GB of model, a 50 GB expert sidecar, 2.4 GB from this repo and 0.9 GB of
  image encoder weights.
- Xcode's Command Line Tools (`xcode-select --install`), `cmake` and `ninja` (`brew install cmake ninja`).
- the `hf` CLI (`uv tool install huggingface_hub` or `pip install huggingface_hub`).
- for the server and images: the [Strata](https://github.com/Niko1221/Strata) app or Terminal install, whose Python
  server and `strata-vision` image encoder this setup was tested with (Strata Terminal 0.1.54).

```sh
BASE=~/Strata-m5pro
hf download baytizzel/Qwen3.8-Flash-Next-IQ3_S-Strata-M5Pro-48gb --local-dir $BASE/recipe
cd $BASE/recipe
bash scripts/fetch-upstream.sh $BASE/models            # 84.5 GB from ISTA-DASLab, pinned and sha256-checked
bash scripts/build-engine.sh $BASE/engine              # the tuned engine fork, pinned commit (about a minute)
bash scripts/make-experts.sh $BASE/engine $BASE/models # writes the 50 GB expert sidecar once, verified against the GGUF
python3 scripts/render-config.py --engine $BASE/engine --models $BASE/models --out $BASE/strata-iq3_s.json
```

`render-config.py` prints the command that starts the server (OpenAI-compatible API on `127.0.0.1:8091`, model loaded
on the first request). The API key is in the written config:

```sh
curl http://127.0.0.1:8091/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-iq3_s","messages":[{"role":"user","content":"Hello!"}]}'
```

Without a Strata app or Terminal install, the fork's own `serve/` package works too (`python3 -m venv .venv &&
.venv/bin/pip install -r requirements.txt` in the engine folder), but that combination was not part of the measured
setup. Images then need `--vision-exe` pointing to a `strata-vision` build.

## The engine changes

The fork is [betizzel/Strata](https://github.com/betizzel/Strata), branch `m5pro-iq3s` (MIT). Each change is an opt-in
switch or a fix; the served config turns them on.

- **`STRATA_METAL_DIRECT=1`.** Unaligned `F_NOCACHE` reads still left about 35% of every expert in the macOS file cache.
  The cache grew to ~18 GB and macOS compressed the expert arena instead. Page-aligned bounce reads cache nothing.
- **Freed host weight copies.** The host code's 6.2 GB of CUDA-shaped weight copies were "given back" with
  `setPurgeableState(Empty)`. That frees nothing for a buffer that has been in a Metal residency set. They are now
  released: footprint before the arena 13.3 → 7.1 GB.
- **`STRATA_METAL_WIRE_ARENA=1`, now step by step during the fill.** Unwired, macOS compressed a just-filled arena: the
  experts are incompressible, so this freed nothing, and the first request waited ~10 s. Wired as it fills, the memory
  comes from the file cache and from other programs' compressible pages.
- **`STRATA_METAL_REGION_FIT=1`, `STRATA_METAL_PROMPT_SMALL=256`.** Prompt buffers live inside the arena instead of
  2.4-4.8 GB of their own. Short prompts (tool results) use a small buffer set and leave the cached experts alone.
- **`STRATA_METAL_KEEP_WARM=1`.** Tiny GPU ticks keep the GPU clocked while a decode window waits for the SSD. GPU
  segment time went from 73 to 62 ms per window.

With these, the arena can be 16-17.5 GB instead of 6 GB. The arena hit rate goes from 0.57 to 0.84, and a 4-token decode
window drops from ~140 ms to ~90 ms.

## Memory and behavior

- **While loaded, the engine holds about 22-25 GB.** The expert arena is wired, so macOS compresses other programs
  rather than the experts.
- **The arena is sized at each load** from the memory free then (`ARENA_MAX_GB=17.5`, `KEEP_FREE_GB=8`). A busier Mac
  gets a smaller arena and slower decoding. A 19 GB arena with 5.5 GB left reached warning pressure in testing.
- **Output matches stock settings token for token.** Greedy and seeded sampling both matched with `--spec 4`, the 4-token
  verify window. Changing the window size changes the batched verification's arithmetic and therefore long greedy outputs.
- **The numbers come from one Mac.** They are a short suite on a desktop in use, not a long-context or vision benchmark.
  Decode is slower on battery when macOS uses Low Power Mode.

## Optional: runtime refusal-direction projection

The tested service also projected a refusal direction out of the residual stream at run time (layers 4-44, strength 1),
so the model refuses less. No weights are changed.

- The direction is alesha-pro's, derived from huihui-ai's abliterated model. It is not included here.
- `bash scripts/fetch-upstream.sh $BASE/models --with-refusal-vector` downloads it from
  [alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF),
  pinned and sha256-checked.
- `render-config.py` then adds the projection. `--no-projection` serves the stock model.
- You are responsible for how you use a model with reduced refusals.

## Upstream files and checksums

| File | Source | sha256 |
| --- | --- | --- |
| `IQ3_S/…-00001-of-00002.gguf` (54.8 GB) | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF @ `ed59f920` | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `IQ3_S/…-00002-of-00002.gguf` (28.8 GB) | same | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (0.9 GB) | same | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |
| `…-00001-of-00002.gguf.experts` (50.3 GB, made locally) | `scripts/make-experts.sh` | `4eb63134d38499d8364989ac335bcfcd8fe7097989e13e78cc076bf595eda0e6` |
| `Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf` (optional) | alesha-pro/… @ `d4af3be5` | `59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38` |

## License and credits

- **Model**: Qwen3.8-Flash-Next by Qwen, under the [Qwen Community License 1.0](LICENSE). The files here are
  derivatives and carry that license and notice.
- **Quantization**: GSQ-RCO IQ3_S by ISTA-DASLab.
- **Engine**: Strata by Niko1221 and contributors (MIT).
- **Refusal direction**: alesha-pro and huihui-ai.

Details: [NOTICE.md](NOTICE.md).
