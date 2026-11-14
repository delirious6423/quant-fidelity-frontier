---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- strata
- apple-silicon
- metal
- macos
- moe
- code
- mtp
- speculative-decoding
- ssd-streaming
---

# Qwen3.8-Flash-Next Coder IQ1_M on a 48 GB Mac (Strata Metal, tuned)

This is a tested setup for serving **Qwen3.8-Flash-Next Coder** (ISTA-DASLab's expert-pruned GSQ-RCO IQ1_M) on an
Apple Silicon Mac with **48 GB** of memory, with windows up to the model's full **262,144 tokens**.

- **Engine.** A tuned fork of [Strata](https://github.com/Niko1221/Strata)'s Metal engine.
- **Streaming.** The engine keeps the most-used routed experts in a wired RAM arena and reads the rest straight from the
  GGUF on the internal SSD.
- **Drafting.** The base model's native multi-token prediction (MTP) layer drafts up to four tokens per step.

On a MacBook Pro M5 Pro (48 GB):

- **45.8 tok/s** mean decode in our benchmark suite, and **54.6 tok/s** on code.
- About 29 GB in use while loaded, at normal memory pressure.
- The same model on llama.cpp ran at 44.9 tok/s on that code request. With the whole model resident, it pushed this
  Mac to critical memory pressure.
- A 244K-token prompt is read at about 670 tok/s. Details: [results/RESULTS.md](results/RESULTS.md).

The weights are **not** re-uploaded here: `scripts/fetch-upstream.sh` downloads ISTA-DASLab's unmodified files at a
pinned revision and checks their sha256.

## What is in this repository

| Path | What | Size |
| --- | --- | --- |
| `mtp/mtp-Qwen3.8-Flash-Next-native-Q2_0.gguf` | The native MTP draft layer from Qwen/Qwen3.8-Flash-Next, packed for the Metal engine (Q2_0 experts). Only the drafter is quantized this far; the target model is ISTA-DASLab's Coder IQ1_M, unchanged | 0.89 GB |
| `pack/coder-iq1_m/` | Strata's pack for the Coder GGUF: dense weights, tensor index, tokenizer, chat template | 1.5 GB |
| `scripts/` | download and verify upstream files, build the engine, write the server config | |
| `config/` | `strata-coder-iq1_m.example.json` (the served config with placeholder paths); router service entries for a 64K and a 128K window | |
| `results/RESULTS.md` | the measurements, what was tried and what was not | |
| `SHA256SUMS` | checksums of the weight files here (`mtp/`, `pack/`) | |

## Setup on a Mac

You need:

- an Apple Silicon Mac. Tested: M5 Pro, 48 GB. Less memory means a smaller expert arena and slower decoding.
- about **61 GB free on the internal SSD**: 58.4 GB of model and 2.4 GB from this repo. There is no expert sidecar to
  write. The n-gram shard is byte-identical to the IQ3_S model's. If you have the
  [IQ3_S recipe](https://huggingface.co/baytizzel/Qwen3.8-Flash-Next-IQ3_S-Strata-M5Pro-48gb) installed, hard-link that
  shard into place first and it is not downloaded again.
- Xcode's Command Line Tools (`xcode-select --install`), `cmake` and `ninja` (`brew install cmake ninja`).
- the `hf` CLI (`uv tool install huggingface_hub` or `pip install huggingface_hub`).
- for the server: the [Strata](https://github.com/Niko1221/Strata) app or Terminal install, whose Python server this setup
  was benchmarked with (Strata Terminal 0.1.54).

```sh
BASE=~/Strata-m5pro
hf download baytizzel/Qwen3.8-Flash-Next-Coder-IQ1_M-Strata-M5Pro-48gb --local-dir $BASE/coder-recipe
cd $BASE/coder-recipe
bash scripts/fetch-upstream.sh $BASE/models            # 58.4 GB from ISTA-DASLab, pinned and sha256-checked
bash scripts/build-engine.sh $BASE/engine-coder        # the tuned engine fork, pinned commit (about a minute)
python3 scripts/render-config.py --engine $BASE/engine-coder --models $BASE/models --out $BASE/strata-coder.json
```

`render-config.py` prints the command that starts the server (OpenAI-compatible API on `127.0.0.1:8092`, model loaded
on the first request). `--max-context 131072` or `262144` gives a longer window. The API key is in the written config:

```sh
curl http://127.0.0.1:8092/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-coder-iq1_m","messages":[{"role":"user","content":"Write a binary search in Rust."}]}'
```

Without a Strata app or Terminal install, the fork's own `serve/` package works too (`python3 -m venv .venv &&
.venv/bin/pip install -r requirements.txt` in the engine folder), but that combination was not part of the measured
setup.

## The engine

The fork is [betizzel/Strata](https://github.com/betizzel/Strata), branch `m5pro-coder` (MIT). It is the IQ3_S branch
(`m5pro-iq3s`) plus one commit, upstream's 256-expert Metal support (`91411e17`, cherry-picked). This Coder keeps 256
of the base model's 512 routed experts a layer. Upstream had not run that code on such a model end to end; this setup
did, at 64K, 128K and 262K.

From the IQ3_S work, each an opt-in switch or a fix that the served config turns on:

- direct (uncached) expert reads;
- freed host weight copies (about 6.2 GB);
- an arena wired as it fills;
- prompt buffers inside the arena;
- a GPU keep-warm while a decode window waits for the SSD.

The [IQ3_S card](https://huggingface.co/baytizzel/Qwen3.8-Flash-Next-IQ3_S-Strata-M5Pro-48gb#the-engine-changes)
explains each one. The Coder-specific settings:

- **`STRATA_METAL_ARENA_MAX_GB=20`.** The arena holds about three quarters of the Coder's experts: 9,081 of 12,288.
  Arena hits are 0.97 and the 4-token decode window takes ~62 ms.
- **`STRATA_METAL_EXPERT_PROFILE=<engine>/data/expert-profile-coder.bin`.** The stream ranks its first fill by this
  profile. The default profile is for 512 experts and is rejected for this model.
- **`--spec 4 --spec-min-p 0.5`.** A verify window ends early where the draft layer is unsure. That is +5.6% over plain
  `--spec 4` here, and upstream's Mac setup uses min-p 0.5 for the Coder too.

## Memory and behavior

- **While loaded, the engine holds about 27-30 GB.** The expert arena is wired, so macOS compresses other programs
  rather than the experts.
- **The arena is sized at each load** from the memory free then (`KEEP_FREE_GB=8`). A busier Mac gets a smaller arena
  and slower decoding: 17.9 GB with apps open gave 46.8 tok/s on code, and 20 GB on an idle desktop gave 54.6.
- **The window costs memory at load.** The KV cache takes about 13.7 KB a token: 0.9 GB at 64K, 1.8 GB at 128K,
  3.6 GB at 262K. That comes out of the arena when memory is short.
- **Output is deterministic per setting.** Every run with the same draft settings produced identical tokens, greedy and
  seeded. Changing the verify window changes the batched verification's arithmetic, and with it the odd near-tie in a
  long greedy answer.
- **One request at a time.** Requests queue.
- **The numbers come from one Mac.** They are a short suite plus needle-in-a-haystack checks on a desktop in use, not a
  quality benchmark.

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
| `IQ1_M/…-00001-of-00002.gguf` (29.6 GB) | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF @ `bba88856` | `bfa3644db4adfff072fc946c20994f2909d2cf89879d83f80a4c811d4016c211` |
| `IQ1_M/…-00002-of-00002.gguf` (28.8 GB, the n-gram table) | same | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf` (optional) | alesha-pro/… @ `d4af3be5` | `59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38` |

## License and credits

- **Model**: Qwen3.8-Flash-Next by Qwen, under the [Qwen Community License 1.0](LICENSE). The files here are
  derivatives and carry that license and notice.
- **Quantization and expert pruning**: GSQ-RCO Coder IQ1_M by ISTA-DASLab.
- **Engine**: Strata by Niko1221 and contributors (MIT), including the 256-expert Metal support.
- **Refusal direction**: alesha-pro and huihui-ai.

Details: [NOTICE.md](NOTICE.md).
