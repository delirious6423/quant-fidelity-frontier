---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags:
  - gguf
  - trellis
  - mirai
  - strata
  - abliterated
  - control-vector
---

# Qwen3.8-Flash-Next-S-abliterated-Strata-GGUF

Mirai's 2-bit trellis build of Qwen3.8-Flash-Next, packed for Strata, with a refusal vector and everything else it
needs to run. It runs on one or two RTX 3090s with the experts in system RAM.

## Credits first

| Part | Authors | Links |
|---|---|---|
| The quantization (Qwen Next S, trellis), its Strata format, kernels and converter | [Mirai](https://huggingface.co/trymirai) | [Qwen3.8-Flash-Next-S-strata-experimental](https://huggingface.co/trymirai/Qwen3.8-Flash-Next-S-strata-experimental) |
| Base model | Qwen team | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Refusal direction, taken from their weights | huihui-ai | [Huihui-Qwen3.8-Flash-Next-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated) |
| Engine | Niko1221 | [Strata](https://github.com/Niko1221/Strata) |
| Picture encoder (mmproj) | ISTA-DASLab | [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |

`S/` holds Mirai's files byte for byte (the SHA256 match their repo). My part: the S-small GGUF (converted with
Mirai's converter from their [S-small](https://huggingface.co/trymirai/Qwen3.8-Flash-Next-S-small-experimental)
trellis files), the refusal vector, the packaged MTP layer, the port of Mirai's Strata patch to Strata 0.2.0
([alesha-pro/Strata](https://github.com/alesha-pro/Strata), upstream as
[Niko1221/Strata#3384](https://github.com/Niko1221/Strata/pull/3384)), and the measurements below.

## The two sizes

| Size | Files | Experts in RAM | Engine process peak RAM | Quality by Mirai (ROMA KL to bf16, top-1) |
|---|---:|---:|---:|---|
| `S` | 42.3 + 2.1 GB | 33.8 GiB | 41 GiB | 0.040, 93.4% |
| `S-small` | 34.4 + 2.1 GB | 26.5 GiB | 33 GiB | 0.069-0.074, 91% |

The routed experts take 1 to 5 bits each, 2.4 on average for S. The second file of each size is the backbone and the
head as trellis, which leaves more VRAM for experts. The quality numbers are Mirai's (37,620 positions against the
bf16 model, measured outside Strata); for comparison they give 0.303 for ISTA's GSQ-RCO IQ2_XS. I did not check them.

## Run it

You need Linux, an NVIDIA card (tested on RTX 3090, 24 GB), the CUDA toolkit 12.8 or newer and about 64 GB of RAM
for S (48 GB for S-small). Upstream Strata cannot read these files yet, and its installers download the upstream
engine, so build the fork from source:

```bash
git clone https://github.com/alesha-pro/Strata && cd Strata
tools/build_qns_linux.sh
```

Download one size, the shared files, and the n-gram table (shard 2 of any ISTA-DASLab GSQ-RCO size, 28.8 GB; this is
the copy in my other repo):

```bash
hf download alesha-pro/Qwen3.8-Flash-Next-S-abliterated-Strata-GGUF \
  --include "S/*" "mtp/*" "mmproj-*" "Huihui-*" --local-dir ../flash-next-s
hf download alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF \
  IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf --local-dir ../ngram
```

Pack it (seconds, once) and write the server config, then start the server:

```bash
.venv/bin/python tools/qns_config.py --model-dir ../flash-next-s --size S --gpu 0 \
  --ngram ../ngram/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
.venv/bin/python -m serve.server --engine strata --config strata-qns-S.json
```

The server is at `http://127.0.0.1:8080`: OpenAI clients use `/v1`, Anthropic clients `/v1/messages`. `--gpu 0,1`
splits the layers over two cards, `--size S-small` takes the smaller build (download `S-small/*` instead of `S/*`),
`--vector off` runs the original model, `--vision off` skips the picture encoder (it takes about 1.3 GB of VRAM),
`--context` sets the window (default 131072, up to 262144).

## The vector

Strata subtracts one direction `r` from the residual stream after each steered layer:

```text
h = h - (h . r) r
```

The weights stay as Mirai published them. The direction comes from Huihui's abliterated checkpoint (strength 1.0) and
is the same file as in my [GSQ-RCO repo](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF),
layers 1-47. It works on this build without changes. A request with `"experimental_speed_projection": false` runs
the original model.

Refusal counts, greedy, thinking off (Strata 0.1.43 with Mirai's patch):

| Set | S, vector | S, stock | S-small, vector | S-small, stock |
|---|---:|---:|---:|---:|
| AdvBench validation, 64 prompts | 1 | 64 | 1 | 63 |
| JailbreakBench held-out, 82 prompts | 1 | 77 | 1 | 77 |
| Alpaca validation (harmless), 64 prompts | 0 | 1 | 0 | 0 |

With thinking on (S, first 24 AdvBench prompts, 2,500 tokens) none refused and 4 did not finish thinking in the
budget. On ISTA's IQ3_S the same vector refuses 0 of 64 and 0 of 82, so 2 of 146 leak here; I did not take a direction
from this build itself.

## Measurements

4x RTX 3090 at 300 W (PCIe 3.0), EPYC 7642 (48 cores, AVX2, two memory channels), 125 GB RAM, Ubuntu 22.04,
CUDA 12.8.

### On this fork

alesha-pro/Strata `flash-next-s`, built by `tools/build_qns_linux.sh`, configs from `tools/qns_config.py`, 262K
context, vector on, pictures on:

- S on one card, S on two cards, S-small on one card: 7 of 7 checks each (an answer with thinking off, three
  reasoning efforts, a tool call and the reply, a picture with text and shapes).
- A 6,210-token document with a code hidden in it: found on every setup.
- S on one card, cold prompt, 512 tokens out, greedy, thinking off, two repeats (tok/s):

| Prompt | Prompt reading | Decode | Drafts accepted |
|---:|---:|---:|---:|
| 4K | 887 / 1,011 | 91.4 / 102.1 | 59% |
| 32K | 1,557 / 1,488 | 98.8 / 101.2 | 58% |
| 131K | 1,186 / 1,378 | 100.2 / 102.7 | 63-68% |
| 261K | 1,424 / 1,298 | 56.6 / 91.1 | 62-70% |

The rig is shared, and another job used the same card during the first 261K repeat. Peak 40.8 GiB of RAM and
23.8 GB of VRAM.

### On Strata 0.1.43 with Mirai's patch

I measured the other setups earlier on a quiet rig, with the same protocol:

| Setup | Prompt reading | Decode | Peak RAM |
|---|---:|---:|---:|
| S, 2 cards, 4K-131K | 830-2,580 | 104-113 | 42.0 GiB |
| S-small, 1 card, 4K-261K | 830-1,590 | 88-98 | 33.4 GiB |
| S-small, 2 cards, 4K-261K | 870-2,810 | 100-118 | 34.6 GiB |

Needles at 27.7K, 110K and 217.7K prompt tokens: 3 of 3 for S on two cards and for S-small on one and two cards. The
217.7K prompt was read in 86 s on two cards (S) and answered at 115 tok/s.

Against ISTA's GSQ-RCO IQ3_S on the same engine and card, S decodes about as fast or faster (82-88 against 74-84 tok/s
on one card on 0.1.43), reads prompts 20-25% slower and needs 41 GiB of RAM against 53 (one card) and 80 (two cards).

## Files

| File | Size | From |
|---|---:|---|
| `S/Qwen3.8-Flash-Next-S-qns.gguf` | 42.3 GB | Mirai, unchanged |
| `S/Qwen3.8-Flash-Next-S-qns-dense.gguf` | 2.1 GB | Mirai, unchanged |
| `S-small/Qwen3.8-Flash-Next-S-small-qns.gguf` | 34.4 GB | converted from Mirai's S-small with their converter |
| `S-small/Qwen3.8-Flash-Next-S-small-qns-dense.gguf` | 2.1 GB | the same |
| `mtp/rt/` | 0.8 GB | Qwen's MTP layer in Strata's runtime format (the draft layer) |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 0.9 GB | ISTA-DASLab, unchanged |
| `Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf` | 480 KB | mine, `r` for layers 1-47 |
| `LICENSE`, `SHA256SUMS` | | |

## Limits

- NVIDIA on Linux only. The fork has not been built for AMD, Intel or Mac, or on Windows.
- Not tested: real photos and documents in vision, a 12 GB card, an own refusal direction for this build.

Removing refusals removes a safety behaviour. What the model writes with the vector on is your responsibility.

## License

The weights are derived from Qwen3.8-Flash-Next and are under the [Qwen Community License 1.0](LICENSE), as Mirai
publish theirs.
