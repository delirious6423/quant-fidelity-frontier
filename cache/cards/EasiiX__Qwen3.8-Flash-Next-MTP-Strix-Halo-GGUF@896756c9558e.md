---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - llama.cpp
  - speculative-decoding
  - mtp
  - qwen4exp
  - strix-halo
---

# Qwen 3.8 Flash-Next MTP sidecar — tuned for AMD Strix Halo (GGUF)

The multi-token-prediction (MTP) draft head of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
exported as a standalone GGUF sidecar for llama.cpp speculative decoding.
The MTP weights ship in the official checkpoint but are stripped from the
common GGUF conversions — this repo provides them ready to use.

**Measured on AMD Strix Halo (Ryzen AI MAX+ 395, gfx1151), UD-IQ3_XXS target,
temperature 0:** decode +50% on code (23.5 → 35.7 t/s, ~85% acceptance),
+49% even at 156K context depth. Speculative decoding is lossless at
temperature 0.

## Files

| File | Size | Note |
|---|---|---|
| `mtp-Qwen3.8-Flash-Next-hc-Q8_0.gguf` | 4.1 GB | **use this one** — includes the hyper-connection tensors required by llama.cpp since upstream PR [#28901](https://github.com/ggml-org/llama.cpp/pull/28901) |
| ~~`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`~~ | 4.1 GB | **DEPRECATED (2026-09-19)** — predates upstream #28901; current llama.cpp builds refuse to load it (`check_tensor_dims: tensor 'output_hc_norm.weight' not found`). Same weights as the `-hc-` file, just missing the hc tensors. Kept only for builds from before 2026-09. |

## Usage

Needs a llama.cpp build with the qwen4exp MTP draft head. Tested branch
(includes Strix-Halo kernel patches + build guide):
**https://github.com/Aristo94/EngramHalo.cpp**

```bash
llama-server -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 32768 -ub 2048 \
  -md mtp-Qwen3.8-Flash-Next-hc-Q8_0.gguf \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75
```

`--spec-draft-p-min 0.75` matters: without the confidence gate, low-acceptance
prose loses more to rejected drafts than it gains.

You can also build the sidecar yourself with that branch's converter
(streams only the MTP tensors, ~8 GB traffic). Use a current checkout —
older checkouts produce the deprecated layout without the hc tensors:

```bash
python convert_hf_to_gguf.py --remote --mtp Qwen/Qwen3.8-Flash-Next \
  --outfile mtp-Qwen3.8-Flash-Next-BF16.gguf --outtype bf16
llama-quantize mtp-Qwen3.8-Flash-Next-BF16.gguf mtp-Qwen3.8-Flash-Next-hc-Q8_0.gguf Q8_0
```

## Credits & license

Draft-head design: Qwen team
([llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739));
qwen4exp support: [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742).
Related community work: dzannotti's MTP GGUF repo.

Derivative of Qwen 3.8 Flash-Next, distributed under the
**Qwen Community License 1.0** (see `LICENSE`). Note the license's
Model-as-a-Service clause before commercial serving.
