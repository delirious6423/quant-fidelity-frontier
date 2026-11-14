---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
  - deepseek
  - speculative-decoding
  - amd
  - strix-halo
  - gguf
---

# osmos-evo-1017

The files [osmos](https://github.com/mrsirg97-rgb/libosmos) needs beside unsloth's GGUFs to run
DeepSeek-V4.1-Flash-1017 on an AMD Strix Halo (Ryzen AI Max+ 395, gfx1151, 128 GB). The full model weights are not
here: the installer fetches them from their own repos, each pinned to a commit and a SHA-256. The exceptions are
osmos's own pack, stored as differences: osmos-evo-A's delta on unsloth's UD-IQ1_M, and osmos-evo's patch on osmos-evo-A.

| file | what | bytes |
| --- | --- | --- |
| `evo-A2/DeepSeek-V4.1-Flash-1017-osmos-evo-A2-patch.gguf` | osmos-evo as a patch on osmos-evo-A: the routed experts' gate and up of layers 10-19 (20 tensors, unsloth UD-IQ2_XXS's) | 23,357,055,232 |
| `evo-A/DeepSeek-V4.1-Flash-1017-osmos-evo-A-delta.gguf` | osmos-evo-A as a delta on unsloth's UD-IQ1_M: the 221 tensors it changes | 33,664,472,480 |
| `dspark-iq2xs/dspark.bin` | DeepSeek's DSpark drafter (the three `mtp` blocks), its experts requantized to IQ2_XS | 4,676,398,848 |
| `dspark-iq2xs/dspark.json` | the drafter's tensor index and target layers (37, 38, 39) | 98,163 |
| `picks/UD-IQ1_S/osmos-picks.bin` | which experts UD-IQ1_S's router picks most, by layer, on calibration text: what osmos keeps in GPU memory first | 122,880 |
| `picks/UD-IQ1_M/osmos-picks.bin` | the same for UD-IQ1_M | 122,880 |

## The packs

| pack | weights | drafter | context | decode, plain / drafted |
| --- | --- | --- | --- | --- |
| `osmos-evo` (the default) | unsloth UD-IQ1_M + this repo's delta + its patch + the MXFP8 Engram table | this repo | 1,048,576 | 17.7 / 32.8 tok/s |
| `osmos-evo-a` (0.7.0) | unsloth UD-IQ1_M + this repo's delta + the same table | this repo | 1,048,576 | 19.0 / 35.3 tok/s |
| `ud-iq1_m` (unsloth's, 0.7.0) | unsloth UD-IQ1_M + the same table | this repo | 1,048,576 | 17.6 / 35.1 tok/s |
| `ud-iq2_xxs` (unsloth's, 0.7.0) | unsloth UD-IQ2_XXS + the same table | this repo | 1,048,576 | 14.2 / 26.0 tok/s |

Decode is greedy at AMD's `performance` profile, the osmos release suite's way: 64K of context, three questions, 256
tokens each, the mean of the three (osmos-evo on 0.7.6, the others on 0.7.0's run; `ud-iq1_s` remains for speed).
Drafted speed varies by prompt. Output is bit-for-bit reproducible. osmos sizes its expert budget to the GPU's free
memory at start (`--budget auto`). Every number and its command is in the osmos repo.

## osmos-evo

osmos-evo-A (below) with the routed experts' gate and up of layers 10-19 at IQ2_XXS - unsloth UD-IQ2_XXS's own
tensors, where osmos-evo-A holds UD-IQ1_M's IQ1_S. Found by probing the one fact osmos-evo-A wavered on in a
1M-token recall test (an access code 188K tokens back, its first token 0.47) and bisecting its layers by swapping in
UD-IQ2_XXS's tensors a group at a time (osmos DESIGN.md, "0.7.6"; `tools/evo`). Scored against a reference built from
the model's own weights (KL divergence a token, lower is closer):

| | Wiki text | agent traces | top-1 (Wiki) | the 188K fact |
| --- | ---: | ---: | ---: | ---: |
| `ud-iq2_xxs` | 0.126 | 0.036 | 92.4% | 0.983 |
| **`osmos-evo`** | **0.158** | **0.042** | **91.2%** | **0.955** |
| `osmos-evo-a` | 0.175 | 0.0485 | 90.5% | 0.469 |
| `ud-iq1_m` | 0.190 | 0.048 | 90.3% | |

About halfway from UD-IQ1_M's quality to UD-IQ2_XXS's (agent traces 0.0485 -> 0.0422, 18 standard errors), for 5.8 GB
more and about 5% of decode against osmos-evo-A; faster than UD-IQ2_XXS by 25% plain and 26% drafted. Every fact of
the 1M recall test is recalled, each answer token the model's first choice.

The installer builds it where it runs: UD-IQ1_M from unsloth, osmos-evo-A's delta and osmos-evo's patch from here,
each assembled with `tools/convert/gguf_assemble.py` and checked against its pinned SHA-256. An install that already
has osmos-evo-A fetches the 23 GB patch alone. On btrfs or XFS every shared byte is cloned.

## osmos-evo-A

osmos-evo's base: UD-IQ1_M with two changes, each scored against the model's own weights before it was kept:

- the routed experts' down projections of layers 20-39, requantized to Q2_K from DeepSeek's MXFP4 originals with
  unsloth's importance matrix;
- the attention (kv, q_a, q_b, output_a, output_b) and the output head, requantized to Q5_K by GPTQ from UD-IQ1_M's
  values.

Every other tensor is UD-IQ1_M's, byte for byte. Scored against a reference built from the original weights (KL
divergence a token, lower is closer): on agent traces it is level with UD-IQ1_M (0.0485 against 0.0479), on Wiki text
closer (0.175 against 0.190; UD-IQ1_S 0.259). At AMD's `low-power` profile it decodes 8% faster than UD-IQ1_M plain
and 15% faster drafted (17.9 / 29.7 tok/s against 16.6 / 25.8), within 2-6% of UD-IQ1_S. The method and every
number are in the osmos repo (DESIGN.md, "0.6.6" and "0.6.8").

The installer builds it where it runs: it downloads UD-IQ1_M from unsloth and the 34 GB delta from here, writes the
pack's shards with `tools/convert/gguf_assemble.py`, and checks each against its pinned SHA-256. On btrfs or XFS
every byte the two share is cloned, so beside UD-IQ1_M it takes almost no extra disk.

An abliterated variant is an opt-in overlay (`--abliterated`) of huihui-ai's patch (MIT), applied exactly at load time.
It is off by default.

## Install

```
git clone --recursive https://github.com/mrsirg97-rgb/libosmos && cd libosmos
python3 tools/install/osmos_install.py osmos-evo --serve   # or --rig to register it in rig (osmos-evo-a: the same)
```

## Where the drafter comes from

DeepSeek's DSpark weights (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, model-00044 to 00046, MIT). Project Maya's
`tools/ds4/drafter_pack.py` (MIT) packed them: dense matrices Q8_0, experts MXFP4. osmos's `tools/requant/requant_dspark`
requantized the 3 x 128 experts to IQ2_XS with ggml's quantizers (llama.cpp, MIT). That cut the drafter from 7.97 to
4.68 GB, leaving GPU memory for the model's own experts. The heads stay BF16.

## License

MIT. The weights derive from DeepSeek-V4.1-Flash-1017 (MIT, DeepSeek). osmos-evo-A's delta derives from unsloth's
UD-IQ1_M quantization and importance matrix (unsloth/DeepSeek-V4.1-Flash-1017-GGUF); osmos-evo's patch is unsloth
UD-IQ2_XXS's tensors from the same repo; its `--abliterated` overlay is
huihui-ai's UD-IQ1_M patch (MIT). Packing is by Project Maya (MIT, Peasant Smith
and the Project Maya contributors). Requantization uses ggml (MIT, the ggml authors).
