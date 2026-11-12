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
here: the installer fetches them from their own repos, each pinned to a commit and a SHA-256. The one exception is
osmos-evo-A's delta, osmos's own pack, stored as the difference from unsloth's UD-IQ1_M.

| file | what | bytes |
| --- | --- | --- |
| `evo-A/DeepSeek-V4.1-Flash-1017-osmos-evo-A-delta.gguf` | osmos-evo-A as a delta on unsloth's UD-IQ1_M: the 221 tensors it changes | 33,664,472,480 |
| `dspark-iq2xs/dspark.bin` | DeepSeek's DSpark drafter (the three `mtp` blocks), its experts requantized to IQ2_XS | 4,676,398,848 |
| `dspark-iq2xs/dspark.json` | the drafter's tensor index and target layers (37, 38, 39) | 98,163 |
| `picks/UD-IQ1_S/osmos-picks.bin` | which experts UD-IQ1_S's router picks most, by layer, on calibration text: what osmos keeps in GPU memory first | 122,880 |
| `picks/UD-IQ1_M/osmos-picks.bin` | the same for UD-IQ1_M | 122,880 |

## The packs

| pack | weights | drafter | context | decode, plain / drafted |
| --- | --- | --- | --- | --- |
| `osmos-evo-a` (the default) | unsloth UD-IQ1_M + this repo's delta + the MXFP8 Engram table | this repo | 1,048,576 | 17.9 / 29.7 tok/s |
| `ud-iq1_s` (speed) | unsloth UD-IQ1_S + an MXFP8 Engram table | this repo | 1,048,576 | 18.3 / 31.7 tok/s |
| `ud-iq1_m` (unsloth's, unchanged) | unsloth UD-IQ1_M + the same table | this repo | 1,048,576 | 16.6 / 25.8 tok/s |

Decode is greedy, at AMD's `low-power` profile on osmos 0.6.8: a short question and a 2K and an 8K prompt, 256 tokens
each, two interleaved rounds on a quiet box. Drafted speed varies by prompt. Output is bit-for-bit reproducible. osmos sizes its expert budget to the
GPU's free memory at start (`--budget auto`). Against Project Maya 1.1.53 on the same files, osmos reads prompts
26% faster at 2K and 11-12% at 8K, drafts 6-8% faster, and runs Maya's 4-agent bench in 20% less time on 13% less
energy an answer token; plain decode is a tie. Every number and its command is in the osmos repo.

## osmos-evo-A

UD-IQ1_M with two changes, each scored against the model's own weights before it was kept:

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
python3 tools/install/osmos_install.py osmos-evo-a --serve   # or --rig to register it in rig
```

## Where the drafter comes from

DeepSeek's DSpark weights (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, model-00044 to 00046, MIT). Project Maya's
`tools/ds4/drafter_pack.py` (MIT) packed them: dense matrices Q8_0, experts MXFP4. osmos's `tools/requant/requant_dspark`
requantized the 3 x 128 experts to IQ2_XS with ggml's quantizers (llama.cpp, MIT). That cut the drafter from 7.97 to
4.68 GB, leaving GPU memory for the model's own experts. The heads stay BF16.

## License

MIT. The weights derive from DeepSeek-V4.1-Flash-1017 (MIT, DeepSeek). osmos-evo-A's delta derives from unsloth's
UD-IQ1_M quantization and importance matrix (unsloth/DeepSeek-V4.1-Flash-1017-GGUF); its `--abliterated` overlay is
huihui-ai's UD-IQ1_M patch (MIT). Packing is by Project Maya (MIT, Peasant Smith
and the Project Maya contributors). Requantization uses ggml (MIT, the ggml authors).
