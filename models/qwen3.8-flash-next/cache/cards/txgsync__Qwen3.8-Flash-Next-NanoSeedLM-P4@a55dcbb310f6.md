---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- qwen4_exp
- qwen3.8-flash-next
- moe
- nanoseedlm
- nslm
- research-preview
- seedlm
- lfsr
- seeds
- mlx
- mlx-vlm
- omlx
- apple-silicon
- custom_code
- cuda
- metal
---

# Qwen3.8-Flash-Next-NanoSeedLM-P4

**Qwen3.8-Flash-Next with SeedLM seed experts, for MLX on a 128 GB Mac.** Its 120.8 B routed expert weights are seeds of
a linear feedback shift register: 68.0 GB, at 4.5 bits each. It runs in mlx-vlm and oMLX as well as in the
NanoSeedLM engines. Its experts are on Unsloth's quality curve: 0.0408 at 4.5 bits, 2% below it. For the lowest KL, use [TCQ4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4).

> **Research preview.** It runs in the [NanoSeedLM](https://github.com/mbarnson/NanoSeedLM) engines, in mlx-vlm with Trust Remote Code, and in oMLX
> with the fix in [jundot/omlx#5112](https://github.com/jundot/omlx/pull/5112).

## Which pack

The five packs are the same model, Qwen3.8-Flash-Next. They differ in three things: how they store the 120.8 B routed
expert weights, their dense weights, and the runtimes that load them.

| Your machine | Pack | Why |
|---|---|---|
| PC: a GPU of 24 GB or more, 96 to 128 GB of RAM | [TCQ4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4) | The lowest KL |
| PC: a 16 GB GPU, 96 to 128 GB of RAM | [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | The middle KL, 10 GB smaller |
| PC: a 12 or 16 GB GPU, 64 GB of RAM | [TCQ3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3) | The only pack for 64 GB of RAM |
| Mac: 128 GB | [TCQ4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4) | The lowest KL |
| Mac: 128 GB, for speed | [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | About 20% faster, 4 times the KL |
| Mac: 96 GB | [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | Room for long contexts |
| Research on seeds | **P4**, [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | SeedLM seeds |
| Less than 64 GB of RAM | None | The experts must fit in RAM |

- On a PC, the experts stay in RAM, and the GPU caches the experts in use. A smaller pack lets the GPU cache more of
  its experts.
- We did not test TCQ3 on a 12 GB GPU or a 64 GB PC yet. A 12 GB GPU caches fewer experts than a 16 GB GPU, so it
  decodes slower.
- P3 is faster on a Mac because its dense weights are 4-bit. They are also why its KL is 4 times that of TCQ4.
- The start script of the NanoSeedLM [engine release](https://github.com/mbarnson/NanoSeedLM/releases) checks the machine and picks a TCQ pack.

## The family

| Pack | Experts | Bits | Memory GB | KL as shipped |
|---|---|---|---|---|
| [TCQ4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4) | Trellis K4 | 4.125 | 70 | 0.0415 |
| [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | Trellis K3, down K4 | 3.458 | 60 | 0.0638 |
| [TCQ3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3) | Trellis K3 | 3.125 | 55 | 0.0778 |
| **P4** | Seeds P4 | 4.500 | 74 | not measured |
| [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | Seeds P3 | 4.000 | 66 | 0.182 |

| Pack | Expert GB | Dense weights | Download GB | Runtimes |
|---|---|---|---|---|
| [TCQ4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4) | 62.3 | 8-bit | 101.9 | Engines, mlx-vlm, oMLX |
| [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | 52.2 | 8-bit | 91.6 | Engines, mlx-vlm, oMLX |
| [TCQ3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3) | 47.2 | 8-bit | 86.5 | Engines, mlx-vlm, oMLX |
| **P4** | 68.0 | 4-bit | 105.7 | CUDA engine, mlx-vlm, oMLX |
| [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | 60.4 | 4-bit | 98.2 | Engines, mlx-vlm, oMLX |

- **Bits** are the bits for each expert weight. **Memory** counts the weights that stay resident, about. The PLE
  n-gram table stays on the disk, and the page cache holds the rows that a session reads.
- **KL as shipped** is against BF16 in llama.cpp, on 64 windows of 2,048 tokens of wikitext-2. Lower is closer to the
  original model. P4 as shipped is not measured yet.
- **Engines** are the NanoSeedLM engines: CUDA on Linux and Windows, Metal on macOS. oMLX needs the fix in
  [jundot/omlx#5112](https://github.com/jundot/omlx/pull/5112). We tested oMLX with P3 and TCQ4, and P4 in the CUDA engine only.

**The trade-offs.**

- **Trellis or seeds.** Both compute each expert weight from a few stored bits with integer arithmetic, and neither
  stores a codebook. On this model the trellis code (TCQ) is closer to BF16 for its bits. Compare the experts with the
  same 8-bit dense weights: TCQ4 gives 0.0398 at 4.125 bits, P4 0.0408 at 4.5 bits and P3 0.0600 at 4.0 bits.
- **8-bit or 4-bit dense weights.** The dense weights are a small part of the bytes, but in P3 they cause most of the
  KL: 0.0600 with 8-bit dense weights, 0.182 with the 4-bit ones that it ships. So the TCQ packs ship 8-bit dense
  weights. The cost is speed: they read 1.84 GB more for each token. On the M4 Max, TCQ4 decodes about 15% slower than
  P3.
- **Runtimes.** Each pack carries its MLX model files, so mlx-vlm runs it with Trust Remote Code. oMLX runs it with
  the fix in [jundot/omlx#5112](https://github.com/jundot/omlx/pull/5112). The NanoSeedLM engines run each pack too. The start scripts of the engine
  release prepare the TCQ packs.
- **The PLE table.** Each pack carries the 4-bit MLX PLE table (32 GB). In the NanoSeedLM engines, Unsloth's Q8_0 table
  (54.4 GB) lowers the KL by about 0.006, and the start scripts download it.

Decode in tokens per second: `nslm-serve --ctx 32768` with the Q8_0 PLE table, 4,096 tokens, thinking on, temperature
0, without MTP drafts, the mean of two runs. The machines are an M4 Max with 128 GB and an RTX 4080 (16 GB) with
128 GB of RAM. Read the runs on [the M4 Max](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-06-long-bench-m4max.md) and
[the RTX 4080](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-06-long-bench-4080.md).

| Pack | M4 Max, Metal | RTX 4080, CUDA |
|---|---|---|
| [TCQ4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4) | 39.0 | 31.4 |
| [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | not measured | 38.0 |
| [TCQ3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3) | not measured | 44.0 |
| **P4** | not measured | not measured |
| [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | 46.9 | not measured |

## Why NanoSeedLM

NanoSeedLM began when Matt Barnson heard about SeedLM ([Shafipour et al., 2024](https://arxiv.org/abs/2410.10714)) at
Apple's machine learning summit in the spring of 2025. SeedLM stores each block of 8 weights as the seed of a linear
feedback shift register (LFSR), plus a few small coefficients. An LFSR is the cheapest pseudo-random generator in
hardware. The paper rebuilt the weights on an FPGA, where an LFSR is a few flip-flops.

The idea: a chip that can calculate a weight from a few bits does not have to store the weight or wait for it.
Decode on home hardware waits mostly for memory. A gaming PC keeps most experts in system RAM, behind PCIe, and a Mac
waits for its memory bandwidth. Cheap arithmetic on the GPU, the CPU or an FPGA can take the place of some memory
reads. Then a model that was state of the art a year ago can run on that hardware at a usable speed.

The first NanoSeedLM model was [K2-Horizon-MoVA-36B-A4B in 22.76 GB](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v),
for a 32 GB Mac. This family takes the idea to Qwen3.8-Flash-Next. On this model, the trellis code of
[QTIP](https://arxiv.org/abs/2406.11235) gives the best quality for its bits. QTIP is the closest working relative of
SeedLM: its weights also come from cheap integer arithmetic, on a bit stream in place of a seed. So the TCQ packs are
the ones to use, and P3 and P4 keep the SeedLM seeds. A planned next step is a CPU kernel for the trellis experts. Then
the CPU of a PC can compute the experts that are not in VRAM. Read the story:
[NanoSeedLM: LLM inference using pseudo-random numbers](https://barnson.org/2026/10/07/nanoseedlm-llm-inference-using-pseudo-random-numbers/).

## How it works

Each block of 8 expert weights is a 16-bit LFSR seed, 4 int4 coefficients and a 4-bit exponent code
([SeedLM](https://arxiv.org/abs/2410.10714)). The GPU runs the register inside the matrix multiply and rebuilds the
weights there, so the pack stores no expert weight. The seed search tried all 65,535 seeds for each block. It weighted
the error of each weight by the mean square of its input activation, from Unsloth's imatrix. All other weights come
unchanged from [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp)
(4-bit affine, group size 32), with its MTP head.

| Part | P4 | mlx-community 4-bit |
|---|---|---|
| Routed experts | Seeds P4, 4.5 bits | 4-bit affine, 5.0 bits |
| All other weights | The same tensors | 4-bit affine, 8-bit routers |
| Size | 105.7 GB | 113.3 GB |

The routed experts are 120.8 B weights (48 layers of 512 experts). The other weights are attention, DeltaNet, the shared
experts and the routers. They also include the PLE table, the embedding, the LM head, the vision tower and the MTP head. The 4-bit affine
weights use groups of 32 and the routers groups of 64.

## Quality

| P4 | Mean KLD against BF16 |
|---|---|
| The experts, with 8-bit dense weights | 0.0408 |
| As shipped, with 4-bit dense weights | Not measured yet |

The 8-bit dense weights of the first row are Unsloth's Q8_0, in llama.cpp. In P3, the 4-bit dense weights add 0.12 to the KL.

With the same dense weights, Unsloth's packs give 0.0416 at 4.5 bits: the straight line between their two
packs nearest in bits. This pack's experts are 2% below that line. Read [KL against BF16](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-04-qwen38-kld-vs-bf16.md)
and [packs as shipped](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-05-dense-as-shipped.md).

## Speed

We did not measure this pack over long runs yet.

## How to run

### NanoSeedLM engines

Download the engine ([releases](https://github.com/mbarnson/NanoSeedLM/releases), or [build it](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/build.md)) and this
folder. We tested P4 in the CUDA engine, not in the Metal engine yet. Then chat, or serve an OpenAI-compatible API. On macOS, use `out/bin` in place of `build/bin`.

```bash
hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4 --local-dir Qwen3.8-Flash-Next-NanoSeedLM-P4
build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-P4 "Why do tide pools matter?"
build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-P4 --ctx 32768 --port 8080
```

The engines memory-map the PLE table, so it stays on the disk. For a lower KL, download Unsloth's Q8_0 table and set
`Q4E_PLE_GGUF` to it, as on the TCQ cards. `Q4E_DRAFTS=2` turns on the MTP drafts. Read
[Run and serve](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/run-and-serve.md).

### mlx-vlm

```bash
python -m mlx_vlm.generate --model txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4 --trust-remote-code \
    --prompt "Why do tide pools matter?" --max-tokens 256
```

The pack is 105.7 GB, and macOS gives the GPU about three quarters of its memory. So keep the PLE table on the SSD: make
a folder view with mlx-vlm, and run that folder.

```python
from mlx_vlm.models.qwen4_exp.ple_storage import prepare_external_ple_model
prepare_external_ple_model("Qwen3.8-Flash-Next-NanoSeedLM-P4", "Qwen3.8-Flash-Next-NanoSeedLM-P4-ssd-ple")
```

### oMLX

Stock oMLX removes `model_file` from each `qwen4_exp` configuration, so it does not load this pack's model files. The
fix in [jundot/omlx#5112](https://github.com/jundot/omlx/pull/5112) keeps `model_file` when it is not the stock `qwen4_exp.py`. Until it merges, use
the branch [fix/qwen4-exp-custom-model-file](https://github.com/mbarnson/omlx/tree/fix/qwen4-exp-custom-model-file). Then enable Trust Remote Code for this model. We tested this fix with P3, not with P4 yet.

## Limits

- This is a research preview. Its output is not identical to the output of the original model.
- mlx-vlm and oMLX need Trust Remote Code. The NanoSeedLM engines do not.
- On Apple Silicon, seed experts decode slower than affine experts of the same size.
- We did not test image or video input.

## Files

- `model-*.safetensors`: the weights. `NAME.seeds`, `NAME.nibbles` and `NAME.exp_bias` hold the routed experts.
- `nanoseedlm_qwen4.py`, `nanoseedlm_seeds.py`: the MLX model files.
- `config.json`: the configuration of the 4-bit MLX model, with `model_file`.
- `LICENSE`: the Qwen Community License 1.0.

## Credits

Base model: Qwen, [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Community License 1.0).
MLX conversion: [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp).
Activation statistics: [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) imatrix. Method: SeedLM
([arXiv:2410.10714](https://arxiv.org/abs/2410.10714)). Code: [NanoSeedLM](https://github.com/mbarnson/NanoSeedLM).
