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
- trellis
- tcq
- mlx
- mlx-vlm
- omlx
- apple-silicon
- custom_code
- cuda
- metal
---

# Qwen3.8-Flash-Next-NanoSeedLM-TCQ4

**Qwen3.8-Flash-Next on a 128 GB Mac, or a PC with a GPU of 24 GB or more and 96 to 128 GB of RAM.** Its 120.8 B routed expert weights take 62.3 GB, at 4.125 bits each. Its KL
divergence against BF16 is 0.0415, 15% below Unsloth's packs of the same size. On an M4 Max, Metal it
decodes at 39.0 tokens per second, and at 38.8 after a prompt of 30,000 tokens.

> **Research preview.** It runs in the [NanoSeedLM](https://github.com/mbarnson/NanoSeedLM) engines: CUDA on Linux and Windows, Metal on macOS. It
> also runs in mlx-vlm with Trust Remote Code, and in oMLX with the fix in [jundot/omlx#5112](https://github.com/jundot/omlx/pull/5112).

**Start.** Unpack the [engine release](https://github.com/mbarnson/NanoSeedLM/releases) for your machine. Then run `START-HERE.bat TCQ4` (Windows) or
`./start-here.sh TCQ4` (Linux, macOS). The script checks the machine, downloads and checks this pack, and starts an
OpenAI-compatible server on port 8080.

## Which pack

The five packs are the same model, Qwen3.8-Flash-Next. They differ in three things: how they store the 120.8 B routed
expert weights, their dense weights, and the runtimes that load them.

| Your machine | Pack | Why |
|---|---|---|
| PC: a GPU of 24 GB or more, 96 to 128 GB of RAM | **TCQ4** | The lowest KL |
| PC: a 16 GB GPU, 96 to 128 GB of RAM | [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | The middle KL, 10 GB smaller |
| PC: a 12 or 16 GB GPU, 64 GB of RAM | [TCQ3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3) | The only pack for 64 GB of RAM |
| Mac: 128 GB | **TCQ4** | The lowest KL |
| Mac: 128 GB, for speed | [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | About 20% faster, 4 times the KL |
| Mac: 96 GB | [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | Room for long contexts |
| Research on seeds | [P4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4), [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | SeedLM seeds |
| Less than 64 GB of RAM | None | The experts must fit in RAM |

- On a PC, the experts stay in RAM, and the GPU caches the experts in use. A smaller pack lets the GPU cache more of
  its experts.
- A 12 GB GPU caches fewer experts than a 16 GB GPU. On an RTX 4080 limited to 12 GB, TCQ3 decodes at 27 tokens per
  second, against 44 with all 16 GB. We did not test a PC with 64 GB of RAM yet.
- P3 is faster on a Mac because its dense weights are 4-bit. They are also why its KL is 4 times that of TCQ4.
- The start script of the NanoSeedLM [engine release](https://github.com/mbarnson/NanoSeedLM/releases) checks the machine and picks a TCQ pack.

## The family

| Pack | Experts | Bits | Memory GB | KL as shipped |
|---|---|---|---|---|
| **TCQ4** | Trellis K4 | 4.125 | 70 | 0.0415 |
| [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | Trellis K3, down K4 | 3.458 | 60 | 0.0638 |
| [TCQ3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3) | Trellis K3 | 3.125 | 55 | 0.0778 |
| [P4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4) | Seeds P4 | 4.500 | 74 | 0.169 |
| [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | Seeds P3 | 4.000 | 66 | 0.182 |

| Pack | Expert GB | Dense weights | Download GB | Runtimes |
|---|---|---|---|---|
| **TCQ4** | 62.3 | 8-bit | 127.5 | Engines, mlx-vlm, oMLX |
| [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | 52.2 | 8-bit | 117.2 | Engines, mlx-vlm, oMLX |
| [TCQ3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3) | 47.2 | 8-bit | 112.1 | Engines, mlx-vlm, oMLX |
| [P4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4) | 68.0 | 4-bit | 131.3 | Engines, mlx-vlm, oMLX |
| [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | 60.4 | 4-bit | 123.8 | Engines, mlx-vlm, oMLX |

- **Bits** are the bits for each expert weight. **Memory** counts the weights that stay resident, about. The PLE
  n-gram table stays on the disk, and the page cache holds the rows that a session reads.
- **KL as shipped** is against BF16 in llama.cpp, on 64 windows of 2,048 tokens of wikitext-2, with Unsloth's Q8_0 PLE
  table. Lower is closer to the original model.
- **Engines** are the NanoSeedLM engines: CUDA on Linux and Windows, Metal on macOS. oMLX needs the fix in
  [jundot/omlx#5112](https://github.com/jundot/omlx/pull/5112). We tested TCQ4 and P3 in oMLX with the 8-bit PLE table, and each pack with the
  earlier 4-bit table.

**The trade-offs.**

- **Trellis or seeds.** Both compute each expert weight from a few stored bits with integer arithmetic, and neither
  stores a codebook. On this model the trellis code (TCQ) is closer to BF16 for its bits. Compare the experts with the
  same 8-bit dense weights: TCQ4 gives 0.0398 at 4.125 bits, P4 0.0408 at 4.5 bits and P3 0.0600 at 4.0 bits.
- **8-bit or 4-bit dense weights.** The dense weights are a small part of the bytes, but in P3 and P4 they cause most
  of the KL. With 8-bit dense weights, P3 gives 0.0600 and P4 0.0408. With the 4-bit ones that they ship, P3 gives 0.182
  and P4 0.169. So the TCQ packs ship 8-bit dense weights. The cost is speed: they read 1.84 GB more for each token. On the M4 Max, TCQ4 decodes about 15% slower than
  P3.
- **Runtimes.** Each pack carries its MLX model files, so mlx-vlm runs it with Trust Remote Code. oMLX runs it with
  the fix in [jundot/omlx#5112](https://github.com/jundot/omlx/pull/5112). The NanoSeedLM engines run each pack too. The start scripts of the engine
  release prepare the TCQ packs.
- **The PLE table.** Each pack carries the PLE n-gram table at 8 bits (57.6 GB, MLX's form, group 32), made from Qwen's
  BF16 table. Every runtime reads it. Its row error against BF16 is 0.88 times that of Unsloth's Q8_0 table
  ([the encoding](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-07-ple8-encoding.md)). A 4-bit table would add about 0.006 to the KL.

Decode in tokens per second: `nslm-serve --ctx 32768` with Unsloth's Q8_0 PLE table, 4,096 tokens, thinking on, temperature
0, without MTP drafts, the mean of two runs. The machines are an M4 Max with 128 GB and an RTX 4080 (16 GB) with
128 GB of RAM. Read the runs on [the M4 Max](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-06-long-bench-m4max.md) and
[the RTX 4080](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-06-long-bench-4080.md). The 12 GB GPU is [the RTX 4080 with 4 GB kept free](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-06-tcq3-12gb.md).

| Pack | M4 Max, Metal | RTX 4080, CUDA |
|---|---|---|
| **TCQ4** | 39.0 | 31.4 |
| [TCQ3-D4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4) | not measured | 38.0 |
| [TCQ3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ3) | not measured | 44.0 |
| [P4](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P4) | not measured | not measured |
| [P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) | 46.9 | not measured |

**MTP drafts** (engine v0.1.1). Set `Q4E_DRAFTS=2` and `Q4E_DRAFT_MIN_P=0.3`. Then the M4 Max decodes TCQ4 32% faster
at temperature 0 and 16% faster at 1.0. At temperature 0 the text does not change. On the RTX 4080, drafts break even,
so they stay off there. The start script turns them on for a Mac. Read the gates on
[the M4 Max](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-06-draft-gates-m4max.md) and [the RTX 4080](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-06-draft-gates-4080.md).

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

The experts are trellis-coded (K = 4 for `gate_proj`, `up_proj` and `down_proj`, 12-bit states, as in [QTIP](https://arxiv.org/abs/2406.11235)). Neighbouring
weights share the bits of one stream. Each weight is a fixed multiply-add of a 12-bit window of the stream, so there is
no codebook to store or look up. The dense weights are 8-bit MLX (oQ8e). On a GPU, the routed experts stay in host
memory, and a cache keeps the experts in use in VRAM.

## Quality

This pack's experts and dense weights are written into Unsloth's Q8_0 GGUF, so llama.cpp runs this pack's weights. The
PLE table in these runs is Unsloth's Q8_0. The pack's own 8-bit table has a lower row error against BF16, so its KL
should be the same within the error. That is not yet measured in llama.cpp.

| | Mean KLD | Median KLD | 99% KLD | Top-1 | Perplexity |
|---|---|---|---|---|---|
| TCQ4 | 0.0415 | 0.0113 | 0.492 | 92.98% | 3.847 |
| BF16 | 0 | 0 | 0 | 100% | 3.802 |

The mean KLD's standard error is 0.0005. "99% KLD" is the 99th percentile. "Top-1" is the share of positions
where the pack's likeliest token is BF16's.

Unsloth's packs, measured in the same way (with their own dense weights):

| Pack | Bits for each expert weight | Mean KLD |
|---|---|---|
| This pack | 4.125 | 0.0415 |
| Unsloth UD-Q4_K_XL | 5.101 | 0.0304 |
| Unsloth UD-IQ4_XS | 3.999 | 0.0509 |
| Unsloth UD-Q3_K_XL | 3.720 | 0.0593 |
| Unsloth UD-IQ3_XXS | 3.331 | 0.0735 |
| Unsloth UD-Q2_K_XL | 3.049 | 0.0994 |

"Below Unsloth's packs of the same size" compares with the straight line between Unsloth's two packs nearest in bits:
0.0486 at 4.125 bits. Read [the measurements](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-05-release-as-shipped.md) and
[the KL of the PLE tables](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-06-ple-table-kld.md).

## Speed

This pack's long runs, as in the family table. The prompt column is the second run.

| Machine, engine | Decode, 4,096 tokens | Decode after 30K tokens | Prompt of 30K tokens |
|---|---|---|---|
| M4 Max, Metal | 39.0 | 38.8 | 475 |
| RTX 4080, CUDA | 31.4 | 30.2 | 1,725 |

## How to run, by hand

### NanoSeedLM engines

Download the engine ([releases](https://github.com/mbarnson/NanoSeedLM/releases), or [build it](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/build.md)) and this
folder. Then chat, or serve an OpenAI-compatible API. The engines read the pack's own 8-bit PLE table. On macOS, use
`out/bin` in place of `build/bin`.

```bash
hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4 --local-dir Qwen3.8-Flash-Next-NanoSeedLM-TCQ4
build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-TCQ4 "Why do tide pools matter?"
build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-TCQ4 --ctx 32768 --port 8080
```

Read [Run and serve](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/run-and-serve.md).

### mlx-vlm

```bash
python -m mlx_vlm.generate --model txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TCQ4 --trust-remote-code \
    --prompt "Why do tide pools matter?" --max-tokens 256
```

mlx-vlm loads the whole pack, 127.5 GB, and macOS gives the GPU about three quarters of its memory. So mlx-vlm needs a Mac
with about 170 GB. On a 128 GB Mac, use oMLX with SSD PLE (below) or the NanoSeedLM engines: both keep the PLE
table on the SSD.

### oMLX

Stock oMLX removes `model_file` from each `qwen4_exp` configuration, so it does not load this pack's model files. The
fix in [jundot/omlx#5112](https://github.com/jundot/omlx/pull/5112) keeps `model_file` when it is not the stock `qwen4_exp.py`. Until it merges, use
the branch [fix/qwen4-exp-custom-model-file](https://github.com/mbarnson/omlx/tree/fix/qwen4-exp-custom-model-file). Then enable Trust Remote Code for this model. We tested this pack in oMLX with SSD PLE on an M4 Max: MTP on gives the same text as MTP off, at about 84 tokens per second against 49.

## Limits

- This is a research preview. Its output is not identical to the output of the original model.
- mlx-vlm and oMLX need Trust Remote Code. The NanoSeedLM engines do not.
- We did not test image or video input.

## Files

- `model-*.safetensors`: the dense weights (8-bit MLX), the vision tower and the MTP head. `model-ple8-*.safetensors`:
  the PLE table (8-bit MLX, group 32, from Qwen's BF16).
- `L{layer}_{gate_proj,up_proj,down_proj}.blkt`: the trellis-coded routed experts of layers 0 to 47 and of the MTP
  head (layer 48). Read [the format](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/reference/model-folder.md).
- `nanoseedlm_qwen4.py`, `nanoseedlm_seeds.py`, `nanoseedlm_trellis.py`: the MLX model files. `nanoseedlm_qwen4.py`
  puts the trellis banks in place of the routed experts.
- `config.json`: the configuration of the 8-bit MLX model, with `model_file`. The tokenizer and the chat template.
- `LICENSE`: the Qwen Community License 1.0.

## Credits

Base model: Qwen, [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Community License 1.0).
Dense weights: [mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp)
and [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp).
PLE table: Qwen's BF16, in 8 bits by `tools/hf_ple8.py`. Activation statistics: Unsloth's imatrix.
Trellis code: QTIP ([arXiv:2406.11235](https://arxiv.org/abs/2406.11235)). Code: [NanoSeedLM](https://github.com/mbarnson/NanoSeedLM).
