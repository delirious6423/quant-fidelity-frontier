---
base_model: Qwen/Qwen3.8-Flash-Next
language:
- en
library_name: gguf
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
tags:
- gguf
- imatrix
- qwen4exp
- mtp
- strix-halo
- gfx1151
- custom-quant
- rdna-boosts
---

# Qwen3.8-Flash-Next: 4.46 bpw custom mixture for AMD RDNA (Strix Halo / R9700)

A hand-tuned, imatrix-driven GGUF quantization of **Qwen/Qwen3.8-Flash-Next** (179.6 B params),
built for **[llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)**, a patch set
for `llama.cpp` that provides AMD RDNA specific optimizations including native BF16 KV support,
Adaptive MTP, multi-tiered MoE expert caching, and a bespoke `qwen4exp` implementation.

This is deliberately a **quality/speed blend, not a bits-per-weight record**: the aim is good PPL
*and* high throughput on AMD hardware people actually have. That means a **128 GiB Strix Halo APU**
(`gfx1151`, unified memory), or **one or more Radeon AI Pro R9700** (`gfx1201`, 32 GiB each) in a
machine with **64-96 GiB of system RAM**.

| | |
|---|---|
| size | **93.24 GiB** (100,114,598,624 B): 91.47 GiB backbone + 1.77 GiB MTP head |
| effective bpw | **4.461** overall · **4.445** excluding the 51.2 B per-layer embedding (PLE) |
| `general.file_type` | `25` = `MOSTLY_IQ4_NL` |
| architecture | `qwen4exp` · 48 layers + 1 MTP layer · hidden 2560 · 512 experts / 10 active · 262144 ctx |
| MTP | draft head **built in**, no `--spec-draft-model` needed |
| quality | wikitext-2 PPL **4.092** (-c 2048); **4.685** on the 512-ctx/580-chunk harness |

## The goals

The constraints that shaped this file, in order of how much they mattered:

1. **Fit comfortably in 128 GiB, with room for context.** On the Strix Halo APU, host DRAM *is*
   device DRAM, so `-ncmoe` expert offload frees nothing and only adds copies. The whole model plus
   a 200 k KV cache must therefore be resident at once. Target: **under ~94 GiB**, leaving ~30 GiB
   for KV cache and compute buffers. That same compactness is what makes a **1-card R9700 box with
   64 GiB of system RAM** (or 2-3 cards) viable, where the experts can instead be streamed from
   host memory.
2. **Spend bits where the imatrix says they matter.** Per-tensor importance in this model varies
   by ~4 orders of magnitude: hyper-connection mixers and SSM/attention aux weights are ~10³×
   more sensitive than the routed experts. The experts (the bulk of the bytes) get 4-bit; the aux
   weights get 5-8 bit.
3. **Only types that gfx1151 has fast kernels for.** The HIP backend in the
   [rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) delivery has type-gated fusions
   and a block-32 on-the-fly dequant path for prefill (MMB). K-quants on the
   aux weights cost prefill throughput for a nominal quality gain, so aux types are block-32
   (`Q5_0`, `Q8_0`, `IQ4_NL`), chosen for *speed*, not just size.
4. **First-generation quantization only.** Never a requant of someone else's GGUF: start from
   the BF16 weights converted from the official safetensors.
5. **Keep the built-in MTP head.** The draft head lives in the model (`blk.48`), so a separate
   MTP file (2.8 GB duplicate) is unnecessary.

## How it was made

Three parts, one splice:

| # | part | role |
|---|---|---|
| 1 | **BF16 GGUF** converted from `Qwen/Qwen3.8-Flash-Next` safetensors | the only weight source |
| 2 | **[AesSedai](https://huggingface.co/AesSedai/Qwen3.8-Flash-Next-GGUF) `imatrix.gguf`** (bundled here) | importance matrix for `llama-quantize`: 926 tensors, 50 x 8192 tokens |
| 3 | **[unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`** | source of the 32 `blk.48.*` MTP-head tensors |
| 4 | **`Qwen3.8-Flash-Next-MTP4-recipe.txt`** (bundled here) | the 29 per-tensor type rules |

1. **Quantize** the BF16 GGUF with `llama-quantize --pure`, base type `IQ4_NL`, guided by the
   AesSedai imatrix and the recipe → experts `IQ4_XS`/`IQ4_NL`, aux `Q5_0`/`Q8_0`, PLE `Q4_0`.
2. **Splice** the MTP head: the 32 `blk.48.*` tensors are replaced with the ones from unsloth's
   `Q4_K_M` head (that's the "MTP4" in the name: the head at ~4-bit instead of Q8_0, saving
   ~1 GiB for equivalent draft acceptance). Verified byte-identical to that file.
3. **Split** into 2 files with `llama-gguf-split --split --split-max-size 52G`.

The recipe was the single biggest lever, and the imatrix was the biggest lever *within* it: the
same rules with a weaker imatrix scored PPL 4.19 instead of ~4.06.

Full detail, exact commands and verification notes: [`RECIPE.md`](RECIPE.md).

## What's inside

| tensor group | type | share of params |
|---|---|---|
| routed + shared experts (`ffn_*_exps`, `ffn_{gate,up}_shexp`) | `IQ4_XS` | 44.9 % |
| PLE (`per_layer_token_embd.weight`, 51.2 B) | `Q4_0` | 28.5 % |
| `token_embd`, `ffn_down_exps` | `IQ4_NL` | 22.8 % |
| `attn_*`, `ssm_*`, `hc_*`, `output_hc` | `Q8_0` | 1.0 % |
| `attn_qkv`, `attn_q`, `attn_gate`, `ssm_out` | `Q5_0` | 1.4 % |
| MTP head `blk.48.*` (from unsloth) | `Q4_K` / `Q6_K` / `Q5_0` / `Q8_0` | 1.0 % |
| `indexer` / `output.weight` / norms | `BF16` / `Q6_K` / `F32` | <0.1 % |

Two structural limits worth knowing: `ffn_down_exps` has reduction dim 640, so **no K-quants
apply** (only block-32 types), and the PLE is 160 wide, so `IQ4_XS` is impossible there too.

**Note on the name:** during development this family was called "5 bpw", but the measured figure
is **4.46 bpw**. The 51.2 B PLE (28.5 % of parameters) sits at Q4_0 and pulls the average down.
`IQ4_NL` was chosen as the label because it is what `general.file_type` already says and it
matches 4.5 bpw nominal. `Q4_K_L`/`Q4_K_XL` are not real `llama.cpp` types.

## Quality

PPL is full **wikitext-2**, `-c 2048`, bf16 KV, flash-attn on, measured on a Strix Halo APU with the
[rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) delivery (ROCm 7.14). Numbers are for
the identical backbone (this file differs only in the MTP head, which does not affect main-model PPL):

| model | PPL | size |
|---|---|---|
| **this file** | **4.092** | **93.2 GiB** |
| AesSedai `Q4_K_M` | 4.024 | 126.1 GiB |
| Unsloth `UD-Q4_K_XL` | 4.034 | 103.7 GiB |
| Unsloth `UD-IQ4_XS` | 4.127 | 87.2 GiB |
| ilintar `IQ4_NL` | 4.167 | 93.2 GiB |

So: within ~0.07 PPL of much larger quants while being 25-35 GiB smaller. On AesSedai's own
512-ctx/580-chunk harness this file scores **4.685** vs their `Q4_K_M` **4.612** at 1.9 bpw less.

**MTP matters more than any of the quant choices.** The draft head is built in, so point at
`-00001-of-00002.gguf` and pass no `--spec-draft-model`.

> **Speed is deliberately not reported.** The throughput figures from earlier drafts of this card
> (prefill, decode, and the multi-turn MTP speedup) predate a large amount of optimisation work in
> the rdna-boosts set and are no longer representative. Re-measure on your own build; the run
> flags below are the parts that survived that tuning.

> KLD is not reported: it needs a saved BF16 reference logits file (~150 GB) that this 128 GiB
> machine cannot regenerate. From its PPL position, expect roughly ~0.06.

## Running it

Needs **[llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)**, or another
build carrying the same `qwen4exp` + MTP work. Stock mainline `llama.cpp` will load the weights but
has no MTP graph for this architecture and no cross-model tensor borrowing, so the built-in draft
head is simply ignored.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout <fork-point from the rdna-boosts README>
bash <path-to-rdna-boosts>/scripts/apply-all.sh .     # -> branch `rdna-boosts`
```

The patch set is **default-on** (no environment variables required) and covers RDNA3/RDNA3.5/RDNA4,
so the same build serves both targets below.

**Strix Halo APU (gfx1151, 128 GiB unified):**

```bash
llama-server \
  --alias Qwen3.8-Flash-Next-MTP4-IQ4_NL \
  --model Qwen3.8-Flash-Next-MTP4-IQ4_NL-00001-of-00002.gguf \
  -c 204800 --parallel 1 -ngl all \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto \
  --spec-type draft-mtp-adaptive --spec-draft-n-min-adaptive 2 --spec-draft-n-max 7 \
  -ot per_layer_token_embd.weight=ROCm0 \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.001
```

**Radeon AI Pro R9700 (gfx1201, discrete) with 64-96 GiB system RAM:** keep the experts in host
memory and stream them instead. That means `-ncmoe` plus the MoE expert cache, which the patch set
natively supports and documents with measured tables in `COMMUNITY-CONFIG.md`:

```bash
llama-server \
  --alias Qwen3.8-Flash-Next-MTP4-IQ4_NL \
  --model Qwen3.8-Flash-Next-MTP4-IQ4_NL-00001-of-00002.gguf \
  -c 65536 --parallel 1 -ngl all -ncmoe <layers> \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn auto \
  --spec-type draft-mtp-adaptive --spec-draft-n-max 7 \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.001
```

* The loader finds part 2 from the `-00001-of-00002` name; download **both** files.
* `-ot per_layer_token_embd.weight=ROCm0` moves the 28.8 GB PLE onto the GPU, which removes a CPU
  dequant and upload from every graph. It is worth it on **unified** memory, where it costs no
  physical RAM. On a discrete card it would eat most of a 32 GiB frame, so leave the PLE in host
  memory there.
* Symmetrically, `-ncmoe` is **wrong** on the APU (host DRAM == device DRAM, so it only adds copies)
  and **right** on the R9700 (it genuinely frees VRAM).
* Don't use `LLAMA_LAZY_BUF_MB` / `--lazy-buffer-size` on the unified box; the managed PLE reader
  was slower than simply pinning the tensor to the device.
* Draft depth is worth tuning per workload: shallower `--spec-draft-n-max` for plain prose, the
  adaptive controller for mixed code and reasoning.
* Vision needs a separate `--mmproj`; this repo is text-only.

## Credits

* **[llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts)**: the AMD RDNA
  patch set this model was built to run on. Native BF16 KV, Adaptive MTP, `qwen4exp` support,
  gfx1151 MMB prefill, fused MoE and k-quant decode paths, multi-tiered MoE expert caching, and
  the `-ncmoe` configuration guidance for discrete cards.
* **[Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**: base model, Qwen community licence.
* **[AesSedai](https://huggingface.co/AesSedai/Qwen3.8-Flash-Next-GGUF)**: the
  [importance matrix](https://huggingface.co/AesSedai/Qwen3.8-Flash-Next-GGUF/blob/main/imatrix.gguf)
  that drove the quantization (bundled here, hash-identical), and the `Q4_K_M` build used as the
  quality reference.
* **[unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)**: the
  [MTP draft head](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf)
  whose `blk.48.*` tensors are copied verbatim into this file.
