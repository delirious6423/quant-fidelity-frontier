---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - nchapman/Qwen3.8-Flash-Next-Swift-1.5-LoRA
  - nazeshinjite/Qwen3.8-Flash-Next-Abliterated-LoRA
library_name: basalt
pipeline_tag: text-generation
tags:
  - basalt
  - lapilli
  - qwen4exp
  - gguf-derived
  - q5_k_m
  - abliterated
  - v100
---

# Qwen3.8-Flash-Next + Swift-1.5 + abliterated — Q5_K_M Basalt pack

A `.basalt` pack for the [Basalt](https://github.com/jesdga95/basalt) engine and its V100 port
[Lapilli](https://github.com/chris83-dev/lapilli-v100), built by applying two published LoRAs to the base
model and quantizing once:

1. **Qwen/Qwen3.8-Flash-Next** — the base checkpoint (Qwen4-Exp / `qwen4exp` architecture).
2. **nchapman/Qwen3.8-Flash-Next-Swift-1.5-LoRA** — applied with `llama-export-lora`.
3. **nazeshinjite/Qwen3.8-Flash-Next-Abliterated-LoRA** — a rank-1 refusal direction (F32, alpha 1.0,
   192 tensors over `ffn_down_shexp` and `ssm_out`), applied the same way.

## The quantization recipe, and why it is not a plain Q5_K_M

llama.cpp's Q5_K_M refuses K-quants on rows whose width is not a multiple of 256. In this model that hits
`ffn_down_exps` and `ffn_down_shexp` (640 columns) and promotes them to **q8_0**, and it hits
`per_layer_token_embd` (160 columns) which the Basalt engine then **rejects outright** — it accepts only
IQ4_NL, Q5_0, Q8_0 or BF16 for that tensor, and it travels raw in the gguf section rather than through
`dense.index`, so `--compat-bf16` does not rescue it.

The published pack therefore overrides two things explicitly:

```text
blk.<N>.ffn_down_exps.weight   = q5_1      for N in 0..47   (640 columns: K-quants refused)
blk.<N>.ffn_down_shexp.weight  = q5_1      for N in 0..47
per_layer_token_embd.weight    = iq4_nl    (the engine accepts only IQ4_NL, Q5_0, Q8_0 or BF16 here)
```

Everything else keeps llama.cpp's Q5_K_M choices (gate/up `q5_K`, attention `q6_K`). The result is
**118,202,878,144 bytes** of GGUF, converted to a **122,228,152,221-byte** `.basalt` with
`tools/convert/convert.py --gguf ... --sections-from BASE.basalt --compat-bf16`.

## Measured, not asserted

`basalt-bench --quality 16` against a KL-divergence base generated from **this model's own F16 weights**
(`out-ablit.gguf`, 354,029,929,664 bytes) over a 16-chunk wikitext-2 test set. Every row below is the same
harness, the same base, the same chunk count; the engine emitted identical warnings for all of them.

| pack | PPL | ln(PPL/base) | KLD mean | same top-1 | bytes |
|---|---|---|---|---|---|
| **this release (down experts q5_1)** | **2.1877** | **0.03224** | **0.11384** | **91.52%** | **122,228,152,221** |
| same model, down experts q8_0 | 2.1984 | 0.03712 | 0.11562 | 91.35% | 128,532,190,671 |
| same model, down experts iq4_nl | 2.2040 | 0.03966 | 0.12574 | 91.27% | 114,663,725,515 |
| F16 reference (the base of these numbers) | 2.1183 | — | — | — | 354,029,929,664 |

Two things worth stating plainly:

- **The iq4_nl compression costs quality here.** An earlier build that used iq4_nl on the down experts
  measured KLD 0.12574 and top-1 91.27%, worse than the higher-precision arms on all three metrics. The
  pack published here is the arm that won that comparison.
- **The q8_0 arm being larger did not make it better** (KLD 0.11562 against 0.11384), which is not the
  intuition higher precision usually gives. The numbers are reported as measured.

## Files

| file | bytes | sha256 |
|---|---|---|
| `qwen38-flash-next-swift15-abliterated-q5km-Q5_K_M.gguf` | 118,202,878,144 | `a66a4d7824262fcd64583b54751016be18cc52d1e3969f2832d2c081679c4071` |
| `qwen38-flash-next-swift15-abliterated-q5km.basalt` | 122,228,152,221 | `8a63c3b22b50b7ac5b7e0b7e5b7cf8fce362f7ccff0cf304b9294767c74ff615` |

The GGUF is the portable form and loads in llama.cpp, LM Studio or ollama. The `.basalt` is the same weights
packed for the Basalt/Lapilli engine. The recipe's `per_layer_token_embd` constraint exists because that
engine refuses q5_1 there; it is harmless in a GGUF reader, so a llama.cpp user could substitute another type
at that tensor without changing anything else.

## What this pack is not

- **Not benchmarked for abliteration.** The refusal direction is a third party's published LoRA; four
  spot probes answered directly with no refusals, and that is the entire behavioural evidence. There is no
  HarmBench/XSTest battery behind this claim.
- **Not validated on hardware other than 2x V100-SXM2 32 GB** with the Lapilli fork.
- **Not 16-bit.** The F16 GGUF is 354 GB and is not published here.

## Licence

The base model's Qwen community licence applies and permits redistribution of derivatives with
attribution; the `base_model` field and `NOTICE` carry that attribution. The two LoRAs keep their own
licences. This repository distributes **quantized weights** — a modified copy, not the original
checkpoint, and the F16 source is not included.
