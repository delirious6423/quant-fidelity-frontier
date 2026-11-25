---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
  - orcarouter/Qwen3.8-Flash-Next-Uncensored
  - jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - qwen
  - qwen3.8
  - flash-next
  - uncensored
  - nvfp4
  - gptq
  - strata
  - moe
  - mtp
  - vision-language
---

# Qwen3.8 Flash-Next Abliterated — NVFP4 + GPTQ experts, for Strata

A ready-to-run [Strata](https://github.com/Niko1221/Strata) model folder (NVFP4 build: [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4)) for OrcaRouter's abliterated (uncensored) Qwen3.8 Flash-Next.
The MoE experts are **re-quantized with GPTQ from the BF16 checkpoint**, so they round better than a
straight NVFP4 conversion. Everything else is the same as Strata's normal NVFP4 build.

> **Strata only.** These files are Strata's own pack layout and GGUFs. They do not load in vLLM,
> SGLang, transformers or stock llama.cpp.

## What's different from the plain NVFP4 pack

| | Plain NVFP4 pack | This repo |
|---|---|---|
| Experts | NVFP4 from the ModelOpt checkpoint, repacked without loss | Re-quantized from **BF16** with GPTQ, using real activations |
| Everything else | — | Identical |
| Expert pack size | 63 GB | 63.3 GB (67,948,118,016 bytes) |

### How it was made

1. Downloaded the ModelOpt NVFP4 checkpoint
   ([jpezzulli/…-ModelOpt-NVFP4](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4))
   and ran Strata's `prepare-model` steps: PLE table (FP8), token embedding (BF16), vision mmproj (F32),
   the native GGUF, the expert pack, and the fine-tune's own MTP draft head.
2. Downloaded the BF16 original
   ([orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored), ~360 GB).
3. Calibration: ran Strata on four texts (`uk`, `en`, `code`, `chat` from Strata's `data/requant_calib/`)
   with `STRATA_DUMP_MOE_INPUT` set, dumping the inputs every MoE layer actually sees.
4. `tools/requant.py pack --method gptq --bf16 <bf16> --calib <dumps> --base pack --out pack-gptq`
   (Strata 0.1.40.2, NVFP4 build).

No fine-tuning, merging or further uncensoring was done. The behaviour is OrcaRouter's model.

### Quality

Measured by the strata-nvfp4 project on this model (OrcaRouter's fine-tune) and this pack type
(GPTQ-only, 63.3 GiB experts), from
[docs/NVFP4.md](https://github.com/sergqwer/strata-nvfp4/blob/main/docs/NVFP4.md):

- **Reference:** an all-Q8_0 pack built from the same BF16 checkpoint (119.5 GiB).
- **Test:** 12 prompts (Ukrainian, English, code; half with thinking), none of them in the calibration
  set. The reference's answers are teacher-forced and the KL divergence is taken over the answer
  positions (top 64 logits), per prompt. The metric's own noise floor is 0.0017.

| Pack | Experts | Median answer KL | vs ModelOpt | Chat decode |
|---|---:|---:|---:|---:|
| ModelOpt NVFP4 (plain pack) | 63.3 GiB | 0.0107 | | 142 tok/s |
| **GPTQ NVFP4 (this repo)** | **63.3 GiB** | **0.0070** | **65% (better in 12/12 prompts)** | **143 tok/s** |

- First token after a 32K prompt (full vocabulary): KL 0.0277 → **0.0137** with GPTQ.
- Same size and same speed as the plain pack; only the rounding is better.

These figures are against a Q8_0 reference, not BF16, so they are not directly comparable to KLD
tables measured against BF16 on other texts.

## Files

| Path | Size | What it is |
|---|---|---|
| `pack-gptq/` | 65 GB | GPTQ expert pack (`experts.bin`), dense weights, index, tokenizer |
| `orca-nvfp4.gguf` | 69 GB | Native GGUF (dense layers, also used by the vision server) |
| `ple-fp8.gguf` | 48 GB | Per-layer n-gram (PLE) table, FP8 as shipped |
| `token-embd-bf16.gguf` | 1.2 GB | Token embedding, BF16 |
| `mmproj-f32.gguf` | 1.7 GB | Vision encoder |
| `mtp/rt/` | 0.8 GB | The fine-tune's MTP draft head (q2_0 experts) |
| `strata-config.json` | — | Launch config (paths to fill in) |

Total ≈ 186 GB.

## Running it

Download (resumable):

```bash
hf download Cracked-ANJ/Qwen3.8-Flash-Next-Abliterated-NVFP4-GPTQ-Strata --local-dir ./orca-gptq
```

Open `strata-config.json`, replace `MODEL_DIR` with that folder and `STRATA_DIR` with your Strata install,
and start Strata with it. The config is the one I use daily: 384k context (YaRN 1.5), int8 KV, MTP with
4 draft tokens, vision on, on a 32 GB GPU with 64 GB RAM.

A lighter setup (this is what the benchmark run used): drop `--rope-scaling`/`--rope-scale`,
`--kv-resident`, `--pool-workers`, `--prompt-cache`, `--pcie-frac` and `--vram-reserve-mib`, and use

```
--spec 4 --spec-min-p 0.5 --mtp MODEL_DIR/mtp/rt --kv int8 --max-context 262144
--expert-cache auto --prefill auto --vision --conversation-cache-mib 4096 --conversation-cache-slots 3
```

The model thinks before it answers; give requests a generous `max_tokens`, or start Strata with
fit-max-tokens so a limit larger than the context left is shortened instead of refused.

## Licence and use

The files are under the **Qwen Community License 1.0** (see `LICENSE`), as are the base model and the
NVFP4 checkpoint this was built from. OrcaRouter's model card says Apache-2.0, but the LICENSE it ships is
the Qwen Community License, so that is what this repo follows.

This is an **uncensored** model. As OrcaRouter states, it is released for legitimate research; do not
deploy it to end users without adding your own safety layer, and you are responsible for how you use it.

## Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for the base model
- [OrcaRouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) for the uncensored fine-tune
- [jpezzulli](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4) for the ModelOpt NVFP4 checkpoint
- [Strata](https://github.com/Niko1221/Strata) and [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) for the engine, the pack format and the GPTQ re-quantizer
