---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - gguf
  - ik_llama.cpp
  - deepseek
  - vision
  - mmproj
  - dspark
  - speculative-decoding
  - text-generation
---

# DeepSeek-V4.1-Flash-ik-llama-GGUF

Side artifacts for running [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
on [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp): the vision
projector (mmproj) and the DSpark speculative-decoding drafter, plus the
conversion scripts that produced them.

## Files

| File | Size | What |
|---|---|---|
| `mmproj-DeepSeek-V4.1-Flash-Q8_0.gguf` | 494 MiB | Vision projector. `clip.projector_type = deepseek41`, 298 tensors. |
| `dspark-DeepSeek-V4.1-Flash-MXFP4.gguf` | 7.91 GB | DSpark drafter. `dflash` architecture, 78 tensors. |
| `scripts/convert_v41_mmproj.py` | 12 KB | Builds the mmproj from the official checkpoint. |
| `scripts/convert_v41_dspark.py` | 19 KB | Builds the drafter from the official checkpoint. |
| `chat_template.jinja` | 8.3 KB | Chat template tuned for ik_llama.cpp. |

SHA256:

```
cb0fd489778ac5990385055f8b858a29061d73e9bf361fed5d0e30099527184e  mmproj-DeepSeek-V4.1-Flash-Q8_0.gguf
c0b5e80ec876cc106d71d9ff3c4ad43b42271dc407922138953f7104e690dce6  dspark-DeepSeek-V4.1-Flash-MXFP4.gguf
0d46607cfb39956c8e073f7e0732fbd0b6d4a3c2b59a3576292a3aeed7419a64  chat_template.jinja
```

Verify with `sha256sum <file>` against the values above.

## Requirements

The mmproj needs ik_llama.cpp with the `deepseek41` vision projector
([PR #2545](https://github.com/ikawrakow/ik_llama.cpp/pull/2545), merged as
`248e777`). The drafter needs V4.1 DSpark draft loading
([PR #2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522), `c84f76cb`) and
the DSV4 draft SwiGLU limits
([PR #2546](https://github.com/ikawrakow/ik_llama.cpp/pull/2546), `91941904`).

## Usage

Attach the mmproj and (optionally) the drafter:

```bash
llama-server \
  --model          /path/to/DeepSeek-V4.1-Flash-Q8_0.gguf \
  --mmproj         /path/to/mmproj-DeepSeek-V4.1-Flash-Q8_0.gguf \
  --model-draft    /path/to/dspark-DeepSeek-V4.1-Flash-MXFP4.gguf \
  --spec-type      dspark:n_max=5 \
  --chat-template-file /path/to/chat_template.jinja \
  --image-min-tokens 1024 \
  --jinja \
  -amb 512 \
  -ctk f16 \
  -ctv f16 \
  -ngl 99 \
  --cpu-moe \
  --ctx-size 512000 \
  --ctx-size-draft 512 \
  --ctx-checkpoints 16 \
  --swa-compress \
  --batch-size 8192 \
  --ubatch-size 8192 \
  --threads 64
```

## Recipe

Both scripts read the official V4.1-Flash safetensors checkpoint
(`model.safetensors.index.json` + `config.json` + shards) and write a GGUF.
They import `gguf-py` from an ik_llama.cpp checkout, located via the
`IK_LLAMA_DIR` environment variable (defaults to the current directory).

### mmproj

```bash
export IK_LLAMA_DIR=/path/to/ik_llama.cpp
python3 scripts/convert_v41_mmproj.py \
  /path/to/DeepSeek-V4.1-Flash-checkpoint \
  mmproj-DeepSeek-V4.1-Flash-Q8_0.gguf
```

The script re-reads the output, spot-checks a sampled tensor set (blocks
0/15/31, patch embed, aligner biases, sentinels) against the checkpoint,
audits the full tensor inventory, and prints `VERIFY: PASS`. It also runs
standalone with `--verify-only <checkpoint> <mmproj.gguf>`.

### DSpark drafter

```bash
export IK_LLAMA_DIR=/path/to/ik_llama.cpp
python3 scripts/convert_v41_dspark.py \
  /path/to/DeepSeek-V4.1-Flash-checkpoint \
  /path/to/DeepSeek-V4.1-Flash-Q8_0.gguf \
  dspark-DeepSeek-V4.1-Flash-MXFP4.gguf
```

The second argument is the target model's GGUF. The drafter shares the target's
vocabulary, so its tokenizer metadata is copied from that file.

## Notes

- mmproj: 32 vision blocks, hidden 1024, 16 heads, patch 14, aligner
  downsample 3, projection dim 5120. ViT and aligner weights are Q8_0; the
  patch embed, norms, biases and the learned sentinels
  (`v.token_embd.img_start`, `v.token_embd.img_end`, `v.image_newline`) stay
  F32.
- drafter: the `ffn_*_exps` expert tensors (3 per block, 128 experts each)
  are MXFP4 (repacked from the checkpoint's HF nibble order to ik_llama.cpp's
  split-half order); attention, markov, `fc`, shared-expert and `hc_*_fn`
  tensors are Q8_0; `conf_proj` is BF16; `hc_*_base/scale` and the other norms
  are F32.
- The drafter deliberately omits `output_hc_fn/base/scale`. V4's drafter has a
  learned head collapse; V4.1's does not. ik_llama.cpp detects this by the
  absence of `output_hc_base.weight` and sets `dflash_dsv41`.
- The drafter carries no YaRN rope scaling. The V4 drafter GGUF ships
  `rope.scaling.type=yarn` (factor 16), but every V4.1 DSpark stage has
  `compress_ratio == 0`, so the reference builds its frequencies from the base
  rope (`rope_theta=10000`). Inheriting the V4 keys drops acceptance to ~2/55.
- The main model GGUF is not here. A 474 GB single file exceeds the per-file
  limit, so it would need `llama-gguf-split` chunking before upload.

## License and attribution

The base model is MIT licensed. The `LICENSE` file in this repo is the
DeepSeek license copied from
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).