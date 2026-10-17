---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
thumbnail: https://huggingface.co/wemoh/Qwen3.8-Flash-Next-strixw/resolve/main/assets/strixite-banner.jpg
tags:
  - strixite
  - strix-halo
  - gfx1151
  - amd
  - rocm
  - hip
  - quantized
  - int4
  - mtp
---

<p align="center"><img src="assets/strixite-banner.jpg" alt="strixite - LLM inference, from scratch, for AMD Strix Halo" width="100%"></p>

# Qwen3.8-Flash-Next - strixw (for strixite on AMD Strix Halo)

|  | on one AMD Strix Halo, 128 GB, with strixite |
|---|---|
| **Decode, real agent use** | **~47-50 tokens/s, flat from 0 to 492k context** |
| **Context** | **512k tokens** |
| **Prefill** | **~1,370 tokens/s** over a real 490k-token conversation |
| **Agent tasks** | **19 / 19** on terminal-bench-mini's core suite |

These are the weights of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), converted to
**strixw**, the weight format of **[strixite](https://github.com/shawnshekari/strixite)** - an inference engine I wrote from scratch in HIP for one chip, AMD's
Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151). They are exactly the files I serve every day.

> **Only strixite can load these files.** strixw is not GGUF or safetensors: every tensor is already in the
> layout of the kernel that reads it. Transformers, llama.cpp, vLLM and others can't use them - for those, start
> from the [original checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
>
> strixite's source code is coming soon to [github.com/shawnshekari/strixite](https://github.com/shawnshekari/strixite).

**Platform:** strixite runs on Linux with an AMD Strix Halo (gfx1151) and 128 GB of memory, tested on Fedora 43.
It's built and tested by one person on one machine, so that's the only setup I can promise works; Windows and macOS
aren't supported, and I don't plan to add them.

## What's in it

The download is ~116 GiB instead of the original checkpoint's ~360 GB, and needs no conversion step.

| file | size | what |
|---|---|---|
| `converted/Qwen3.8-Flash-Next.U-gdn_in-g128/weights.strixw` | 65.9 GiB | the model: 4-bit routed experts, 8-bit dense projections, MTP head included |
| `converted/Qwen3.8-Flash-Next.U-gdn_in-g128/vision.strixw` | 0.84 GiB | the vision tower (27-layer ViT + merger, unquantized) for image input - strixite v0.3 and later |
| `converted/Qwen3.8-Flash-Next.ngram-q8/ngram.table` | 48.3 GiB | the per-layer n-gram embedding table (int8 rows); read from SSD on demand, not loaded into memory |
| `Qwen3.8-Flash-Next/tokenizer.json`, `generation_config.json`, `config.json`, `preprocessor_config.json` | small | copied unchanged from the original checkpoint |
| `Qwen3.8-Flash-Next/LICENSE`, `LICENSE` | small | the Qwen Community License 1.0 |
| `sha256.txt` | small | checksums of every file above |

The folder layout matches strixite's default model directory (`~/models/strix-infer` - the engine's working name),
so downloading there needs no configuration:

```sh
hf download wemoh/Qwen3.8-Flash-Next-strixw --local-dir ~/models/strix-infer
cd ~/models/strix-infer && sha256sum -c sha256.txt
```

## Quantization

Asymmetric integer quantization per group along the input dimension (a BF16 scale and a BF16 minimum per group),
chosen class by class from a quantization study against the FP32 reference outputs:

| weights | format |
|---|---|
| routed experts (gate / up / down), GDN input projections | 4-bit, groups of 128 |
| every other dense projection: attention q / k / v / o, GDN output, shared expert, hyper-connection mixes, PLE key / value, LM head | 8-bit, groups of 64 |
| MTP head's input projection | 4-bit, groups of 64 |
| router | BF16 (its logits are computed in FP32) |
| token embedding | BF16 |
| n-gram table | int8 per row, with a per-row scale |
| norms, gates, conv kernels, small vectors | FP32 / BF16 as in the checkpoint |

Most of the model's parameters are the 512 routed experts, of which each token reads only a few; the dense
projections are read for every token. Keeping the dense path at 8 bits and the experts at 4 is what keeps accuracy
close to the original at ~4.25 bits per routed weight.

Image input (strixite v0.3 and later, on by default) uses `vision.strixw` and `preprocessor_config.json`; if you
downloaded before they were added, run the same `hf download` again - it fetches only the new files.

## Accuracy and speed

All measured on my machine (Strix Halo, 128 GB) with strixite, 2026-09-28 .. 2026-10-03.

- **Accuracy vs the unquantized model:** over 242 positions of three reference prompts, mean KL divergence from the
  FP32 outputs 0.028, and the top-1 token agrees at 226 of 242 positions.
- **Agent tasks:** terminal-bench-mini's core19 suite, 19 of 19 tasks passed.
- **Decode:** ~33 tokens/s plain and 45-55 tokens/s with the built-in multi-token prediction (MTP) at 2k-131k context;
  in real agentic use it stays at ~47-50 tokens/s out to 492k tokens of context.
- **Prefill:** ~1,370 tokens/s over a real 489,566-token agent conversation.
- **Context:** up to 512k tokens (YaRN factor 2 in the engine; the model was trained to 262,144).

## Memory and storage

- ~66 GiB of weights resident in GPU memory (Strix Halo's unified memory - GTT, no VRAM carve-out needed), plus up to
  ~13 GiB of KV cache at 512k context.
- The n-gram table (48.3 GiB) stays on disk and is read row by row as tokens need it, through a cache in RAM: put it
  on a fast NVMe drive.

## How it was made

- Source: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540` (BF16 safetensors).
- `weights.strixw`: strixite's converter (`convert_qwen4exp`, commit `eb57cdb` of my private development repo),
  strixw format version 4, layout
  `exp_gu=q4g128,exp_down=q4g128,sh_gu=q8g64,sh_down=q8g64,gdn_in=q4g128,gdn_out=q8g64,attn_qkv=q8g64,attn_o=q8g64,hc_down=q8g64,hc_up=q8g64,ple_kv=q8g64,lm_head=q8g64,mtp_fc=q4g64,embed=bf16`.
- `ngram.table`: strixite's `convert_ngram_table` (the checkpoint's 128 table shards into one BF16 file), then
  `transcode_ngram_table --dtype q8` (int8 rows with a BF16 per-row scale; median row error 0.66% of the BF16 row).
- `vision.strixw`: the same converter with `--tower-only` (commit `83e0ce9`), the checkpoint's vision tower
  unquantized (BF16, 0.45B parameters) - 0.84 GiB in memory while serving, plus working buffers sized per image.
- The converters will be part of the strixite release, so anyone can rebuild these files from the original
  checkpoint and check them.

strixite started as "how hard can it be?" It turned out an engine that holds its own against a closed-source one -
faster at decode, on its home turf - is within reach of one person, with a good model, a good coding assistant, and a
lot of patience. I hope it encourages you to grab your favorite LLM and framework and build something from scratch.

## Limitations

- Only for strixite, and only for Strix Halo (gfx1151) with 128 GB.
- Quantization moves the outputs slightly away from the original model (numbers above); with sampling on, text
  differs from what the BF16 model would write.
- Images yes (PNG, JPEG, WebP, GIF, BMP, TGA, up to ~2.1 megapixels after resizing), video no.
- Everything the original model card says about the model's own behaviour, limits and intended use applies here too.

## License

These weights are a derivative of Qwen3.8-Flash-Next and are distributed under the
[Qwen Community License 1.0](LICENSE), the same license as the original - its full text is in `LICENSE`. Note its
terms for commercial use: a "Model as a Service" or "AI Work Assistant" business needs a separate license from
Qwen, and products above 100 million monthly active users or US$20 million monthly revenue must display the model
name. See the license text for the exact wording.

Copyright for the original model belongs to Qwen. The conversion is mine; the strixite code is a separate project
under its own license (AGPL-3.0).
