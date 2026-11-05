---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: halogen
# .hgn weights cannot be loaded by the Hub widget or by transformers/vLLM/llama.cpp.
inference: false
language:
- en
- zh
tags:
- hgn
- halogen
- halogen-flash
- qwen
- qwen3_8
- moe
- quantized
- 4-bit
- strix-halo
- gfx1151
---

# Qwen3.8-Flash-Next (HGN v2, from Unsloth UD-IQ4_XS)

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) converted from [Unsloth's `UD-IQ4_XS` GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) into the **HGN v2** weight format used by [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) (AMD Strix Halo, gfx1151).

> [!NOTE]
> **When to use this.** peonist-ai publishes its own HGN v2 checkpoint of this model, converted from the BF16 weights: [peonist-ai/halogen-qwen3.8-flash-next](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next). That is the better choice for running the stock model on halogen. This repository is a reference conversion from an already-quantized GGUF, with the extra rounding step that implies (see "Precision" below).

**These weights are in `.hgn` format and will not load in transformers, vLLM or llama.cpp.** They are meant to be mounted into the halogen-flash-server container.

## Status (read this first)

This is a community conversion. It has not been qualified by the authors of any upstream project.

| Check | Result |
|---|---|
| Conversion (`bf16-to-hgn convert`, profile `v2`, GGUF input) | Completed and published: 1,658 source tensors to 2,349 output tensors, manifest included |
| `bf16-to-hgn verify` (source hashes, layout, decoded values) | Not yet recorded |
| Loads and generates in halogen-flash-server | Not yet recorded |
| Speed and quality benchmarks | Not yet recorded |
| CUDA output compared byte-for-byte with the CPU backend | Not done. The converter documents CUDA as qualified on sm_89; this run used sm_86 (RTX 3090) |

If you test it, please report results on the discussion tab.

## Files

| File | Size | Purpose |
|---|---|---|
| `model.hgn` | about 62.1 GiB | Checkpoint, identity `qwen3.8-flash-next-v2`. Includes the MTP draft head |
| `qwen38-flash-next-ngram.hgn` | about 47.7 GiB | The 51B-parameter n-gram lookup table (FP8). **Required; keep beside `model.hgn`** |
| `qwen38-flash-next-vision.hgn` | about 0.84 GiB | Vision tower (BF16). Optional; without it the engine is text-only |
| `config.json` | | Model configuration |
| `tokenizer/` | | Flat tokenizer directory (also contains `config.json`) |
| `conversion-manifest.json` | | Converter manifest: policy, spec version and per-file details |
| `fix_gguf_types.py` | | The script used to prepare the GGUF (see "How it was made") |
| `LICENSE` | | Qwen Community License 1.0, which must accompany copies of the weights |

Total about 111 GiB. The model needs about 62 GiB resident plus the memory-mapped n-gram table.

## How it was made

```
Qwen/Qwen3.8-Flash-Next                       base model
  -> unsloth/Qwen3.8-Flash-Next-GGUF          Unsloth Dynamic UD-IQ4_XS (4 shards)
  -> this repo                                HGN v2 via bf16-to-hgn
```

1. **Merge the shards.** Unsloth's GGUF is split with an empty first shard (metadata only), which `bf16-to-hgn` rejects, so the four shards were merged into one file with llama.cpp's `llama-gguf-split --merge` (1,224 tensors).
2. **Rewrite unsupported tensor types.** The UD-IQ4_XS recipe mixes quantization types per tensor, and it includes types the converter cannot read: IQ3_XXS and Q2_0. [`fix_gguf_types.py`](fix_gguf_types.py) dequantizes every tensor in a type the converter does not support and stores it as F16; all other tensors and all metadata are copied unchanged, and the `split.*` keys are dropped. It needs the `gguf` package from a recent llama.cpp checkout (the PyPI release did not know Q2_0).
3. **Convert.** [Quat3rnion/bf16-to-hgn](https://github.com/Quat3rnion/bf16-to-hgn) with profile `v2`, policy `bf16-hgn-v2-rtn-nf4-signed-rms-trellis-v1`, `--ht-beam 128`, seed 0, format per [hgn-spec](https://github.com/jtsylve/hgn-spec) 1.1.0. CUDA backends on an RTX 3090; the conversion passes (231.2G values) took 37 minutes 48 seconds.

- **MTP head and vision tower:** the converter takes these as native HGN sidecars rather than from the GGUF. Both come from peonist-ai's stock checkpoint: the 31 `mtp.*` tensors were copied byte-for-byte from `qwen3.8-flash-next-v2.hgn` into a standalone sidecar, and the vision tower is peonist-ai's `qwen38-flash-next-vision.hgn`. Because this model is the stock trunk, the stock head is the matching head. The converter then re-encoded the head's three Q4C-P tensors, which carries about 0.8–0.9% error energy relative to the stock head in earlier conversions with this setup.
- **Assets:** the stock Flash-Next `config.json` and tokenizer files.

Reproduce:

```bash
llama-gguf-split --merge Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf Qwen3.8-Flash-Next-UD-IQ4_XS-merged.gguf
python fix_gguf_types.py Qwen3.8-Flash-Next-UD-IQ4_XS-merged.gguf Qwen3.8-Flash-Next-UD-IQ4_XS-fixed.gguf

bf16-to-hgn convert --source-format gguf --source Qwen3.8-Flash-Next-UD-IQ4_XS-fixed.gguf \
  --assets assets --mtp mtp.hgn --vision qwen38-flash-next-vision.hgn \
  --profile v2 --output Qwen3.8-Flash-Next-UD-IQ4_XS-HGN --threads 8 --ht-beam 128 \
  --ht-backend cuda --i4r-backend cuda --q4cp-backend cuda
```

## Precision: what this is and is not

This is **not** a quantization from BF16. The source is Unsloth's dynamic GGUF, which spends different numbers of bits on different tensors, from 2-bit (Q2_0) and 3-bit (IQ3_XXS) up to higher-precision formats on sensitive tensors. The converter decodes those values and re-encodes them into HGN v2, a 4-bit-class format: I4R routed experts, HT4 trellis dense matrices and LM head, Q6G64 hyper-connection mixers and Q8G64 for the MTP head's dense projections, with the router, norms and vision tower kept in BF16.

- The per-tensor bit allocation of the GGUF is not carried over as a format choice, only the values it produced. Information lost by the GGUF's 2-bit and 3-bit tensors is not recovered, and tensors the GGUF kept at higher precision are reduced to the HGN formats.
- The tensors stored as F16 in step 2 are the exact decoded values of the original blocks up to F16 rounding (about 0.05% relative), which is far smaller than the quantization error of any of these formats.
- No perplexity or benchmark comparison against the GGUF or against peonist-ai's BF16-derived checkpoint has been run for this conversion.

## Use with halogen-flash-server

Requires an AMD Strix Halo machine (gfx1151) with enough unified memory for roughly 62 GiB of weights plus the memory-mapped table, and a halogen-flash-server release of 0.15 or later. Mount the downloaded directory read-only, as for any halogen checkpoint:

```bash
hf download johnlockejrr/unsloth-Qwen3.8-Flash-Next-HGN --local-dir ~/halogen-models

podman run --rm -p 8731:8731 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ipc=host --ulimit memlock=-1:-1 \
  -e HALOGEN_CHECKPOINT=/models/model.hgn \
  -e HALOGEN_VISION_TOWER=1 \
  -v ~/halogen-models:/models:ro \
  ghcr.io/peonist-ai/halogen-flash-server:<version>
```

Notes:
- Replace `<version>` with a current server release. `HALOGEN_VISION_TOWER=1` is optional and only needed for images.
- `HALOGEN_CHECKPOINT` points at `model.hgn` by name because the checkpoint is not named `qwen38-flash-next-v2.hgn`. This checkpoint has not been loaded in the engine yet. With the earlier v2 conversions from the same converter, server 0.18.22 found the n-gram table beside `model.hgn` without further settings.
- On Docker, replace `--group-add keep-groups` with `--group-add video --group-add render`.
- The server exposes an OpenAI-compatible API on port 8731.

## Model overview

Qwen3.8-Flash-Next is an experimental preview of the architecture that will underpin Qwen4. Highlights from the Qwen team:

- **Hybrid attention with QSA:** Gated DeltaNet paired with Qwen Sparse Attention, which operates at the micro-block level rather than selecting individual tokens, cutting long-context latency.
- **Gated Residual:** an element-wise, data-dependent read gate and a per-branch scalar write gate on widened residual streams.
- **N-gram embedding:** scales parameters through short n-gram lookups, which is cheaper than MoE and easier to offload; the lookup table is the separate `qwen38-flash-next-ngram.hgn` file here.
- **Training recipe:** Muon and AdamW applied to specific weight categories, with no batch-size warmup.

| | |
|---|---|
| Type | Causal language model with vision encoder |
| Parameters | 125B with 6B activated, plus a 51B n-gram embedding and 4B MTP |
| Layers / hidden size | 48 layers, hidden dimension 2560 |
| Layout | 12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE)) |
| Experts | 512, with 10 routed and 1 shared active |
| Gated residual | 4 branches, bottleneck rank 320 |
| MTP | 1 layer |
| Context | 262,144 tokens natively, extensible up to 1,000,000 |

Benchmark results are published by the Qwen team on the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and in their [blog post](https://qwen.ai/blog?id=qwen3.8-flash-next) and [technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf). **Those numbers describe the original unquantized model and do not describe this conversion.**

## Best practices

From the Qwen team's recommendations (server support for individual parameters varies):

- **Thinking mode (default):** `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`.
- **Instruct (non-thinking) mode:** `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`.
- **Output length:** allow long outputs for agentic tasks. Qwen suggests up to 262,144 tokens for reasoning and 131,072 for the final response where the framework supports separate limits.
- **Thinking controls:** the model thinks by default; the chat template supports `enable_thinking`, `preserve_thinking` (on by default, keeps earlier thinking blocks) and `reasoning_effort`. Qwen's card has examples for OpenAI-compatible clients.
- **Ultra-long texts and long video:** Qwen's card describes YaRN rope scaling and video preprocessor settings for other frameworks. Check the halogen-flash-server documentation for what it supports.

For reference, the halogen-flash-server 0.18.22 log shows request defaults of `temperature=1.0`, `reasoning_effort=medium` and, when a sampled request omits them, `top_k=20` and `top_p=0.95`.

## Credits

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team
- GGUF quantization: [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (Dynamic UD-IQ4_XS)
- HGN format specification: [jtsylve/hgn-spec](https://github.com/jtsylve/hgn-spec)
- Converter: [Quat3rnion/bf16-to-hgn](https://github.com/Quat3rnion/bf16-to-hgn)
- Engine, stock HGN checkpoint, MTP head and vision tower: [peonist-ai](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next) ([halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server))
- HGN conversion: johnlockejrr

This repository is an independent community conversion and is not affiliated with or endorsed by Qwen, Unsloth, Peonist or any other project above.

## Citation

If you find the model useful, please cite the Qwen team's work:

```bibtex
@techreport{qwen2026design,
    title       = {On the Design of {Qwen3.8-Next} Architecture: Evaluation, Efficiency, and Training Stability},
    author      = {{Qwen Team}},
    institution = {Alibaba Group},
    month       = {August},
    year        = {2026}
}

@misc{qwen3.8flashnext,
    title  = {{Qwen3.8-Flash-Next}: A New Architecture, Towards Ultimate Cost-Efficiency},
    author = {{Qwen Team}},
    month  = {August},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.8-flash-next}
}
```

## License

The weights are a derivative of Qwen3.8-Flash-Next and are provided under the **Qwen Community License 1.0**; the full text is in `LICENSE` and its copyright and permission notice must be kept with all copies. As the license states, commercial products or services with more than 100,000,000 monthly active users or US$ 20,000,000 monthly revenue must prominently display the model name in the user interface, and a licensee that conducts a Model as a Service or AI Work Assistant business needs a separate license from Qwen before using the Software or its derivatives for any commercial purpose. This summary is not legal advice; read `LICENSE` and verify the terms yourself before commercial use.

The halogen-flash engine is separate software distributed under its own terms. "halogen" is a trademark of Peonist, LLC; see `TRADEMARKS.md` in the server repository for permitted use.
