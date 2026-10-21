---
license: other
license_name: glm-5.3
license_link: LICENSE
base_model: zai-org/GLM-5.3
base_model_relation: quantized
library_name: mlx-serve
pipeline_tag: text-generation
tags:
- glm_moe_dsa
- mlx
- mlx-serve
- mlx-stream
- exl3
- streaming
- moe
---

# GLM-5.3, streaming repack, EXL3 3.42 bpw experts (mixed K3/K4)

**This is Z.AI's model.** It is a repack of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)
(744B `glm_moe_dsa` MoE: 78 layers plus one MTP layer, 256 routed experts per layer, 8 active per token). The
weights, the architecture and the tokenizer belong to Z.AI and are released under their GLM-5.3 License; the
`LICENSE` file here is theirs. This repack changes only how the weights are stored: no training, no fine-tuning, no
re-quantization.

The routed experts come from an existing EXL3 quantization:
[davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw)
(revision `99c6f951333d2b38f1efefa533c7afadf0d376e3`): a mixed K = 3 / K = 4 trellis (EXL3, turboderp's format,
mcg codebook, Hadamard input and output transforms `suh` / `svh`), where each layer's experts are split between K3
and K4 by the source's sensitivity ranking. This repack copies those expert tensors byte for byte into a different
layout; `convert-report.json` records that every record read back equal to its source (77,824 of 77,824 expert
records, 1,024 of 1,024 MTP records).

## What changed and why

- **Routed experts (75 MoE layers × 256 experts), EXL3 3.42 bpw, in `experts.bin`.** Each layer has two bank
  layers, one for its K3 experts and one for its K4 experts; each expert is 4 minis (tensor-parallel slices of its
  gate, up and down projections) and each mini is one contiguous record of 3,579,904 bytes (K3) or the K4 size at
  a 4,096-byte-aligned offset. 317.3 GB of expert records.
- **MTP layer (78)** in `mtp/`, same form, so the server's MTP draft lane runs the release's own head.
- **Non-expert weights** in MLX 8-bit affine form from pipenetwork's conversion, unchanged in value (20.1 GB).

Measured on one M5 Ultra (256 GB, internal SSD, 16 GB/s), a 16,387-token coding prompt with 1,024 generated tokens,
the server's MTP lane at depth 3, expert cache of 85 units: decode 13.3 to 13.7 tok/s, prefill about 720 tok/s,
0.86 GB of expert bytes read from SSD per generated token (about 50 % of each layer's experts resident). The 3.0 bpw
sibling pack
([OpensourceWTF/GLM-5.3-streaming-repack-exl3-3.0bpw](https://huggingface.co/OpensourceWTF/GLM-5.3-streaming-repack-exl3-3.0bpw))
decodes about 65 % faster on the same machine for about +4 % perplexity on code and +7 % on prose relative to this
pack; this pack is the higher-quality choice.

## What it is for

This pack is served by [mlx-serve](https://github.com/ddalcu/mlx-serve) with the
[mlx-stream](https://github.com/davidtai/mlx-stream) plugin (`glm_moe_dsa` arch). The trunk, the shared experts,
the routers and the MTP layer's residents load once. The routed experts the router picks stream from `experts.bin`
into a bounded slot cache on the GPU, and EXL3 Metal kernels run them. A Mac with 256 GB of memory serves the model
this way although it cannot hold all the experts. Other loaders (Transformers, vLLM, ExLlamaV3, mlx-lm) do not read
this layout; use the sources linked above for those.

The expert cache keeps a per-expert request-rate tally in `~/.mlx-stream/glm_moe_dsa-<pack dir>/expert-tally.bin`,
written at the end of each request and read at load; the first request of a fresh install plans without it.

## How to run

Build mlx-serve with the mlx-stream plugin, following the plugin's README. Download this repo to a local directory
(the `experts.bin` sidecar is one file; keep the directory layout), then start:

```sh
mlx-serve --serve --model /path/to/<this repo> --port 8080 --ctx-size 16448 --wired-margin-gib 2
```

Per-model settings used for the numbers below (`~/.mlx-serve/model-settings.json`, keyed by the model path):

```json
{"expert_units_cap": 85, "mtp_depth": 4, "mtp_acceptance": "typical", "mtp_typical_delta": 0.2}
```

`expert_units_cap` bounds the expert cache in units of 2,147,328,000 bytes; set it to what your machine's free memory
allows (the server refuses a load that does not fit). `mtp_depth` 0 turns the MTP draft lane off.

## Files

- `experts.bin`: the routed experts, one record per (layer, expert, mini), 4,096-byte aligned; `expert-manifest-exl3-v1.json`
  gives the format, the source, every layer's record layout and every record's offset.
- `mtp/`: the MTP layer (layer 78) in the same form (`mtp-experts.bin`, `mtp-manifest-exl3-v1.json`,
  `mtp-residents.safetensors`), plus `mtp-residents.safetensors` at the top level.
- `model-0000N-of-00004.safetensors` + `model.safetensors.index.json`: the non-expert weights (attention, indexer,
  routers, shared experts, embeddings, head, norms) in MLX 8-bit affine form, taken unchanged from
  [pipenetwork/GLM-5.3-MLX-mixed-4_8bit](https://huggingface.co/pipenetwork/GLM-5.3-MLX-mixed-4_8bit)
  (revision `e1034d395c26b9bcd3631df783e65737701ff2d6`), which converted them from the BF16 release.
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja`, `LICENSE`:
  the release's files (the config carries the MLX `quantization` block for the residents).
- `convert-report.json`: the build's record counts and the byte-for-byte verification of every record against the source.

## License

GLM-5.3 is released by Z.AI under the GLM-5.3 License; `LICENSE` is their file. This pack adds no training and
changes no weight values; it changes only how the weights are stored.
