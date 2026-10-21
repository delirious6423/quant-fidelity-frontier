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

# GLM-5.3, streaming repack, EXL3 3.0 bpw experts (uniform K3)

**This is Z.AI's model.** It is a repack of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)
(744B `glm_moe_dsa` MoE: 78 layers plus one MTP layer, 256 routed experts per layer, 8 active per token). The
weights, the architecture and the tokenizer belong to Z.AI and are released under their GLM-5.3 License; the
`LICENSE` file here is theirs. This repack changes only how the weights are stored: no training, no fine-tuning, no
re-quantization.

The routed experts come from an existing EXL3 quantization:
[davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw)
(revision `eeab94eb6e95b4e4d13d94af55ab3c420d6f52d3`): a uniform K = 3 trellis (EXL3, turboderp's format, mcg
codebook, Hadamard input and output transforms `suh` / `svh`), data-free. This repack copies those expert tensors
byte for byte into a different layout; `convert-report.json` records that every record read back equal to its
source (77,824 of 77,824 expert records, 1,024 of 1,024 MTP records).

## What changed and why

- **Routed experts (75 MoE layers × 256 experts), EXL3 3.0 bpw, in `experts.bin`.** Each expert is 4 minis
  (tensor-parallel slices of its gate, up and down projections); each mini is one contiguous record of 3,579,904
  bytes at a 4,096-byte-aligned offset. 76,800 expert records plus 1,024 for the MTP layer: 278.6 GB.
- **MTP layer (78)** in `mtp/`, same form, so the server's MTP draft lane runs the release's own head.
- **Non-expert weights** in MLX 8-bit affine form from pipenetwork's conversion, unchanged in value (20.1 GB).

Against the 3.42 bpw sibling pack on the same machine, the uniform K3 experts are 12 % smaller, which lets about
70 % of each layer's experts stay resident in 256 GB instead of 50 %. Measured on one M5 Ultra (256 GB, internal SSD,
16 GB/s), a 16,387-token coding prompt with 1,024 generated tokens, the server's MTP lane at depth 4:

| | this pack (cache 96 units) | 3.42 bpw pack (cache 85 units) |
|---|---|---|
| decode tok/s | 22.5 to 23.0 | 13.3 to 13.7 |
| prefill tok/s | about 745 | about 720 |
| expert bytes read from SSD per generated token | 0.39 to 0.47 GB | 0.86 GB |

Quality against the 3.42 bpw pack, teacher-forced on 128 positions of two 1,024-token texts (KL over the matched
top-20, an upper bound; the 3.42 pack's own measurement floor is 0.011 to 0.032): code KL 0.0285 mean, +0.041 nats
per token (about +4 % perplexity); prose KL 0.073 mean, +0.066 nats per token (about +7 %). The source card's
expectation for this quantization is a KLD of 0.04 to 0.05 from BF16; this measurement is consistent with it on code
and at or above its top on prose.

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
{"expert_units_cap": 96, "mtp_depth": 4, "mtp_acceptance": "typical", "mtp_typical_delta": 0.2}
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
