---
license: other
license_name: glm-5.3
base_model: zai-org/GLM-5.3
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - glm-5.3
  - glm-5
  - nvfp4
  - modelopt
  - sglang
  - blackwell
  - moe
---

# GLM-5.3-NVFP4

**TL;DR:** [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), the 753B-total,
~40B-active GLM MoE sparse-attention model, compressed from 1506.7 GB BF16 to 465 GB with
NVIDIA ModelOpt NVFP4 routed-expert weights. Built for SGLang and vLLM on RTX PRO 6000
Blackwell (SM120) and other NVFP4-capable hardware.

This is a pure weight quantization of the official GLM-5.3 BF16 release. GLM-5.3 keeps the
GLM-5.2 base architecture (78 transformer layers, 256 routed experts per MoE layer, DeepSeek-style
sparse attention with a 2048-token top-k indexer, 1,048,576-token context). The conversion
follows the same experts-only recipe as the official nvidia/GLM-5.2-NVFP4 release.

## Facts and figures

| | |
|---|---|
| Source checkpoint | [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), revision `304b8051cfb2b260b61ce0cbe330e02a98e73639` |
| Method | ModelOpt NVFP4, routed expert gate/up/down weights only, block size 16, two-level scaling, max algorithm, no calibration |
| Precision retained | BF16 attention, indexer, router, dense and shared experts, embeddings, LM head, layer 0, and the layer 78 MTP/nextn block |
| KV cache | FP8 declared in `hf_quant_config.json` (runtime BF16 recommended, see below) |
| Size | 465 GB tensor bytes (433 GiB on disk), down from 1506.7 GB BF16, about 69% smaller |
| Export layout | 163 safetensors shards, 232,385 indexed tensors, 19,200 quantized expert bundles (layers 3-77) |
| Conversion | 292 seconds with 3 GPU workers, streamed shard by shard on 3x RTX PRO 6000 Blackwell 96 GB |
| Context length | 1,048,576 tokens (unchanged from the source config) |
| ModelOpt revision | `0.46.0.dev65+g977d34dc3` (`NVFP4QTensor`) |
| Serving validation | Not yet performed on this export. The identical architecture is proven in production on 7x RTX PRO 6000 Blackwell with nvidia/GLM-5.2-NVFP4 (see below) |

## Quantization recipe

The conversion streams every source shard and quantizes all 57,600 routed-expert projection
tensors (75 MoE layers, 256 experts each, gate/up/down). It uses ModelOpt's `NVFP4QTensor` max
quantizer with 16-value blocks. `gate_proj` and `up_proj` share the second-level FP32 global
scale (fused gate/up runtime layout), `down_proj` gets its own. The exported weights use packed
U8 data, FP8 E4M3 block scales, and FP32 global scales. No calibration dataset was used; the
quantization is weight-only max scaling, which matches the layout served by SGLang and vLLM as
`modelopt_fp4`.

Everything else stays BF16: attention (including the DSA indexer), the router, dense layers 0-2,
shared experts, embeddings, the LM head, and the complete layer 78 MTP/nextn draft block.
The exclude-module list in `hf_quant_config.json` mirrors the official nvidia/GLM-5.2-NVFP4
release (158 entries).

## Run it with SGLang

The proven recipe for this architecture on 7x RTX PRO 6000 Blackwell 96 GB (SM120), taken from
the live nvidia/GLM-5.2-NVFP4 deployment:

```bash
python3 -m sglang.launch_server \
  --model-path GLM-5.3-NVFP4 \
  --served-model-name GLM-5.3-NVFP4 \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 --pp-size 7 \
  --quantization modelopt_fp4 \
  --trust-remote-code --dtype auto \
  --context-length 1048576 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 12288 \
  --attention-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cutlass \
  --disable-flashinfer-autotune \
  --disable-custom-all-reduce \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --sleep-on-idle
```

Pipeline-parallel partition on 7 GPUs: `10,12,12,12,12,12,8` (required by the DSA
`index_topk_freq=4` / `index_skip_topk_offset=3` layer pattern). BF16 KV cache is strongly
recommended for this architecture: the FP8-KV path lacks per-model scaling factors and measured
75% lower decode throughput on the GLM-5.2 sibling deployment.

This exact export has not yet been smoke-tested in SGLang; the command above is the validated
GLM-5.2-NVFP4 production profile and is expected to carry over unchanged.

## Run it with vLLM

```bash
vllm serve GLM-5.3-NVFP4 \
  --quantization modelopt \
  --kv-cache-dtype auto \
  --max-model-len 131072 \
  --tool-call-parser glm45
```

Not yet validated on this export. Start with a reduced context before scaling up.

## Quality and limitations

- No benchmark run has been performed on this export yet. The GLM-5.2 evidence base (same
  architecture, official NVFP4 quant of the same base) showed NVFP4 experts-only quantization at
  or under 1% on common benchmarks; treat that as the expectation, not a measurement.
- `hf_quant_config.json` declares `kv_cache_quant_algo: FP8` for loader compatibility with the
  official NVIDIA GLM-5.2 release, but serve with BF16 KV cache (see the SGLang note above).
- MTP speculative decoding weights (layer 78) are preserved in BF16 but the SGLang MTP path for
  this architecture requires a patched fork; serve without speculative decoding on stock builds.
- NVFP4 requires Hopper/Blackwell-class GPUs (compute capability 9.0+, SM120 validated).
  Ampere and older cannot run this checkpoint.

## Provenance and reproducibility

Converted on 2026-08-29 from `zai-org/GLM-5.3-BF16` revision
`304b8051cfb2b260b61ce0cbe330e02a98e73639` with a streaming experts-only ModelOpt NVFP4
conversion (recipe `streaming/nvfp4_experts_only_input_scale1-kv_fp8-glm_moe_dsa`):
`NVFP4QTensor.get_weights_scaling_factor` / `quantize`, block size 16, input scale 1.0,
shared gate/up global scale, max algorithm, zero calibration samples. Full run metadata is in
`run-manifest.json` inside the repo.

## License and credits

GLM-5.3 License (Copyright (c) 2026 Z.AI), inherited from
[zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3). Quantization and validation by
**Robert Ressl**
([Hugging Face](https://huggingface.co/ressl) · [Website](https://ressl.ch/) · [LinkedIn](https://www.linkedin.com/in/robertressl/) · [Patreon](https://www.patreon.com/cw/ressl)).
Built with the NVIDIA TensorRT Model Optimizer and SGLang.

**❤️ Support this work:** if these models are useful to you, consider supporting on
[Patreon](https://www.patreon.com/cw/ressl), more at [ressl.ch](https://ressl.ch/).
