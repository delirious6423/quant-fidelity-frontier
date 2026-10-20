---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - eaq
  - quantized
  - experimental
  - moe
  - deepseek
  - multimodal
---

# DeepSeek-V4.1-Flash-1017 · EXL3 · EAQ v1 · 2.9 bpw

EAQ v1 (EXL3 format) quantization of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision `1bae0e10`, the refreshed V4.1-Flash weights), converted with exllamav3 `v1.6.1` plus the EAQ patch set (converter version 1.6.1) and a port of the DeepSeek-V4.1 architecture (see Conversion). **Quality versus the source model is NOT MEASURED**; only structural checks and a smoke test were done (see Conversion).

This is **not** a drop-in replacement for the original FP8/FP4 checkpoint. The Engram embedding tables are **not** in this repo: you still need shards 47 and 48 from the base model (about 95 GiB each).

**Format: plain EXL3 (EAQ v1).** EAQ v1 currently uses the standard EXL3 format unchanged: `quant_method` is `exl3`, codebook `mul1`, `out_scales=always`, and the tensor keys and the main-model per-tensor bit widths are identical to the standard EXL3 2.9 bpw builds of this model (checked against [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw): 192,495 tensor keys; the widths of all 46,702 main-model quantized tensors match; tokenizer files byte-identical). The 46 MTP / DSpark dense tensors differ: 4 bit here, 6 bit in the LibertAI build. What EAQ v1 changes is only how the weights were produced (the EAQ conversion; see Conversion). Any runtime that loads EXL3 DeepSeek-V4.1 checkpoints should load this one without EAQ-specific code, including the TensorFold DeepSeek-V4.1-Flash-1017 2x DGX Spark recipe (set its `MODEL_ID` to this repo).

| | |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Format | EXL3 (`trellis` / `suh` / `svh` / `mul1`), codebook **mul1**, `out_scales=always` |
| Average bpw | **2.90** over the body (integer widths; routed experts 3 bit, layers 18-22 at 2 bit; attention 5, shared experts 4-5, indexer wk 8) |
| Head bits | 6 |
| MTP / DSpark bits | 4 |
| Size | 196 GiB (210.6 GB, `total_size` in the index) in 39 shards (the previous 2.9 bpw build of the V4.1-Flash base: 210.7 GB in 39 shards) |
| DSpark | In-checkpoint (`dspark_block_size=5`, 128 draft experts / top-3, layers 37-39) |
| Vision | Included (tower left native) |
| Engram tables | **Not quantized.** Use original shards **47** and **48** of the base model |

## Engram tables (required)

EXL3 quantized only the Engram `wkv` linears. The FP8 row tables stay in the original DeepSeek tree:

- `model-00047-of-00048.safetensors` (~95 GiB)
- `model-00048-of-00048.safetensors` (~95 GiB)

Take them from **the same revision** (`1bae0e10`) of the base model, not from the earlier V4.1-Flash repo: the 1017 refresh changed the weights.

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --include "model-00047-of-00048.safetensors" \
  --include "model-00048-of-00048.safetensors" \
  --local-dir ./DeepSeek-V4.1-Flash-1017
```

Point the runtime at that directory (`engram_table_dir` in `config.json`). This repo ships `engram_table_dir` as `null` in `config.json`; set it to your local copy of the base model directory.

## Conversion (what was actually done)

- Source: the FP8 (dense, 32x32 blocks, ue8m0) + FP4 (experts) 1017 checkpoint, dequantized to FP16 on load (bit-exact against the DeepSeek reference decode on 11 sampled tensors; FP4 differs from the reference table only in the sign of zero).
- Converter: clean upstream exllamav3 v1.6.1 + EAQ patches, plus a port of `deepseek_v41` (attention with shared compressed KV, two-level indexer, Engram, shifted mHC pre-mix, DSpark drafter, vision) from the MiaAI-Lab fork. The port was checked against the fork's forward on real layer 0-3 weights (strict router-set match failed because the earlier fork rounds the router `gate.bias` to fp16; with that rounding emulated the port matches the fork: router set 96.8-99.6 %, MoE rel L2 7e-5 on identical-expert tokens; **fixed in this build**: the converter keeps the router `gate.bias` in fp32, bit-equal to the source, where the earlier fork rounded it to fp16).
- Allocation: the exllamav3 integer allocator at 2.9 bpw (integer widths, high-quality promotion), frozen as an explicit per-tensor recipe (sha256 `5e6a394f...`); per-tensor widths equal those of the earlier [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) build for all 46,702 comparable tensors. MTP uniformly 4 bit. The pack has 43 extra tensors versus the earlier build (`*.ffn.gate.bias_vl`, the vision router selection bias of the 1017 source); no tensor is missing.
- Smoke test (conversion machine, pack only): the 197 GB pack loads module by module with finite logits and greedy decoding of a sanity prompt answers correctly. This only shows the pack is not broken.
- What is NOT claimed: no KL/perplexity result, no comparison with the earlier 2.9 bpw build.

## Download

```bash
hf download Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw \
  --local-dir ./DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw
```

## Runtime

Stacks that load EXL3 (`trellis` / `suh` / `svh` / `mul1`) with DeepSeek-V4.1 support:

- **TensorFold** with the DeepSeek-V4.1-Flash-1017 2x DGX Spark recipe (the same EXL3 path as other EXL3 2.9 bpw builds).
- **ExLlamaV3 / TabbyAPI** once the architecture is supported in your build.

Official sampling from the base card: temperature 1.0, top_p 0.95, context up to 1M tokens. Thinking defaults on. KV cache is the native V4.1 FP4 layout; do not force an unrelated `fp8` KV dtype.

## Base model

DeepSeek-V4.1-Flash is a multimodal MoE (552B backbone) with CED, CSA2, Engram memory and in-checkpoint DSpark. Architecture, evals and prompt format live on the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Scores there are for the unquantized model, not this build.

## Credits

- **[DeepSeek-AI](https://huggingface.co/deepseek-ai)**: the base model, [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).
- **[turboderp](https://github.com/turboderp-org)**: the EXL3 format and the [exllamav3](https://github.com/turboderp-org/exllamav3) converter this build is based on (MIT).
- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and contributors: the runtime.
- **[LibertAIDAI](https://huggingface.co/LibertAIDAI)**: the EXL3 2.9 bpw build of this model used as the format comparison.
- `chat_template.jinja` adapted from [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (comment lines removed).

## License

MIT, inherited from the base model. Quantization adds no extra restrictions.

Quantized by [Mia's AI Lab](https://huggingface.co/Mia-AiLab). Format and tooling: [exllamav3](https://github.com/turboderp-org/exllamav3) (turboderp).
