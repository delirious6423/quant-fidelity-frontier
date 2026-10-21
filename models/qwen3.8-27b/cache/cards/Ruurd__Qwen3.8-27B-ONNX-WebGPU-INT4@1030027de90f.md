---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: onnxruntime
pipeline_tag: text-generation
inference: false
tags:
- onnx
- webgpu
- int4
- qwen
- browser
- jspi
---

# Qwen3.8-27B ONNX WebGPU INT4

Community conversion of **Qwen/Qwen3.8-27B** for text generation in current Chrome/Edge through **ONNX Runtime Web + WebGPU**. All model operators run locally on the GPU with CPU execution-provider fallback disabled. JavaScript performs tokenization, sequence metadata and sampling.

**Download approximately 15.26 GB. Tested on a 48 GiB Apple Silicon Mac with Chrome 154 and hardware Metal 3 WebGPU.** Short prompts produced approximately 5–6 tokens/second. This is a functional browser proof of concept; other GPUs, long contexts and quality parity with the original checkpoint have not been validated.

## Run outside Clinstruct

The [standalone browser example and reproducible exporter](https://github.com/RuurdKuiper/qwen38-27b-onnx-webgpu) include the tokenizer, greedy generation loop, recurrent/KV cache bindings and JSPI/Blob loader. Follow its README to download this pack and serve the example on localhost. The local server only serves files; it performs no inference.

These files require the supplied generation implementation or an equivalent implementation of the graph metadata. This is not a Transformers.js pipeline export, and loading `input_ids → logits` without all cache/state tensors is insufficient. The native `model.onnx` and `genai_config.json` are also retained for independent GenAI runtime validation.

Clinstruct users can select **Qwen 3.8 27B INT4** and use **Download model**, then **Load model**. The application distribution does not include these weights.

## Conversion

- Original checkpoint revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Official `onnxruntime-genai` builder **0.17.1**, ONNX Runtime Python **1.30.0**, ONNX **1.23.2**, Transformers **5.18.0**, Torch **2.14.1**.
- WebGPU INT4, DEFAULT weight-only quantization, block size **32**, `op_types_to_quantize=MatMul/Gather` (including untied embeddings).
- Non-paged linear attention, `state_window=0`, decoder only, `exclude_mtp=true`; no vision, MTP, speculative decoding or CUDA GatedDeltaNet.
- A memory-bounded serializer calls the official quantizer per weight and writes **15 external-data shards, each ≤1 GiB**. It does not replace the model architecture or inference kernels.
- `model.browser.onnx` replaces six attention-mask metadata nodes with host inputs `seqlens_k` and `total_sequence_length`; numerical weights and decoder computation are unchanged relative to the native export.
- Pinned browser runtime: `onnxruntime-web@1.31.0-dev.20260914-8d85527a0`; Transformers.js **4.3.0** supplies tokenization only.

## Graph, state and browser requirements

The browser loader retains **128 state tensors on GPU**: 32 ordinary KV tensors, 48 convolution tensors and 48 recurrent tensors. State persists between decoding steps and resets between prompts. Current support is **batch size one, contiguous unpadded text**, with Qwen's three-axis position IDs.

Use current Chrome or Edge on a secure origin (localhost is sufficient), hardware WebGPU with `shader-f16`, and **WebAssembly JSPI**. Use an ordinary browser profile: a large Blob download failed in an incognito validation context. The 48 GiB test machine is a tested configuration, not a measured minimum-memory requirement.

## Validation

Independent native GenAI generation answered Paris/156. The browser produced sensible text for short prompts, answered `12 * 13` as `156`, generated 32 consecutive tokens, and retained all 128 states on GPU at every step. Clinstruct extracted Paris and Utrecht. No CPU/server inference was used in browser tests. Full quality evaluation, vision, arbitrary batch sizes, other hardware and long contexts remain untested.

`manifest.json` lists graph inputs/outputs, operators, external data and byte sizes. `SHA256SUMS` supplies file-integrity checks. The standalone repository includes test code and measured results, so additional hardware reports are welcome.

## License and attribution

Original model/tokenizer copyright **2026 Alibaba Cloud**, Apache License 2.0. Conversion and browser integration by **Ruurd Kuiper**. See `LICENSE` and `NOTICE` for attribution and changed-artifact notices. This is a community conversion and does not imply endorsement by Alibaba or Microsoft.
