---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
tags:
- vllm
- int5
- w5a16
- hqq
- smoothquant
- quantized
- moe
- a100
pipeline_tag: text-generation
library_name: vllm
---

# GLM-5.3-Flash — W5 (int5, g32, HQQ), A100-ready

5-bit (**W5A16, routed-experts-only, group size 32**) quantization of **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**, packaged to serve on **4× NVIDIA A100 80GB (SM80)** under vLLM. The base model ships for Hopper/Blackwell and does not boot on A100; this repo is the A100 path: the 5-bit weights, a 5-bit Marlin MoE kernel, and small vLLM patches that route GLM's attention onto kernels A100 has.

- bf16 → **W5**: ≈599 GiB → **226.4 GiB** (243.1 GB), fits TP-4 on 4× A100 80GB with a ~2.1M-token fp8 KV pool
- KL-divergence vs bf16: **0.042** (wikitext) / **0.081** (multi-turn agentic text)

## Use this model (vLLM)

Validated on the GLM-5.3-Flash vLLM image **`vllm/vllm-openai@sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4`** (vLLM `0.1.dev20051+g487ecf187`). Pin the digest: the `glm53-flash` tag has moved since. Official vLLM releases were not tested; once official GLM-5.3-Flash support lands (we are waiting on 0.29), these patches will need re-basing onto it.

```bash
cd serving
KDA_PATCH=/path/to/third_party_kda.patch ./prepare.sh   # fetch + patch kernel sources, build the kernel, patch vLLM files
MODEL=/path/to/GLM-5.3-Flash-W5-HQQ-g32 GPUS=0,1,2,3 ./serve.sh
```

`prepare.sh` fetches 10 Marlin source files from vLLM commit `7fd9cc0`, applies `kernel/w5_kernel.diff`, builds `osc_w5.so` inside the image, copies the target files out of the image and applies `vllm_glm53_a100.diff`. `serve.sh` bind-mounts the patched files and starts an OpenAI-compatible server on port 8000.

- **`KDA_PATCH`** — our validation also applied a third-party model-code patch for this image (KDA/model/MTP projection splitting), posted in [discussion #2 of cyankiwi/GLM-5.3-Flash-AWQ-INT4](https://huggingface.co/cyankiwi/GLM-5.3-Flash-AWQ-INT4/discussions/2). It is not redistributed here; save it to a file and pass its path. We have not tested this checkpoint without it.
- **Sampling:** `temperature 1.0, top_p 0.95` (the shipped `generation_config.json`).
- Parsers: `--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice`.
- First load takes several minutes (the 1-bit high planes are repacked on GPU); `serve.sh` raises the engine-ready timeout.
- Check the log for `5-bit experts engaged (kU5B16, 4+1 planes)`.

## Files

| file | what |
|---|---|
| `w5g32-layer-{03..44}.safetensors` (42) | the 5-bit routed experts, one shard per MoE layer (194.9 GiB) |
| `model-000NN-of-00120.safetensors` (49) | everything else, unquantized bf16: attention, dense layers, shared experts, router, embeddings, `lm_head`, MTP layer, vision tower (31.5 GiB) |
| `model.safetensors.index.json`, `config.json` | weight map; model + `compressed-tensors` quantization config |
| `tokenizer*`, `chat_template.jinja`, `generation_config.json`, `processor_config.json` | from the base model |
| `serving/kernel/w5_kernel.diff`, `serving/kernel/build_w5.py` | the 5-bit Marlin MoE kernel, as a diff against vLLM `7fd9cc0` sources, and its build script |
| `serving/vllm_glm53_a100.diff` | the vLLM patches (8 files), as a diff against the image's own files |
| `serving/prepare.sh`, `serving/serve.sh` | build + launch |

## Method

Only the routed experts (288 × 42 MoE layers, ~95% of parameters) are quantized. Per expert, from the bf16 weights:

1. **SmoothQuant fold, α = 0.65**, on the `down_proj` input: per-channel scales from activation maxima are divided out of `up_proj` and multiplied into `down_proj` (exact through SwiGLU).
2. **int5 symmetric**, values in [−15, 15], stored with bias 16.
3. **Per-group scales, g = 32** (bf16), initialised by an **MSE clip search** (13 points over 0.70–1.00 of the group max).
4. **HQQ refit**: proximal half-quadratic solver, p = 0.7, β = 10 growing ×1.01 per step, 20 iterations.
5. **{4,1} plane pack**: each 5-bit value is a 4-bit low plane in the standard Marlin int4 layout plus a 1-bit high plane (`weight_packed_hi`). 5.5 bits per expert weight including scales.

The MTP layer (layer 45) has no activation statistics and stays bf16 (it is in the config's `ignore` list).

## Serving patches

**Kernel (`w5_kernel.diff`).** A `kU5B16` type for vLLM's Marlin MoE kernel: the low plane goes through the stock int4 path, the high plane is streamed alongside it at a quarter of the bytes and merged during dequant. Output is bit-identical to stock int8 Marlin on the same 5-bit values. Built as a separate op library, so the image's own kernels are untouched.

**vLLM (`vllm_glm53_a100.diff`).**
1. Route 5-bit compressed-tensors experts to that kernel and repack the high planes at load (3 files).
2. Allow the MLA prefill geometry GLM uses (qk_nope 256, rope 0, v 256) on FlashAttention-2.
3. Keep the sparse-attention indexer built (its weights must load) but run attention on the dense MLA path (`OSC_FORCE_DENSE_MLA=1`): A100 has no sparse-MLA kernel.
4. Skip a Hopper-only DeepGEMM call in the indexer on SM80.
5. Allow `fp8_e5m2` KV for Triton MLA on SM80 (`OSC_ALLOW_FP8_KV_SM80=1`) and view the cache as e5m2.

## Serving notes (A100) — learned the hard way

- **Dense attention changes long-context behaviour.** Up to 2,048 tokens of context it matches the reference exactly (top-2048 of ≤2048 keys selects all). Beyond that every key is attended instead of the indexer's top 2,048. We have not measured the long-context quality effect.
- **fp8 KV must be `fp8_e5m2`.** Triton cannot compile e4m3 on SM80, so `fp8` / `fp8_e4m3` fail.
- **Leave MTP speculative decoding off.** As configured it was ~3× slower at 8 concurrent users; the boot reported no CUDA-graph memory, so graphs were probably lost on the speculative path.
- **KV pool size varies ~7% between identical boots** (2.09M vs 2.23M tokens at `--max-model-len 262144`).
- `--prefix-match-unit` made no measurable difference; `--mamba-cache-mode align` is set automatically when prefix caching is on.
- **The high-plane repack must be chunked.** Unpacking a whole layer's 1-bit plane at once is a ~10 GB allocation that stalls the allocator; it runs 8 experts at a time (`OSC_W5_HI_CHUNK`).
- **Bind mounts pin the inode.** Re-run `prepare.sh` only while the server is down.

## Quality

Mean KL-divergence vs the bf16 model, full vocabulary, 2,048-token windows:

| | wikitext-2 | multi-turn agentic text |
|---|---|---|
| **this W5 checkpoint (served)** | **0.0424** | **0.0806** |
| 8-bit g32 + HQQ (reference point) | 0.0153 | — |

- **Coherency:** 52/52 on a fixed prompt battery, on four separate boots.
- **Throughput:** TP-4, 4× A100 80GB, `fp8_e5m2` KV, 8 concurrent multi-turn sessions: **36 / 32 / 28 / 29 tok/s per user** at 0–10k / 10–25k / 25–50k / 50–100k tokens of context.
- **Memory:** 55.3 GiB per GPU for weights + runtime at TP-4.

Text only: the vision tower is included unquantized but we did not test image input.

## License & credits

- Base model: **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** © Z.AI, MIT ([LICENSE](./LICENSE)). Quantized from `zai-org/GLM-5.3-Flash-BF16`.
- Kernel and patch diffs modify **[vLLM](https://github.com/vllm-project/vllm)** sources (Apache-2.0).
- Third-party model-code patch referenced above: its author, in the linked discussion.
- W5 quantization, kernel and A100 patches by **[@spectator2026](https://huggingface.co/spectator2026)**.
