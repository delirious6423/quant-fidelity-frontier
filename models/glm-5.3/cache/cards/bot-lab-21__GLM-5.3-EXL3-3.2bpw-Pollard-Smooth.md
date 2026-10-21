---
license: other
license_name: glm-5.3-license
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
base_model: zai-org/GLM-5.3
base_model_relation: quantized
quantized_by: bot-lab-21
pipeline_tag: text-generation
library_name: vllm
tags: [glm, glm-5.3, moe, exl3, exllamav3, trellis, 3-bit, smoothquant, pollard, vllm, tensor-parallel, dgx-spark, gb10, mtp, speculative-decoding]
---

# GLM-5.3 — EXL3 3.2 bpw, Pollard method, SmoothQuant-folded, for four DGX Sparks

[zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (744B total / ~40B active MoE, `glm_moe_dsa`: 78 layers, 256 routed
experts + 1 shared, top-8, MLA attention with a DeepSeek-style sparse indexer, one MTP layer) quantized from the **bf16** release
to an **EXL3 trellis at 3.21 bits per weight**, with SmoothQuant folded in first and the allocation chosen by exllamav3's budgeted
allocator on in-domain calibration data. Sized to serve on **four** 128 GB DGX Spark (GB10) nodes at tensor parallelism 4 with the
model's own MTP layer as the speculative draft.

## What is in the box

| part | precision | note |
|---|---|---|
| routed experts (256 × 75 MoE layers) | EXL3 trellis, ~3 bpw (allocator-chosen per tensor) | the bulk of the 292 GB |
| attention (`q_a`, `q_b`, `kv_a`, `o_proj`), indexer `wq_b`, shared experts, dense MLP (layers 0–2) | EXL3, 5 bpw | attention is quantized, unlike the public expert-only builds |
| `lm_head` | EXL3, 6 bpw | |
| MTP layer 78 (block + shared head norm) | EXL3, 8 bpw | the speculative draft |
| `eh_proj` (MTP), embeddings, norms, routers, indexer `wk` / `weights_proj`, `kv_b_proj` | bf16 / fp16 | passthrough; `eh_proj` restored to bf16 because vLLM builds it unquantized |
| SmoothQuant | folded (α = 0.5) into the two norm seams of every layer | exactly invertible |

292 GB in 46 safetensors shards plus one 144 MB `eh_proj` shard; `quantization_config.json` (exl3, version 1.4.8,
`target_bits_per_weight` 3.2, `average_bits_per_weight` 3.21, `head_bits` 6, `mtp_bits` 8 — `bits` is kept as the integer 3 because
the Hub's config validator requires an integer there); GLM-5.3 tokenizer and the updated upstream chat template (commit aca966e).

## Measured quality (same fleet, same probes, same day; "int4 TP8" = our production int4/int8 GPTQ build of the same model)

| measurement | this model, 4 nodes | int4 TP8, 8 nodes |
|---|---|---|
| live perplexity, 6 fixed held-out texts, 4,079 tokens | **4.831** | 4.82 – 4.84 |
| HumanEval / HumanEval+ pass@1 (greedy, EvalPlus) | **0.957 / 0.927** | 0.963 / 0.945 |
| MBPP / MBPP+ pass@1 | **0.979 / 0.841** | 0.971 / 0.828 |
| correctness probe (counting, factual, prose, 4- and 8-way concurrent) | all passed | all passed |

Probes are teacher-forced or greedy through the served API. The perplexity texts were excluded from calibration by content hash.

## Speed and memory (vLLM 0.28, TP4, fp8 KV, MTP k=3, `--gpu-memory-utilization 0.84`)

| | this model, 4 nodes | int4 TP8, 8 nodes |
|---|---|---|
| single-stream decode, mixed workload | 24.0 tok/s | 40.0 |
| 4-stream aggregate, mixed workload | 58.2 tok/s | 85.2 |
| speculative draft (in-checkpoint MTP), replay of real traffic | 2.23 accepted/step at k=3, 74 % draft-accept, 3.23 tok/step | 1.58 – 1.84 at k=5 |
| KV cache | 396K tokens (fp8, ~57 KB/token incl. indexer cache) | 900K (nvfp4) |
| weights per node | 81 GiB | ~50 GiB |

## Serving

Built and measured with vLLM 0.28 plus an EXL3 runtime plugin that provides the sparse-MLA attention backend for GB10 and the
EXL3 GEMM/MoE kernels (`--attention-backend CUSTOM --block-size 256 --kv-cache-dtype fp8`, speculative config `method: mtp`,
`num_speculative_tokens: 3`, `--max-model-len 180000`).

**Runtime requirement.** This artifact quantizes attention and the head, which public expert-only EXL3 builds do not. The EXL3
runtime therefore has to (1) decode every fused shard at its *stored* width — exllamav3 pads output widths to multiples of 128 and
applies a 128-column output Hadamard, so `kv_a_proj_with_mqa` is stored 640 wide for 576 declared — and trim afterwards, and (2)
shard a quantized `lm_head` on 128-row boundaries of the global vocabulary under tensor parallelism. A runtime that narrows the
trellis to the declared width, or starts a vocab shard mid-block, loads this model without error and emits gibberish. Check with a
teacher-forced perplexity read before trusting it: the numbers above were taken on a runtime that satisfies both. NVFP4 KV is not
available on the sparse-MLA path we used.

GLM-5.3's chat template has no `enable_thinking` switch. Control thinking with `chat_template_kwargs`
`{"reasoning_effort": "low" | "high", "clear_thinking": false}`; passing `chat_template_kwargs` replaces the server defaults.

## Recipe (short)

1. Streaming bf16 forward over 384 × 2048 in-domain rows with hooks on the two norm seams of each layer → per-channel max|x|;
   SmoothQuant fold at α = 0.5 (norm weight ÷ s, consumer columns × s). Exactly invertible.
2. exllamav3 1.4.8 (aarch64 build) budgeted allocator, `-b 3.2 -hb 6 -mb 8 -hq`, same calibration rows (`-cd`). No hand-tiered
   recipe: the allocator's choices beat ported per-layer recipes on this model (Pollard Weights' measurement, reproduced).
3. Band-parallel conversion on ten nodes (8 layers each, `--resume` from the previous band's hidden state, static strategy),
   merge, compile.
4. Restore bf16 `eh_proj`; resolve MTP-side tensors by shard headers, not the index.

The full recipe and receipts follow in a companion repository.

## Files

`model-00001…00046-of-00046.safetensors`, `model-mtp-eh_proj.safetensors`, `model.safetensors.index.json`, `config.json`,
`quantization_config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`.

## Prior work this build stands on

- **GLM-5.3** — Z.ai. The model, its architecture, tokenizer, chat template and license. https://huggingface.co/zai-org/GLM-5.3
- **Pollard Weights** — WestWaters. The quantization method and playbook followed here: Hessian error-feedback quantization on
  in-domain calibration, MoE-aware allocation, the `hf-smooth` SmoothQuant fold, and the EXL3 lane with its measured finding that
  exllamav3's budgeted allocator beats hand-ported per-layer recipes. Our measurements from this run were contributed back
  (PRs #36, #39, #40). https://github.com/WestWaters/pollard-weights
- **exllamav3 / EXL3** — turboderp-org. The trellis format, converter (v1.4.8) and budgeted allocator that produced these
  tensors. https://github.com/turboderp-org/exllamav3
- **QTIP** — Tseng, Sun, Ouyang, De Sa. *QTIP: Quantization with Trellises and Incoherence Processing*, NeurIPS 2024 — the
  trellis-coded quantization with Hadamard incoherence processing that EXL3 implements. https://arxiv.org/abs/2406.11235
- **SmoothQuant** — Xiao, Lin, Seznec, Wu, Demouth, Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for
  Large Language Models*, ICML 2023. https://arxiv.org/abs/2211.10438
- **GPTQ** — Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
  Transformers*, ICLR 2023 — the Hessian error-feedback rounding behind the int4/int8 comparison build. https://arxiv.org/abs/2210.17323
- **cuda-exl3** — Zeuss5. vLLM plugin providing the GB10 sparse-MLA attention backend and EXL3 GEMM/MoE kernels the numbers
  above were measured on (v1.0.0). https://github.com/Zeuss5/cuda-exl3
- **vLLM** — vllm-project, v0.28.0. https://github.com/vllm-project/vllm — and the GB10 container lineage by ciprianveg
  (`gb10-glm-5.2`) our serving images derive from. https://github.com/ciprianveg/gb10-vllm
- **drowzeys — keys-GLM-5.3-EXL3-Abliterated** and its 4 × DGX Spark serving recipe: the first public four-node GLM-5.3 EXL3
  build (experts 3.0 bpw, everything else bf16; TP4 DCP1, 200K, ~15 tok/s, KL 0.10 vs bf16), the baseline this build was measured
  against, and the source of two findings we confirm (context is nearly free, decode-context parallelism is the tax).
  https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-Abliterated ·
  https://github.com/drowzeys/keys-GLM-5.3-EXL3-3.0BPW-abliterated-vLLm-cuda-Exl3
- **Mia AI Lab — GLM-5.3-Flash EXL3 on 2 × DGX Spark**: the vLLM + EXL3 container and sparse-MLA overlay that recipe serves from.
  https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
- **0xSero — GLM-5.3-500B-EXL3-3.0bpw**: another public expert-only EXL3 build of this model.
  https://huggingface.co/0xSero/GLM-5.3-500B-EXL3-3.0bpw
- **DeepSeek-V3** (multi-token prediction, the MTP layer used as the draft) https://arxiv.org/abs/2412.19437 and
  **DeepSeek-V3.2 / DeepSeek Sparse Attention** (the indexer + top-k sparse MLA that GLM-5.3's attention follows)
  https://github.com/deepseek-ai/DeepSeek-V3.2
- **EvalPlus** — Liu, Xia, Wang, Zhang. *Is Your Code Generated by ChatGPT Really Correct?*, NeurIPS 2023 — HumanEval+ / MBPP+.
  https://github.com/evalplus/evalplus

## License / provenance

Derivative of zai-org/GLM-5.3 (GLM-5.3 License, © 2026 Z.AI). Quantization and measurement by bot-lab-21. Quantizer method:
Pollard Weights (Apache-2.0); converter: exllamav3 (MIT); serving: vLLM (Apache-2.0) with the cuda-exl3 plugin. Produced with the
help of an AI assistant operating the cluster.
