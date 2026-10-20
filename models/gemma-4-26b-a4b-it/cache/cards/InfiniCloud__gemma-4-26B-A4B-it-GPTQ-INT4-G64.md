---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
- compressed-tensors
- quantization
- int4
- vllm
---

# gemma-4-26B-A4B-it-GPTQ-INT4-G64

English | [日本語](README.ja.md)

**A GPTQ INT4 version of Gemma 4 26B A4B for deployments with less VRAM.** For users who want Gemma's language and image-input capabilities, it reduces weight-file size from approximately 51.6 GB for the original BF16 model to 15.7 GB, a reduction of about 70%. Image-processing components and other excluded parameters remain in BF16. File size differs from runtime VRAM; measured conditions and memory use are documented below.

## Who this model is for, and why we built it

InfiniCloud created this quantized version to **provide a Gemma deployment option for L4 24GB environments in InfiniCloud AI (Shiraito)**. It is intended primarily for two uses:

- **Running Gemma on an L4 24GB.** As a candidate for deployment on one L4, it reduces the memory occupied by weights to leave room for context and inference processing.
- **Running a primary and a secondary model on an 80GB-class GPU.** With Gemma serving as the secondary model alongside a primary model, it reduces Gemma's share of the VRAM used by the two models' weights.

It is particularly intended for users who want **4-bit weight savings on GPUs without native NVFP4 arithmetic, such as L4**. The chosen GPTQ INT4 format does not require native NVFP4 arithmetic. It still requires kernels that support the selected GPU and inference engine. See [NVIDIA's NVFP4 overview](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) and [vLLM's hardware support by quantization method](https://docs.vllm.ai/en/v0.29.0/features/quantization/#supported-hardware).

L4 deployment and simultaneous operation of two models on an 80GB-class GPU are intended uses of this distribution. Measurements were performed on Blackwell 96GB; testing those hardware configurations and integration with Shiraito itself remains a subsequent step.

## Why distribute a prequantized model

Users can also [quantize to 4 bits at load time](https://docs.vllm.ai/en/v0.29.0/features/quantization/bnb/#inflight-quantization-load-as-4bit-quantization). Quantization methods, selected layers, and calibration conditions can affect answer quality and memory use, so a deployment decision requires evaluating the chosen configuration.

This distribution provides **GPTQ-calibrated weights, quantization settings, and evaluation results together**. Users can load the same checkpoint without running GPTQ calibration themselves, review its quality differences from BF16 and recorded VRAM use, and decide whether it is a suitable deployment candidate. Startup still requires model loading and preparation for the inference kernels.

To support that decision, we compared BF16 and INT4 on GSM8K and Japanese MGSM, checked Japanese text, arithmetic JSON, and image JSON requests, and ran an 8k-context memory test. **These checks establish the observed quality tradeoff and operating conditions for the goal of using less memory.** Evaluation settings and reproduction instructions are included. Inference-speed comparisons have not been performed.

**Use vLLM 0.29.0, the inference engine version tested for this checkpoint.** Ordinary Transformers loading has limitations with this compressed format. Detailed conditions and a usage example follow below.

## VRAM and model placement

The inference target is **one dedicated NVIDIA L4 marketed as 24GB**, tensor parallel size 1, with all experts resident and no CPU weight offload. Initial L4 validation is planned for **8192 total context tokens and one concurrent request**, eager execution, BF16 KV, prefix caching disabled, and at most one image with no video/audio per request. Total context includes input, generated tokens, and image tokens. **L4 has not been tested; its actual CUDA-visible capacity, minimum required VRAM, and throughput are not measured.**

For deployment alongside a primary model on an 80GB-class GPU, budget for both models' weights, KV caches, and workspaces, and set each inference process's memory budget accordingly. The primary model that can be paired with Gemma, the context lengths, and concurrent request counts depend on both configurations. This distribution offers a way to reduce the VRAM allocated to the secondary model in that arrangement.

The analytic estimate of serialized parameter payload is **15,652,016,944 bytes (14.577 GiB)**. It includes packed INT4 weights, BF16 group scales, retained BF16 parameters, and original-shape integers. The generated weight file was separately measured at **15,656,878,852 bytes (14.582 GiB)**. **Neither figure is runtime VRAM.** The payload estimate excludes persistent nonparameter buffers and file headers; runtime additionally needs loading copies, kernel preparation, activation/encoder memory, MoE workspaces, cache pools, and allocator overhead. A4B describes activated parameters; this deployment keeps all 128 experts resident.

| Total context per request | Concurrent requests | Ideal steady-state BF16 KV contents |
| --- | ---: | ---: |
| 8192 | 1 | 0.352 GiB |
| 8192 | 8 | 2.813 GiB |
| 16384 | 1 | 0.508 GiB |
| 16384 | 8 | 4.063 GiB |

The KV values assume sliding-window eviction with a 1024-token window. They exclude in-flight prefill tokens, block rounding, and reserved cache pools. Full-attention K=V source weights do not eliminate separate physical K/V cache storage in vLLM. The larger-context and concurrency rows are sizing examples, not validated L4 operating points. Do not infer a supported context by subtracting the payload estimate from nominal 24GB. See [VRAM planning formulas and source evidence](evidence/vram-planning-preflight.json).

Generation and completed inference checks used one **RTX PRO 6000 Blackwell Max-Q 96GB**. The inference startup profiles reported:

| vLLM startup diagnostic | BF16 smoke | INT4 smoke | INT4, 21 GiB budget |
| --- | ---: | ---: | ---: |
| Model loading | 48.54 GiB | 15.1 GiB | 15.1 GiB |
| Weights plus non-PyTorch consumption | 48.88 GiB | 16.26 GiB | 16.26 GiB |
| Peak activation during profiling | 0.97 GiB | 0.59 GiB | 0.58 GiB |

These diagnostic categories overlap and must not all be added together. They are separate from the reserved KV pool and do not establish minimum VRAM. The initial smoke servers used a 0.80 memory budget on Blackwell. The subsequent **21 GiB planning-budget test passed** with context 8192, one sequence, eager execution, BF16 KV, and at most one image per request. Its reported KV pool was **4.16 GiB**.

All three budget-test smoke cases passed. A separate synthetic text request returned HTTP 200 with **8064 input + 128 generated = 8192 total tokens**. It deliberately used `ignore_eos=True` and finished with `length`; the client elapsed time was 6.527 seconds. This was a memory/transport exercise, not an answer-quality or performance benchmark.

Across 185 device-wide samples at one-second intervals over 184.813 seconds, the maximum observed GPU usage was **22,379 MiB (21.854492 GiB)**, about **0.854 GiB above the 21 GiB budget**. The budget controls vLLM's memory planning; it is **not a hard GPU memory cap**. The observation includes device-wide usage, does not capture every instantaneous peak, and is not a minimum VRAM requirement. It does not establish L4 kernel compatibility or performance. See [the budget test and measurement evidence](evidence/vram-budget-check.json).

## Verification and evaluation status

The following measurements and checks were completed on 2026-09-12 to support deployment decisions. Quality comparisons and memory observations are reported with their conditions.

| Check | Gemma BF16 | Gemma INT4 G64 |
| --- | --- | --- |
| vLLM full-model reload and three fixed requests | Passed, 3/3 | Passed, 3/3 |
| Japanese city-name smoke | Passed | Passed |
| Arithmetic JSON: required key, integer type, value | Passed | Passed |
| Image JSON: required keys, shape/color order, structure | Passed | Passed |
| GSM8K, 1319 questions | Complete; results below | Complete; results below |
| Japanese MGSM, 250 questions | Complete; results below | Complete; results below |
| Quantized weight file bytes / SHA256 / finite scales | Not applicable | Passed |
| Blackwell, 21 GiB planning-budget test | Not applicable | Passed: 3/3 smoke and 8192-token exercise |
| L4 startup / peak VRAM / throughput | Not measured | Not measured |
| Dedicated performance benchmark | Not measured | Not measured |

Both BF16 and INT4 smoke runs returned `東京`, `{"answer": 12}`, and `{"shapes":[{"color":"red","shape":"square"},{"color":"blue","shape":"circle"}]}`. All requests returned HTTP 200, finish reason `stop`, and zero reasoning tokens. Requests used temperature 0, seed 42, and at most 256 generated tokens. Responses and checks are retained in [BF16 smoke evidence](evidence/bf16-smoke.json) and [INT4 smoke evidence](reproduce/evidence/vllm-smoke-summary.json); the [image fixture](evidence/smoke_shapes.png) is included.

These three cases check their specified responses and JSON structures. They do not establish a general JSON compliance rate or InfiniCloud AI (Shiraito) integration support. No JSON Schema constrained decoding, output repair, or retry result is claimed.

The completed arithmetic comparison uses the same source tokenizer/template for BF16 and INT4: GSM8K and Japanese MGSM native CoT, both 5-shot, thinking disabled, temperature 0, output limit 2048, and context 8192. See [full-test-set results and paired checks](evaluation/quality-comparison.json) and [evaluation reproduction notes](evaluation/README.md). Differences below are INT4 minus BF16:

| Full test set / extraction | BF16 | INT4 G64 | Difference |
| --- | ---: | ---: | ---: |
| GSM8K / strict | 94.920394% (1252/1319) | 93.555724% (1234/1319) | −1.364670 pp (−18) |
| GSM8K / flexible | **94.844579% (1251/1319)** | **93.479909% (1233/1319)** | **−1.364670 pp (−18)** |
| Japanese MGSM / strict | 4.0% (10/250) | 4.0% (10/250) | 0 pp (0) |
| Japanese MGSM / flexible | **82.8% (207/250)** | **81.2% (203/250)** | **−1.6 pp (−4)** |

Both metrics score the correctness of the **extracted answer**. Neither is a JSON compliance rate or a pure format-match rate. Their difference shows sensitivity to extraction rules, particularly for Japanese MGSM; the low strict score alone does not establish a comparable loss of arithmetic ability. The [evaluation plan](evidence/evaluation-plan.json) records the preparation settings.

Paired checks matched problem IDs, problem text, prompts, targets, dataset revisions, task definitions, and generation controls. On GSM8K, both extraction methods had 11 previously incorrect answers become correct and 29 previously correct answers become incorrect. Japanese MGSM had 5 become correct and 9 become incorrect under flexible extraction, and 2 each under strict extraction. Both models had **zero empty responses**. The saved lm-eval samples lack generated-token and finish counts, so reaching the 2048-token limit is **unobserved**, not zero. No significance test was performed. These results describe this arithmetic comparison; they do not establish general task quality, general JSON reliability, or an inference-speed difference.

The [generation audit](reproduce/evidence/completed-generation.json) found no logged GPTQ-to-RTN fallbacks. After generation, [metadata normalization](reproduce/evidence/completed-normalization.json) changed only the quantization target patterns in `config.json` to support both Hugging Face and vLLM names, preserving the same 11,725 Hugging Face targets. The [CPU numerical scan](reproduce/evidence/numerical-validation-summary.json) confirmed the unchanged weight-shard SHA256, finite floating-point tensors, positive finite scales, and **447 protected BF16 tensors bitwise equal to the source**. These checks validate the saved artifact; answer quality is measured separately.

## Model and quantization

InfiniCloud produced this quantized derivative of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52), revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, authored by Google DeepMind. An unchanged [original model card](source/README.md) is included.

The recipe is **GPTQ INT4 W4A16**, symmetric group size **64**, `actorder=weight`, with BF16 activations. It targets language attention, the shared dense MLP, and all routed experts in all 30 decoder layers. There are 128 experts per layer and 8 are selected per token. Expert intermediate width 704 and shared-MLP width 2112 require G64 in this recipe because neither divides evenly into groups of 128.

The model-structure preflight selected **11,725 Linear modules**, containing **24,483,430,400 weight elements** out of **25,805,933,872 unique loaded parameter elements**. Completed artifact validation confirmed all **11,725 packed projections**, their scales and original shapes, including all **11,520 expert projections**. The vision tower and projection, routers, input embeddings/tied `lm_head`, normalization, and other nonselected parameters are retained in BF16. The `Gemma4ForConditionalGeneration` multimodal architecture and original processor are retained. KV-cache quantization is outside this recipe.

Calibration uses [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a), revision `8049631c405ae6576f93f445c6b8166f76f5505a`, split `train_sft`: 512 conversations, seed 42, at most 2048 tokens per conversation, using Gemma's tokenizer/template with thinking disabled. The prepared dataset contains 584,705 tokens; 50 conversations reached the length limit. Its token SHA256 is `3fd58c06750f31b8dd1e592b8fed2ecfccb151507d8865886e83af2fe8f7f59a`. This is predominantly English text calibration; Japanese-specific and image calibration were not performed.

`moe_calibrate_all_experts=True` sends every calibration token through every expert; only the originally selected experts contribute to the routed output. This avoids uncalibrated experts, but its statistics differ from using only tokens routed to each expert in normal inference. Calibration workload is not an inference-speed measurement. See [the quantization configuration](quantization-config.json).

The quantization run took **2 hours 22 minutes 33 seconds** including its measurement wrapper. One-second device-wide polling observed **15,673 MiB** maximum GPU usage. Partial CPU observations reached **100.224 GiB RSS/HWM**, including file-backed pages; the final sample preceded process exit by about 30 seconds. These are **generation resource observations**, not inference requirements, guaranteed lifetime peaks, or minimum host RAM. See [completed generation and measurement scope](reproduce/evidence/completed-generation.json).

## Runtime and saved configuration

| Purpose | Versions |
| --- | --- |
| Quantization environment | llmcompressor 0.13.0; compressed-tensors 0.18.0; Transformers 5.14.1; PyTorch 2.13.0+cu130 |
| BF16 / INT4 inference | vLLM 0.29.0; compressed-tensors 0.17.0; Transformers 5.14.1; PyTorch 2.13.0; driver 580.178.04 |
| Quality evaluation | lm-eval 0.4.13 |

The recorded BF16 and INT4 smoke servers used context 8192, maximum 8 sequences, batched-token limit 8192, eager execution, chunked prefill, prefix caching disabled, memory budget 0.80, and one image / no video/audio per request. The registered vLLM reasoning parser was **`gemma4`**; default chat-template arguments were **`{"enable_thinking":false}`**. [BF16 serving settings](evidence/bf16-serving.json) and [INT4 serving settings](reproduce/evidence/vllm-smoke-summary.json) distinguish this Blackwell smoke profile from the one-request L4 target. Full quality runs use a maximum of 32 sequences; they are not L4 capacity tests.

The [source config](source/config.json) declares Transformers `5.5.0.dev0`; artifact serialization used **5.14.1**. Per-layer attention settings use the source generation's `layer_types`, `head_dim`, `global_head_dim`, and global KV-head fields. Those settings, vision processing, and the multimodal architecture are retained. This version record is not a claim that arbitrary other Transformers versions can load the artifact.

The [source chat template](source/chat_template.jinja), [processor config](source/processor_config.json), and [tokenizer config](source/tokenizer_config.json) are included for comparison. The source semantics for `response_template`, `response_schema`, special tokens, and BOS are retained. The template already emits BOS; quality evaluation uses `add_bos_token=False`. The [source generation config](source/generation_config.json) has stop IDs `[1,106,50]`, `do_sample=true`, temperature 1.0, top-k 64, and top-p 0.95. Evaluation uses explicit generation settings with `generation_config=vllm`; preserving the source file does not mean those sampling defaults were used for the smoke.

The initially saved tokenizer retained calibration-time truncation at 2048 tokens. This checkpoint issue was corrected by restoring `tokenizer.json` and `tokenizer_config.json` from the pinned source revision. A 4800-token check then matched the source token IDs without truncation. All smoke and quality jobs explicitly select the source tokenizer, so their tokenization was unaffected. See the [tokenizer restoration audit](reproduce/evidence/tokenizer-source-restoration.json); quantized weights were unchanged.

## vLLM usage: reference Blackwell settings

**Loading limitation:** ordinary Transformers 5.14.1 `from_pretrained` is not a supported inference path for this linearized-expert compressed format. In an isolated CPU test with a tiny Gemma checkpoint, the loader returned with missing fused-expert weights and unexpected individual-expert weights, then used newly initialized expert parameters. Finite output alone did not establish a correct reload. Reloading with `run_compressed=False` or `load_context` did not resolve the problem. The actual INT4 checkpoint passed its inference smoke checks using **vLLM 0.29.0 on Blackwell**. See [loading compatibility](reproduce/evidence/transformers-compatibility.json) and [INT4 smoke evidence](reproduce/evidence/vllm-smoke-summary.json).

The following is a **post-publication example** for the reference Blackwell environment with vLLM 0.29.0. It follows the successful budget-test command, replacing local checkpoint/tokenizer paths with the distribution repository. The measured run used the pinned source tokenizer; the distributed tokenizer files have been restored to that same source. Direct loading from the public repository was not part of the local test. Select the physical GPU assigned to the job; the reference run used GPU 1.

**The utilization value below applies only to the tested Blackwell device's CUDA-visible 101,963,071,488 bytes:** `21 * 1024**3 / 101963071488 = 0.2211445572886036`. **Do not reuse this fraction on L4.** The L4 configuration above remains an untested target; no L4 utilization fraction or supported capacity is established here.

```bash
MODEL_ID='InfiniCloud/gemma-4-26B-A4B-it-GPTQ-INT4-G64'
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.cli.main serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port 8099 \
  --served-model-name gemma4-test \
  --tokenizer "$MODEL_ID" \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.2211445572886036 \
  --generation-config vllm \
  --seed 42 \
  --enforce-eager \
  --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}'
```

## Distribution files and reproduction

The distribution includes the [license](LICENSE), [attribution/change notice](NOTICE), source documents, [recipe](quantization-config.json), completed-check summaries, and a [standalone reproduction bundle](reproduce/README.md), alongside the compressed checkpoint.

The reproduction instructions load the **original BF16 source** inside llmcompressor's `load_context` for expert linearization and GPTQ calibration. This is distinct from reloading the resulting compressed checkpoint with Transformers, which remains unsupported by the tested combination. The bundle preserves the [original execution archive](reproduce/historical/README.md) and distinguishes that run's later metadata normalization from a future run using the corrected target patterns from the start.

The model package contains the quantized `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `processor_config.json`, `chat_template.jinja`, and the compressed weight file. See the [final artifact validation and file inventory](artifact-validation.json) and [SHA256 checksums](SHA256SUMS) for the exact distributed files and integrity checks. Actual L4 testing and performance benchmarking remain separate, unperformed checks.

## License and attribution

The original model and this quantized derivative use **Apache License 2.0**. See [LICENSE](LICENSE), [NOTICE](NOTICE), and Google's [model-specific license link](https://ai.google.dev/gemma/docs/gemma_4_license). The unchanged source model card identifies Google DeepMind as the original author. InfiniCloud's modifications, completed checks, and remaining limitations are recorded above.
