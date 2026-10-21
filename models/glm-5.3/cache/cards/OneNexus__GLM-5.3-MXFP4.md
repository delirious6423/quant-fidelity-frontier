---
language:
- en
- zh
library_name: transformers
license: other
license_name: glm-5.3
pipeline_tag: text-generation
base_model:
- zai-org/GLM-5.3
base_model_relation: quantized
tags:
- glm_moe_dsa
- quark
- mxfp4
- rocm
- sglang
---

# GLM-5.3-MXFP4

## Model Overview

- **Model architecture:** full GLM-5.3 (`GlmMoeDsaForCausalLM`)
  - **Input:** text
  - **Output:** text
- **Source checkpoint:** [zai-org/GLM-5.3-BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), revision `304b8051cfb2b260b61ce0cbe330e02a98e73639`
- **Validated hardware:** 4× AMD Instinct MI350 GPUs (gfx950)
- **Validated runtime:** stock InferenceX/SGLang ROCm path
  - SGLang image tag `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260908`
  - TP4/EP4, EAGLE MTP, TileLang DSA, AITER MXFP4 MoE, FP8 E4M3 KV cache, and HiCache
  - AMD Quark `0.12.post1+1b229f7` checkpoint format

This is the first evaluated OneNexus MXFP4 release for the full GLM-5.3 model. The promoted checkpoint’s internal candidate name is **Strong7**. It was quantized from the BF16 checkpoint, not from the published FP8 checkpoint.

The 282 model shards contain **438,001,945,864 bytes (407.92 GiB)** of indexed model weights. This is 42.04% smaller than the official GLM-5.3 FP8 checkpoint and 70.93% smaller than the BF16 source.

## Model Quantization

AMD Quark applies OCP MXFP4 E2M1 quantization to the routed MoE expert weights. Weights use static 1×32 block scaling with E8M0 scales; expert activations are quantized dynamically with the same 1×32 layout. No calibration dataset is required for the initial MXFP4 conversion.

The following paths remain in BF16:

- attention and DSA projections;
- router gates, dense/shared MLP projections, and `lm_head`;
- the MTP layer, layer 78.

Strong7 additionally applies a checkpoint-only, folded-intermediate SmoothQuant refinement to seven routed experts in layer 6: expert IDs **16, 96, 103, 159, 195, 208, and 253**. The transform preserves MXFP4 storage and requires no custom runtime operation. Relative to the initial Quark conversion, only 28 tensors across four shards change; 115,874 non-target tensors remain unchanged.

The Hugging Face metadata uses normalized `model.layers.*` module names and the same Quark MXFP4 convention as [amd/GLM-5.2-MXFP4](https://huggingface.co/amd/GLM-5.2-MXFP4). The checkpoint keeps the `glm_moe_dsa` model type, `GlmMoeDsaForCausalLM` architecture, 282-shard layout, and native Quark weight/scale pairs expected by the stock SGLang loader.

The reference Quark recipe is:

```bash
cd Quark/examples/torch/language_modeling/llm_ptq/
python quantize_quark.py \
  --model_dir zai-org/GLM-5.3-BF16 \
  --output_dir GLM-5.3-MXFP4 \
  --quant_scheme mxfp4 \
  --exclude_layers "*self_attn*" "*mlp.gate" "*lm_head" \
    "*mlp.gate_proj" "*mlp.up_proj" "*mlp.down_proj" \
    "*layers.78.*" \
  --file2file_quantization
```

Strong7 adds the targeted folded SmoothQuant refinement described above. Machine-readable release details are in `mxfp4_smoothquant_optimization.json`; the exact quantization exclusions and tensor format are in `config.json`.

## InferenceX Drop-in Compatibility

This checkpoint was validated as a drop-in weight replacement in the stock InferenceX GLM MXFP4 serving recipe. During validation, only the model path, served-model name, chat-template path under the model directory, and listening port changed. The image layers and configuration, environment, TP4/EP4 topology, EAGLE settings, HiCache settings, KV format, DSA backends, and all other serving flags were identical. No SGLang source overlay was mounted.

## Deployment

### Stock InferenceX/SGLang recipe on four MI350 GPUs

```bash
hf download OneNexus/GLM-5.3-MXFP4 \
  --local-dir "$PWD/GLM-5.3-MXFP4"


docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --security-opt label=disable \
  --ipc=host \
  --shm-size 32g \
  -p 30000:30000 \
  -v "$PWD/GLM-5.3-MXFP4:/model:ro" \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_SET_CPU_AFFINITY=1 \
  -e SGLANG_USE_ROCM700A=1 \
  -e SGLANG_MOE_PADDING=1 \
  -e SGLANG_ROCM_DISABLE_LINEARQUANT=0 \
  -e SGLANG_ROCM_FUSED_DECODE_MLA=1 \
  -e SGLANG_OPT_USE_TOPK_V2=false \
  -e NCCL_MIN_NCHANNELS=112 \
  -e ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
  lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260908 \
  sglang serve \
    --model-path /model \
    --served-model-name glm-5.3-mxfp4 \
    --tp 4 \
    --ep-size 4 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-hierarchical-cache \
    --hicache-ratio 1.5 \
    --hicache-write-policy write_through \
    --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct \
    --page-size 64 \
    --trust-remote-code \
    --context-length 1048576 \
    --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang \
    --chunked-prefill-size 131072 \
    --max-prefill-tokens 16384 \
    --mem-fraction-static 0.80 \
    --cuda-graph-max-bs 2 \
    --max-running-requests 2 \
    --watchdog-timeout 1800 \
    --kv-cache-dtype fp8_e4m3 \
    --moe-runner-backend auto \
    --chat-template /model/chat_template.jinja \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --mm-feature-transport cpu \
    --host 0.0.0.0 \
    --port 30000
```

The loader auto-detects `quantization=quark`; an explicit quantization flag is not required for the validated recipe.

## Evaluation

### Agentic benchmarks

Independent evaluations were run on AMD Instinct MI355X infrastructure using the official benchmark harnesses and sampling settings.

#### DeepSWE v1.1

[DeepSWE](https://deepswe.datacurve.ai/) was evaluated with the official `mini-swe-agent` harness, `temperature=0.95`, `top_p=1.0`, a 6-hour task timeout, and a 400K context limit. The table reports the multi-run score and the reported variation.

| Model / reference | Pass@1 |
|---|---:|
| DataCurve official | 69.0% ± 3.0% |
| GLM-5.3 FP8 | 70.6% ± 4.9% |
| **GLM-5.3 MXFP4** | **68.9% ± 4.0%** |

#### Terminal-Bench 4.0

| Model / endpoint | Pass@3 |
|---|---:|
| Official GLM-5.3 API | 55.87% |
| **GLM-5.3 MXFP4** | **55.87%** |

### SGL-eval accuracy (16K output cap)

This MXFP4 release and the BF16 oracle were evaluated with [sgl-eval](https://github.com/sgl-project/sgl-eval) using `temperature=0`, `seed=0`, `reasoning_effort=max`, and a 16,384-token maximum output limit. The MXFP4 endpoint used the stock InferenceX recipe above on four MI350 GPUs. “16K” below is the maximum output-token limit, not the input-context limit.

Definitions:

- **Completed:** requests with a recorded evaluator result.
- **Raw accuracy:** correct ÷ completed.
- **Truncated:** requests ending because the maximum output-token limit was reached.
- **Excluding truncation:** correct ÷ (completed − truncated).
- **Recovery:** MXFP4 accuracy ÷ BF16 accuracy.

#### Results

| Benchmark | Model | Completed | Correct | Raw accuracy | Truncated | Excluding truncation | Recovery (raw / excl. trunc.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K, flexible extract | BF16 oracle | 1,319/1,319 | 1,288 | 97.65% | 2 (0.15%) | 1,288/1,317 = 97.80% | — |
| GSM8K, flexible extract | **MXFP4** | 1,319/1,319 | **1,286** | **97.50%** | **2 (0.15%)** | **1,286/1,317 = 97.65%** | **99.84% / 99.84%** |
| MMLU | BF16 oracle | 500/500 | 446 | 89.20% | 32 (6.40%) | 446/468 = 95.30% | — |
| MMLU | **MXFP4** | 500/500 | **443** | **88.60%** | **34 (6.80%)** | **443/466 = 95.06%** | **99.33% / 99.75%** |
| GPQA | BF16 oracle | 198/198 | 107 | 54.04% | 87 (43.94%) | 107/111 = 96.40% | — |
| GPQA | **MXFP4** | 198/198 | **105** | **53.03%** | **91 (45.96%)** | **105/107 = 98.13%** | **98.13% / 101.80%** |

Across all 2,017 questions, BF16 scores 1,841 correct and MXFP4 scores **1,834**. The MXFP4 release therefore recovers **99.62%** of the BF16 combined correct-count rate.

Among rows where both MXFP4 and BF16 produce parsed answers, selected-answer agreement is:

- GSM8K: 1,303/1,316 = **99.01%**;
- MMLU: 446/450 = **99.11%**;
- GPQA: 88/89 = **98.88%**.

The paired row-identity SHA-256 values are `847cdb1e2b395e2542ac293f4abbb76a5a6614458ec6dfe7da8ae7e3032034f7` for GSM8K, `cd7e1e554596ede9b565164f69cce2ef6b7b977202bb4674a519c82c1d6842ab` for MMLU, and `e40093cba458dad6ba216ff3aa83516250627f3463f5188786e7c71d3b0c1565` for GPQA.

### Reproduction

After starting the endpoint, install `sgl-eval` and run:

```bash
sgl-eval run gsm8k \
  --num-examples 1319 \
  --num-threads 32 \
  --max-tokens 16384 \
  --temperature 0 \
  --seed 0 \
  --reasoning-effort max \
  --base-url http://localhost:30000/v1 \
  --model glm-5.3-mxfp4

sgl-eval run mmlu \
  --num-examples 500 \
  --num-threads 32 \
  --max-tokens 16384 \
  --temperature 0 \
  --seed 0 \
  --reasoning-effort max \
  --base-url http://localhost:30000/v1 \
  --model glm-5.3-mxfp4

sgl-eval run gpqa \
  --num-examples 198 \
  --num-threads 16 \
  --max-tokens 16384 \
  --temperature 0 \
  --seed 0 \
  --reasoning-effort max \
  --base-url http://localhost:30000/v1 \
  --model glm-5.3-mxfp4
```

For a strict paired comparison, reuse one frozen dataset artifact for BF16 and MXFP4 rather than relying on the same seed to recreate the same question mapping.

## Limitations

- This is a post-training MXFP4 checkpoint. It can differ numerically and behaviorally from BF16, especially on long reasoning traces near an output-token cap.
- Validation used the stock SGLang ROCm/AITER path described above. Other engines, hardware, and kernel implementations require independent compatibility and accuracy checks.

## License

This checkpoint is distributed under the source model’s GLM-5.3 license. See `LICENSE` and the [GLM-5.3-BF16 model card](https://huggingface.co/zai-org/GLM-5.3-BF16) for source-model details and citation information.
