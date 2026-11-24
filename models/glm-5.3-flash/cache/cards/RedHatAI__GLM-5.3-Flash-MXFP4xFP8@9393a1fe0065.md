---
language:
- en
- zh
library_name: transformers
license: mit
pipeline_tag: image-text-to-text
tags:
- mxfp4
- fp8
- vllm
- llm-compressor
- compressed-tensors
- glm5_next
base_model: zai-org/GLM-5.3-Flash
---

# GLM-5.3-Flash-MXFP4xFP8

## Model Overview
- **Model Architecture:** Glm5NextForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** Mixed (MXFP4/FP8)
  - **Activation quantization:** FP8
- **Release Date:** 2026-11-23
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), ready for inference with vLLM.
It was evaluated on reasoning and agentic coding benchmarks to compare its quality with the original FP8 checkpoint (see [Evaluation](#evaluation)).

This is a mixed-precision quantization: the routed MoE experts use MXFP4 weights, and every other quantized layer keeps FP8 weights. All quantized layers use FP8 activations.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) to mixed MXFP4 and FP8 precision, ready for inference with vLLM.

MXFP4 quantization usually pairs MXFP4 weights with MXFP4 activations, a combination that runs best on Blackwell GPUs. In this model, the routed experts, which hold most of the parameters, instead pair MXFP4 weights with FP8 activations that are quantized dynamically per group. This makes the checkpoint well suited to inference on Hopper GPUs. The attention projections of the sparse-attention layers, the dense MLPs, the shared experts and the multi-token prediction (MTP) layer stay in FP8, with block-wise weight scaling and FP8 dynamic activations, as in the original checkpoint.

The original checkpoint already stores these weights in FP8. This model cuts the routed expert weights from 8 bits to about 4.25 bits per parameter, which reduces the disk size and GPU memory requirements by approximately 45% (from ~328 GB to ~182 GB).

Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```
vllm serve RedHatAI/GLM-5.3-Flash-MXFP4xFP8 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype bfloat16 \
  --no-enable-flashinfer-autotune \
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice
```

This model requires vLLM 0.32.0 or newer, the first release that runs MXFP4 weights with FP8 activations on Hopper. On Hopper GPUs this model must run with a BF16 KV cache; FP8 KV cache is supported only on Blackwell. For more serving options, see the [vLLM recipe for GLM-5.3-Flash](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash).

### Enable Speculative Decoding

The MTP layer is kept in FP8 and can be used for speculative decoding:

```
vllm serve RedHatAI/GLM-5.3-Flash-MXFP4xFP8 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype bfloat16 \
  --no-enable-flashinfer-autotune \
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a mixed-precision recipe. The routed expert layers use MXFP4 weight quantization (group size 32) with per-group (group size 128) FP8 dynamic activation quantization. The sparse-attention projections, dense MLPs, shared experts and MTP layer use FP8 block-wise (128×128) weight quantization with FP8 dynamic activations. The quantized model was exported in the [compressed-tensors](https://github.com/vllm-project/compressed-tensors) format.

## Evaluation

This model was evaluated on AIME 2026, GPQA Diamond, MMLU-Pro, SWE-bench Verified, DeepSWE 1.1 and Terminal-Bench 2.1 using [Inspect AI](https://inspect.aisi.org.uk/) ([inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) and [inspect_harbor](https://github.com/meridianlabs-ai/inspect_harbor)), served with vLLM 0.32.0 (OpenAI-compatible API). The agentic benchmarks use Inspect's ReAct agent. Evaluations were run on 8x H200 GPUs.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>zai-org/GLM-5.3-Flash (FP8)</th>
      <th>RedHatAI/GLM-5.3-Flash-MXFP4xFP8</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>Reasoning</b></td>
      <td>AIME 2026 (avg@8)</td>
      <td>94.2 ± 3.1</td>
      <td>92.9 ± 2.9</td>
      <td>98.7%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (avg@3)</td>
      <td>89.4 ± 2.0</td>
      <td>90.6 ± 1.8</td>
      <td>101.3%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (pass@1)</td>
      <td>87.6 ± 0.3</td>
      <td>87.3 ± 0.3</td>
      <td>99.7%</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Agentic Coding</b></td>
      <td>SWE-bench Verified (pass@1)</td>
      <td>75.6 ± 1.9 (378/500)</td>
      <td>76.2 ± 1.9 (381/500)</td>
      <td>100.8%</td>
    </tr>
    <tr>
      <td>DeepSWE 1.1 (pass@1)</td>
      <td>68.1 ± 4.4 (77/113)</td>
      <td>59.3 ± 4.6 (67/113)</td>
      <td>87.0%</td>
    </tr>
    <tr>
      <td>Terminal-Bench 2.1 (pass@1)</td>
      <td>80.9 ± 4.2 (72/89)</td>
      <td>79.8 ± 4.3 (71/89)</td>
      <td>98.6%</td>
    </tr>
  </tbody>
</table>

Scores are percentages, and ± is the standard error. Both models were evaluated with the same protocol, vLLM release and serving setup (one TP=8 replica, BF16 KV cache, MTP speculative decoding), so only the checkpoint differs. These numbers come from an open-source evaluation stack (Inspect AI's ReAct agent) and are **not directly comparable** to the figures on the GLM-5.3-Flash model card, which uses different agent harnesses (mini-swe-agent for DeepSWE, Claude Code for Terminal-Bench), sampling settings and timeouts.

DeepSWE 1.1 shows the largest gap, 10 tasks out of 113. Each run is a single pass over a small set of long-horizon tasks, so the difference is about 1.4 standard errors. On Terminal-Bench 2.1, the tasks give the agent internet access by design. All trajectories were audited, and any sample that fetched its task's published solution or tests was counted as a failure. This affected 2 tasks for the FP8 model and 3 for this model. Before that correction, the verifier scored both models 74/89.

### Reproduction

All benchmarks were run with [Inspect AI](https://inspect.aisi.org.uk/) against a vLLM 0.32.0 server, with thinking at maximum effort (`--reasoning-effort max`) and the sampling parameters from the model's `generation_config.json` (temperature 1.0, top_p 0.95, top_k unset). AIME 2026 was run for 8 epochs and GPQA Diamond for 3, and each is reported as the mean over epochs. All other benchmarks were run for a single epoch (pass@1). The single-turn benchmarks allow up to 163,840 output tokens. The agentic benchmarks allow 65,536 tokens per turn, with a per-sample time limit of 4 hours (2 hours for Terminal-Bench 2.1).

<details>

#### Serving

```
vllm serve RedHatAI/GLM-5.3-Flash-MXFP4xFP8 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --kv-cache-dtype bfloat16 \
  --no-enable-flashinfer-autotune \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort": "max"}' \
  --language-model-only
```

```
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=local
```

#### AIME 2026, GPQA Diamond, MMLU-Pro (inspect_evals)

```
inspect eval inspect_evals/aime2026 --model vllm/RedHatAI/GLM-5.3-Flash-MXFP4xFP8 \
  --epochs 8 --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 --reasoning-effort max \
  --max-tokens 163840 -M client_timeout=7200

inspect eval inspect_evals/gpqa_diamond --model vllm/RedHatAI/GLM-5.3-Flash-MXFP4xFP8 \
  --epochs 3 --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 --reasoning-effort max \
  --max-tokens 163840 -M client_timeout=7200

inspect eval inspect_evals/mmlu_pro --model vllm/RedHatAI/GLM-5.3-Flash-MXFP4xFP8 \
  --temperature 1.0 --top-p 0.95 --reasoning-effort max \
  --max-tokens 163840 -M client_timeout=7200
```

#### SWE-bench Verified (inspect_evals)

The official SWE-bench images are used instead of the package's default images, which keep later release tags of each project and so expose the reference fix.

```
inspect eval inspect_evals/swe_bench \
  -T image_name_template='swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest' \
  --model vllm/RedHatAI/GLM-5.3-Flash-MXFP4xFP8 \
  --temperature 1.0 --top-p 0.95 --reasoning-effort max \
  --max-tokens 65536 --message-limit 200 --time-limit 14400 \
  --max-connections 48 --max-sandboxes 48 \
  -M client_timeout=7200
```

#### DeepSWE 1.1 (inspect_harbor)

DeepSWE tasks declare no network access for the agent, but stock inspect_harbor 1.0.0 does not enforce that. The run therefore used a fork that does ([meridianlabs-ai/inspect_harbor#200](https://github.com/meridianlabs-ai/inspect_harbor/pull/200)). The tasks were loaded from a Harbor registry file pinned to a commit that carries a verifier fix (datacurve-ai/deep-swe#100).

```
inspect eval inspect_harbor/harbor \
  -T registry_path=harbor_registry.json -T dataset_name_version=deep-swe-1-1@1.1 \
  -T override_cpus=0 \
  --model vllm/RedHatAI/GLM-5.3-Flash-MXFP4xFP8 \
  --temperature 1.0 --top-p 0.95 --reasoning-effort max \
  --max-tokens 65536 --time-limit 14400 \
  --max-connections 48 --max-sandboxes 48 \
  -M client_timeout=7200
```

#### Terminal-Bench 2.1 (inspect_harbor)

```
inspect eval inspect_harbor/terminal_bench_2_1 \
  -T ref=sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a \
  -T override_cpus=0 \
  --model vllm/RedHatAI/GLM-5.3-Flash-MXFP4xFP8 \
  --temperature 1.0 --top-p 0.95 --reasoning-effort max \
  --max-tokens 65536 --time-limit 7200 \
  --max-connections 89 --max-sandboxes 89 \
  -M client_timeout=7200
```

</details>
