---
tags:
- fp4
- vllm
- llm-compressor
- compressed-tensors
- 8-bit
- mxfp4
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-MXFP4

## Model Overview
- **Model Architecture:** Qwen3_5ForConditionalGeneration
  - **Input:** Text / Image / Video
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Release Date:** 2026-09-18
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

### Model Optimizations

This model was obtained by applying GPTQ with AWQ activation smoothing via [LLM Compressor](https://github.com/vllm-project/llm-compressor), quantizing the weights and activations of the linear operators to the MXFP4 (FP4) data type and the KV cache to FP8, ready for inference with vLLM.

This optimization reduces the number of bits per parameter from 16 to 4 for the quantized weights and activations, reducing disk size and GPU memory requirements by approximately 75% (non-quantized components are kept in their original BF16 precision).

## Deployment

### vLLM Serving

```
vllm serve RedHatAI/Qwen3.8-27B-MXFP4 \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

Additional model-specific options (long-context scaling, speculative decoding via the built-in MTP head, etc.) are documented in the [vLLM recipe for Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with calibration samples from `perfectblend`, as presented in the code snippet below.

<details>

```python
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.8-27B"

# Load model.
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)


recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        targets="Linear",
        scheme="MXFP4",
        ignore=[
            "re:visual.*",
            "re:model.visual.*",
            r"re:.*lm_head",
            "re:.*embed_tokens$",
            r"re:.*linear_attn\.in_proj_a$",
            r"re:.*linear_attn\.in_proj_b$",
        ],
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "symmetric": True,
            "strategy": "tensor",
            "dynamic": False,
            "observer": "static_minmax",
        },
    ),
]

# Apply quantization.
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset="perfectblend",
    splits="train[:512]",
    max_seq_length=4096,
    num_calibration_samples=512,
    moe_calibrate_all_experts=True,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

</details>

## Evaluation v1

This model was evaluated on GSM8K Platinum, IFEval, MATH-500, AIME 2025, and GPQA Diamond using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness) and [lighteval](https://github.com/neuralmagic/lighteval), all served with vLLM (OpenAI-compatible API). Evaluations were run on 1× NVIDIA B200.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>Qwen/Qwen3.8-27B</th>
      <th>RedHatAI/Qwen3.8-27B-MXFP4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Instruction Following</b></td>
      <td>IFEval (prompt-level strict)</td>
      <td>91.19%</td>
      <td>90.88%</td>
      <td>99.7%</td>
    </tr>
    <tr>
      <td rowspan="4"><b>Reasoning</b></td>
      <td>GSM8K Platinum (strict-match)</td>
      <td>96.25%</td>
      <td>95.18%</td>
      <td>98.9%</td>
    </tr>
    <tr>
      <td>MATH-500 (pass@1)</td>
      <td>83.67%</td>
      <td>84.67%</td>
      <td>101.2%</td>
    </tr>
    <tr>
      <td>AIME 2025 (pass@1)</td>
      <td>96.67%</td>
      <td>92.50%</td>
      <td>95.7%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (pass@1)</td>
      <td>89.56%</td>
      <td>87.71%</td>
      <td>97.9%</td>
    </tr>
  </tbody>
</table>

### Reproduction

All tasks were evaluated 0-shot with seeds 1234/2345/3456 (3 repetitions); AIME 2025 used 8 seeds (1234/2345/3456/4567/5678/6789/7890/8901). The model was served with vLLM (the [Deployment](#deployment) command) and the results were obtained using the following commands:

<details>

#### GSM8K Platinum & IFEval (lm-eval, 0-shot, 3 repetitions)

```
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/Qwen3.8-27B-MXFP4,max_length=69632,base_url=http://127.0.0.1:3233/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
  --num_fewshot 0 \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,seed=1234,max_gen_toks=32000" \
  --apply_chat_template \
  --output_path results_gsm8k_platinum.json
```

Run once per seed (changing `--seed` and the `seed` in `gen_kwargs` each time); use `--tasks ifeval` for IFEval.

#### MATH-500, AIME 2025, GPQA Diamond (lighteval, 3 repetitions; 8 for AIME 2025)

**litellm_config.yaml:**
```yaml
model_parameters:
  provider: hosted_vllm
  model_name: hosted_vllm/RedHatAI/Qwen3.8-27B-MXFP4
  base_url: http://127.0.0.1:3233/v1
  concurrent_requests: 32
  timeout: 3600
  generation_parameters:
    temperature: 1.0
    top_p: 0.95
    top_k: 20
    seed: 1234
    max_new_tokens: 65536
```

Run once per seed (changing `seed` in the config each time):
```
lighteval endpoint litellm litellm_config.yaml 'math_500|0' \
  --output-dir results/ --save-details

lighteval endpoint litellm litellm_config.yaml 'aime25|0' \
  --output-dir results/ --save-details

lighteval endpoint litellm litellm_config.yaml 'gpqa:diamond|0' \
  --output-dir results/ --save-details
```

</details>


## Evaluation v2

We also assess performance of this model against a comparable [amd/Qwen3.8-27B-Quark-AWQ-MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) checkpoint and observe the following results:

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>RedHatAI/Qwen3.8-27B-MXFP4</th>
      <th>amd/Qwen3.8-27B-Quark-AWQ-MXFP4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>Reasoning & Knowledge</b></td>
      <td>GPQA Diamond</td>
      <td style="background-color:#d9ead3; text-align:center"><b>87.4%</b></td>
      <td style="text-align:center">86.5%</td>
    </tr>
    <tr>
      <td>MMLU-Pro</td>
      <td style="background-color:#d9ead3; text-align:center"><b>84.4%</b></td>
      <td style="text-align:center">84.1%</td>
    </tr>
    <tr>
      <td>AIME 2026</td>
      <td style="background-color:#d9ead3; text-align:center"><b>97.8%</b></td>
      <td style="text-align:center">94.4%</td>
    </tr>
    <tr>
      <td rowspan="4"><b>Coding & Agentic</b></td>
      <td>LiveCodeBench</td>
      <td style="background-color:#d9ead3; text-align:center"><b>90.3%</b></td>
      <td style="text-align:center">87.1%</td>
    </tr>
    <tr>
      <td>Terminal-Bench 2.1</td>
      <td style="background-color:#d9ead3; text-align:center"><b>55.4%</b></td>
      <td style="text-align:center">53.2%</td>
    </tr>
    <tr>
      <td>SWE-bench Verified</td>
      <td style="background-color:#d9ead3; text-align:center"><b>75.8%</b></td>
      <td style="text-align:center">72.3%</td>
    </tr>
    <tr>
      <td>DeepSWE 1.1</td>
      <td style="background-color:#d9ead3; text-align:center"><b>33.9%</b></td>
      <td style="text-align:center">22.1%</td>
    </tr>
  </tbody>
</table>


Evaluations were collected through [Inspect-AI](https://github.com/UKGovernmentBEIS/inspect_ai), and the results above are average scores of three seeds except for MMLU-Pro and DeepSWE where we run only one seed due to extensive runtime of the benchmark.
Both checkpoints were evaluated in the same environment: vLLM `vllm-0.31.0+rocm723` and on AMD MI355X GPUs.

