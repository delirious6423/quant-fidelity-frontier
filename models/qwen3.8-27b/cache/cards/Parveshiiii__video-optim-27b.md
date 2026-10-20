---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - quark
  - quantized
  - mxfp4
  - awq
  - qwen3_5
---

# Qwen3.8-27B-Quark-AWQ-MXFP4

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized to MXFP4 weights + MXFP4 activations with [AMD Quark](https://github.com/amd/Quark), using the **AWQ** (Activation-aware Weight Quantization) algorithm.

- **Weights:** MXFP4 (OCP microscaling FP4, group size 32, E8M0 shared scale)
- **Activations:** MXFP4, dynamic (per-block, computed at runtime)
- **Algorithm:** AWQ
- **Calibration:** 128 samples, seq_len 512, `pileval` dataset (Quark default)
- **Base model:** Qwen/Qwen3.8-27B (bf16)
- **Vision tower:** left unquantized (bf16) — only the language-model decoder (`model.language_model.layers`) is quantized; `model.visual.*` is excluded

## Benchmark results

| Benchmark | Setting | This model (AWQ) | BF16 base | Recovery % |
|---|---|---:|---:|---:|
| GSM8K, 5-shot (flexible-extract / strict-match) | Thinking: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0, max_gen_toks=3072 | 94.996% / 95.30% | 93.33% / 93.33% | 101.8% |
| GSM8K, 5-shot (flexible-extract / strict-match) | Non-thinking: temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, max_gen_toks=1024 | 89.92% / 89.76% | 90.67% / 89.76% | 99.2% |

Recovery % is computed on the flexible-extract metric (`this model / bf16 base`).

## Eval commands

Thinking mode:
```bash
lm-eval run --model local-completions \
  --model_args model=awq,base_url=http://localhost:8002/v1/completions,tokenizer=amd/Qwen3.8-27B-Quark-AWQ-MXFP4,num_concurrent=16,max_retries=5,timeout=900,tokenized_requests=False,max_length=16384 \
  --gen_kwargs max_gen_toks=3072 temperature=1.0 top_p=0.95 top_k=20 min_p=0.0 presence_penalty=0.0 repetition_penalty=1.0 do_sample=True \
  --tasks gsm8k \
  --output_path ./results/awq_gsm8k \
  --log_samples
```

Non-thinking mode: the raw-completions endpoint used above doesn't support passing `enable_thinking=False` through a chat template, so non-thinking mode is approximated with a custom task that pre-closes an empty `<think></think>` block in the prompt:

`gsm8k_nothink.yaml`:
```yaml
tag:
  - math_word_problems
task: gsm8k_nothink
dataset_path: openai/gsm8k
dataset_name: main
output_type: generate_until
training_split: train
fewshot_split: train
test_split: test
doc_to_text: "Question: {{question}}\nAnswer:<think>\n\n</think>\n\n"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\$"
      - "(?s).*#### "
      - "\\.$"
generation_kwargs:
  until:
    - "Question:"
    - "</s>"
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0
repeats: 1
num_fewshot: 5
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "regex"
        group_select: -1
        regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"
      - function: "take_first"
metadata:
  version: 3.0
```

```bash
lm-eval run --model local-completions \
  --model_args model=awq,base_url=http://localhost:8002/v1/completions,tokenizer=amd/Qwen3.8-27B-Quark-AWQ-MXFP4,num_concurrent=16,max_retries=5,timeout=900,tokenized_requests=False,max_length=16384 \
  --include_path ./custom_tasks \
  --gen_kwargs max_gen_toks=1024 temperature=0.7 top_p=0.80 top_k=20 min_p=0.0 presence_penalty=1.5 repetition_penalty=1.0 do_sample=True \
  --tasks gsm8k_nothink \
  --output_path ./results/awq_gsm8k_nothink \
  --log_samples
```

## Quantization command

```bash
HIP_VISIBLE_DEVICES=<gpu> python3 quantize_quark.py \
  --model_dir Qwen/Qwen3.8-27B \
  --output_dir ./Qwen3.8-27B-w_mxfp4_a_mxfp4-awq \
  --quant_scheme mxfp4 \
  --num_calib_data 128 \
  --seq_len 512 \
  --quant_algo awq \
  --model_export hf_format \
  --data_type auto \
  --device cuda \
  --trust_remote_code
```

## Serving

```bash
vllm serve amd/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --served-model-name awq \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 16384
```

## License

Modifications Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved.
