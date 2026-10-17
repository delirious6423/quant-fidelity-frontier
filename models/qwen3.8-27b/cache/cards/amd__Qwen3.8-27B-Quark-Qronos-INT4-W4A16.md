---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - quark
  - amd
  - quantization
  - int4
  - qronos
  - w4a16
---

# Qwen3.8-27B-Quark-Qronos-INT4-W4A16

INT4 weight-only (W4A16) quantized version of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced with the **Qronos** algorithm using [AMD Quark](https://github.com/amd/Quark).

- **Weights:** INT4, group size 128
- **Activations:** BF16 (unquantized)
- **Algorithm:** Qronos (Hessian-based post-training quantization)
- **Calibration:** 128 samples, sequence length 512
- **Base model:** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0)

## Benchmark results

| Benchmark | Setting | This model (Qronos) | BF16 base | Recovery % |
|---|---|---|---|---|
| GSM8K, 5-shot (flexible-extract / strict-match) | Thinking: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`, `max_gen_toks=8192` | 94.62% / 94.69% | 93.33% / 93.33% | 101.4% |
| GSM8K, 5-shot (flexible-extract / strict-match) | Non-thinking: `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`, `max_gen_toks=8192` | 89.99% / 89.84% | 90.67% / 89.76% | 99.2% |
| Wikitext perplexity | Greedy | 8.6819 | 8.4364 | 97.2% |
| BFCL Overall Acc (single_turn)* | Greedy (harness default) | 23.80% | 24.38% | 97.6% |

\* BFCL Overall Acc reflects only `single_turn` categories, not the full Gorilla-leaderboard formula (multi-turn/web-search/memory categories were not run and would count as 0 against the public leaderboard's own Overall Acc). Sub-metrics: Non-Live AST 85.17% (base 88.52%), Live AST 82.09% (base 83.05%), Relevance Detection 62.50% (base 75.00%), Irrelevance Detection 70.79% (base 72.22%).

Recovery % = quantized / BF16-base, using flexible-extract for GSM8K rows and base/quantized (inverted, since lower is better) for perplexity — both measured by us against verified-upstream `Qwen/Qwen3.8-27B` weights, not vendor-reported numbers. GSM8K uses [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness); non-thinking mode is approximated by pre-closing an empty `<think></think>` block in the prompt, since the harness task is a raw few-shot completion rather than a chat-templated request. BFCL run via the official [`bfcl_eval`](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) harness.

## Eval command

GSM8K, thinking mode, via [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)'s native vLLM backend:

```bash
lm-eval run \
  --model vllm \
  --model_args pretrained=amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.4,enforce_eager=True,trust_remote_code=True \
  --tasks gsm8k \
  --num_fewshot 5 \
  --gen_kwargs max_gen_toks=8192,do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=0.0,repetition_penalty=1.0 \
  --batch_size auto \
  --log_samples
```

GSM8K, no-thinking mode. The prompt suffix \<think\>\n\n\</think\>\n\n is prepended to each answer field to suppress the model's thinking preamble (see custom task yaml below).

```yaml
# gsm8k_nothink.yaml
task: gsm8k_nothink
dataset_path: openai/gsm8k
dataset_name: main
output_type: generate_until
training_split: train
fewshot_split: train
test_split: test
doc_to_text: "Question: {{question}}\nAnswer: <think>\n\n</think>\n\n"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore: [",", "\\$", "(?s).*#### ", "\\.$"]
generation_kwargs:
  until: ["Question:", "</s>", "<|im_end|>"]
  do_sample: false
  temperature: 0.0
repeats: 1
num_fewshot: 5
filter_list:
  - name: "strict-match"
    filter: [{function: "regex", regex_pattern: "#### (\\-?[0-9\\.\\,]+)"}, {function: "take_first"}]
  - name: "flexible-extract"
    filter: [{function: "regex", group_select: -1, regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"}, {function: "take_first"}]
metadata: {version: 3.0}
```

```bash
lm-eval run \
  --model vllm \
  --model_args pretrained=amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.4,enforce_eager=True,trust_remote_code=True \
  --tasks gsm8k_nothink \
  --include_path <dir containing gsm8k_nothink.yaml> \
  --num_fewshot 5 \
  --gen_kwargs max_gen_toks=8192,do_sample=True,temperature=0.7,top_p=0.80,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0 \
  --batch_size auto \
  --log_samples
```

`--gen_kwargs` on the CLI overrides the YAML task's own `generation_kwargs` defaults (greedy) with the instruct-mode recommended sampling parameters used for the non-thinking scores above.

## Quantization command

```bash
python3 quantize_quark.py \
  --model_dir Qwen/Qwen3.8-27B \
  --output_dir Qwen3.8-27B-Quark-Qronos-INT4-W4A16 \
  --quant_scheme int4_wo_128 \
  --num_calib_data 128 \
  --seq_len 512 \
  --quant_algo qronos \
  --model_export hf_format \
  --data_type auto \
  --device cuda
```

Run from `Quark/examples/torch/language_modeling/llm_ptq` using [AMD Quark](https://github.com/amd/Quark) with native `qwen3_5` architecture support for Qronos (contributed upstream).

## Serving

Requires a Quark-compatible inference runtime with `W4A16Int4` scheme support ([#48606](https://github.com/vllm-project/vllm/pull/48606) and [#46110](https://github.com/vllm-project/vllm/pull/46110)).

```bash
vllm serve amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3
```

## Citation

Quantized using the Qronos algorithm ([Zhang et al., ICLR 2026](https://arxiv.org/abs/2505.11695)):

```bibtex
@inproceedings{zhang2026qronos,
  title={Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization},
  author={Zhang, Shihao and Zhang, Haoyu and Colbert, Ian and Saab, Rayan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://arxiv.org/abs/2505.11695}
}
```

# License
Modifications Copyright(c) 2026 Advanced Micro Devices, Inc. All rights reserved.