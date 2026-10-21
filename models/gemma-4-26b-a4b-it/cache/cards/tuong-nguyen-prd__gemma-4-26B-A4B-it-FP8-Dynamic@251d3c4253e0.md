---
base_model:
- google/gemma-4-26B-A4B-it
tags:
- gemma4
- fp8
- vllm
- compressed-tensors
name: RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic
---

This is a preliminary version (and subject to change) of the FP8 quantized google/gemma-4-26B-A4B-it model. The model has both weights and activations quantized to FP8 with vllm-project/llm-compressor.

Run it with ```vllm serve RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic --max-model-len 96000``` on vllm main.

On a single B200:

```
lm_eval  --model local-chat-completions   --tasks gsm8k_platinum_cot_llama   --model_args "model=RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=2400"   --num_fewshot 5   --apply_chat_template   --fewshot_as_multiturn   --output_path results_gsm8k_platinum.json   --seed 1234   --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=64000,seed=1234"
```

Original:

```
|         Tasks          |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|------------------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k_platinum_cot_llama|      3|flexible-extract|     5|exact_match|↑  |0.9702|±  |0.0049|
|                        |       |strict-match    |     5|exact_match|↑  |0.9702|±  |0.0049|
```


FP8:

```bash
|         Tasks          |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|------------------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k_platinum_cot_llama|      3|flexible-extract|     5|exact_match|↑  |0.9669|±  |0.0051|
|                        |       |strict-match    |     5|exact_match|↑  |0.9669|±  |0.0051|
```