---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- qwen3_5
- text-only
- llama.cpp
- gguf
---
# Qwen3.8-27B tuneLLM GGUF (text only)

Unadapted, text-only quantization of `Qwen/Qwen3.8-27B`, pinned upstream revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`. The model weights occupy **16,742,036,832 bytes**.
Vision and MTP components are excluded; this package provides text generation.
No task adapter, merged adapter, or activation steering is included.
Modification notice: upstream weights/configuration were modified by text-only
extraction, exclusion of vision/MTP components, and quantization.

## Measured scope

| Runtime representation | General MMLU | GSM8K numeric fallback | GSM8K strict |
| --- | ---: | ---: | ---: |
| HF/bitsandbytes NF4 (native) | 30/32 | 26/32 | 26/32 |
| llama.cpp Q4_K_M | 29/32 | 27/32 | 26/32 |
| llama.cpp BF16 GGUF (local reference) | 29/32 | 26/32 | 25/32 |

These are fixed, small descriptive subsets, with batch one, greedy nonthinking
generation and a 512-token generation cap. MMLU uses custom zero-shot four-letter likelihoods.
The same screen includes 35 QASPER development questions; exact denominators,
custom text-evidence scores and caveats are in `evaluation.json`. These overlap
development/retention fixtures and are not an independent final benchmark.
The complete-text QASPER primary subset has 27 questions per arm.
NF4 and llama.cpp differ in both representation and backend. Compare Q4 with
the local BF16 GGUF reference for the same-backend measurement; do not attribute
every NF4/Q4 difference to quantization.
No matched BF16 Transformers reference was run; NF4's own quantization effect
was not isolated.

The original NF4 cached/full-prefix strict logit diagnostic exceeded tolerance
(maximum observed absolute difference 0.125). Its failure is retained. Short
token replay and functional reload checks do not establish numerical equivalence.
Timing includes model load and the recorded tasks; concurrent jobs and a sharded
BF16 reference limit speed comparisons. No isolated TTFT benchmark is claimed.

## Use

The only model binary here is `Qwen3.8-27B-text-Q4_K_M.gguf`.
Use the pinned llama.cpp build in `provenance.json`. After downloading the file
from an immutable release revision, start the checked completion server:

```bash
llama-server -m Qwen3.8-27B-text-Q4_K_M.gguf -ngl 99 -c 8192 \
  --host 127.0.0.1 --port 8080 --parallel 1 -t 4 -fa on
```

Then render the original nonthinking prompt explicitly, as in the evaluation:

```python
import requests
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.8-27B", revision="37561816e66b5b9153b1b64c10968c18c1e6f4a0")
prompt = tokenizer.apply_chat_template(
    [{"role":"user", "content":"Explain a band gap in two sentences."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
token_check = requests.post("http://127.0.0.1:8080/tokenize", json={
    "content":prompt, "add_special":False, "parse_special":True}, timeout=30)
token_check.raise_for_status()
assert token_check.json()["tokens"] == ids
response = requests.post("http://127.0.0.1:8080/completion", json={
    "prompt":ids, "n_predict":128, "temperature":0,
    "seed":42, "cache_prompt":False, "repeat_penalty":1.0,
    "samplers":["temperature"]}, timeout=300)
response.raise_for_status()
print(response.json()["content"])
```

The checked evaluation used the llama.cpp server completion API with exact
HF/llama.cpp prompt-token parity and prompt-cache reuse disabled. The BF16 GGUF
reference remains local and is not part of this payload. No importance matrix
was used. This file comes from the original higher-precision model, independently
of the bitsandbytes NF4 checkpoint.

Scores apply to this explicit HF-rendered, thinking-off token route. Default
llama.cpp chat-template behavior can differ and was not evaluated here.


Recorded mid-study Python environment: PyTorch 2.12.0+cu130, Transformers 5.23.0,
bitsandbytes 0.49.2, PEFT 0.21.0, Accelerate 1.12.0.
The original screen omitted per-arm package-version fields; the separately
dated runtime witness and environment lock preserve that provenance limit.
The GGUF runtime is pinned by llama.cpp revision and binary hashes in provenance.
The HF NF4 path uses reference PyTorch GDN/causal-convolution kernels;
optimized FLA/causal-conv packages were not installed. Hardware: RTX A6000 48GB;
one GPU per NF4/Q4 arm, two GPUs for the local BF16 GGUF reference.

Apache 2.0 upstream license is included. The screen is an evaluation of public
datasets; this payload redistributes aggregate results and item IDs, not dataset
text. Model and dataset
provenance, licenses, hashes and reproduction commands are supplied separately.
This model may produce unsupported scientific statements; the small screen does
not establish scientific reliability.

Part of [tuneLLM](https://github.com/RenormalizationTech/tuneLLM).

## Published-artifact verification

Anonymous download of commit `8cce64fc0767c8efea134468b23079ef2d693a8c` verified every published file. The downloaded model then passed the recorded GPU functional reload check. See `verification.json` for the scope and original commit. This documentation update records that earlier check; runtime files retain their verified hashes. The original strict cache-equivalence failure remains unresolved.
