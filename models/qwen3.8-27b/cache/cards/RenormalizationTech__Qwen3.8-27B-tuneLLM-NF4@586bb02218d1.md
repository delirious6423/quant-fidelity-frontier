---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3_5
- text-only
- bitsandbytes
- nf4
---
# Qwen3.8-27B tuneLLM NF4 (text only)

Unadapted, text-only quantization of `Qwen/Qwen3.8-27B`, pinned upstream revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`. The model weights occupy **17,653,667,512 bytes**.
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

```python
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

repo = "RenormalizationTech/Qwen3.8-27B-tuneLLM-NF4"
revision = HfApi().model_info(repo).sha  # Resolve once; retain this immutable SHA.
print("Resolved release:", revision)
tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
model = Qwen3_5ForCausalLM.from_pretrained(
    repo, revision=revision, dtype=torch.bfloat16,
    device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=False,
).eval()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain a band gap in two sentences."}],
    add_generation_prompt=True, enable_thinking=False,
    return_dict=True, return_tensors="pt",
).to(model.device)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    ids = model.generate(**inputs, do_sample=False, use_cache=True,
                         max_new_tokens=128, logits_to_keep=1)
print(tokenizer.decode(ids[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

Load the saved quantization configuration directly; do not quantize these stored
weights again. Use the explicit text-only class shown above. PEFT preparation
casts floating modules to FP32 and defines a different measured runtime; the
separate adapters require that preparation. The baseline checkpoint has no adapter.


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

Anonymous download of commit `e7644d6f288cb8a9ba51856ca2c9fa68946f70fe` verified every published file. The downloaded model then passed the recorded GPU functional reload check. See `verification.json` for the scope and original commit. This documentation update records that earlier check; runtime files retain their verified hashes. The original strict cache-equivalence failure remains unresolved.
