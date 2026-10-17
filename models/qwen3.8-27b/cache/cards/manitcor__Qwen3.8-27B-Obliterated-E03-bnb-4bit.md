---
license: apache-2.0
base_model:
  - manitcor/Qwen3.8-27B-Obliterated-E03
  - Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: transformers
language:
  - en
tags:
  - qwen
  - qwen3.8
  - bitsandbytes
  - nf4
  - 4-bit
  - abliteration
  - mechanistic-interpretability
  - refusal-research
---

# Qwen3.8-27B-Obliterated-E03 BitsAndBytes NF4

This is the independently qualified Transformers/BitsAndBytes NF4 derivative
of [`manitcor/Qwen3.8-27B-Obliterated-E03`](https://huggingface.co/manitcor/Qwen3.8-27B-Obliterated-E03).
It targets single-GPU inference on a 24 GB NVIDIA RTX 4090 while preserving the
behavior measured for the BF16 E03 checkpoint.

This is post-training quantization, not knowledge distillation. No teacher,
student, fine-tuning, or additional weight-training stage was used.

The model is a text-only derivative of `Qwen/Qwen3.8-27B`. Qwen3.8 uses the
`qwen3_5` hybrid architecture: 48 Gated DeltaNet/linear-attention layers and 16
full-attention layers. The original multimodal vision tower is not included.

## Immutable provenance

- Upstream base: `Qwen/Qwen3.8-27B`
- Upstream revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Qualified BF16 source: `manitcor/Qwen3.8-27B-Obliterated-E03`
- BF16 source revision: `56bbc4a80c17353254c0ed0f31828e3980970495`
- OBLITERATUS E03 run: `run-c18babdb7be34396a07fbb994c23d4cb`
- OBLITERATUS merge: `cd8b0b785c8b32fd059f645b5b8c1d53f67f01f2`

All 28 source shard sizes and SHA-256 hashes were matched against the immutable
Hub revision before quantization.

## Inherited OBLITERATUS configuration

Quantization did not repeat or alter the refusal-direction intervention. The
qualified BF16 parent was created with the following OBLITERATUS configuration:

| Setting | Value |
|---|---|
| Method | `qwen38_e03` |
| Direction extraction | SVD |
| Refusal directions | 4 |
| Layer selection | Middle 60%, layers 12–50 (39 layers) |
| Projection target | Output projections |
| Matrices modified | 312 |
| Norm preservation | enabled |
| Regularization | 0.10 |
| Refinement passes | 1 |
| Chat template | enabled |
| Activation winsorization | enabled |
| KL co-optimization | enabled, 0.5-nat budget |
| Bias projection | disabled |
| Quantization during surgery | none (BF16) |

The intervention used 500 harmful/harmless pairs, a separate 142-pair
optimizer-tune split, and a single-use 200-pair final holdout. Full
machine-readable configuration and partitioned evaluation records are in the
immutable BF16 parent revision above.

## Quantization

| Setting | Value |
|---|---|
| Backend | BitsAndBytes |
| Weight format | NF4, 4-bit |
| Nested/double quantization | enabled |
| Compute dtype | BF16 |
| Quantization storage | UINT8 |
| CPU offload during qualification | none |
| Quantized `Linear4bit` modules | 496 |
| Saved model shards | 9 |
| Saved artifact size | approximately 17.67 GB |

Conversion used PyTorch 2.10.0, Transformers 5.8.1, BitsAndBytes 0.50.0,
Accelerate 1.14.0, and Hugging Face Hub 1.28.0 on an NVIDIA A100 80 GB.

### Reproduction recipe

The conversion loaded BF16 revision
`56bbc4a80c17353254c0ed0f31828e3980970495` with native BF16 weights,
`device_map={"": 0}`, and the following quantization configuration:

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

source = "manitcor/Qwen3.8-27B-Obliterated-E03"
revision = "56bbc4a80c17353254c0ed0f31828e3980970495"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.uint8,
)
model = AutoModelForCausalLM.from_pretrained(
    source, revision=revision, quantization_config=bnb,
    torch_dtype=torch.bfloat16, device_map={"": 0},
    low_cpu_mem_usage=True,
)
model.save_pretrained("qwen38-e03-bnb4", safe_serialization=True,
                      max_shard_size="2GB")
```

Verify the source against `bf16-source-artifact-inventory.json` before
conversion. After saving, verify `quantized-artifact-inventory.json`, reload in
a fresh process, require exactly 496 `Linear4bit` modules, no CPU parameters,
and finite logits, then run the declared final-test gates. Conversion took
206.685 seconds with a 17,266,162,944-byte model footprint.

The original conversion, evaluation, and BF16-comparison programs had SHA-256
digests `454b07e9eba62df9060f9734135414f751146e97b5f48e2e2a74a1e8c89084d1`,
`d05bbe81c9964e61f7162f3d559aaa93692c46fe9f416ab8fb143b8d70836e77`,
and `2442826c567373c23252c2cd3ed6317d2362f99ed1c636e345b5d43dae8c9b46`.

## Independent qualification

The saved quantized artifact was reloaded in a fresh process. The evaluator
used the untouched 200-prompt final-test partition and the corrected
full-completion coherence scorer.

| Metric | BF16 E03 | NF4 artifact |
|---|---:|---:|
| Refusal rate | 1.0% | **0.5%** |
| Coherence | 100% | **100%** |
| Capability checks | 83.3% | **83.3%** |
| Reference perplexity | 3.2099 | **3.4106** |
| Perplexity increase from BF16 | — | **6.25%** |
| Harmful-output degeneracy | 3.0% | **2.5%** |
| Mean first-token KL vs BF16, 33 prompts | — | **0.01943** |
| Maximum first-token KL | — | **0.3652** |

Structural/runtime checks passed:

- all 496 expected modules reloaded as BitsAndBytes `Linear4bit`;
- no parameters were CPU-offloaded;
- all sampled logits were finite;
- no NaN/Inf or repetitive-token collapse was accepted;
- peak CUDA allocation during A100 reload was approximately 22.74 GB.

Ten deterministic 64-token comparisons were also retained. Quantization
changed the exact token sequences in all ten, so this artifact should not be
treated as byte-for-byte equivalent to BF16 despite passing the semantic and
capability gates. See `EVALUATION.md` and `qualification/` for evidence.

The comparison record's `elapsed_seconds` field contains a wall-clock epoch
timestamp due to a recorder bug; it is not a duration and must not be used in
performance comparisons.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "manitcor/Qwen3.8-27B-Obliterated-E03-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
)

messages = [{"role": "user", "content": "Explain eigenvalues concisely."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda:0")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Use a current Transformers/BitsAndBytes stack. Loading this repository without
BitsAndBytes support is not a supported fallback. Do not silently dequantize or
CPU-offload it when validating performance.

## Hardware notes

The artifact was designed for a 24 GB RTX 4090. A fresh A100 reload used about
17.66 GB steady CUDA allocation and peaked near 22.74 GB. Available context and
KV-cache capacity depend on the runtime, prompt length, and concurrent GPU
users. Start with batch size 1 and validate 2K, 4K, then 8K contexts.

## Limitations and safety

This model was deliberately modified to reduce refusal behavior. It can comply
with requests that the upstream model would reject. Deployers are responsible
for access controls, monitoring, policy enforcement, and compliance with law.

The automated refusal evaluator is lexical and the capability suite is small.
The model retains a measured 2.5% harmful-output degeneracy rate, and one of 33
first-token comparisons had KL divergence 0.3652. Independent human,
classifier, domain, multilingual, long-context, and adversarial evaluations are
recommended before consequential deployment.

The inherited `config.json` declares `mtp_num_hidden_layers: 1`, but the
serialized model contains 64 text layers and no MTP tensors. The qualified
Transformers/BitsAndBytes path loads it; converters that infer required tensors
from this optional field may fail. Bind results to an immutable revision and do
not present a locally edited config as the same artifact.

## License and attribution

Released under Apache License 2.0, following the upstream Qwen checkpoint.
This repository includes the upstream license and retains provenance for both
the BF16 E03 intervention and this NF4 conversion.
