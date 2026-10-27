---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
- qwen3_5
- slq
- mixed-precision
- gguf
- llama.cpp
- quantization
---

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/64b93e6bd6c468ac7536607e/mj6xac74jHGLqymiovObc.png" alt="The Kaitchup — AI on a Budget" width="100%" />
  <p><a href="https://kaitchup.substack.com/subscribe"><strong>Subscribe to my newsletter</strong></a> · <a href="https://buymeacoffee.com/kaitchup"><strong>Support</strong></a></p>
</div>

# Qwen3.8-27B-SLQ-GGUF

This is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), compressed with a **customized version of SLQ (Statistically-Lossless Quantization)** and assembled as GGUF. The search combines existing quantized donor weights and chooses a source separately for 498 complete matrices, including embeddings and the output head.

Choose one of three models:

| Model | GGUF file | Active text weights (GB) | Complete file (GB) | Measured EAR@10 ↑ |
|---|---|---:|---:|---:|
| EAR98 | [Qwen3.8-27B-SLQ-EAR98-Q5_K_S.gguf](Qwen3.8-27B-SLQ-EAR98-Q5_K_S.gguf) | 18.514 | 18.876 | 98.000383% |
| EAR98.5 | [Qwen3.8-27B-SLQ-EAR98.5-Q5_K_M.gguf](Qwen3.8-27B-SLQ-EAR98.5-Q5_K_M.gguf) | 20.341 | 20.703 | 98.500187% |
| EAR99 | [Qwen3.8-27B-SLQ-EAR99-Q6_K.gguf](Qwen3.8-27B-SLQ-EAR99-Q6_K.gguf) | 23.335 | 23.697 | 99.000026% |

- **EAR targets:** 98%, 98.5%, and 99%, selected using the BF16 reference's top 10 tokens.
- **Sizes:** decimal GB (1 GB = 1,000,000,000 bytes). Active text weights include fixed text tensors and exclude MTP, metadata and alignment.
- **Embeddings / output head:** Q4_K / Q8_0 for EAR98; Q4_K / BF16 for EAR98.5; Q6_K / BF16 for EAR99.
- Each file retains 351,008,768 bytes of quantized MTP tensors from the UD-Q5_K_XL donor. MTP was disabled during evaluation. No vision tensors or multimodal projector are included.
- **Made by:** [The Kaitchup](https://kaitchup.substack.com/).
- **License:** [Apache 2.0](#license).

The filename suffixes **Q5_K_S**, **Q5_K_M**, and **Q6_K** are Hub compatibility and dominant-precision-family labels for these **custom SLQ allocations**.

Method: [SLQ paper](https://arxiv.org/abs/2605.02404) · [SLQ repository](https://github.com/IST-DASLab/SLQ).

## Candidate sources

The search considered **UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q4_0, Q8_0, and BF16** from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), plus **IQ3_S** from [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF). Identical tensor payload alternatives were canonicalized. The final three models select no GSQ donor tensors; its candidates remained eligible during search.

Credit goes to the Qwen team for the base model, Unsloth for the GGUF donors, ISTA-DASLab for GSQ/RCO, and Michael Helcig, Eldar Kurtic and Dan Alistarh for SLQ. The base and donor repositories declare Apache 2.0 licensing; the full license is included [below](#license). Original copyright and attribution rights remain with their authors.


## Precision by layer

### EAR98

![EAR98: actual tensor formats and stored bits per weight](assets/precision-ear98.svg)

[SVG](assets/precision-ear98.svg)

### EAR98.5

![EAR98.5: actual tensor formats and stored bits per weight](assets/precision-ear98.5.svg)

[SVG](assets/precision-ear98.5.svg)

### EAR99

![EAR99: actual tensor formats and stored bits per weight](assets/precision-ear99.svg)

[SVG](assets/precision-ear99.svg)

## EAR comparison

**EAR means Expected Acceptance Rate.** It measures the overlap between the original BF16 model's next-token probabilities and those of the compared model. For EAR@K, both distributions are separately renormalized over the BF16 reference's K most likely tokens. Full-vocabulary EAR uses all 248,320 token IDs. An EAR of 0.99 means 99% average probability overlap on that measured support; it does not mean 99% benchmark accuracy or identical generated answers.


![EAR versus active text-weight size: 512 sequences of length 2,048](assets/ear-vs-size.svg)

[Zoomable SVG](assets/ear-vs-size.svg)

SLQ's “statistically lossless” objective concerns approximate fidelity, not identical numerical outputs. These releases meet their stated measured EAR@10 targets. The same calibration corpus guided model selection, so these values are calibration fidelity rather than an independent held-out benchmark. Statistical equivalence on downstream tasks has not been established for these releases.

## Instructions: llama.cpp

Use a llama.cpp build with Qwen3.8 architecture support. Native text scoring and standalone GGUF parity were validated on an NVIDIA GH200 with llama.cpp commit `45451c366d4256f94fbc8845645e31a6b9155054`. This validation did not cover image/video inference or speculative decoding.

Install the Hugging Face CLI, then download one file:

```bash
pip install -U huggingface_hub
hf download kaitchup/Qwen3.8-27B-SLQ-GGUF \
  Qwen3.8-27B-SLQ-EAR98-Q5_K_S.gguf --local-dir .
```

Start an OpenAI-compatible text server with the downloaded model:

```bash
llama-server \
  --model Qwen3.8-27B-SLQ-EAR98-Q5_K_S.gguf \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --port 8080
```

To use EAR98.5 or EAR99, download the corresponding file from the table and change `--model`.

Send a chat request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Is Earth smaller than Venus?"}],
    "max_tokens": 256
  }'
```


## Support

If these models are useful, [support the work](https://buymeacoffee.com/kaitchup) or [subscribe to The Kaitchup](https://kaitchup.substack.com/subscribe).

