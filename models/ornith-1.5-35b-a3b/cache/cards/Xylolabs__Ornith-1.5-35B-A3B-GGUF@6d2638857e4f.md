---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language:
- en
- ko
tags:
- gguf
- llama-cpp
- quantized
- ornith
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Ornith 1.5 35B A3B text GGUF reference</h1>

[![Base model](https://img.shields.io/badge/base-ornith--ai%2FOrnith--1.5--35B--A3B-blue)](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
[![Formats](https://img.shields.io/badge/formats-mixed%20Q4__K__M-5B2C83)](#formats)
[![License](https://img.shields.io/badge/license-MIT-green)](https://huggingface.co/Xylolabs/Ornith-1.5-35B-A3B-GGUF/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-llama.cpp-black)](https://github.com/ggml-org/llama.cpp/tree/cd959bae99ff4bed22bef2604f45241c09bbb8c8)

`gguf` · `llama-cpp` · `quantized` · `ornith`

</div>

A preserved mixed Q4_K_M text-model reference of [Ornith 1.5 35B
A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), prepared by Xylolabs with llama.cpp.

This export needs about 21.70 GB of device memory for its GGUF weights alone, plus room for the KV cache and
buffers.

## Formats

`Ornith-1.5-35B-A3B-Q4_K_M.gguf` downloads 21.713 GB and averages 4.890 bits over all 35,505,251,456 stored
parameters. Actual types and counts: F32: 310, Q4_K: 379, Q6_K: 64.

All 753 tensors are retained, including 40 main layers, 20 additional MTP tensors, embedding, untied output
head and original tokenizer/chat-template metadata. MTP weights count toward the complete file size;
generation here uses the main text model. Vision encoder/projector weights are omitted. No new training,
calibration or quantization was performed for publication.

## Quickstart

With a [llama.cpp build](https://huggingface.co/docs/hub/gguf-llamacpp), download the file and run:

```bash
llama-cli -m Ornith-1.5-35B-A3B-Q4_K_M.gguf -c 2048 -n 32 --temp 0 --jinja \
  --reasoning off --single-turn -p "Name the capital of France. Reply with only the city."
```

This command is checked with llama.cpp v0.9.0. It runs the main text model with reasoning disabled. The
example uses 2,048 context tokens; original context metadata remains unchanged. KV, recurrent state and
working buffers need additional memory.

## Evaluation

**The current Q4 reference answers 31/33 plus 7/7, matching the stored BF16 totals, with zero counted
regressions or broken answers.** Mean KL is 0.079161 and top-1 agreement 87.63%. This is the Q4 baseline
itself, with a self-ratio of 1.000×.

The original contract is four 2,048-token English/Korean evaluation sequences and 8,188 scored positions.
Every sequence ID, token, first scored position, 33+7 prompt, stop ID and generation budget matches. The full
calibration/evaluation fingerprint matches. A teacher rebuilt in bounded chunks from the four original stored
PT outputs equals every original scored slice, without a new BF16 forward. Current llama.cpp GPU checks use
context 8,192.

The exact public reasoning-off CLI completes one query. MTP decoding and image inputs are not evaluated. These
checks cover retained texts and prompts; an integration needs its own task evaluation. Speed is not reported
here.

## License

The pinned primary Ornith card declares MIT and links a LICENSE file that is absent. LICENSE therefore
supplies standard MIT terms naming the original model authors and explicitly identifies that provenance. The
primary card describes Qwen3.5/Gemma4 family lineage; full Apache 2.0 terms and family-author attribution are
retained in APACHE_LICENSE and NOTICE. This is a family-level lineage statement, with no additional exact
initialization claim. Provided as is, without warranty or upstream endorsement.

## Citation

Use the citation provided by the [original Ornith model
card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B):

```bibtex
@misc{ornith_1_5,
    title = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
    url = {https://ornith.ai/ornith_1_5.html},
    author = {{Ornith Team}},
    year = {2026}
}
```

