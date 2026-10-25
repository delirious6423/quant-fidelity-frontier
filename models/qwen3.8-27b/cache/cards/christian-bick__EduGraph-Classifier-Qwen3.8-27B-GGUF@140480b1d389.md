---
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: gguf
base_model: Qwen/Qwen3.8-27B
datasets:
  - christian-bick/edugraph-exercises
tags:
  - gguf
  - llama-cpp
  - edugraph
---

# EduGraph Classifier — Qwen3.8-27B GGUF

This research model labels **one educational task image** with explicit EduGraph ontology identifiers in `areas`, `scopes`, and `abilities`. It is a language-adapter fine-tune of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, merged into the BF16 base and converted to Q4_K_M GGUF. The vision projector remains BF16. The two GGUF files in this repository are a pair; use both.

The training corpus covers **in-scope Common Core State Standards for Mathematics from Kindergarten through Grade 6 (K–6)**, using dataset v0.31.0-01 and ontology v0.31.0. The [Coverage Explorer](https://coverage.edugraph.io) documents competency coverage and scope exclusions. This describes curriculum scope, not verified model mastery of every standard. The selected adapter was trained on 2,330 images.

## Files and provenance

- Language model: `model-Q4_K_M.gguf` (SHA-256 `22506fbed9ad33421d7322719eb21dbe9c457aa3cdb816e8a48b44295e2723ff`)
- Vision projector: `mmproj-BF16.gguf` (SHA-256 `3248a0391bed86eef3842121dab70d4657da244bde47ae02ce37580351bd0138`)
- Pinned llama.cpp converter/runtime commit: `0faee5004297c3bcfa40b7bf11750127b8c1fd7d`
- Selected training run: `edugraph-20261024-qwen38-27b-runpod-k6-v1`; GGUF validation run: `edugraph-20261025-qwen38-27b-k6-gguf-v2` at code commit `7c9db04e3bc3cf9b018bf8912bac82ddc680c29e`
- Training dataset: [christian-bick/edugraph-exercises](https://huggingface.co/datasets/christian-bick/edugraph-exercises) release `v0.31.0-01` at revision `00ae2441669a42d1357c70d6f194bc71286eb464`; ontology `0.31.0`
- `provenance.json` contains the source archive and curated-file checksums. The [immutable benchmark record](https://github.com/christian-bick/edugraph-classify/blob/7d942c7fc09a020704c9ffeae436e654cd1f4ea6/docs/RUNPOD-GGUF-K6-20261025.md) documents the measured results; the execution code remains pinned separately above.

## Use with llama.cpp

Use the included `chat_template.jinja`, `prompt.json`, and `closed_schema.json` together. The template injects the fixed classifier system prompt and disables thinking. The measured single-request path used a 4,096-token context, 4,096-token microbatch, GPU offload, greedy generation, a 512-token output ceiling, and an RGB JPEG image encoded at quality 75. Start a local server:

```bash
llama-server -m model-Q4_K_M.gguf --mmproj mmproj-BF16.gguf \
  --jinja --chat-template-file chat_template.jinja \
  --ctx-size 4096 --batch-size 4096 --ubatch-size 4096 --parallel 1 \
  --n-gpu-layers all --fit off --host 127.0.0.1 --port 8080
```

Send one user message containing the text from `prompt.json`'s `user` field and the image as a JPEG data URL to `/v1/chat/completions`. Set `temperature: 0`, `max_tokens: 512`, `chat_template_kwargs: {"enable_thinking": false}`, and `response_format: {"type": "json_object", "schema": ...}`. Before sending the schema, construct a copy with `properties` in **`areas`, `scopes`, `abilities` order**. The saved schema may be serialized in a different key order; this request-time ordering is essential to reproduce the measured result. See [`completion_request`](https://github.com/christian-bick/edugraph-classify/blob/7c9db04e3bc3cf9b018bf8912bac82ddc680c29e/src/edugraph_classify/inference_benchmark.py) for the exact request builder.

The model returns explicit labels only. Do not treat ontology `integrates` relations or derived ancestors as model predictions. Verify labels against the pinned ontology version before downstream use.

## License and attribution

The EduGraph fine-tuned and Q4_K_M-converted model weights are released under Apache License 2.0. The included `LICENSE` is the exact license text from the [pinned upstream checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE) (SHA-256 `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`), including the Alibaba Cloud copyright notice.

**Modifications:** EduGraph trained a language adapter, merged it into the base weights, converted the result to GGUF, and quantized the language model. The vision projector remains BF16. The dataset and ontology have separate upstream ownership and terms.


## Evaluation

All validation rows below are the **same 64 original images** and gold labels; exact-set match compares complete explicit label sets. This is a diagnostic cohort used for checkpoint selection, not a fresh blind test.

| Configuration | Exact-set match |
|---|---:|
| Selected NF4 base plus adapter (training runtime) | 47/64 |
| This Q4_K_M GGUF (llama.cpp, ordered schema) | **43/64** |

The GGUF run measured 95.38% micro label F1 and 0/64 invalid outputs. Its immutable report archive has SHA-256 `51b3a6f5f2d6cca09fc1961d9f9422c52934c27b0d21466a787bb6199294a21e`. The separate **306/414 final-assessment result belongs only to the NF4-plus-adapter configuration**; this GGUF export has not been scored on that final cohort. Differences between configurations include precision, conversion, serving, and JSON constraint mechanics, so they do not isolate quantization loss.

## Limitations

Training and evaluation used synthetic educational-task images and ontology `0.31.0`. Both GGUF repeats produced identical outputs and 43/64 exact matches, versus 47/64 for the selected NF4 source: four fewer exact cases (6.25 percentage points). No matching standalone BF16 diagnostic was run. These are regression measurements: 147 of 403 training labels are absent from official validation, and 224 official validation images reuse bytes from a previously inspected assessment. That overlap covers selection and final validation together, not only this 64-image benchmark. Realized-range annotations, label completeness and a fresh assessment remain follow-up work. This GGUF was tested on the checkpoint-selection cohort; its performance on the separate 414-image final cohort is unmeasured. Runtime measurements apply to the tested hardware and do not establish fit or latency on other devices. Classroom performance and unfamiliar task types remain unmeasured.
