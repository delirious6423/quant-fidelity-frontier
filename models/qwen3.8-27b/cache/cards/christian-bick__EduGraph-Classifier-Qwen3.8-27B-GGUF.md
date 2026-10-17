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

## Files and provenance

- Language model: `model-Q4_K_M.gguf` (SHA-256 `bd2ec1357e84b27b58ec0b1c83e143028f9b54e42989dc9f3db7c59aed870639`)
- Vision projector: `mmproj-BF16.gguf` (SHA-256 `3248a0391bed86eef3842121dab70d4657da244bde47ae02ce37580351bd0138`)
- Pinned llama.cpp converter/runtime commit: `0faee5004297c3bcfa40b7bf11750127b8c1fd7d`
- Selected training run: `edugraph-20261002-qwen38-27b-runpod-full-a40-v1`; GGUF validation run: `edugraph-20261004-qwen38-27b-gguf-benchmark-v4` at code commit `0a6cc246889ea1087df88126d26d5642e9b285a2`
- Training dataset: [christian-bick/edugraph-exercises](https://huggingface.co/datasets/christian-bick/edugraph-exercises) release `v0.30.0-03` at revision `9b509867e898490615be3f59bc2f31fac429389c`; ontology `0.30.0`
- `provenance.json` contains the source archive and curated-file checksums. The [immutable benchmark record](https://github.com/christian-bick/edugraph-classify/blob/60e2e2b710b262cc527137ffc75aa61fe868d679/docs/RUNPOD-GGUF-BENCHMARK-20261004.md) documents the corrected v4 score; the execution code remains pinned separately above.

## Use with llama.cpp

Use the included `chat_template.jinja`, `prompt.json`, and `closed_schema.json` together. The template injects the fixed classifier system prompt and disables thinking. The measured single-request path used a 2,048-token context, 1,024-token microbatch, GPU offload, greedy generation, a 512-token output ceiling, and an RGB JPEG image encoded at quality 75. Start a local server:

```bash
llama-server -m model-Q4_K_M.gguf --mmproj mmproj-BF16.gguf \
  --jinja --chat-template-file chat_template.jinja \
  --ctx-size 2048 --ubatch-size 1024 --parallel 1 \
  --n-gpu-layers all --fit off --host 127.0.0.1 --port 8080
```

Send one user message containing the text from `prompt.json`'s `user` field and the image as a JPEG data URL to `/v1/chat/completions`. Set `temperature: 0`, `max_tokens: 512`, `chat_template_kwargs: {"enable_thinking": false}`, and `response_format: {"type": "json_object", "schema": ...}`. Before sending the schema, construct a copy with `properties` in **`areas`, `scopes`, `abilities` order**. The saved schema may be serialized in a different key order; this request-time ordering is essential to reproduce the measured result. See [`completion_request`](https://github.com/christian-bick/edugraph-classify/blob/0a6cc246889ea1087df88126d26d5642e9b285a2/src/edugraph_classify/inference_benchmark.py) for the exact request builder.

The model returns explicit labels only. Do not treat ontology `integrates` relations or derived ancestors as model predictions. Verify labels against the pinned ontology version before downstream use.

## License and attribution

The EduGraph fine-tuned and Q4_K_M-converted model weights are released under Apache License 2.0. The included `LICENSE` is the exact license text from the [pinned upstream checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE) (SHA-256 `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`), including the Alibaba Cloud copyright notice.

**Modifications:** EduGraph trained a language adapter, merged it into the base weights, converted the result to GGUF, and quantized the language model. The vision projector remains BF16. The dataset and ontology have separate upstream ownership and terms.


## Evaluation

All validation rows below are the **same 64 original images** and gold labels; exact-set match compares complete explicit label sets. This is a diagnostic cohort used for checkpoint selection, not a fresh blind test.

| Configuration | Exact-set match |
|---|---:|
| Selected NF4 base plus adapter (training runtime) | 43/64 |
| Merged BF16 (Transformers runtime) | 40/64 |
| This Q4_K_M GGUF (llama.cpp, corrected property order) | **37/64** |

The GGUF run measured 93.69% micro label F1 and 0/64 invalid outputs. Its immutable report archive has SHA-256 `41bdd7535b410fc3f23a4095975ba9bfcbe6af692290a275455cf597ffe0b0d1`. The separate **181/250 final-assessment result belongs only to the NF4-plus-adapter configuration**; this GGUF export has not been scored on that final cohort. Differences between BF16 and GGUF also include conversion, serving, and JSON constraint mechanics, so they do not isolate quantization loss.

## Limitations

Training and evaluation used synthetic educational-task images and ontology `0.30.0`. Real-world classroom performance, unfamiliar label combinations, and a fresh post-refinement assessment are unmeasured. The GGUF run fit an L40S; its memory and timing do not establish fit, load behavior, or cold-start performance on a 24 GB GCP L4. This model is a research candidate and should be reviewed for the intended educational setting before use in decisions affecting learners.
