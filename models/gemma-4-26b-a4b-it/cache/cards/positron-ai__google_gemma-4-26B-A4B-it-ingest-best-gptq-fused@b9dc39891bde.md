---
base_model: google/gemma-4-26B-A4B-it
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
tags:
- positron-ai
- quantized
- gptq
---

# CI ingest checkpoint

Quantized checkpoint for CI ingestion and regression tests.

GPTQ: 4-bit, group size 64, symmetric, activation order disabled.

Per-expert gate/up projections are fused, with gate rows first.

Requires a consumer that supports this fused expert layout.

Published as a test fixture; no serving-qualification or accuracy claim.
