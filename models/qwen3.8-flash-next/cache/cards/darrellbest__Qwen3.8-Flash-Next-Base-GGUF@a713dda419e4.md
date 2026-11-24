---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
---

# Qwen3.8-Flash-Next-Base-GGUF

Untuned Q8_0 reference GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

- `Q8_0/`: Unsloth's Q8_0 build, unchanged (6 shards). Credit: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
- `MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf`: the multi-token-prediction head in BF16.
- `mmproj-F16.gguf`: the vision projector.

The Coder-iQ builds (`darrellbest/Qwen3.8-Flash-Next-Coder-iQ-*`) are calibrated from this reference.

License: Qwen Community License 1.0 (see `LICENSE`).
