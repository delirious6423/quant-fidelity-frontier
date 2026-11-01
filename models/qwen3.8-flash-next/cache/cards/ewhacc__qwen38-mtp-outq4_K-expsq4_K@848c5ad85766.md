---
license: other
license_name: qwen
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
---
# Qwen3.8-Flash-Next MTP head, GGUF

This is the MTP (multi-token prediction) layer of Qwen3.8-Flash-Next, quantized in GGUF.
Use it as the draft model for speculative decoding in llama.cpp or [tandem.cpp](https://github.com/ewhacc/tandem.cpp).

- Made with `convert_hf_to_gguf.py --mtp`, then `llama-quantize` (routed experts and output in Q4_K, the rest in Q8_0).
  Only the MTP layer was extracted and quantized; nothing else was changed.
- Usage: see the tandem.cpp README (`--spec-type draft-mtp -md qwen38-mtp-outq4_K-expsq4_K.gguf --spec-draft-n-max 3`).
- Licence: Qwen Community License 1.0, see LICENSE.