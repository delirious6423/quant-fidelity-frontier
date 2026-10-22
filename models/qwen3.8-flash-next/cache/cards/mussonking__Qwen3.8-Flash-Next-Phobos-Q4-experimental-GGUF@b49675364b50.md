---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags: [gguf, experimental, qwen]
---
# Qwen3.8-Flash-Next Phobos Q4 (experimental, archived)

4-bit Qwen3.8-Flash-Next quant restricted to the expert types of the Phobos/Strata engine:
experts gate/up IQ4_XS, down IQ4_NL; attention / shared experts / ssm_out Q8_0; small tensors BF16.
Exact command in `quantize.sh` (llama-quantize with the unsloth imatrix, from the unsloth BF16 GGUF).
Experimental, archived as-is, no support offered. License: Qwen Community License 1.0 (see LICENSE).
