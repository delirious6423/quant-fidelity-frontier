---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
tags: [qwen, qwen3.8, gguf, mtp, speculative-decoding, strix-halo]
---

# Qwen3.8-Flash-Next MTP draft head (mtp- sidecar GGUF)


The native multi-token-prediction head of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (the 4B MTP
module the base model ships with), converted to a standalone `mtp-` sidecar GGUF
(Q8_0, 4.1 GB) directly from the official weights.

**Requires a patched llama.cpp** (as of 2026-08-27, upstream + PR #27742 cannot load or
run this): https://github.com/apepojken/llama.cpp/tree/qwen4exp-spec-mtp. With the patches, pass it as the draft model:

```
--spec-type draft-mtp --spec-draft-model mtp-Qwen-Qwen3.8-Flash-Next-Q8_0.gguf \
--spec-draft-ngl 999 --spec-draft-n-max 6
```

Measured on Strix Halo (Ryzen AI Max+ 395, Vulkan/RADV), 125B target at UD-Q3_K_XL:
decode 23.2 -> 49.8 tok/s (2.14x), draft acceptance 0.785 at n-max 6 (0.905 at n-max 2).

Conversion: `convert_hf_to_gguf.py Qwen/Qwen3.8-Flash-Next --remote --mtp --outtype q8_0`
on the patched tree (the converter gains an `--mtp` export path for this architecture).
The head runs dense attention in the draft context; its indexer tensors are included but
unused. Vision/PLE are not part of the head.
