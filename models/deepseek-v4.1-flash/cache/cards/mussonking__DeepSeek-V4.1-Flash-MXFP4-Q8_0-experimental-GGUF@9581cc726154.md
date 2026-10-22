---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags: [gguf, experimental, deepseek]
---
# DeepSeek-V4.1-Flash MXFP4 / Q8_0 GGUF (experimental, archived)

GGUF conversion of the official DeepSeek-V4.1-Flash weights: MoE experts kept in their native MXFP4,
other tensors Q8_0, 9 shards. Made 2026-10-09 as a numerical reference for an engine port.
**Requires a llama.cpp fork with the `deepseek_v41` architecture; mainline llama.cpp does not load it.**
Archived as-is, no support offered.
