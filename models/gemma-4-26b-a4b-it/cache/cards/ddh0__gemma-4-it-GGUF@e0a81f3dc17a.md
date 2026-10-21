---
license: apache-2.0
base_model:
- google/gemma-4-31B-it
- google/gemma-4-26B-A4B-it
- google/gemma-4-E4B-it
- google/gemma-4-E2B-it
---

These are miscellaneous GGUF quantizations of the instruct-tuned Gemma 4 series of models, released by Google.

For more information about Gemma, you should refer to the original model cards.

The chat template baked into these GGUFs is technically outdated, however, inference in llama.cpp should still work exactly as it should, thanks to these fixes:
- [llama.cpp#21704](https://github.com/ggml-org/llama.cpp/pull/21704): `common : better align to the updated official gemma4 template`
- [llama.cpp#21760](https://github.com/ggml-org/llama.cpp/pull/21760): `common/gemma4 : handle parsing edge cases`

For the latest official chat template, refer to the original model repo.
