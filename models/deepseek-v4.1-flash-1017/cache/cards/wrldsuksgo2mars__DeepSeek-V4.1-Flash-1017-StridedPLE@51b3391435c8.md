---
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags: ["transformers", "safetensors", "deepseek_v41", "text-generation", "image-text-to-text", "license:mit", "eval-results", "endpoints_compatible", "8-bit", "fp8", "region:us"]
---
# DeepSeek V4.1 Flash 1017 StridedPLE

Byte-preserving storage repack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, under the original MIT license (see LICENSE).

Only the two Engram embedding tables (layers 1 and 14) change storage. Each `layers.{layer}.engram.embed.strided_{part:05}` tensor is U8 `[rows, 264]`: bytes 0..256 are the original F8_E4M3 weight row, bytes 256..264 are the original F8_E8M0 scales. These bytes are opaque storage, not quantized integer values. All values are byte-identical; there is no requantization. Every row was compared and the deinterleaved planes matched full source tensor SHA256 hashes. Every untouched tensor retains its source bytes. Shards 1..46 are unchanged; 47/48 retain all other tensors unchanged.

`config.json.engram_storage` declares `layout=fp8_row_interleaved_v1`, `weight_bytes=256`, `scale_bytes=8`, `tensor_suffix=strided`, plus an ordered `parts` map per layer. Each part declares `tensor` and `rows`; cumulative row counts define its global range. Each table has three contiguous row-range parts in separate shards below 45 GB; concatenate their row spaces in declared order. Rewritten safetensors headers also carry `engram_layout=fp8_row_interleaved_v1` in metadata. The original quantization_config is unchanged. A loader supporting this storage layout (CuteAFD) is required; unmodified Transformers does not load the interleaved table. GPU math is unchanged: the CPU gathers one contiguous row then splits existing weight/scale staging. No speed improvement is claimed. `MAPPED_SCALE_PRELOAD_GIB` does not preload interleaved scales; `ENGRAM_WARM` warms the whole table.

The original model implementation, tokenizer, license, and other repository assets are preserved. Refer to the source model card for model capabilities and usage requirements.
