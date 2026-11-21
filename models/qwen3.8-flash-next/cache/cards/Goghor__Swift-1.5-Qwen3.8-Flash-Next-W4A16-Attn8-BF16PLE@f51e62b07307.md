---
library_name: transformers
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
---

Quantized Qwen3.8-Flash-Next (Swift 1.5 fine-tune, text + image + video, 262k context).
The PLE n-gram table is **BF16**. License: `LICENSE` (Swift Open License v1.0) and
`LICENSE-QWEN`. Provenance and per-file changes: `NOTICE`, `MANIFEST.json`.

| Component | Storage |
|---|---|
| Routed experts (512/layer, top-10) | AutoRound INT4 W4A16, group 128, packed `auto_round:auto_gptq` |
| Attention (36 `linear_attn` ×3 + 12 `self_attn` ×4) | INT8, per-channel, symmetric, round-to-nearest |
| **PLE n-gram table** | **BF16**, 128 shards, 102.4 GB |
| `embed_tokens`, `lm_head`, norms, vision tower, hyper-connection | BF16 |
| MTP draft head (1 layer) | experts INT4 g128, dense/indexer INT8 g128 (klee100) |

## PLE n-gram table (BF16)

- `model-plebf16-00000…00032.safetensors` (33 files). Keys
  `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_<0-127>.weight`,
  128 tensors of `2500012 x 160` = 51.2B parameters.
- `config.json` has no `text_config.ple_embedding_dtype` key, so the engine loads the
  table at model dtype. The PLE projections are BF16 in `intel-ple-small.safetensors`.
- Rows taken from `ukisai/Swift1.5-Qwen3.8-Flash-Next`. No scale tensors: the FP8
  build's `ngram_embedding.weight_scale` key and its `model-plefp8-*.safetensors` are absent.
- BF16 is the base model's dtype for this table: AutoRound lists `ple` in
  `--ignore_layers`, so the quantizer never touched it.

Serving with the table pinned in host RAM (`PLE_MODE=cpu`) costs **95.4 GiB**, not the
47.7 GiB of the FP8 table. Use the sibling `todiadiyatmo/…-Attn8-FP8PLE` to save that RAM.

Checkpoint: 174.7 GB, 227,834 tensors, 86 files. INT4 base:
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (auto-round W4A16, tuning mode; see
[auto-round](https://github.com/intel/auto-round) and [arXiv:2309.05516](https://arxiv.org/abs/2309.05516)).

The GSM8K/MMLU/PIQA/HellaSwag table in the Intel card (INT4 = 99.64% of BF16) measures
the plain INT4 model, not this fine-tuned build.
