---
license: mit
base_model: zai-org/GLM-5.3-Flash
library_name: vllm
tags:
- glm5_next
- ci
- test-asset
- pruned
---

# GLM-5.3-Flash-4L (vLLM CI asset)

A **layer-pruned slice of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**
(snapshot `03eb5366286afd40d2221b1d9c63a6dd1ba4832e`) meant for vLLM CI / repro of the
GLM-5.3-Flash serving stack (hybrid KDA + DeepSeek-sparse-MLA with kpool indexer, mHC, fp8 MoE, MTP,
PD disaggregation, KV offload). **It is not a usable language model**; its outputs are meaningless.

## What was kept / removed

Only whole decoder layers were removed; every width-like dimension that kernels see is untouched.

| | GLM-5.3-Flash | this checkpoint |
|---|---|---|
| decoder layers | 45 (3x KDA + 1x DSA-MLA period) | **4** = original layers 0, 1, 2 (KDA + dense MLP) and 3 (DSA-MLA + MoE) |
| MTP layer | `layers.45` | kept, renumbered to `layers.4` (`num_nextn_predict_layers=1`) |
| routed / shared experts | 288 / 1, top-8 | unchanged |
| KDA heads / head_dim | 64 / 128 | unchanged |
| MLA / indexer dims | 64 heads, q_lora 1536, kv_lora 512, nope 256, indexer 32 heads, kpool 4, topk 2048 | unchanged |
| mHC, fp8 block-128 quant, vocab / tokenizer, vision tower (24 blocks) | | unchanged |
| size | 328 GB | 18.5 GiB (5 safetensors shards) |

Config edits: `num_hidden_layers`, `layer_types`, `indexer_types`, `mlp_layer_types`,
`first_k_dense_replace` (=3), `linear_attn_config.kda_layers/full_attn_layers`, and
`quantization_config.modules_to_not_convert` renumbered for the kept layers. Weights are exact
copies of the original tensors.

## Behaviour in vLLM (vllm main, 2026-09-17, GB300)

- KV cache groups: `{MLA + indexer k_cache}`, `{kpool tail}`, and 3 Mamba groups aliasing the MLA slot,
  exactly like the full model.
- Auto block size: 4352 tokens (TP1), 2176 (TP2), 1152 (TP4, same as the full model).
- Requires SM90/SM100 (DeepGEMM paged-MQA indexer, FlashMLA/FlashInfer sparse MLA); does not run on L4.
- PD with NixlConnector needs `VLLM_SSM_CONV_STATE_LAYOUT=DS`.

```bash
vllm serve JaredforReal/GLM-5.3-Flash-4L --enforce-eager --max-model-len 32768
# with MTP:  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## License

MIT, inherited from GLM-5.3-Flash (see `LICENSE`). Derived work; all credit to Z.AI.
