---
license: other
license_name: glm-5.3
license_link: LICENSE
base_model: zai-org/GLM-5.3
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- furnace
- e2m1
- fp4
- glm_moe_dsa
---

# GLM-5.3 E2M1 Furnace

The full [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/tree/ba32f8c19c8738e2c57c71798c601d477a7bf844) converted to Furnace's packed E2M1 format. This is not GLM-5.3-Flash. Weight payload: **377.35 GiB**.

All routed experts, including the native MTP layer, use canonical low-nibble-first E2M1 packing with FP32 scales per 128×128 block. Attention, dense layers, shared experts, embeddings and normalization retain higher precision; FP8 non-routed tensors are reconstructed to BF16. This is a lossy weight conversion, not additional training. It is not a standard MXFP4 or GGUF checkpoint.

Requires [Furnace](https://github.com/m8than/furnace), tested at commit `9f016aab7e82c391f0d80c0bf711f46afa970173` on four AMD MI325X GPUs. Structural validation covers all 117,953 tensors and source checksums; every uploaded weight shard was checked against its size and SHA256. A short TP4 sampled-generation smoke test passed with native MTP. See `validation-structural.json` and `validation-tp4.json`.

```bash
SGLANG_USE_AITER=1 sglang serve \
  --model-path m8than/GLM-5.3-E2M1-Furnace \
  --served-model-name zai-org/GLM-5.3 \
  --tp 4 --quantization moe_e2m1 --dtype bfloat16 \
  --attention-backend dsa --dsa-prefill-backend aiter --dsa-decode-backend aiter \
  --kv-cache-dtype bfloat16 --page-size 64 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-kv-cache-dtype bfloat16 --speculative-draft-attention-backend dsa \
  --mem-fraction-static 0.85 --context-length 32768 --max-total-tokens 32768 \
  --chunked-prefill-size 8192 --max-running-requests 4 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --host 0.0.0.0 --port 8000
```

This conservative launch does not enable `--superfast`: that profile is restricted to GLM-5.2 in the tested Furnace revision. The conversion preserves the source context configuration, but this smoke test did not validate long-context operation. GLM-5.3 accepts `reasoning_effort` of `low`, `high`, or `max`; its default is `max`.

No full quality evaluation or throughput benchmark is claimed. The conversion manifest's relative L2 weight error is not a model accuracy or perplexity measurement.

The upstream GLM-5.3 license and copyright notice are included unchanged in `LICENSE`; the same license applies to this derivative.
