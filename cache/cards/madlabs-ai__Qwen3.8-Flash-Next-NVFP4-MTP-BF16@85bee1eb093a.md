---
pipeline_tag: image-text-to-text
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - nvidia/Qwen3.8-Flash-Next-NVFP4
  - RadixArk/Qwen3.8-Flash-Next-NVFP4
license: other
license_name: nvidia-open-model-license
license_link: https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
tags:
  - qwen3.8
  - flash-next
  - nvfp4
  - modelopt
  - speculative-decoding
  - mtp
  - sglang
---

# Qwen3.8-Flash-Next-NVFP4-MTP-BF16

> **Serve this with SGLang, not `transformers`.** The `qwen4_exp`
> architecture is not in `transformers` (checked against 5.12.1: `qwen4_exp`
> is absent from `CONFIG_MAPPING_NAMES`), so the auto-classes cannot load it.
> See the serving section below.

NVIDIA's NVFP4 body with the official **BF16 multi-token-prediction head** grafted
on, so the checkpoint can serve with NEXTN/MTP speculative decoding on a single
RTX PRO 6000 Blackwell (96 GB, SM120, TP=1).

This is a **transplant, not a re-quantization**. No weight was re-derived.

## Why it exists

`nvidia/Qwen3.8-Flash-Next-NVFP4` is a ModelOpt `MIXED_PRECISION` export whose MTP
experts are FP8 block-scaled. SGLang's `_mtp_quant_config()` force-returns `None`
for `modelopt_mixed`, so the draft head cannot be loaded and speculative decoding
is unavailable — which is most of the reason to run this model at TP=1.

The fix is to take the 31 BF16 MTP tensors from the RadixArk export (the same
trained head, kept unquantized) and graft them onto the NVIDIA body.

## What was actually changed

* Pure-body shards are byte-identical to NVIDIA's.
* Mixed shards were rewritten to drop the FP8 MTP tensors.
* The 31 BF16 MTP tensors were taken from `RadixArk/Qwen3.8-Flash-Next-NVFP4`.
* `model.safetensors.index.json` was rebuilt with a **real** `metadata.total_size`
  (a zero there is rejected by SGLang's NVMe PLE preparer).
* `config.json` sets `text_config.ple_embedding_dtype = "float8_e4m3fn"`.

That last line is the one that makes it boot. SGLang decides the PLE embedding
dtype from that field alone (`qwen4_exp.py`), not from any quantization metadata.
RadixArk's config has it; NVIDIA's omits it. Without it the loader builds a
bfloat16 embedding and dies with `SSD Stream layer 0 is float8_e4m3fn, but SGLang
constructed a bfloat16 embedding`.

## VRAM cost

**Zero.** Measured delta against the RadixArk export is 0.00 GiB: the two bodies
are identical in size and dtype, and the MTP head is BF16 in both. 20/20 sampled
expert tensors differ in value (NVIDIA's quantization is genuinely different);
every non-expert tensor is byte-identical.

So this gets NVIDIA's quantization quality at RadixArk's memory footprint and
speculative-decoding capability.

## Serving (SGLang)

Verified on pennyroyal (`jpezzulli/sglang-rtxpro6000` @ `pennyroyal-v2.5.0`),
single RTX PRO 6000, TP=1:

```
--quantization modelopt_fp4          # NOT modelopt_mixed: the FP8 groups are gone
--speculative-algorithm NEXTN --speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--mamba-ssm-dtype bfloat16 --page-size 64
--reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

Pass `--quantization modelopt_fp4` explicitly. The stock NVIDIA export resolves
to `modelopt_mixed` and must *not* be given `--quantization` at all; this one is
the opposite, because the FP8 groups were stripped.

Measured single-stream decode **184.6 tok/s** with MTP, spec accept length
2.5–3.3 depending on workload, 92–99% prefix-cache hit on long agent contexts.

### Tool calling

If you drive this with an agent harness, set `SGLANG_TOOL_STRICT_LEVEL=2`. The
chat template satisfies both the `qwen3_coder` (XML) and `qwen` (native JSON)
detector rules, and under long context the model drifts to native-JSON tool
calls inside the XML envelope — emitting `<parameter=arguments>` instead of the
real parameter. At SGLang's default there is no grammar constraint for
`tool_choice: "auto"`, so nothing stops it. We measured 56 malformed calls per
380 requests before, and 1 in 56,000 after.

## Provenance and licence

This is a derivative work of three upstream artifacts, and their terms all apply:

| component | source | licence |
|---|---|---|
| base model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | see the Qwen model card |
| NVFP4 body | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | NVIDIA Open Model License |
| BF16 MTP head | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | see that model card |

Not affiliated with, endorsed by, or produced by NVIDIA, Alibaba/Qwen, or
RadixArk. All credit for the weights belongs to them; the only contribution here
is the graft and the config fix.
