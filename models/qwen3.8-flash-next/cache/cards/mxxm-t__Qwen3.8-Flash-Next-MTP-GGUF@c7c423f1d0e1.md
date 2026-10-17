---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - qwen
  - qwen3
  - mtp
  - nextn
  - speculative-decoding
  - draft-model
  - llama.cpp
---

# Qwen3.8-Flash-Next MTP draft head (GGUF)

The **NextN / MTP draft head** from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), extracted
and packaged as a standalone GGUF so it can be used as the draft model for
speculative decoding with the full Qwen3.8-Flash-Next as the target.

> **This is not a standalone language model.** It is a single prediction block
> (~2.6B parameters plus embeddings). It cannot generate text on its own and will
> produce nonsense if loaded as a normal model. It only does something useful when
> paired with the full Qwen3.8-Flash-Next as the target model.

## Files

| file | size | notes |
|---|---|---|
| `qwen4exp-mtp-head-Q8_0.gguf` | 4.14 GB | **recommended** |
| `qwen4exp-mtp-head-BF16.gguf` | 7.77 GB | full precision reference |

At temperature 0 both produce **identical draft tokens** on the prompts tested
(181 drafted, 136 accepted, 75.1% acceptance for both), so the Q8_0 file costs
nothing in draft quality and is the one to use.

## Usage

The head is a draft model for `llama-server`:

```
llama-server \
  -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md qwen4exp-mtp-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1 \
  -ngl 99 -ngld 99
```

`--spec-draft-n-max 2` is the measured optimum; deeper drafts accept less per
position and lose more than the extra depth gains.

## Requirements

These files were built with and are supported by
**[mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp)**, a llama.cpp fork with
qwen4exp MTP support, AMD gfx906 / MI50 tuning, tensor-parallel `-sm tensor` and a
custom AllReduce. That is the build to use them with.

```
git clone https://github.com/mxxm-t/mx-llama.cpp
```

Support for this architecture is also being upstreamed - review is in progress at
[ggml-org/llama.cpp#27836](https://github.com/ggml-org/llama.cpp/pull/27836). The
tensor names here (`blk.N.nextn.hc_head_*`) match that convention, so these files
are intended to keep working with upstream llama.cpp once it lands. Stock llama.cpp
releases do not support qwen4exp MTP yet.

## What acceptance to expect

Draft acceptance is strongly text-dependent, measured between roughly 46% and 90%
across different prompts on the same model and settings. That variation is normal
for this head, not a defect. Throughput follows acceptance closely, because a
rejected draft is far more expensive than an accepted one.

Whether speculative decoding is a net win depends heavily on your hardware and
split mode. On multi-GPU tensor-parallel setups the verify step can cost far more
than the drafting saves. Measure on your own configuration before assuming a gain.

## Provenance

Extracted from the original `mtp.*` tensors of Qwen3.8-Flash-Next and converted
with `convert_hf_to_gguf.py --mtp`. No weights were retrained, fine-tuned or
otherwise altered - only re-quantized (Q8_0) or format-converted (BF16).

## License

Released under the [Qwen Community License 1.0](LICENSE), inherited from the base
model. A copy of the licence is included in this repository as required by its
terms. Note clause 2: a separate licence from Qwen is required for Model as a
Service or AI Work Assistant commercial use.
