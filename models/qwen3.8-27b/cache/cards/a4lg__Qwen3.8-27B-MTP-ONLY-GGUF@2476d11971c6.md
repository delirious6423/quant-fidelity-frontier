---
library_name: gguf
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- qwen3.5
- qwen3.8
- speculative-decoding
- mtp
- mtp only
- gguf
---

MTP-only GGUF subset of Qwen3.8 models
=======================================

This is a supplement for Qwen3.8-27B-based models (including fine-tunes and
abliterated models) **without MTP tensors**.

This repository contains an **MTP-only subset** of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
which provides the draft model for speculative decoding in the GGUF format.

It accelerates token generation using speculative decoding with the draft model
from the original Qwen model.  In most cases, this is sufficient to accelerate
Qwen-based derivative models even if this draft model is not trained from them.

Note that however, the performance metrics heavily depend on the derivative
model you use, your machine and your MTP settings.

**Benchmark it before blindly trusting it.**


Using this Model
-----------------

It can be used in two ways:

1.  As a separate draft model file (Method 1)
2.  As a donor for grafting the draft model into a Qwen-based model
    (Method 2; Recommended)

### Method 1: Separate Draft Model File

It is easy to begin with but memory-inefficient as it does not share
some tensors with the original model.

If you find the draft model can accelerate a Qwen-based model you use, grafting
the draft model (Method 2) is recommended (note: switching to Method 2 may
slightly change the acceptance rate).

If you use `llama-server`, you may configure like this:

```
llama-server \
    --model       Qwen3.6-27B-finetune-Q4_K_M.gguf \
    --model-draft Qwen3.6-27B-MTP-ONLY-Q4_K_M.gguf \
    ... \
    --spec-type draft-mtp
    --spec-draft-n-max 4
```

`--model` specifies the original Qwen-based model and
*new* `--model-draft` specifies a file from this repository.

You also need `--spec-type draft-mtp` to enable the draft model.

Once the draft model is enabled, you may configure the rest of MTP options
as you like (in this example, custom `--spec-draft-n-max` is specified).

### Method 2: Grafting the Draft Model (Using this as a Donor)

This is recommended.

First, download [`convert.py`](https://gist.github.com/buzz/1c439684d5e3f36492ae9f64ef7e3f67/7e1d6f929e141f4977f89f99e6a7df39f41eeffa)
written by [@buzz](https://github.com/buzz) to transplant MTP tensors.
You may also need to install some dependencies required by this script.

Then, you can run this script like:

```sh
# ./convert.py INPUT MTP OUTPUT
python3 convert.py \
    Qwen3.6-27B-finetune-Q4_K_M.gguf \
    Qwen3.6-27B-MTP-ONLY-Q4_K_M.gguf \
    Qwen3.6-27B-finetune-Q4_K_M+MTP.gguf
```

The second argument of `convert.py` is a GGUF file (donor) from this repository.

Once the grafted GGUF file is created, you may use this like a regular
Qwen model with embedded draft model.

This is an example for `llama-server` users.

```
llama-server \
    --model Qwen3.6-27B-finetune-Q4_K_M+MTP.gguf \
    ... \
    --spec-type draft-mtp
    --spec-draft-n-max 4
```

The output of `convert.py` must be specified as the model file name
(**DO NOT** use `--model-draft` in this case).

`--spec-type draft-mtp` enables the draft model transplanted into the main one.


Additional Quantization (`Q4_K_M`, `Q6_K` and `Q8_0`)
------------------------------------------------------

Quantized GGUF files are provided so that deploying the draft model easier.

To maximize the accuracy, tensor types are based on Unsloth quants
(see the table below):

| This Repo               | Unsloth      |
| ----------------------- | ------------ |
| `Q4_K_M`                | `UD-Q4_K_XL` |
| `Q5_K_M` (not provided) | `UD-Q5_K_XL` |
| `Q6_K`                  | `UD-Q6_K_XL` |
| `Q8_0`                  | `UD-Q8_K_XL` |

However, since `UD-Q5_K_XL` and `UD-Q6_K_XL` has exactly the same
quantization on MTP-related tensors, the `Q5_K_M` version is not
provided on this repository.

*It is not required to match the quantization level.*
For instance, you may pair `Q6_K`-quantized draft model with
the `Q4_K_S`-quantized main model.


Conversion Process
-------------------

*   Tools: [llama.cpp](https://github.com/ggml-org/llama.cpp) ([b10481](https://github.com/ggml-org/llama.cpp/releases/tag/b10481))

The `--mtp` option of `convert_hf_to_gguf.py` is the crucial part of this
conversion process because this option does exactly what the author expects:
create an MTP-only GGUF subset.

For additional quantization, the `llama-quantize` tool (llama.cpp)
is used without options.


License and Copyright
----------------------

For all GGUF files under this repository,
[the license terms of the original Qwen model (Apache License version 2.0)](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE)
applies (as the author of this repository did not perform any changes
significant enough for own copyright):

> Copyright 2026 Alibaba Cloud

No `NOTICE` files are attached in the original model.

This README file is licensed under the terms of [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

> Copyright 2026 a4lg.


Links: Sources
---------------

*   The Original Model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
*   Tensor Types (Unsloth): [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/27af057ecb382ddfea5d12837360a8980560e3ed)


Links: All MTP Subsets: Qwen3.5/3.6/3.8
----------------------------------------

*   [Qwen/Qwen3.5-9B](https://huggingface.co/a4lg/Qwen3.5-9B-MTP-ONLY-GGUF)
*   [Qwen/Qwen3.5-27B](https://huggingface.co/a4lg/Qwen3.5-27B-MTP-ONLY-GGUF)
*   [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/a4lg/Qwen3.5-35B-A3B-MTP-ONLY-GGUF)
*   [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/a4lg/Qwen3.5-122B-A10B-MTP-ONLY-GGUF)
*   [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/a4lg/Qwen3.5-397B-A17B-MTP-ONLY-GGUF)
*   [Qwen/Qwen3.6-27B](https://huggingface.co/a4lg/Qwen3.6-27B-MTP-ONLY-GGUF)
*   [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/a4lg/Qwen3.6-35B-A3B-MTP-ONLY-GGUF)
*   [Qwen/Qwen3.8-27B](https://huggingface.co/a4lg/Qwen3.8-27B-MTP-ONLY-GGUF)
*   [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/a4lg/Qwen3.8-2.4T-A95B-MTP-ONLY-GGUF)
