---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
- hone
- qwen3_5_moe
- long-context
---

# Ornith-1.5-35B-A3B for hone (`groupwise-int-compact`)

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) converted to the `.hone`
container for [hone](https://github.com/tnculp/hone), a single-GPU inference server. This file is
what hone's installer downloads by default; it is not usable with other runtimes.

| | |
|---|---|
| File | `ornith_1_5_35b_a3b_compact.hone` |
| Size | 20,533,031,424 bytes |
| sha256 | `a23f0e8c9d7d6c1bc8728782c63c64d7aab4146c5b531d0ac245f07a746cab22` |
| Identity | `ornith-1.5-35b-a3b / groupwise-int-compact` |
| Source | `ornith-ai/Ornith-1.5-35B-A3B` @ `10fbf86fed7ecee4a061f8b499a618f46001cac1` |

## What the conversion changed

The weights are re-encoded, not retrained. The `groupwise-int-compact` profile stores most tensors as
grouped INT8 and moves the 40 routed MoE down-projections, the token embedding and the output head to
grouped Q4 (group 64), so the full 262,144-token context fits beside the weights on a 24 GB card. It
costs about 2.5 % perplexity against the unquantized-profile artifact (+0.5 % on code). The MTP head,
a proposal head for speculative decoding, the vision tower and the tokenizer and chat template are
included. `conversion.json` records every source file hash and the exact converter recipe.

## Use

Install hone and let it fetch this file (`install.ps1` downloads Ornith by default), or run it directly:

```powershell
hone-serve.exe ornith_1_5_35b_a3b_compact.hone --max-context 262144 --kv-capacity 262144 `
  --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 2 --lm-head-draft --host-kv-mib 2048
```

## Licence

Ornith-1.5-35B-A3B is released by its authors under the MIT licence (`LICENSE`); it is a continued
training of Qwen3.6-35B-A3B (Apache-2.0, `LICENSE-APACHE-2.0`). This conversion is distributed under
the same terms.
