---
license: other
license_name: glm-5.3
base_model: zai-org/GLM-5.3
base_model_relation: quantized
library_name: colibri
pipeline_tag: text-generation
tags:
  - colibri
  - moe
  - int4
  - cpu
  - expert-streaming
language:
  - en
  - zh
---

# GLM-5.3 for colibrì (int4-gs64)

An int4 group-scaled conversion of
[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3), packed in the
container format that [colibrì](https://github.com/JustVugg/colibri) streams
from disk.

**This is not the official checkpoint.** The weights are Z.ai's; this repository
only changes how they are stored. It will not load in transformers or vLLM. It
loads in colibrì's `colibri` engine and nothing else.

## What it is for

Running a 744B-class model on a machine with about 25 GB of RAM, by keeping the
routed experts on disk and reading only the ones each token selects.

| | |
|---|---|
| on disk | 419.3 GB across 141 shards |
| resident dense | 10.6 GB |
| total RSS | about 13.6 GB with an 18 GB budget |
| shape | 78 layers (75 sparse), 256 experts, 8 active per token |
| routed experts | int4, group size 64, 21.2 MB each |
| dense / io | 8-bit |
| MTP | not included in this conversion |

## Use

```bash
git clone https://github.com/JustVugg/colibri && cd colibri
make -C c colibri
c/coli chat --model /path/to/this/folder
```

The engine detects the family from `config.json` and picks `colibri`
automatically.

## One thing the output gets wrong today

Z.ai state it plainly on the upstream card:

> GLM-5.3 uses the same base model as GLM-5.2. Every gain comes from
> post-training.

That is visible in the files. The two checkpoints have identical geometry, and
their `config.json` differ only by the `transformers_version` that wrote them
and one extra key. There is nothing architectural to tell them apart.

The practical consequence is that colibrì currently announces this checkpoint as
**"GLM-5.2 744B"**, because that is what the configuration says and no heuristic
over the configuration can ever say otherwise. The engine loads and runs the
right weights; only the label is wrong. Naming it correctly requires taking the
name from outside the config, and that is tracked in the colibrì repository.

## Provenance and what has been verified

Converted from the official `zai-org/GLM-5.3` release with colibrì's converter
at `ebits=4, xbits=4, io_bits=8, group_size=64`.

All 141 shards were checked for header validity and truncation before upload:
116,915 tensors, no problems found.

What has been confirmed by running it: the model loads, generates coherent text,
and produces byte-identical output across two different expert-fetch paths under
a fixed seed. What has **not** been done: a token-exact comparison against the
reference implementation. If you run that comparison, please open an issue in
the colibrì repository with the result, whichever way it goes.
