---
library_name: transformers
pipeline_tag: text-generation
base_model: zai-org/GLM-5.3-Flash
tags:
  - glm5-next
  - model-surgery
  - dummy
  - test-model
---

# GLM-5.3-Flash 9B Surgery Dummy

Test-only checkpoint for GLM-5.3-Flash post-training integration. It is not a
usable chat or benchmark model. For cheaper smoke tests, use
[GLM-5.3-Flash-0.1B-A0.1B](https://huggingface.co/inference-optimization/GLM-5.3-Flash-0.1B-A0.1B).

GLM-5.3-Flash branches: [VERL](https://github.com/imvladikon/verl/tree/glm-5.3-flash),
[Slime](https://github.com/imvladikon/slime/tree/glm-5.3-flash),
[SGLang](https://github.com/imvladikon/sglang/tree/glm-5.3-flash), and
[Megatron-LM](https://github.com/imvladikon/Megatron-LM/tree/glm-5.3-flash).

Known issue: the compatibility-only visual stub has `intermediate_size=64`, so
TP=4 produces a 16-wide partition that is not divisible by the FP8 block size
128; keep `model.visual.*` in BF16 by excluding it from FP8 conversion.

The checkpoint keeps the original text width but reduces 45 decoder layers to
10 and 288 routed experts to 32, for 8,895,622,684 text parameters. Vision is
disabled. Surgery provenance is in `surgery_plan.json` and
`surgery_manifest.json`.

**Experimental text-only test model. Not an official Z.ai release and not yet
quality-recovered. Do not use it as a production or benchmark model.**

This checkpoint preserves GLM-5.3-Flash width/kernel geometry while reducing
the decoder from 45 to 10 layers and each routed MoE from 288 to 32 experts.
It has 8,895,622,684 text parameters. Source layers are
`[0, 1, 2, 3, 8, 18, 25, 31, 38, 44]`. Each target routed expert is a
four-donor functional mosaic: 512 individually selected, coupled SwiGLU units
come from each donor (gate/up rows plus matching down columns), a closed-form
down-projection scale matches synthetic output variance, and all three expert
matrices receive fresh per-128x128 FP8 E4M3 scales. Router rows use balanced
router-space clusters. No donor forward pass, activation cache, distillation,
or parameter training is used. Student-only evaluation remains required before
calling the model functionally useful.

Vision is intentionally disabled. A zeroed 49,056-parameter
visual compatibility stub exists only because the stock Transformers wrapper
currently constructs a visual submodule. It is not a vision model.

The exact source revision, tensor map, expert clusters, and provenance hashes
are stored in `surgery_plan.json`; output shard hashes are in
`surgery_manifest.json`.
