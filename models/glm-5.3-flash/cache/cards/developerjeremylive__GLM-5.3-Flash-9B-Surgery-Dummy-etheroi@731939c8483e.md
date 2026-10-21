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
