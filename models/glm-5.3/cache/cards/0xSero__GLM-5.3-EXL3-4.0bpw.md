---
license: other
base_model: zai-org/GLM-5.3
pipeline_tag: text-generation
tags:
  - glm
  - glm-5.3
  - moe
  - exl3
---

> [!WARNING]
> **Known KL regression — do not use this checkpoint yet.** Measured KL vs BF16 is **~1.02 nats** (reproduced: 1.0223 then 1.0215; median token KL 0.33), ~10× worse than expected for an experts-only + fp16-sensitive build. The regression is confirmed real (verified-complete checkpoint, measured twice), and traced to a crash-and-resume finalization during the encode. A clean full re-encode is pending; until then use the [3bpw ladder](https://huggingface.co/collections/0xSero/glm-53-reap-exl3-quantization-suite-6a9f03389fe6a96671619ba3) or the [W4A16 cuts](https://huggingface.co/collections/0xSero/glm-53-reap-w4a16-hopper-6a9f128893dbd3dbf0a2fce7).

> [!TIP]
> **[Support this work →](https://donate.sybilsolutions.ai)** · [X](https://x.com/0xsero) · [GitHub](https://github.com/0xsero) · [REAP paper](https://arxiv.org/abs/2510.13999) · [Cerebras REAP](https://huggingface.co/collections/cerebras/cerebras-reap)

# GLM-5.3 — EXL3 4.0bpw (experts-only quant, BF16 sensitive layers)

**The full GLM-5.3 (all 256 experts) where only the routed experts are quantized to 4-bit and every sensitive layer stays full-precision BF16 — 405 GB.** This is the high-fidelity build and the unpruned reference for the 4bpw series.

## What this is

GLM-5.3 is a 753B mixture-of-experts model (8 of 256 experts per token, ~40B active). Here **only the routed experts are EXL3 4bpw** (mcg codebook); attention (including the DSA indexer), the shared expert, the dense layers, and the head are kept **BF16 byte-exact** — the davidsyoung/brandonmusic standard. Compression is spent only where the model is most redundant, so this sits far closer to the original than a fully-quantized 3-bit checkpoint.

|  |  |
|---|---|
| Base | zai-org/GLM-5.3-BF16 |
| Quant | routed experts EXL3 4.0bpw (mcg); attention + DSA indexer + shared expert + dense + head = BF16 |
| Experts | 256 / 256 (unpruned) |
| Size | 405 GB |
| KL vs BF16 | **1.0223 nats** (sealed 25-prompt panel, full 154k vocab; target: davidsyoung ~0.024) |

Pruned, VRAM-friendlier cuts of this model are in the [EXL3 collection](https://huggingface.co/collections/0xSero/glm-53-reap-exl3-quantization-suite-6a9f03389fe6a96671619ba3). Criterion and all KL runs: [fidelity study](https://huggingface.co/datasets/0xSero/glm-5.3-reap-fidelity-study).

## Running it

```bash
pip install exllamav3   # or via TabbyAPI; tensor-split across your GPUs
```

## Credits

- **[Z.AI / zai-org](https://huggingface.co/zai-org)** — [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), the base model.
- **[Cerebras Research](https://github.com/CerebrasResearch/reap)** — REAP ([arXiv:2510.13999](https://arxiv.org/abs/2510.13999)).
- **[turboderp / exllamav3](https://github.com/turboderp-org/exllamav3)** — the EXL3 format and runtime.

Observations: [`glm-5.3-reap-observations-v1`](https://huggingface.co/datasets/0xSero/glm-5.3-reap-observations-v1) · Built on 8× NVIDIA RTX PRO 6000 Blackwell.

## License
Inherits the [GLM-5.3 license](https://huggingface.co/zai-org/GLM-5.3).
