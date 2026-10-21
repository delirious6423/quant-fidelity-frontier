---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: Avifenesh
pipeline_tag: text-generation
library_name: safetensors
language:
  - en
  - zh
tags:
  - nvfp4
  - fp4
  - fp8
  - 4-bit
  - modelopt
  - w4a4
  - hybrid
  - quantized
  - glm5_next
  - moe
  - blackwell
  - b200
  - conversational
  - tool-calling
---

# GLM-5.3-Flash NVFP4 B200 hybrid (Puzzle Table 2)

Hybrid-precision mint of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
for **NVIDIA B200 / sm_100**, not for the RTX PRO 6000 pin. Recipe follows NVIDIA
Nemotron-Labs-3-Puzzle-75B-A9B Table 2 (NVFP4 column): per-operator hybrid, not a
uniform 4-bit checkpoint.

Built with the tiyuvta serving engine, by **tiyuvta**
([inference.tiyuvta.ai](https://inference.tiyuvta.ai)).

> **Run GLM-5.3-Flash today through an API.**
> [Open GLM-5.3-Flash on tiyuvta](https://inference.tiyuvta.ai/models/glm-5-3-flash?c=hf-glm53-b200)
> and use model id `zai/glm-5.3-flash`. That endpoint serves the PRO 6000 mint
> ([tiyuvta/GLM-5.3-Flash-NVFP4](https://huggingface.co/tiyuvta/GLM-5.3-Flash-NVFP4)
> @ `2bade47d`). This B200 hybrid is a different numeric program and a different
> layout. It is not a candidate for that pin.
> Speculative decoding on the live endpoint is drafted by
> [DFlash 2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (inco.ai).
> [Get an API key and send the first request.](https://inference.tiyuvta.ai/quickstart?c=hf-glm53-b200)

- **189.1 GB** across 19 safetensors shards. 38,770 source tensors classified fail-closed:
  37,290 NVFP4, 204 FP8, 86 FP32, 1,190 BF16.
- Stored tensors: **113,554** = 3 x 37,290 NVFP4 triples + 2 x 204 FP8 pairs + 86 FP32 + 1,190 BF16.
- Source is the vendor's **BF16 twin** @ `f12e0fe1f6b2ea274c11a569582edfd99d993c5e`, not their FP8 release.
- PRO 6000 Four-Over-Six sibling (same keep-list as the live pin, modelopt scale layout):
  [tiyuvta/GLM-5.3-Flash-NVFP4-4o6](https://huggingface.co/tiyuvta/GLM-5.3-Flash-NVFP4-4o6).
- Upstream technical report: [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
  (arXiv 2602.15763).

## How it was made

| | |
|---|---|
| Source | `zai-org/GLM-5.3-Flash-BF16` @ `f12e0fe1f6b2ea274c11a569582edfd99d993c5e` (656 GB) |
| Tool | `nvidia-modelopt` 0.46.0 for the NVFP4 math; FP8/FP32/BF16 paths are explicit |
| NVFP4 | max-calib (not Four-Over-Six). Packed weights **K-inner** `[N, K/2]`. Scale plane **Swizzle32x4x4** (CUTLASS / vLLM `swizzle_blockscale`), padded N%128 K%4 |
| FP8 | e4m3 per-tensor, `amax/448`, on the KDA six only |
| FP32 | router `mlp.gate.weight` and `e_score_correction_bias`, upcast from BF16 |
| Rotation | **none.** No QuaRot, no Hadamard. Per-16 UE4M3 already isolates outliers |
| `input_scale` | **deferred.** W4A4 tensor-core wants a 256 x 65K activation calib into the checkpoint. This pass does not have those forwards. The serving engine's dp4a path already does dynamic q8_1 without them |

`try_tensorrt=True` is a no-op on the mint GPU (sm_89). The Swizzle32x4x4 plane is
written explicitly. That is the opposite of the 4/6 remint, which keeps
`try_tensorrt=False` so the serving engine's existing modelopt-layout reader can eat it.
**This file is not a drop-in for that reader.** B200 tensor-core and a dp4a
indexer that addresses K-major / swizzled scales are the consumers.

`LAYOUT.json` in this repo is the operator map, not a comment.

## Precision split (Puzzle Table 2 -> GLM-5.3-Flash)

| Puzzle operator | GLM-5.3-Flash tensors | this checkpoint |
|---|---|---|
| Routed and shared MoE GEMMs, dense MLP | `mlp.experts.{i}.{gate,up,down}_proj`, `mlp.shared_experts.*`, dense `mlp.{gate,up,down}_proj` on layers 0-2 | **NVFP4** max-calib, K-inner, Swizzle32x4x4 |
| Linear-attention GEMMs | KDA six the fused `qmatvec_kda6` kernel consumes: `self_attn.{q,k,v}_proj`, `f_a_proj`, `g_a_proj`, `b_proj` (34 layers) | **FP8** e4m3 |
| Attention QKV / output | MLA/DSA `q_a` / `q_b` / `kv_a_proj_with_mqa` / `o_proj` (11 full-attn + MTP) | **BF16** (the live pin quantized these) |
| Absorb | `kv_b_proj` | **BF16** |
| Router | `mlp.gate.weight`, `e_score_correction_bias` | **FP32** |
| conv1d, indexer, mHC, norms, embed, `lm_head`, vision, MTP glue, KDA `o_proj` | | **BF16** |

KDA `o_proj` stays BF16: it is not in the kda6 six. Low-rank `f_b` / `g_b` stay BF16.

Versus the live pin / the 4/6 remint, in one list:

1. Experts stay NVFP4, but calib, pack, and scale layout change (W4A4-shaped, max-calib, K-major, swizzled).
2. KDA six: BF16 keep -> **FP8**.
3. MLA QKV/o: NVFP4 -> **BF16 keep**.
4. Router: BF16 -> **FP32**.
5. `input_scale` for expert W4A4 tensor-core is not in this file yet.

## The gate this has passed (and what it has not)

Census on the mint: **37,290 NVFP4 / 204 FP8 / 86 FP32 / 1,190 BF16**, every
source tensor classified as exactly one bucket or the mint aborted. KDA layers
asserted as exactly those six names and shapes.

**Not yet a serving pin.** Kernel-check, argmax-margin, and vendor-default sampled
decode on sm_100 are still owed on these exact bytes. Do not load this in place of
`2bade47d` on PRO 6000. Do not treat weight mint as W4A4 serving: the activation
scale plane is missing.

## FR-Spec ranks (the `.txt` files)

Same three traffic classes as the q38 and ornith cards. Tokenizer is unmodified
from `zai-org/GLM-5.3-Flash` (`tokenizer.json` sha256
`19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`).
These ranks are a **vocab artifact** of that tokenizer, minted 2026-08-30:
plain text, one token id per line, most-frequent first, 32,768 lines. They drive
the serving engine's load-time trim on a safetensors trunk. Verification
still uses the full vocabulary, so a trim cannot change greedy output.

| file | class | sha256 |
|---|---|---|
| `glm53-ranks-sxc32768.gguf.txt` | agentic (serving-default candidate) | `1804027e6148414c46cdab1a4f8773d063b1af8435d37a231ecd31d5574a1632` |
| `glm53-ranks-prose-32768.gguf.txt` | prose | `9498ed34d46452c281ad6dafa970a0b4f52634b5cdfb0200fd69fd8579e41e9a` |
| `glm53-ranks-mixed-32768.gguf.txt` | mixed | `8461ad2dfea86670e307d51502e23ce25ad13a6bb0724228b9cf79a919163e85` |

A per-requant own-generation remint of ranks from *this* checkpoint is still
owed after the B200 serving gate.

## Running it

This is a hybrid `glm5_next` checkpoint with a CUTLASS-swizzled NVFP4 scale
plane. A stock `transformers` load is not the path. The live PRO 6000 repack
path on the serving engine is not the path either.

Only a reader that indexes K-inner packed weights and Swizzle32x4x4 scales
will do the right matmul. Native chat dialect is the checkpoint's own
`chat_template.jinja` (`[gMASK]<sop>`, `<|user|>` / `<|assistant|>`, GLM tool
calls). Do not pattern-match ChatML. Honour all three eos ids in
`generation_config.json`.

## Files

| file | what it is |
|---|---|
| `model-*-of-00019.safetensors` | the weights, 189.1 GB |
| `LAYOUT.json` | operator map: pack, swizzle, kda6 names, `input_scale: deferred` |
| `config.json` | `glm5_next` |
| `chat_template.jinja`, tokenizer sidecars | unmodified from source |
| `generation_config.json` | unmodified, three eos ids |
| `glm53-ranks-*.gguf.txt` | FR-Spec ranks, one token id per line |

The vision tower is present in the source architecture and is not exercised by
this mint's census.

## Attribution and licence

MIT, following upstream [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
All model capability belongs to the GLM-5 team; this repository contributes a
hybrid quantization. Speculative decoding on the live tiyuvta endpoint is drafted
by DFlash 2 (inco.ai), used with the authors' written permission.

```bibtex
@misc{glm5team2026glm5vibecodingagentic,
      title={GLM-5: from Vibe Coding to Agentic Engineering},
      author={GLM-5-Team},
      year={2026},
      eprint={2602.15763},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.15763},
}
```
