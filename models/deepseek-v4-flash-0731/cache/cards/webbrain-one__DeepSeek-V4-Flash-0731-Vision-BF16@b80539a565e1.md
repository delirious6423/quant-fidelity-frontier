---
license: other
library_name: custom
inference: false
pipeline_tag: image-text-to-text
tags:
- multimodal
- vision-language
- deepseek-v4
- moonvit
- bf16
- fp8
- sglang
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
- moonshotai/Kimi-K2.6
---

# DeepSeek V4 Flash 0731 Vision (BF16 source overlay)

![DeepSeek V4 Flash with MoonViT](deepseek-vision-improved.gif)

**DeepSeek V4 Flash 0731 with sight.** This development checkpoint connects
DeepSeek's reasoning and agentic backbone to the MoonViT vision encoder from
[Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) through WebBrain's
trained, routing-aware PatchMerger projector.

This is a complete copy of DeepSeek's `DeepSeek-V4-Flash-0731` text checkpoint
with WebBrain's frozen MoonViT tower, trained PatchMerger projector, routing
bridge, and custom SGLang integration added in the same repository.

> [!IMPORTANT]
> **The `BF16` suffix describes the added vision tower and projector, not every
> text tensor.** The text checkpoint is the official upstream release, whose
> config declares `torch_dtype: bfloat16` and FP8 quantization and whose stored
> tensors use mixed formats. No text tensor was converted or relabeled.

## Why vision at WebBrain

At [WebBrain](https://www.webbrain.one), we build browser agents that need to
understand the visual state of the web—not just extracted text. Screenshots,
charts, dashboards, rich editors, and the location and appearance of controls
are part of real browser work, so vision is a practical product requirement.

In our
[American–Chinese open-model frontier benchmark](https://www.webbrain.one/blog/american-chinese-open-model-frontier-gap-benchmark),
DeepSeek V4 Flash stood out as a strong and cost-efficient model, while the
upstream checkpoint remains text-only. This project attaches a MoonViT vision
bridge while keeping the language backbone and vision tower frozen.

## Contents

| Component | Provenance | Status |
|---|---|---|
| Text checkpoint | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) at [`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/commit/9e165c30e2704aec5d9d593cce3eebd58bbef1cb) | 48 shards; copied server-side without tensor changes |
| Vision tower | MoonViT-3d extracted from [`moonshotai/Kimi-K2.6`](https://huggingface.co/moonshotai/Kimi-K2.6) at `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` | 329 BF16 tensors; frozen |
| Projector | WebBrain PatchMerger, trained with the text backbone and tower frozen | 6 BF16 tensors; 40,119,040 parameters |
| Serving | External DeepSeek/MoonViT SGLang model and processor | Experimental; source patch required |

The added components are byte-identical to the published WebBrain vision
artifacts:

| File | Size | SHA-256 |
|---|---:|---|
| `vision_tower.safetensors` | 833,765,768 bytes | `1382c41f1a4afc91791ade630e2b1e1cef68cc5a1e09668a45970a5d5e1b8f15` |
| `mm_projector.safetensors` | 80,238,568 bytes | `7024d9d5c9714c7abbc09abda015f083b7d7b107745eb78879f019bf4721577a` |

The initial mirror commit, `e181152a5285f5bcaad16f032267b192fd7cf99f`,
matched all 74 upstream paths, sizes, Git blobs, and LFS SHA-256 values
(166,898,660,330 bytes). The final package intentionally replaces the model
card and augments `config.json` and `model.safetensors.index.json` so the vision
components can be discovered. The 48 text shards remain unchanged.

See [`VISION_ADAPTER_MANIFEST.json`](VISION_ADAPTER_MANIFEST.json) for
machine-readable provenance.

## Build status

- [x] Pin and exactly mirror the official 0731 text checkpoint.
- [x] Preserve all 48 text shards and the 0731 DSpark/MTP configuration.
- [x] Add and fingerprint-verify the frozen BF16 MoonViT tower.
- [x] Add and fingerprint-verify the trained BF16 PatchMerger projector.
- [x] Add the routing palette, external processor/model package, source patch,
  launch wrapper, and regression tests.
- [x] Augment the source config and index without changing upstream text tensor
  mappings or quantization metadata.
- [x] Pass a full-model GPU loader/startup gate and live image-generation smoke
  test with the established deployment path.
- [ ] Run fresh text-parity and live image-generation smoke tests on the target
  hardware and SGLang image.

## Integration details

The projector performs:

```text
LayerNorm -> 2x2 patch merge -> Linear(4608, 4608) -> GELU -> Linear(4608, 4096)
```

Text token routing IDs are preserved. Image positions receive deterministic
routing IDs from the checked-in 64-ID palette. The integration currently
supports one literal `<image>` marker and targets at most 512 merged image
tokens.

This is not a stock Transformers or stock SGLang multimodal architecture. Use
the checked-in external model/processor package and the SGLang revision pinned
in [`docs/SGLANG_DEPLOYMENT.md`](docs/SGLANG_DEPLOYMENT.md).

## Validation status

- Exact upstream mirror metadata was verified before the vision overlay.
- Both added safetensors files were verified by LFS SHA-256 after publication.
- The source config's architecture, 0731 DSpark fields, and FP8 quantization
  metadata are preserved; only vision metadata was added.
- The index still references the same 48 text shards and additionally maps the
  335 vision/projector tensor names.
- **A full GPU loader/startup and image-generation smoke test has not yet been
  rerun for this assembled 0731 checkpoint.** Earlier validation of the same
  vision adapter with another text package is not claimed as validation here.

## Usage

Download the complete repository, then point the included wrapper at that local
snapshot:

```bash
export DEEPSEEK_VISION_MODEL_PATH=/models/DeepSeek-V4-Flash-0731-Vision-BF16
export DEEPSEEK_VISION_PYTHONPATH="$DEEPSEEK_VISION_MODEL_PATH/sglang_ext"
export DEEPSEEK_VISION_TP=5
scripts/launch_sglang_moonvit.sh
```

Begin with SGLang's native `/generate` endpoint as documented in the deployment
guide. OpenAI chat image parts, production concurrency, CUDA graphs, and broad
quality behavior remain unvalidated.

## Licenses

The DeepSeek text files retain their upstream terms in `LICENSE` and
`LICENSE_DEEPSEEK_V4_FLASH`. The MoonViT-derived tower is covered by
`LICENSE_KIMI_K2.6`. The WebBrain projector and integration source are provided
under the terms included with this repository. Downstream users are responsible
for satisfying all applicable licenses.

## Method credit

The overall construction and model-card approach was inspired by
[Baseten's GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4):
keep the text backbone and MoonViT tower frozen, train a compact PatchMerger
projector between them, and publish provenance and hardware constraints
explicitly. Credit to the Baseten team for demonstrating this practical recipe.

This project adds a DeepSeek-specific routing bridge so mixed text/image
embeddings preserve hash-routed text behavior. It does not reuse Baseten model
weights, benchmark results, or deployment artifacts.

## Acknowledgements

Built on
[DeepSeek AI's DeepSeek V4 Flash 0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
and [Moonshot AI's Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6),
with the vision-attachment method inspired by
[Baseten's GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4).
These teams were not involved in this experimental development checkpoint;
please do not direct issues with this repository to them.

## Want this model on your inference provider?

Ask your inference provider—such as
[OpenRouter](https://openrouter.ai/) or another managed inference service—to
deploy this exact repository with its multimodal processor and serving plugin.
Deploying only the upstream text model will not enable image input. Providers
should complete the fresh 0731 GPU validation gates above before offering it as
a production endpoint.

## Experimental status, roadmap, and get involved

> [!CAUTION]
> **Experimental vision checkpoint.** Repository assembly, provenance, hashes,
> and source preservation have been verified, but a fresh full GPU loader and
> image-generation smoke test has not been run for this 0731 package. This is
> not a production-readiness or quality claim. Fine-grained OCR, small-object
> and control identification, GUI grounding, hallucination calibration,
> concurrency, and broader hardware coverage still need evaluation. Do not use
> this model as the sole decision source for safety-critical automation. The
> roadmap includes larger and more diverse datasets, higher-resolution OCR/UI
> examples, and broader parameter-efficient tuning. Interested in contributing
> evaluation or training data, sponsoring compute, or working with us as a
> design partner? [Tell us here](https://forms.gle/bNoeJ6cvLYQ4VgKd7).
