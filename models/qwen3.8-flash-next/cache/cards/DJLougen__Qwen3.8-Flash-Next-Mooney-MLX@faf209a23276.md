---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
thumbnail: https://huggingface.co/DJLougen/Qwen3.8-Flash-Next-Mooney-MLX/resolve/main/assets/mooney.png
tags:
  - mlx
  - tensorfold
  - quantization
  - mixed-precision
  - ternary
  - qwen4_exp
---

# Qwen3.8-Flash-Next-Mooney-MLX

## 2.25-bit expert storage · 2.52 bits/weight across the transformer · 101.1 GB complete checkpoint (4.49 bits/weight overall)

## Thank you, [Zach Mueller](https://x.com/TheZachMueller) and Lambda

**This model would not exist without [Zach Mueller](https://x.com/TheZachMueller) and [Lambda](https://lambda.ai).** Lambda generously provided the 8× A100 80 GB machine used for the BF16 reference, Mooney quantization, recovery training and original quality evaluations. This MLX checkpoint preserves the weight values from that work. I'm deeply grateful to Zach and to Lambda for making it possible. Thank you.

![Qwen3.8-Flash-Next-Mooney](assets/mooney.png)

**Original Mooney quality — measured on vLLM, not a new TensorFold/MLX benchmark:**

![Original Mooney verifier results: 95.5% of BF16's score](assets/quality_overall.png)

**Support on [Ko-fi](https://ko-fi.com/djlougen) or X Money: [@DJLougen](https://x.com/DJLougen)**

The standalone **MLX-format version of [Qwen3.8-Flash-Next-Mooney](https://huggingface.co/DJLougen/Qwen3.8-Flash-Next-Mooney)**, packaged specifically for the Mooney-aware [TensorFold fork](https://github.com/DJLougen/TensorFold/tree/mooney).

**This is not a stock `mlx-lm` checkpoint.** Its experts are stored in a rotated basis. The runtime must read `mooney_rotation.json` and apply those rotations during inference; loading the arrays without the rotations does not run the correct model.

**Release status:** the checkpoint repack and file integrity are verified. **Apple Silicon inference is not yet working in the real-checkpoint smoke test below.** This is an experimental MLX-format checkpoint release, not a validated Mac deployment.

*Official source: this repository. If you'd like to share the model, please link here rather than re-uploading.*

## What this is

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) @ `de4b8e4d43b917e7706784d8bb445c9af86a3540` is an approximately 180B-parameter MoE checkpoint, including its large PLE n-gram embedding table, MTP head and vision tower.

This release contains the complete checkpoint in **22 safetensor shards**:

- **Routed experts:** the recovered Mooney low-bit weights, stored as MLX affine 2-bit/group-128 arrays, with their rotation manifest.
- **Other transformer weights:** MLX affine 8-bit/group-32; small tensors keep their original stored precision.
- **PLE n-gram table:** MLX affine 8-bit/group-32, split into 128 logical table shards. The TensorFold SSD path reads requested rows rather than keeping the entire table resident.
- **MTP head:** included in the checkpoint; converted from Unsloth's Q8_0 head without another quantization step.
- **Vision tower and projector:** included at BF16. Inclusion of the tensors does not mean image inference is supported on the MLX backend; see **Limitations**.

The conversion is **value-preserving relative to the quantized source**, not lossless relative to BF16. It repacks quantization codes and preserves their decoded values; it does not train or quantize the model again. Safetensor bytes and storage sizes differ from the original representation.

## Where the bits go

Counted from this release's safetensor headers, including affine scales and biases. File sizes use decimal GB; these are **storage sizes, not measured RAM usage**.

| Part | Logical parameters | Stored representation | Effective bits/weight | Tensor payload |
|---|---:|---|---:|---:|
| Routed experts | 120.80B | 2-bit/group-128 + FP16 scales and biases | **2.25** | 33.97 GB |
| Other transformer weights | 4.95B | 8-bit/group-32 + small unquantized tensors | 9.13 | 5.65 GB |
| **Transformer total** | **125.74B** | Mixed precision | **2.52** | **39.62 GB** |
| PLE n-gram table | 51.20B | 8-bit/group-32 + FP16 scales and biases | 9.00 | 57.60 GB |
| MTP head | 2.61B | 8-bit/group-32 + small unquantized tensors | 9.01 | 2.94 GB |
| Vision tower and projector | 0.45B | BF16 | 16.00 | 0.90 GB |
| **All tensor payloads** | **≈180.00B** | | **4.49** | **101.06 GB** |

The complete checkpoint is **101.08 GB** including safetensor headers, tokenizer and checkpoint metadata. The model card, images and small verification artifacts add a little more.

Why the expert storage is **2.25 bits/weight**, rather than the original build's 2.125: MLX affine storage has both a FP16 scale and a FP16 bias per group. This extra metadata preserves the source values; it is not a new quality tradeoff. The same distinction makes 8-bit/group-32 affine storage **9 bits/weight** including metadata.

Exact byte counts, per-file SHA-256 hashes and the runtime reference are in [`manifest.json`](manifest.json). [`card_data.json`](card_data.json) records the inputs behind this card's numbers.

## Why it matters

Mooney concentrates extreme compression in the large expert bank while keeping the other weights at higher precision. This release makes those same quantized values available as MLX arrays, without expanding the experts back to BF16 or requiring the original file format at inference time.

PLE is **57.60 GB of the download**. Keeping it on SSD is important, but does not make the rest of the model small: the transformer alone has **39.62 GB of stored tensor payload**, before runtime allocations. This is **not a full-model 36 GB Mac claim**. Apple Silicon load-time peaks, working memory, KV/state memory and throughput have not been measured for the full checkpoint.

With PLE on SSD, all non-PLE tensor payloads add up to **43.46 GB**, including MTP and vision. With neither of those auxiliary components loaded, the transformer payload is **39.62 GB**. Without SSD PLE, the complete tensor payload is **101.06 GB**. These are byte counts from the checkpoint, not measured resident memory or load-time peaks; runtime allocations require additional headroom.

## Quality: the original Mooney weights

The figures below are the [original Mooney card's vLLM measurements](https://huggingface.co/DJLougen/Qwen3.8-Flash-Next-Mooney/blob/0b76f492970f59a4e231f14f28cc4e754a17127b/README.md), not a rerun on this TensorFold/MLX release.

The source campaign used **536 verifier-checked tasks**, twice per model, with unit tests, exact-answer checks, JSON schemas and tool-call checks:

| Model | Run 1 | Run 2 | Mean |
|---|---:|---:|---:|
| BF16 original | 500/536 | 496/536 | **92.91%** |
| Mooney source weights | 475/536 | 476/536 | **88.71%** |

That is **95.5% of BF16's score**, about **4.20 percentage points below BF16**, on that suite. The main gaps were arithmetic, longer reasoning chains and coding.

![Original Mooney quality by task category, measured on vLLM](assets/quality_by_category.png)

The repack's exhaustive value checks establish that the stored quantized weights are preserved. They do **not** establish identical generated text across runtimes. Different runtime arithmetic can change logits and token choices. A full TensorFold verifier-suite rerun, Apple Silicon quality results and TensorFold long-context quality results are not being claimed here.

## Download and verify

Use Python 3.11 or newer. While this repository is private, authenticate with a Hugging Face account that has access; skip `hf auth login` if you are already authenticated.

```bash
python3 -m venv mooney-env
. mooney-env/bin/activate
python -m pip install huggingface_hub
hf auth login
hf download DJLougen/Qwen3.8-Flash-Next-Mooney-MLX --local-dir Mooney-MLX
```

The download contains one precision/layout only, including PLE, MTP and vision weights. Keep all 22 shards and the index together. To verify the checkpoint files after the download, run this Python block from the directory containing `Mooney-MLX`:

```python
import hashlib
import json
from pathlib import Path

root = Path("Mooney-MLX")
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["repo_id"] == "DJLougen/Qwen3.8-Flash-Next-Mooney-MLX"
files = manifest["files"]
assert len(files) == 37
assert sum(name.endswith(".safetensors") for name in files) == 22
for name, expected in files.items():
    assert Path(name).name == name
    path = root / name
    assert path.is_file() and path.stat().st_size == expected["bytes"], name
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == expected["sha256"], name
print(f"Verified {len(files)} checkpoint files against manifest.json")
```

## Required runtime: TensorFold's Mooney support

Use the [exact Mooney-aware TensorFold revision `a798424`](https://github.com/DJLougen/TensorFold/tree/a79842453567d23f450fe943a2e3cb2c41cbf099), not stock `mlx-lm`, stock `mlx-vlm`, or an upstream TensorFold release that lacks Mooney support.

The full pinned runtime SHA is `a79842453567d23f450fe943a2e3cb2c41cbf099`, also recorded in [`manifest.json`](manifest.json). Keep the model's rotation manifest with the weights. A generic MLX loader can read the arrays but does not implement the required expert rotations.

The MTP-enabled real-checkpoint Metal smoke test still fails in an attention-cache kernel: the generated code expects BF16 cache pointers but receives FP32 cache arrays. The earlier FP16 affine-metadata compile issue was fixed and its targeted component tests passed. **No successful Mac generation or full-model Apple Silicon deployment is being claimed**, so this card deliberately does not provide a Mac serving command. The checkpoint format and the existing CUDA checks are separate from this remaining runtime blocker.

## Verification performed

| Check | Result | Scope |
|---|---|---|
| Fresh checkpoint file hashes | **37 files checked**, including all **22 safetensor shards** | SHA-256 and byte sizes for this exact checkpoint |
| Safetensor/index structure | **3,479 indexed arrays**, **1,707 logical tensors**, **144 rotation entries** | Contiguous byte spans, shape/dtype sizes and reconciled payload totals |
| Quantized repack values | **886/886 quantized tensors checked; no failures** | Decoded values vs the quantized source, including PLE and MTP |
| Dense repack checks | **807 dense tensors checked; no verifier failures** | Source/layout transformations, not BF16 model equivalence |
| Existing full-model TensorFold native identity check | **4 × 256 token IDs matched** | One prompt on CUDA; drafted vs serial within TensorFold, greedy/seeded × eager/graphs; actual source revision `8d002cc`, not an Apple Silicon or cross-engine identity result |
| Latest Metal component checks | **86 passed, 6 hardware-specific skips** | Affine FP16 metadata, rotation, projection and MTP-shape fixtures at the pinned runtime; not a real checkpoint deployment |
| Real four-layer checkpoint smoke test | **Failed** | Complete PLE address space on SSD, MTP and fused decode enabled; remaining attention-cache dtype compile error |
| Full-model Apple Silicon deployment | **Not validated** | No full-model load, generation, memory or speed claim |

The exhaustive repack acceptance and source/runtime identities are recorded in [`verification.json`](verification.json). These checks do not make Mooney equivalent to BF16, and do not replace behavioral evaluation of each runtime.

## Limitations

- **Custom runtime required.** The rotation metadata is part of the model, not an optional optimization. Stock MLX loading is not sufficient.
- **Apple Silicon is not release-qualified yet.** The real-checkpoint Metal smoke test still fails in attention-cache kernel compilation. Synthetic component passes are not a working Mac deployment.
- **No TensorFold speed or full-model memory claims.** The original Mooney runtime's tok/s and RAM figures do not transfer to this release.
- **No verified full-model 36 GB Mac support.** SSD PLE avoids keeping the large table resident, but does not remove the transformer weights or runtime overhead.
- **No image input on TensorFold's MLX Flash-Next path.** Vision weights are included for completeness; they are not a promise of supported Mac image inference.
- **MTP is included, but Mac MTP behavior and speed are not validated.** Do not interpret the existing CUDA identity diagnostic as an Apple Silicon result.
- **Quantization is lossy vs BF16.** The value-preserving conversion preserves the quantized model's strengths and weaknesses; it does not recover the remaining quality gap.
- **Use the pinned fork.** `tensorfold update` targets the upstream project and can replace the Mooney-aware version; do not use it to update this checkpoint's runtime.

## Why Mooney

Mooney faces — from Craig Mooney's 1957 closure study — are photographs reduced to black-and-white patches. Much of the detail is gone, yet the whole remains recognizable. The name reflects the aim here: preserve useful model behavior while discarding most of the weight precision.

## License

These weights are derived from **Qwen3.8-Flash-Next** and distributed under the **Qwen Community License 1.0**. The upstream `LICENSE` is included verbatim. See it for the commercial-use conditions; this release does not replace the base model's license with the runtime's license.

## Credits

Expert quantization builds on **PrismML's Bonsai** work. Original Mooney release: [DJLougen/Qwen3.8-Flash-Next-Mooney](https://huggingface.co/DJLougen/Qwen3.8-Flash-Next-Mooney). Runtime: [TensorFold](https://github.com/ashhart/TensorFold) and the [Mooney-aware fork](https://github.com/DJLougen/TensorFold/tree/mooney). MTP source: Unsloth's Q8_0 conversion ([source repository](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)), preserved without further quantization. Compute generously provided by [Lambda](https://lambda.ai), thanks to [Zach Mueller](https://x.com/TheZachMueller).

## Support This Work

**Support on [Ko-fi](https://ko-fi.com/djlougen) or X Money: [@DJLougen](https://x.com/DJLougen)**
