---
library_name: mlx
pipeline_tag: image-text-to-text
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mlx
- qwen4_exp
- qwen3.8-next
- 4-bit
- embedded-mtp
- vision-language-model
- darkbloom
---

# Qwen 3.8 Next (Flash-Next) — MLX 4-bit with embedded MTP

An MLX-quantized version of Qwen3.8-Flash-Next for compatible Apple silicon
runtimes, with native Qwen4 multimodal support and an embedded multi-token
prediction (MTP) head. The text model, vision tower, learned n-gram tables and
MTP weights are included in one indexed checkpoint. No separate draft-model
download is required.

## Artifact

| Field | Value |
|---|---|
| Base model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Architecture | Native Qwen4: `qwen4_exp` / `qwen4_exp_text` |
| Inputs | Text, images and video; image/video processor assets included |
| Output | Text, including reasoning and structured tool calls in compatible runtimes |
| Quantization | MLX affine, predominantly 4-bit/group-64, with selected 5-, 6- and 8-bit modules |
| Safetensor files, including headers | 106,294,664,646 bytes, approximately 98.99 GiB, across 21 shards |
| Indexed arrays | 3,748, including 76 MTP entries and 333 BF16 vision entries |
| Learned n-gram / PLE tables | 128 parts; 384 weight, scale and bias entries |
| License | [Qwen Community License 1.0](LICENSE) |

The output projection uses 8-bit storage. Other per-module precision choices,
scales and offsets are recorded in [config.json](config.json); the 4-bit label
does not mean every array uses four bits. Disk size is not a minimum-RAM
specification: runtime buffers, context length, batching and offload policy
affect memory requirements.

## Multimodal support

The full vision tower, tokenizer, chat template, image processor and video
processor configuration are included. This is a native Qwen4 vision-language
checkpoint, not a text-only export or a Qwen3.5 architecture alias. Use a runtime
that explicitly supports `qwen4_exp` and its native multimodal preprocessing.
Audio support is not claimed.

The learned n-gram/PLE tables are retained. Compatible runtimes can keep these
tables on SSD without changing their stored values. SSD offload, paged attention
and prefix caching are runtime features, not automatic consequences of
downloading the checkpoint.

## Embedded MTP

The checkpoint includes one trained MTP prediction layer alongside the target
model. An MTP-aware runtime can propose tokens with this layer and verify them
against the target model before accepting output. Speculative depth is a
runtime setting, not the number of trained heads or a change to the weights.

Embedding the head does not enable speculation by itself. Check the runtime's
proposal and acceptance diagnostics. In the assessed d-inference implementation,
eligible single-request text generation can use MTP; constrained required/named
tool calls, media requests and multirow decoding use target-only paths under
the existing compatibility gates.

## Download and runtime compatibility

Download the complete repository with
[huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/download):

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="EigenLabs/Qwen3.8-Flash-Next-MLX-4bit-mtp",
    local_dir="Qwen3.8-Flash-Next-MLX-4bit-mtp",
)
```

For reproducible deployments, supply a full immutable commit as `revision`.
Load the resulting directory through a runtime with native Qwen4 support;
generic MLX model loading does not establish architecture, multimodal or MTP
compatibility.

Darkbloom integration is tracked in
[mlx-swift-lm #149](https://github.com/Layr-Labs/mlx-swift-lm/pull/149) and
[d-inference #1030](https://github.com/Layr-Labs/d-inference/pull/1030).
These are draft integration PRs, not a claim that every shipping provider
already supports the model. The native serving identity is
`DarkBloom/Qwen3.8-Flash-Next-Q4-mtp`. Availability through a provider catalog
depends on a separately approved runtime and immutable artifact revision.

## Validation and limitations

The complete package was checked for index coverage, embedded MTP, vision and
PLE inventories, and full per-file SHA-256 integrity. Local d-inference checks
on these weights passed 59 standard compatibility cases and 24 string-tool and
generated-tool-ID history cases in each MTP OFF/AUTO configuration, covering
reasoning OFF/ON and streamed/nonstreamed Chat/Responses. The tested 81K context
and uncached lifecycle cases also passed.

Known remaining failures include some adversarial literal-string copying and
reasoning/tool cases, brief-video recognition, and related multimodal tool
history cases. They are not established as exclusively model or engine defects.
The trained tokenizer uses NFC Unicode normalization, so byte-exact preservation
of decomposed Unicode input is not guaranteed.

These results do not establish BF16 equivalence, every cache or concurrency
configuration, all physical hardware tiers, complete native state/MTP
qualification, signed/trusted deployment, or hosted OpenRouter certification.
Validate the intended workload and runtime before production use.

## Attribution

Original model and training: Qwen. EigenLabs packaging and Darkbloom
integration retain the supplied model license. This is not an official Qwen
release.

See the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
for the upstream model's intended use and limitations, and
[artifact-provenance.json](artifact-provenance.json) and
[SHA256SUMS.json](SHA256SUMS.json) for package provenance and file hashes.
