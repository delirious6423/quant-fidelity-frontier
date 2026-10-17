---
library_name: cortiq
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- cmf
- cortiq
- moe
- q4tp
- fp8
---

# DeepSeek-V4.1-Flash → CMF — one file, one Rust runtime

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
is converted to the [CMF container](https://github.com/infosave2007/cmf):
one memory-mapped weight file, served by a Rust binary without a Python ML
framework. The runtime supports text and images, dynamic expert staging on
Vulkan, and CPU fallback.

| file | weights | size | measured RTX PRO 6000 96 GB |
|---|---|---:|---:|
| `dsv41-q4tp.cmf` | Q4TP text/experts + native FP8 Engram + F16 vision | **498.99 GB** | **11.26 tokens/s** after the initial stream fragment, repeated request |

The model downloads as 59 verified parts and reconstructs into one file.
Use the matching runtime supplied here; stock cortiq 0.6.6 predates this
V4.1 integration. [Download](#download) · [Run](#run) · [Measured results](#performance)

This conversion derives from
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277),
revision `dba1be0a40aa45a94ad051997016db3960a90277`.

Q4TP applies to ordinary text and expert matrix weights. Sensitive
attention/control tensors retain unquantized higher precision. Vision and
aligner matrices and controls remain unquantized F16, preserving the source
BF16 precision class without introducing vision quantization loss. The two
Engram lookup tables and their scale bytes remain lossless in their original
FP8 E4M3 / E8M0 representation. They account for 202,758,032,400 bytes, so
the complete artifact is a mixed-precision model.
The CMF metadata retains the original nested text, vision and quantization
configuration.

The accepted CMF file is `dsv41-q4tp.cmf`, exactly 498,986,143,769 bytes, with
SHA-256
`a681cafb28bea46271d1e28df5e532288ef545661541762ed06351dd1cf1cd0c`.

## Download

`parts-q4tp/part_*` concatenate into a single CMF file. The reconstruction
manifest records every part's offset, byte size and SHA-256, plus the SHA-256
of the complete model.

Get the parallel downloader and Linux runtime:

```sh
curl -L https://huggingface.co/infosave/DeepSeek-V4.1-Flash-cmf/resolve/main/download_q4tp.py -o download_q4tp.py
curl -L https://huggingface.co/infosave/DeepSeek-V4.1-Flash-cmf/resolve/main/cortiq-linux-x86_64 -o cortiq-linux-x86_64
```

Use the downloader on Linux or macOS (Python 3.10+ is only needed to fetch
and reconstruct the file):

```sh
python3 download_q4tp.py --workers 8 --output dsv41-q4tp.cmf
```

The script resolves one immutable Hub revision, downloads parts concurrently
directly into the destination file, and checks both the parts and final model.
It needs no second full copy of the weights. Run the same command after an
interruption to reuse verified ranges. Keep the `.partial.parts` directory
until the download completes.

## Run

Download `cortiq-linux-x86_64` and make it executable. The Linux binary,
complete source archive, patch against public 0.6.6, dependencies, and build
instructions are supplied together; see [BUILD.md](BUILD.md).

The tested host is one RTX PRO 6000 Blackwell with 96 GB VRAM, approximately
234 GiB available system RAM, and a 13.6-core CPU quota. The 498.99 GB model
is memory-mapped from local disk and experts are staged dynamically. It
does not fit wholly in VRAM. Allow at least 500 GB for the reconstructed
model and additional space for the runtime and filesystem; fast local NVMe
is preferable to a network volume.

Tested 96-GB profile (88 GiB configured GPU budget):

```sh
chmod +x cortiq-linux-x86_64
export XDG_RUNTIME_DIR=/tmp WGPU_BACKEND=vulkan CMF_GPU=1
export CMF_GPU_VRAM_MB=90112 CMF_THREADS=12 RAYON_NUM_THREADS=12
export CMF_DSV41_DYNAMIC_MOE=1 CMF_DSV41_GLOBAL_SEGMENTS=16
export CMF_DSV41_POOL_PCT=85 CMF_DSV41_FETCH_MAX=6
export CMF_DSV41_FETCH_MIN_SEEN=1 CMF_DSV41_GPU_ATTN=1
export CMF_DSV41_FUSED_Q=1 CMF_SERVE_SLOTS=1
./cortiq-linux-x86_64 serve ./dsv41-q4tp.cmf --host 127.0.0.1 --port 8080 --o1 off
```

The server exposes an OpenAI-compatible `/v1` API. For CPU inference set
`CMF_GPU=0`; the weight file and its disk requirement remain the same.
The 16-segment profile is specific to the tested large GPU. Smaller budgets
use 8 segments; a 16-GiB configured-budget check is reported separately
from physical hardware testing.

Example request with the same non-thinking mode used for the measurements:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek_v41-cortiq","messages":[{"role":"user","content":"Explain an LRU cache briefly."}],"temperature":0,"max_tokens":64,"enable_thinking":false}'
```

## Performance

The selected fused-Q profile was measured with a single 64-token response,
one serving slot, temperature zero and thinking disabled:

| Measurement | First request | Repeated request |
|---|---:|---:|
| Output after the initial buffered fragment | 7.08 tokens/s | 11.26 tokens/s |
| Full request elapsed time | 18.57 s | 8.00 s |

The stream's first event contains 22 tokens; 42 tokens arrive afterward.
The rate above divides those 42 tokens by their arrival span, verified with
the checkpoint tokenizer. It is not a whole-request rate or an engine-only
decode timer. A first request means a newly started server, not a guarantee
that the operating-system page cache is cold. These results do not reach
20 tokens/s and are not a universal throughput guarantee.

In the same-binary comparison, disabling fused Q produced 11.09 tokens/s
on the repeated request; the 1.5% difference is small. The larger accepted
improvement came from reusing the GPU attention/output frame: a prior
matched comparison measured 11.75 versus 4.31 tokens/s with that frame
enabled versus CPU attention. That comparison already used dynamic GPU
MoE in both arms, so it is not a CPU-only comparison.

The implementation also reuses the shared dynamic expert pool, supports
16 pool segments for this 96-GB profile, and avoids a discarded host copy
when the optional RAM tier is disabled. A larger 95% expert-pool budget
and parallel refill-copy experiment did not produce useful measured gains
and were not included in the recommended profile.

## KV cache

The global cache stores main attention rows in E2M1 with E4M3 scales per
16 values, and index rows in E2M1 with E8M0 scales per 32 values. The four
CSA2 source banks use 890 bytes per token in total, before fixed metadata
and the bounded sliding windows. Actual storage at 4096 tokens is
3,645,440 bytes. Selected main rows are decoded once per layer into a
bounded scratch buffer shared by the attention heads; the complete history
is never unpacked for an attention call.

This reduces global-cache storage from the earlier f32 representation's
6400 bytes per token while preserving its decoded values. It does not reduce
the 498.99 GB weight-file size. The official production engine also uses
DSpark speculative decoding; this runtime retains those weights but does
not yet execute DSpark or its batched verifier.

The [official technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
describes additional production kernel fusion and speculative execution.
Those production throughput claims do not transfer automatically to this
portable CPU/Vulkan runtime and its weight-offloading hardware setup.

## Validation

The accepted artifact gate covers all 48,498 retained tensors against 96,085
pinned source-index entries, including 40 layers, 384 routed experts per
layer, two Engram layers, the full expert/mHC inventory, and three DSpark
entries. CMF bounds, metadata, canonical coverage, duplicate detection, and
native Engram U8 geometry passed. All MTP tensors are retained; retaining
these tensors does not imply that speculative decoding is enabled.
The artifact receipt is [artifact-summary.json](reports/artifact-summary.json).

All 59 uploaded weight parts and the reconstruction manifest were checked
against the accepted artifact at Hub revision
`dbabe4108fb2020630202b788f52590fd044fdfd`.

The Engram component check covers the complete token mapping, hash parameters,
text/image-break/padding cases, contiguous and chunked lookup, and actual
selected FP8 rows from both layers. All 577,536 gathered BF16 values match
the source exactly. The maximum residual-output difference is 0.015625;
the mean absolute difference is at most 2.29e-7. These are component
measurements, not full-model logit parity.

The final fused-Q GPU profile passed the semantic review of eight bounded
cases: arithmetic, JSON, Russian, Python code, a counting problem, Chinese,
conversation context, and an image. Seven of eight strict automated checks
passed. Arithmetic correctly returned 3945 but included working despite an
integer-only instruction. The generated parentheses-checking function passed
15 assertions. The image answer correctly counted 2 red squares, 2 blue
circles and 1 green triangle, read `CMF 41` and `8123`, and took 85.92 seconds.

An earlier GPU profile without fused Q gave an incorrect total in the counting
problem; the recommended fused-Q profile and CPU control both answered 18
correctly. The earlier failure is retained in the reports. These small checks
are behavior samples, not a benchmark-suite score or full-checkpoint parity.

CPU multi-prompt checks, packed-cache component checks, and a two-prompt run
with a configured 16-GiB GPU budget passed their documented semantic checks.
The budget test ran on the physical 96-GB card; it does not establish speed or
memory behavior on physical 16-GB hardware. The final fused-Q CPU fallback
also matched the prior path exactly on the tiny fixture.

Runtime identities and measured text/image/resource results are recorded in
[release-manifest.json](release-manifest.json) and the accompanying `reports/` files.

### Image example

![The image used for the vision check](https://huggingface.co/infosave/DeepSeek-V4.1-Flash-cmf/resolve/main/examples/vision-check.png)

Asked to return the counts, title and number as JSON, the model returned:

```json
{"red_squares": 2, "blue_circles": 2, "green_triangles": 1, "title": "CMF 41", "number": 8123}
```

The accepted vision component comparison is component-level evidence only:
the F16 Vulkan run produced finite output with cosine `0.999695` against the
official component oracle. OpenRouter responses are a behavioral comparison only; they do
not establish numerical parity with the original checkpoint. The available
eight-case comparison was semantically correct in 8/8 cases and passed the
strict automated format checks in 7/8 cases; the arithmetic case included an
intermediate line despite requesting only the integer.

## License

The source model uses the MIT license. CMF and the cortiq runtime are available
from [infosave2007/cmf](https://github.com/infosave2007/cmf).
