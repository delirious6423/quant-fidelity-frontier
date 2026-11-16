---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: kayceesamuel
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - nf4dq
  - quantized
  - llama.cpp
  - ciu
---

# Qwen3.8 27B, quantised to NF4DQ

**13.23 GiB.** Smaller than Q4_K_M, faster at decode, and closer to Q4_K_M
than to Q3_K_M on perplexity.

NF4DQ is a new ggml block type, not a recipe built from existing ones. It
stores 1024 weights in 532 bytes, 4.1562 bits each: 4-bit NF4 indices into an
int8 codebook, a 4-bit scale index per 32-weight sub-block, and one fp16
super-scale per block.

---

## Read this before downloading

**This file will not load in stock llama.cpp, Ollama, or LM Studio.** NF4DQ is
ggml type 43 and nothing upstream knows what that is. You need one of the
three options below. Downloading it without one gets you a 13.23 GiB file that
every tool refuses to open.

### The easy way: CIU

CIU is a small local inference app. It works out what fits in your machine's
memory, fetches the right engine build, downloads the model, and serves an
OpenAI-compatible API on `127.0.0.1:8674` so your existing tools can point at
it. It is free and runs entirely on your machine.

| Platform | Download |
|---|---|
| macOS, Apple Silicon | [ciu-macos-arm64.zip](https://github.com/runciu/ciu/releases/latest/download/ciu-macos-arm64.zip) |
| Windows 10/11, 64-bit | [ciu-windows-x64.zip](https://github.com/runciu/ciu/releases/latest/download/ciu-windows-x64.zip) |

Unzip, open it, and pick this model from the list. It handles the rest.

This is the largest of the three NF4DQ models, so it wants a 24 GB NVIDIA card
or a Mac with 24 GB or more of unified memory. CIU checks before downloading
and will tell you if it does not fit.

The builds are not code-signed yet, so the first run needs one extra step.
`FIRST-RUN.txt` in the zip has it, and it is one line on either platform.

Intel Macs and Linux are not supported yet.

### The manual way: build the fork

```bash
git clone -b nf4dq https://github.com/runciu/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release    # add -DGGML_CUDA=ON for NVIDIA
cmake --build build -j
./build/bin/llama-server -m Qwen3.8-27B-NF4DQ.gguf -ngl 99 -c 8192 -fa on -np 1
```

Prebuilt `llama-server` binaries are on the fork's
[releases page](https://github.com/runciu/llama.cpp/releases).

CUDA and Metal are supported. There is no Vulkan backend yet.

One thing worth knowing: type 43 is the next free slot upstream. If upstream
ever claims it, a stock build will read these files as that type and decode
garbage rather than refusing to load. If output looks wrong, check which
binary you are running before anything else.

`-np 1` matters. llama-server defaults to 4 sequence slots and `-c` is **per
slot**, so without it a request for 8k context allocates 32k tokens of cache.

### Trying it on Colab

A T4 is not enough for this one. On an A100 or an L4 runtime:

```python
!git clone -q -b nf4dq https://github.com/runciu/llama.cpp
!cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;89" \
    && cmake --build build -j --target llama-server llama-completion
```

```python
from huggingface_hub import hf_hub_download, list_repo_files
REPO = "kayceesamuel/Qwen3.8-27B-NF4DQ"
NAME = [f for f in list_repo_files(REPO) if f.endswith(".gguf")][0]
MODEL = hf_hub_download(REPO, NAME, local_dir="/content")
```

```python
!/content/llama.cpp/build/bin/llama-completion -m {MODEL} \
    -ngl 99 -c 4096 -n 200 --temp 0 -no-cnv \
    -p "Explain photosynthesis in one paragraph."
```

If you want the smaller sibling instead, there is a
[9B](https://huggingface.co/kayceesamuel/Qwen3.5-9B-NF4DQ) that fits a T4.

---

## Measurements

Perplexity on wikitext-2 at context 512, A100-80GB, full test file. Same
source weights, same test file, same context length, same binary.

| Build | Size | PPL @512 |
|---|---|---|
| BF16 | 50.9 GiB | 6.9530 |
| Q4_K_M | 15.652 GiB | 6.9787 |
| **NF4DQ** | **13.230 GiB** | **7.0890** |
| Q3_K_M | 12.643 GiB | 7.3362 |

Against llama.cpp's own K-quant size-quality curve, interpolating between
Q3_K_M and Q4_K_M puts a K-quant at 4.16 bpw near 7.29. NF4DQ measured 7.089.

Throughput against Q4_K_M, same A100, same `llama-bench` run:

| Build | Size | prefill | decode |
|---|---|---|---|
| **NF4DQ** | **13.23 GiB** | 734.52 t/s | **60.81 t/s** |
| Q4_K_M | 17.66 GiB | 1300.04 t/s | 48.67 t/s |

Smaller and 25% faster at decode. Prefill is slower because the fork has no
NF4DQ tile loader in `mmq.cu` yet.

On an NVIDIA L4 with 23 GB the whole model runs in about 16.9 GB at 8k
context, at 16.84 t/s decode, or 30.13 t/s with multi-token prediction
(`--spec-type draft-mtp --spec-draft-n-max 4`). That headroom is the point:
Q4_K_M leaves far less room on a 24 GB card.

---

## What NF4DQ is

NF4 weight levels with a second quantisation stage that stores each 32-weight
sub-block's scale as a 4-bit index into a fixed codebook rather than an fp16
value. That second stage is what takes it below Q4_K's 4.50 bits per weight.

Design notes, the fitting data behind every constant, and the approaches that
were measured and rejected are in the
[fork's design doc](https://github.com/runciu/llama.cpp/blob/nf4dq/docs/nf4dq-design.md).

---

## Questions, problems, or it did not work

Email **hello@runciu.com**. Genuinely, if you downloaded this and could not
run it, that is worth telling me about, because the most likely cause is
something I can fix.

Issues: [github.com/runciu/ciu/issues](https://github.com/runciu/ciu/issues)

---

## Licence

Apache 2.0, matching the base model. Quantisation does not create a new work.
