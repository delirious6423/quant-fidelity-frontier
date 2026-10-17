---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - qwen4exp
  - mtp
  - speculative-decoding
  - llama.cpp
---

# Qwen3.8-Flash-Next IQ3E-Q8D-MTP (GGUF) For V100(32GB)x2

This is a single-file GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
with the model's multi-token-prediction (MTP) head added back in. The mix of quantizations
is chosen so that every routed expert fits in 2× 32 GB of VRAM, and the MTP head lets
llama.cpp decode speculatively. This model and patch is specifically tuned for dual Nvidia Tesla V100(32GB VRAM).

> **This file needs a patched llama.cpp.** It uses the `qwen4exp` architecture with
> MTP (`nextn`) tensors. Stock llama.cpp, Ollama and LM Studio cannot load it. See
> [Runtime](#runtime).

| | |
|---|---|
| File | `Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf` (85,923,705,344 bytes, about 80 GiB) |
| Routed experts | `UD-IQ3_XXS` from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| Dense (non-expert) tensors | the Q8_0 dense tensors of unsloth's `UD-IQ4_XS` |
| MTP head | converted from the original checkpoint, quantized to Q8_0 |
| Draft-only LM head | Q4_0, used only by the MTP draft passes. The target model keeps its own Q6_K output head. |
| Modality | text, and images with unsloth's vision projector (not in this repo; see [Image input](#image-input)) |

## Where the tensors come from

- unsloth GGUF revision `c8b5954a88c2775c546b92593eda40ea041d3176`.
- Qwen checkpoint revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Only the `mtp.*`
  tensors are taken from it.

unsloth's GGUFs have no MTP head. The head was converted separately with
`convert_hf_to_gguf.py --mtp`, which streams only the `mtp.*` tensors (about 8 GB) from
the original checkpoint. It was quantized to Q8_0 and then appended to the trunk. The
routed experts and the per-layer embedding (PLE) table come from `UD-IQ3_XXS`. The other
non-expert tensors come from `UD-IQ4_XS`, whose dense tensors are Q8_0. On V100 these run
faster in small-batch GEMVs than Q6_K does.

## Runtime

> [!WARNING]
> The file loads only with a patched llama.cpp **`v0.5.0`**. Stock llama.cpp cannot load it.

The patches are in [`patches/`](patches/):

| Patch | Use it for |
|---|---|
| [`llama-v0.5.0-v100-v3.patch`](patches/llama-v0.5.0-v100-v3.patch) | **Recommended.** Applies to a clean `v0.5.0`. |
| [`llama-v0.5.0-v100-v2-to-v3.patch`](patches/llama-v0.5.0-v100-v2-to-v3.patch) | Upgrades a tree that has the v2 patch applied to v3. |
| [`llama-v0.5.0-v100-v2.patch`](patches/llama-v0.5.0-v100-v2.patch) | Previous version (v2). No image input. Kept for reference. |
| [`llama-v0.5.0-clean-20260924-to-v2.patch`](patches/llama-v0.5.0-clean-20260924-to-v2.patch) | Upgrades a tree that has the v1 patch applied to v2. |
| [`llama-v0.5.0-clean-20260924.patch`](patches/llama-v0.5.0-clean-20260924.patch) | First version (v1). Layer split only. Kept for reference. |

> [!CAUTION]
> v4 and v5 patches were published briefly and then withdrawn because they had regressions.
> Use v3. The withdrawn patches are still in the repo history
> ([commit `21dbbdb8`](https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF/commit/21dbbdb8e3f17ecfec94a1d780a07c9dc9448afd)).

All versions add:

- `qwen4exp` support, including the FreeToken MoE port. The FreeToken sources are Apache-2.0
  and their license is in [`patches/LICENSE-freetoken`](patches/LICENSE-freetoken).
- MTP speculative decoding for this model.
- The V100 (`sm_70`) performance work: sparse QSA attention, fused kernels, and GPU argmax.

v2 adds:

- **Tensor split** (`--split-mode tensor`) for this model. Both GPUs work on every layer, and their partial results
  are summed by an in-graph AllReduce that writes straight into the peer GPU over NVLink or PCIe peer-to-peer.
  Decode is about 20% faster than with layer split.
- Fixes that let tensor split handle long prompts. v1 aborted on prompts over about 1,000 tokens in this mode.
- Faster fused hyper-connection kernels for up to 8 tokens, and MTP draft passes padded to one fixed shape.
- An even expert split under tensor split: 320 + 320 of the 640 expert columns, where 128-column steps gave
  256 + 384.

v3 adds:

- **Image input at the full 256K context.** The vision encoder does not fit next to the 256K context, so it is
  loaded on demand: for each image it borrows 2 GB of model weights on GPU 0, which are released while the encoder
  runs and copied back afterwards. See [Image input](#image-input).
- **MTP with images.** With v2, the server aborted about 200 tokens into any answer about an image.
- **Fast decode with an image in the context.** The sparse attention's fast path now works on contexts that hold an
  image. Without it, an image in the context halved decode speed at 56K and cut it to a third at 133K.
- **A fix for a race in a fused kernel.** In some layouts the kernel wrote its output over an input it was still
  reading. At long context with images this gave NaN scores and could crash the server. It also cost tensor split
  its perplexity gap against layer split (see [Quality notes](#quality-notes)).

Text decode speed and greedy text output are the same as with v2.

### Build

```bash
git clone --branch v0.5.0 https://github.com/ggml-org/llama.cpp
cd llama.cpp
hf download pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF \
  patches/llama-v0.5.0-v100-v3.patch patches/LICENSE-freetoken --local-dir .
git apply patches/llama-v0.5.0-v100-v3.patch

# NCCL: GGML_CUDA_NCCL is on by default, but CMake must find it. One way is the pip wheel:
pip install nvidia-nccl-cu12==2.27.3
NCCL_HOME=$(python -c 'import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])')

# CUDA 12.8 or later is required (the FreeToken code uses cuda_fp4.h).
# Volta needs CUDA 12.x: CUDA 13 cannot build for sm_70.
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF \
  -DNCCL_INCLUDE_DIR=$NCCL_HOME/include -DNCCL_LIBRARY=$NCCL_HOME/lib/libnccl.so.2
cmake --build build --target llama-server
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU's architecture. With a pip-installed NCCL, add `$NCCL_HOME/lib` to
`LD_LIBRARY_PATH` before starting the server.

The decode AllReduce is the patch's own kernel and does not use NCCL, but it needs peer access between the two
GPUs. The benchmarked build was linked with NCCL. If NCCL is missing, CMake prints a warning and llama.cpp falls
back to its internal AllReduce for the other reductions. That setup was not benchmarked.

To upgrade a tree that has the v2 patch applied, download `patches/llama-v0.5.0-v100-v2-to-v3.patch` instead,
`git apply` it on top, and rebuild. From v1, apply `patches/llama-v0.5.0-clean-20260924-to-v2.patch` first.

### Download

```bash
hf download pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF \
  Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf --local-dir models/merged
```

### Serve

These are the launch settings used for the numbers under [Performance](#performance), on
2× V100 32 GB:

```bash
export GGML_CUDA_P2P=1 GGML_CUDA_GRAPH_OPT=1 GGML_CUDA_DISABLE_GRAPHS=1 \
       GGML_FT_GRAPH_CACHE=1 GGML_FT_GRAPH_CACHE_MIN_NODES=32 \
       LLAMA_QWEN4EXP_HC_FUSION=1 LLAMA_HC_FUSED=2 LLAMA_MTP_PAD=5 \
       LLAMA_MTP_N_UBATCH=32 LLAMA_GPU_ARGMAX=1

build/bin/llama-server \
  --model models/merged/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf \
  --alias Qwen3.8-Flash-Next \
  --host 127.0.0.1 --port 8080 --api-key <your-key> \
  --ctx-size 262144 --parallel 1 \
  --threads 48 --threads-batch 48 \
  --batch-size 2048 --ubatch-size 256 \
  --n-gpu-layers auto --fit on --fit-target 512 \
  --split-mode tensor --main-gpu 0 --numa distribute \
  --moe-expert-cache 0 --moe-expert-cache-inserts 2 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode mmap \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --jinja --reasoning off --reasoning-budget 0 --no-reasoning-preserve \
  --metrics
```

Compared with v1, this adds `LLAMA_HC_FUSED=2`, `LLAMA_MTP_PAD=5` and `GGML_CUDA_DISABLE_GRAPHS=1`, and changes
`--split-mode layer --ubatch-size 512` to `--split-mode tensor --ubatch-size 256`. At startup the server logs two
warnings that are expected under tensor split: `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` and
`backend sampling not supported with SPLIT_MODE_TENSOR`.

To use layer split instead (for example, if the GPUs have no peer access), keep the same environment and use
`--split-mode layer --ubatch-size 512`.

#### Image input

The vision projector is unsloth's `mmproj-F16.gguf` (0.9 GB) at the same revision as the tensors above:

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF mmproj-F16.gguf \
  --revision c8b5954a88c2775c546b92593eda40ea041d3176 --local-dir models/vision

export MTMD_ON_DEMAND=1 GGML_CUDA_EVICT_MIB=2048   # in addition to the environment above
# the same llama-server command as above, plus:
  --mmproj models/vision/mmproj-F16.gguf
```

The 256K setup leaves no room on the GPUs for the encoder, so the two switches load it on demand:
- the encoder's weights stay in host RAM;
- for each image, the last 2 GB of the model weights on GPU 0 are released and the encoder runs in that memory;
- afterwards the encoder frees everything and the weights are copied back to the same addresses. They were copied
  to host RAM once (2 GB of pinned memory).

Each image costs 0.2–0.5 s extra. Text requests are not affected. The encoder runs on the first GPU (`-mmdev` is not
needed).

Without the switches, `--mmproj` needs a smaller context (about 128K) so the encoder can stay on the GPU.
`--no-mmproj-offload` keeps the 256K context but runs the encoder on the CPU: 3 s for a small image, 5.5 minutes for
a 4,000-token screenshot.

#### Command-line flags

| Flag | Value | Why |
|---|---|---|
| `--ctx-size` / `--parallel` | `262144` / `1` | full 256K context in one slot. MTP folded catch-up needs a single sequence. |
| `--batch-size` / `--ubatch-size` | `2048` / `256` | with micro-batch 512 the full 256K context runs out of memory under tensor split. If 128K context is enough, `--ctx-size 131072 --ubatch-size 512` prefills faster (861 against 646 tok/s to 64K) at the same decode speed. |
| `--split-mode tensor` | | every layer is split across both GPUs, with an AllReduce over NVLink or PCIe peer-to-peer |
| `--n-gpu-layers auto --fit on` | | every layer goes on the GPUs. `--fit` is not implemented for tensor split and does nothing there; with layer split it places the layers. |
| `--fit-target` | `512` | layer split only: MiB kept free per GPU for CUDA-graph and pool allocations `--fit` cannot see |
| `--numa distribute` | | spread CPU threads over both sockets |
| `--flash-attn on`, `--cache-type-k/-v q8_0` | | required by the fast QSA paths. The sparse attention kernel reads the Q8_0 cache directly. |
| `--load-mode mmap` | | the 27 GB PLE table stays in host RAM through mmap |
| `--moe-expert-cache` | `0` | patch flag: VRAM expert cache for setups where the experts do not fit. Not needed when they fit. |
| `--spec-type draft-mtp --spec-draft-n-max 3` | | MTP speculative decoding with the embedded head. Depth 3 beat 2 and 4. |
| `--spec-mtp-adaptive` | (off) | patch flag: adapts the draft depth between 2 and 3. Experimental, not used here. |
| `--reasoning off --reasoning-budget 0` | | non-thinking mode, as benchmarked |

#### Environment switches

| Variable | Set to | Default | Effect |
|---|---|---|---|
| `GGML_CUDA_P2P` | `1` | off | peer access between the GPUs (NVLink, or PCIe peer-to-peer) |
| `GGML_CUDA_DISABLE_GRAPHS` | `1` | unset | launches kernels directly instead of replaying CUDA graphs. Faster with tensor split, and the same speed with layer split. |
| `GGML_CUDA_GRAPH_OPT` | `1` | off | CUDA graph optimisation |
| `GGML_FT_GRAPH_CACHE` | `1` | off | keeps CUDA-graph variants for alternating batch shapes (verify and draft) |
| `GGML_FT_GRAPH_CACHE_MIN_NODES` | `32` | `1000` | caches the small MTP draft graphs too |
| `LLAMA_QWEN4EXP_HC_FUSION` | `1` | off | fused hyper-connection graph |
| `LLAMA_HC_FUSED` | `2` | off | v2: hyper-connection mix and combine as fused kernels, for up to 8 tokens |
| `LLAMA_MTP_PAD` | `5` | off | v2: pads every MTP draft pass to 5 rows, so the draft graph has one shape and is never rebuilt |
| `LLAMA_MTP_N_UBATCH` | `32` | ubatch | smaller micro-batch for the MTP context, using about 1.3 GB less VRAM |
| `LLAMA_GPU_ARGMAX` | `1` | off | greedy token picked on the GPU. Logits are copied to the host only when read. |
| `LLAMA_MTP_FOLD` | | on | folds MTP catch-up into the next draft pass. `0` disables it. |
| `LLAMA_QSA_SPARSE_KERNEL` | | on | block-sparse QSA attention kernel, used for prompt batches as well. `0` falls back to gather + flash attention. |
| `LLAMA_QSA_DENSE` | | off | `1` forces the dense (masked) QSA path, as a reference for debugging |
| `MTMD_ON_DEMAND` | `1` (with `--mmproj`) | off | v3: the vision encoder's weights stay in host RAM and go to the GPU only while an image is encoded |
| `GGML_CUDA_EVICT_MIB` | `2048` (with `--mmproj`) | `0` | v3: MiB of model weights per GPU that the encoder may borrow. They are released during an encode and restored after it. |

**Memory.** Every routed expert is on the GPUs, and only the 27 GB PLE table stays in host RAM. At 262,144-token
context with micro-batch 256, each GPU uses 32.3 GB.

## Performance

These results come from runs on 2026-09-26 with the launch settings above: 2× Tesla V100-SXM2-32GB (an NVLink
pair in a DGX-1), 2× Xeon Platinum 8168, CUDA 12.9.1 (`sm_70`), llama.cpp `v0.5.0` with
`patches/llama-v0.5.0-v100-v2.patch`.

Settings: 256K context, one slot, greedy decoding, non-thinking mode, 384 generated tokens,
MTP with 3 draft tokens.

| Scenario | Decode | ms per step |
|---|---:|---:|
| Short prompts (prose, code, technical, Japanese) | **121.9 tok/s** (104–136) | 24.0 |
| KV cache filled to 64K (all / follow-ups) | **119.9 / 123.0 tok/s** | 26.3 |
| KV cache filled to 250K (all / follow-ups) | **112.8 / 114.5 tok/s** | 27.9 |

| Prefill | Speed |
|---|---:|
| From empty to 64K | 646 tok/s |
| From empty to 250K | 589 tok/s |
| About 520-token follow-up on a 64K / 250K cache | 307–328 / 279–306 tok/s |

A speculative step is one 4-token verify plus 3 draft passes. Tokens per second depend on how
many drafts the text accepts: code takes about 3.3 tokens per step and prose about 2.5. The
time per step does not depend on the text, so it compares better across runs. Over three server launches the
short-prompt step time was 24.0, 24.2 and 26.1 ms. The 250K row comes from the 26.1 ms launch.

On a wider set of 16 prompts (320 tokens each), tensor split decodes at 126–127 tok/s (24.5–24.7 ms per step) and
layer split at 107–108 tok/s (28.7–28.9 ms per step).

These numbers were measured with v2. v3 measured the same: 24.3 ms per step on the short prompts, with or without
the vision encoder loaded, and 124.6–127.3 tok/s on the 16-prompt set.

### With images

v3 with the image-input settings above, 256K context. The two images are a newspaper photo and an IDE screenshot
(`tools/mtmd/test-1.jpeg` and `docs/android/imported-into-android-studio.jpg` in llama.cpp).

| Scenario | Result |
|---|---:|
| Idle GPU memory | 32.3 GB per GPU, as without vision |
| Prompt with a 300-token / 4,080-token image | 1.2 s / 10.4 s |
| Decode after an image, short context | 103–123 tok/s |
| Image plus 56K of text (first answer / follow-up) | 114 / 109 tok/s |
| Image plus 133K of text (first answer / follow-up) | 90 / 113 tok/s |
| 4,080-token image after 255K tokens of text | prefill 572 tok/s, follow-up 114.6 tok/s, peak 32.4 GB on GPU 0 |
| Two images in one request | both described correctly, 11.9 s for a 4,400-token prompt |

v2 could not run MTP with images. Without MTP, decode after an image ran at about 60 tok/s.

### Split mode and interconnect

Same host, same v2 build, short prompts and a 64K cache:

| Setup | Short prompts | ms per step | 64K (all / follow-ups) | Prefill to 64K |
|---|---:|---:|---:|---:|
| **Tensor split, NVLink pair** (recommended) | **121.9 tok/s** | **24.0** | 119.9 / 123.0 tok/s | 646 tok/s |
| Tensor split, no NVLink (PCIe peer-to-peer across the two CPU sockets) | 117.7 tok/s | 24.9 | 119.6 / 123.6 tok/s | 619 tok/s |
| Layer split, NVLink pair | 102.6 tok/s | 28.7 | 97.5 / 99.4 tok/s | 660 tok/s |
| Layer split, no NVLink | 102.5 tok/s | 28.8 | 101.5 / 103.4 tok/s | 710 tok/s |
| v1 README: layer split, `llama-custom` build, Xeon 8260 host, 2026-09-24 | 88.2 tok/s | 32.2 | 88.0 / 91.9 tok/s | 646 tok/s |

Tensor split needs peer access between the GPUs, but not NVLink: over PCIe it lost only about 4% on short
prompts. Layer split prefills faster, including short follow-up prompts (about 500 against 280–330 tok/s).

For comparison, stock `UD-IQ4_XS` with experts partly in host RAM and no MTP decodes at 30.2 tok/s
(llama-bench tg128, on the Xeon 8260 host).

## Quality notes

- These are low-bit experts (`IQ3_XXS`). Expect some quality loss compared with the higher
  unsloth quants.
- Tested: a 3-case deterministic API smoke test (marker, arithmetic, multi-turn recall) passes
  with MTP enabled, under both split modes. The patched CUDA kernels match the CPU backend in `test-backend-ops`.
  Prompts of 2.4K, 12.9K, 14.5K and 250K tokens work under tensor split.
- Perplexity on 4K-token chunks: tensor split 2.8506–2.8520 with v3, layer split 2.8525. v2's tensor split
  gave 2.8608 (+0.3%). That gap was not summation order, as this card said before. It was the fused-kernel race
  fixed in v3.
- Image prompts are not bit-for-bit reproducible between runs (text is). Blocks of identical image tokens tie at
  the sparse attention's cut-off, and which one wins can change. The answers were the same in testing.
- MTP drafts are verified by the target model. With greedy decoding, speculation does not
  change the output. Kernels that change the summation order can shift near-tied tokens.

## License

The weights are derived from Qwen/Qwen3.8-Flash-Next and are released under the same
**Qwen Community License 1.0**. The full text is in [`LICENSE`](LICENSE), and it must be
included with any copy. Commercial products or services above 100M monthly active users or
US$20M monthly revenue must display the model name prominently. See the license for the exact
terms.

The patches in `patches/` are not covered by the model license. They modify llama.cpp, which is
MIT-licensed. Their FreeToken-derived files (`freetoken-*`, marked `SPDX-License-Identifier:
Apache-2.0`) are under Apache-2.0, and that license is in `patches/LICENSE-freetoken`.
