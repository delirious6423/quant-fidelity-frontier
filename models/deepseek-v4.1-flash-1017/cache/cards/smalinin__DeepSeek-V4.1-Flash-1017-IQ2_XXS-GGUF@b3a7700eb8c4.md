---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
inference: false
tags:
- gguf
- deepseek
- deepseek-v4.1
- mixture-of-experts
- llama.cpp
- imatrix
- mixed-precision
quantized_by: smalinin
---

# DeepSeek-V4.1-Flash-1017 - Q2 Antirez-like + Q5_K Engram

[**DeepSeek-V4.1-Flash-1017**](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1407)

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V4.1-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<hr>

<p align="center">
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf"><b>Technical Report</b> 👁️</a>
</p>


An imatrix-assisted, mixed-precision GGUF of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), prepared for the CUDA build of [smalinin/llama.cpp, branch `my_build`](https://github.com/smalinin/llama.cpp/tree/my_build).

> [!IMPORTANT]
> Our current [`my_build` runtime](https://github.com/smalinin/llama.cpp/tree/my_build) also supports loading DeepSeek-V4.1-Flash-1017 GGUF models from [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) **without rewriting or repacking their GGUF files**.


This model card covers **one initial release: Q2 Antirez-like with Q5_K Engram tables**.

- Routed expert **gate/up: IQ2_XXS** in all 40 layers.
- Routed expert **down: Q2_K** in all 40 layers.
- Both large **Engram embedding tables: Q5_K**.
- Attention, shared experts, the output head and Engram WKV retain Q8_0; other protected tensors retain BF16 or F32 as detailed below.

"Antirez-like" describes the asymmetric expert-precision recipe; it does not mean this is antirez's own GGUF release.

> [!IMPORTANT]
> Use the current CUDA build of the linked `my_build` branch. Compatibility with a stock upstream llama.cpp build is not claimed. 
> This is the **1017 checkpoint**, not the earlier DeepSeek-V4.1-Flash release.

## Size and BPW

The release contains **8 GGUF shards and 1,046 tensors**. Sizes and types below were checked against the completed local GGUF headers, not inferred from the quantization name.

| Component | Tensor payload, GB | Tensor payload, GiB | Effective BPW |
| --- | ---: | ---: | ---: |
| Backbone and all other tensors, excluding the two Engram embedding tables | 162.530 | 151.368 | 2.356018 |
| Two Engram embedding tables | 135.172 | 125.889 | 5.500000 |
| All stored tensors | **297.702** | **277.257** | **3.181876** |

The complete eight-file download is approximately **297.708 GB / 277.262 GiB**, including GGUF headers and alignment. `GB = 10^9` bytes; `GiB = 2^30` bytes.

BPW means `8 * tensor payload bytes / stored tensor elements`, including quantization scales and block overhead but excluding GGUF headers. It is an average over all stored weights, not just the active experts. Engram WKV, gate weights, token embeddings, output and auxiliary tensors are included in the backbone row.

Compared with our 1017 Antirez-like variant with Q8_0 Engram, this package keeps the same backbone and replaces only the two large embedding tables with Q5_K. This saves storage and host-cache space, but Q5_K is lossy: quality parity with Q8_0 Engram or the official checkpoint is not guaranteed.

## Tensor precision distribution

This is an excerpt of the audited tensor inventory for **this Q2 Antirez-like + Q5_K Engram release**. Layer numbering starts at 0. Patterns describe tensor families; indexer and compressor tensors exist only in the layers that use them.

| Tensor group | Actual stored precision |
| --- | --- |
| Routed expert gate: `blk.*.ffn_gate_exps.weight` | **IQ2_XXS**, all 40 layers |
| Routed expert up: `blk.*.ffn_up_exps.weight` | **IQ2_XXS**, all 40 layers |
| Routed expert down: `blk.*.ffn_down_exps.weight` | **Q2_K**, all 40 layers |
| Attention: `attn_kv`, `attn_q_a`, `attn_q_b`, `attn_output_a`, `attn_output_b` | Q8_0, all 40 layers |
| Shared expert gate/up/down: `blk.*.ffn_{gate,up,down}_shexp.weight` | Q8_0, all 40 layers |
| Main output head: `output.weight` | Q8_0 |
| Token embeddings: `token_embd.weight` | BF16 |
| Engram embedding tables: `blk.1.engram_embd.weight`, `blk.14.engram_embd.weight` | **Q5_K** |
| Engram WKV: `blk.{1,14}.engram_wkv.weight` | Q8_0 |
| Engram Q/K: `blk.{1,14}.engram_{q,k}.weight` | BF16; combined in F32 by our runtime at load time |
| Indexer Q: `blk.*.indexer.attn_q_b.weight` | Q8_0 |
| Indexer K and projection: `blk.*.indexer.attn_k.weight`, `blk.*.indexer.proj.weight` | BF16 |
| Attention compressor gate/KV: `blk.*.attn_compressor_{gate,kv}.weight` | BF16 |
| Expert router: `blk.*.ffn_gate_inp.weight` | BF16 |
| mHC / hyper-connection tensors: `blk.*.hc_*.weight` | F32 |
| Norms, attention sinks and expert-routing biases | F32 |

There are no separate ordinary dense-FFN or `attn_gate` tensor families in this backbone. A DSpark speculative sidecar is a separate model file, not part of the precision table or size totals above.

## Engram shards, RAM and VRAM

For **this eight-shard release**, the Engram tables are in:

| GGUF shard | Tensor | Precision |
| --- | --- | --- |
| **2 of 8** | `blk.1.engram_embd.weight` | Q5_K |
| **5 of 8** | `blk.14.engram_embd.weight` | Q5_K |

Each table is approximately 67.586 GB. A single tensor cannot be split across GGUF files, so these two shards are much larger than some backbone shards.

With the recommended **`--load-mode mmap --lazy-mode auto`** configuration, these huge lookup tables are **not loaded wholesale into VRAM**. The runtime accesses the needed rows from the host-backed memory mapping. The example below also explicitly keeps the two table tensors on CPU.

Accessed pages can become resident in the operating system's RAM page cache and can later be evicted. This is **not** a promise that all 135 GB of tables are permanently resident in RAM. Cold pages or insufficient host cache can add storage I/O and reduce speed; SSD/NVMe storage is preferable for this workload.

The smaller Engram WKV and Q/K tensors are ordinary model tensors and can still be placed on GPU. Lazy table access does not mean all Engram computation runs on CPU.

`--lazy-mode auto` enables on-demand access for eligible tensors larger than 4 GiB. `--lazy-mode off` makes those tensors resident according to their placement; it can greatly increase RAM or VRAM requirements. Keep the explicit mmap/lazy settings unless intentionally testing a different memory policy.

## Runtime compatibility, including Unsloth models

Our current [`my_build` runtime](https://github.com/smalinin/llama.cpp/tree/my_build) also supports loading DeepSeek-V4.1-Flash-1017 GGUF models from [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) **without rewriting or repacking their GGUF files**.

The compatibility layer understands both our two-table Engram layout and Unsloth's combined `per_layer_token_embd.weight` layout, including its offsets, precombined gate and metadata conventions. It loads those weights into **our computation graph**; loading an Unsloth quant does not switch to Unsloth's runtime implementation.

Unmodified **Unsloth UD-Q2_K_XL** has been tested in our runtime, including generation, tools and cache-restoration checks. This is not a claim that every quantization, backend or configuration has been validated. Use a complete matching shard set: **do not mix individual Unsloth shards or Engram files with this release**. Reverse compatibility of our GGUF layout with Unsloth's runtime is not claimed.

### Speed relative to Unsloth's runtime

In the maintainer's local CUDA comparisons, **our strict runtime was roughly 10% slower than Unsloth's runtime in generation throughput**, with workload-dependent variation. The following 8K examples were around 11-12% lower:

| Workload, 8,192 input tokens | Our strict runtime, tok/s | Unsloth runtime, tok/s |
| --- | ---: | ---: |
| Target only | 41.47 | 46.70 |
| BF16 DSpark, `spec-draft-n-max=2` | 48.12 | 54.77 |

These are historical local measurements from October 22, 2026: the **same Unsloth UD-Q2_K_XL GGUF** in both runtimes, five 48-GiB Ada GPUs plus one 24-GiB Ampere GPU, layer split, F16 KV, `index_topk=2048`, batch 2048 / ubatch 512, one slot, one warm-up and three measured repeats, 256 generated tokens. The sidecar row used the same BF16 DSpark file in both runtimes.

This is a **runtime comparison, not a throughput benchmark of the smaller Antirez-like quant published here**. Generated continuations were not necessarily identical across runtimes. The approximate 10% gap is not a universal performance guarantee, a prefill-speed claim, or an isolated measurement of any single kernel.

### Why we did not merge the entire Unsloth update

We evaluated a targeted port in an isolated `test-unsloth` branch and kept the existing strict runtime as the default:

1. **The alternate batched graph changes numerical behavior.** Batched matrix operations and attention reductions can produce different logits from serial execution, sometimes changing the selected top-1 token. It was not a drop-in, bitwise-equivalent replacement for our tested verification path.
2. **Reproducibility needed further investigation.** In the evaluated original Unsloth build, repeated greedy 8K runs did not always produce the same continuation. This does not by itself prove poor model quality or a particular bug, but that build could not serve as a bitwise reference for the port.
3. **The corrected port's gain did not meet our acceptance criterion.** After stability corrections, short-workload DSpark throughput improved by about 4%, below the predefined 5% gate. Its sampled text/code NLL stayed within our review threshold, but that did not establish serial/batch equivalence or unrestricted correctness.

We therefore retained our tested numerical, causal-mask and rollback policy instead of merging the complete alternate graph. Selected compatibility work and independently checked fixes can still be adopted. "Strict" describes that policy, not a guarantee of identical floating-point results on every GPU or arbitrary workload.

This decision concerns the locally evaluated Unsloth build `b72050d03` and our isolated port at that time; it is **not** a claim that all Unsloth releases are incorrect, or that our runtime universally produces higher-quality answers.

## Build and run

### Build the CUDA runtime

```bash
git clone --branch my_build https://github.com/smalinin/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j --target llama-server
```

Use the CUDA architecture and compiler settings appropriate for your GPUs. Keep the executable and its accompanying shared libraries from the same build.

### Download the model

```bash
hf download smalinin/DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF \
  --include '*.gguf' \
  --local-dir ./DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF
```

Download **all eight shards**, keep them together, and pass only shard `00001-of-00008` to the server. The other seven are discovered automatically. The wildcard in the example below selects that first shard regardless of the chosen filename prefix; keep only one matching eight-shard variant in this directory.

### Example server command

This is a starting configuration for a sufficiently large multi-GPU CUDA system, not a hardware-independent benchmark profile. Adjust context, threads and placement for your available RAM/VRAM.

```bash
DS41_MODEL_DIR=./DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF

./build/bin/llama-server \
  --model "$DS41_MODEL_DIR/DeepSeek-V4.1-Flash-1017-EngramQ5-IQ2_XXS_GU-Q2_K_Down-00001-of-00008.gguf" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 256000 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --parallel 1 \
  --threads 20 \
  --n-gpu-layers auto \
  --fit on \
  --n-cpu-moe 0 \
  --load-mode mmap \
  --lazy-mode auto \
  --flash-attn on \
  --jinja \
  --reasoning on \
  --reasoning-format deepseek \
  --chat-template-kwargs '{"reasoning_effort":80}' \
  --override-kv deepseek41.attention.indexer.top_k=int:2048 \
  --temp 1.0 \
  --top-p 1.0
```


The GGUF stores `deepseek41.attention.indexer.top_k=512`; the example explicitly overrides it to **2048**. 
This changes inference behavior and cost, not the saved files. Do not confuse it with the separate candidate-block 
selection settings.

The example uses the embedded Jinja chat template. `reasoning_effort` is a template value from 1 to 100, not a hard token limit; higher effort can produce longer answers. Use `--reasoning off` when reasoning is not wanted.

## Optional vision and speculative decoding

The main GGUF shards do not contain a vision projector or a DSpark sidecar. This initial model release should not 
be interpreted as bundling those additional files.

- Image input requires a separately obtained, compatible **1017 vision projector**, supplied with `--mmproj`.
- Speculative decoding requires a separately obtained, compatible **1017 DSpark sidecar**, supplied with `--spec-draft-model` and `--spec-type draft-dspark`. Start with `--spec-draft-n-max 2` or `3` and compare end-to-end TG against target-only on your hardware; a sidecar is not automatically faster.

Use only files for the matching checkpoint and verify their compatibility with the runtime. Sidecar/projector sizes are not included in the model totals above.

## Limitations

- IQ2_XXS routed experts and Q5_K Engram are aggressive, lossy quantization choices. This card does not establish quality superiority over Unsloth, antirez, higher-bit quants or official weights.
- Different quantization recipes spend precision on different tensors. More bits in one group do not establish better overall quality.
- Hardware, context length, batch sizes, GPU topology and host page-cache state can materially change speed and memory requirements.
- A context capacity in model metadata is not a guarantee that every capacity has been validated by this release. The example starts at 64K; it is not an acceptance claim for the full native context window.

## Samples

### Q2 Antirez-like with Q5_K Engram

```create html with svg "pelican on a bicycle"```
<p align="center">
  <img src="pelic_q2.png" width="400px" >
</p>

- one shoot call
``` create HTML page with "Indian Railways WDP-4D dual-cab diesel locomotive" using Three.js ```
<p align="center">
  <img src="wdp_q2.png" >
</p>

- used opencode, toolcall, vision
``` create HTML page with "Indian Railways WDP-4D dual-cab diesel locomotive" using Three.js ```
<p align="center">
  <img src="wdp_q2_tools.png" >
</p>



## Acknowledgements

Thanks to [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) for the checkpoint and reference implementation, [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and the DeepSeek-V4.1 contributors, antirez for the expert-precision design inspiration, and [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) for their GGUF and runtime work that informed compatibility testing.


