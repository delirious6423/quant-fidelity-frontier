---
license: apache-2.0
base_model:
  - peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF
  - ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - rocmfp4
  - llama.cpp
  - strix-halo
  - gfx1151
  - rocm
  - amd
  - ryzen-ai-max
  - agentic-coding
  - moe
---

# Tiel-Coder-35B-A3B — ROCmFP4 STRIX\_LEAN GGUF — AMD Ryzen AI Max+ 395 / gfx1151

> ### ⚠️ Read this before comparing any number here to a discrete GPU
>
> Every measurement is from an **AMD Ryzen AI MAX+ 395 "Strix Halo"** — an **integrated GPU with
> unified memory**. No discrete VRAM: the Radeon 8060S addresses system RAM through the **GTT
> aperture**, shared with the CPU. Capacity is huge; bandwidth (~215 GB/s measured) is the trade.

Requantized from the excellent
**[peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF)**
— the Tiel-Coder fine-tune and the original GGUF conversion are their work, not mine. Go star
their repo. The underlying base is
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

STRIX\_LEAN is my default publish tier for Strix Halo: ROCmFP4 weights with the Strix attention K/V handling and Q5_K token embeddings, plus a Q6_K output head. `general.file_type` = **106** (`Q4_0_ROCMFP4_STRIX_LEAN`). **17.46 GiB.**

## ⚠️ Honest provenance: this is a requantization of a Q8, not a BF16 build

Their repo publishes GGUFs, not unquantized weights, so this was built from their
**`UD-Q8_K_XL`** (35.81 GiB) using `--allow-requantize` — **quantization applied on top of
quantization**. That is lossier than quantizing from BF16/F32, and you should weigh it against a
Q4 built from full-precision weights. I used their **highest** available tier as the source to
keep the loss as small as this path allows. Flagging it because you cannot tell from the file.

## What's inside

| tensor group | type |
|---|---|
| MoE expert weights | `TYPE_101` (ROCmFP4, 4.251 bpw) |
| attention | half `TYPE_100`, half `TYPE_101` |
| `token_embd.weight` | `Q5_K` |
| `output.weight` (lm head) | **`Q6_K`** — verified by exact tensor name |
| norms / biases | `F32` |

`output.weight` is never left at 4-bit: every sampled token passes through the lm head, so its
error lands directly in the argmax. Verified by **exact** name — `output.weight` is a substring of
`attn_output.weight`, so a loose check reports success on a 4-bit head.

## Building a runtime that loads these files

`qwen35moe` **and** the ROCmFP4 tensor types are both already in the public ROCmFPX fork — no patch
needed:

```bash
git clone https://github.com/charlie12345/ROCmFPX.git
cd ROCmFPX && git checkout d3ca537
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-quantize -j$(nproc)
```

Verified: a binary built from that clean clone loads these files and generates. Stock upstream
llama.cpp will **not** work — it has no ROCmFP4 tensor types.

## Measured — Ryzen AI MAX+ 395, gfx1151, ROCm 7.2.4, full offload

- prompt processing: **1156 tok/s** — 1209.9 / 1156.3 / 1155.0 / 1155.9 / 1153.6
- generation: **58.75 tok/s** — 59.59 / 58.77 / 58.75 / 58.70 / 58.69
- GTT resident: **18.0 GiB**

Method: one fixed 4,000-token prompt from a real source corpus, reused across samples with
`cache_prompt: false`; run 1 discarded as warm-up; median of the 4 settled samples.

### vs the Q8_K_XL it was built from

Same box, same method, same prompt:

| | this build | their `UD-Q8_K_XL` |
|---|---|---|
| size | **17.46 GiB** | 35.81 GiB |
| GTT resident | **18.0 GiB** | 36.9 GiB |
| prompt processing | **1156** | 794 |
| generation | **58.75** | 46.5 |

Half the footprint and faster on both axes — the trade is the requantization loss described above.
Quality was not benchmarked; if you need maximum fidelity, use their Q8_K_XL.

## Usage

```
llama-server \
  --model Tiel-Coder-35B-A3B-Q4_0-ROCmFP4-STRIX_LEAN.gguf \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 999 --flash-attn on --fit off \
  --ctx-size 32768 --threads 16 --jinja
```

Model supports up to **262,144** context. A vision tower (`mmproj`) is available in the source repo.

<!-- CREDITS:START -->

## Acknowledgements

**[peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll)** — the Tiel-Coder fine-tune and the
GGUF conversion this is built from. Their work; I only requantized it.

**[ornith-ai](https://huggingface.co/ornith-ai)** — Ornith-1.5-35B-A3B, the underlying base model.

**ROCmFPX** — defines the ROCmFP4 tensor formats and carries the `qwen35moe` support used here.

**[llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml-org and contributors.**

**AMD ROCm** — the compute platform targeted (ROCm 7.2.4, gfx1151).

<!-- CREDITS:END -->
