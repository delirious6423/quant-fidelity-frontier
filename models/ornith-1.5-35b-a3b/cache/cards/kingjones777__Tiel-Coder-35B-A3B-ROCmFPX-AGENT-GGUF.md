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

# Tiel-Coder-35B-A3B — ROCmFPX AGENT GGUF — AMD Ryzen AI Max+ 395 / gfx1151

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

AGENT is the tool-call tier: `Q6_0_ROCMFPX_AGENT` routing with **Q8_0 on both the output head and the token embeddings**. It exists for agentic/function-calling work, not for speed — see the comparison below. `general.file_type` = **114** (`Q6_0_ROCMFPX_AGENT`). **30.08 GiB.**

## ⚠️ Honest provenance: this is a requantization of a Q8, not a BF16 build

Their repo publishes GGUFs, not unquantized weights, so this was built from their
**`UD-Q8_K_XL`** (35.81 GiB) using `--allow-requantize` — **quantization applied on top of
quantization**. That is lossier than quantizing from BF16/F32, and you should weigh it against a
Q4 built from full-precision weights. I used their **highest** available tier as the source to
keep the loss as small as this path allows. Flagging it because you cannot tell from the file.

## What's inside

| tensor group | type |
|---|---|
| MoE expert weights | `q6_0_rocmfpx` / `q8_0_rocmfpx` routing |
| attention | ROCmFPX Q6/Q8 agent routing |
| `token_embd.weight` | **`Q8_0`** — verified by exact tensor name |
| `output.weight` (lm head) | **`Q8_0`** — verified by exact tensor name |
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

- prompt processing: **599 tok/s** — 608.7 / 599.7 / 598.9 / 598.3 / 598.1
- generation: **46.61 tok/s** — 46.73 / 46.69 / 46.55 / 46.61 / 46.50
- GTT resident: **30.4 GiB**
- **tool call verified**: emitted `get_weather({"city":"Paris"})` from a real `tools` payload

Method: one fixed 4,000-token prompt from a real source corpus, reused across samples with
`cache_prompt: false`; run 1 discarded as warm-up; median of the 4 settled samples.

### vs the Q8_K_XL it was built from

Same box, same method, same prompt:

| tier | size | GTT | pp | gen |
|---|---|---|---|---|
| **AGENT (this, 114)** | **30.08 GiB** | 30.4 | **599** | **46.61** |
| [STRIX_LEAN (106)](https://huggingface.co/kingjones777/Tiel-Coder-35B-A3B-ROCmFP4-STRIX_LEAN-GGUF) | 17.46 GiB | 18.0 | 1156 | 58.75 |
| [FAST (103)](https://huggingface.co/kingjones777/Tiel-Coder-35B-A3B-ROCmFP4-FAST-GGUF) | 17.37 GiB | 18.0 | 1174 | 60.28 |
| their `UD-Q8_K_XL` | 35.81 GiB | 36.9 | 794 | 46.5 |

**Pick this tier only if you want the tool-call routing.** It is ~0.5× the prompt processing and
~0.79× the generation of my 4-bit tiers, and 13 GiB larger. What it buys is `Q6_0_ROCMFPX_AGENT`
routing plus Q8_0 embeddings and head. For general coding, STRIX_LEAN or FAST is the better pick.
Quality was not benchmarked; if you need maximum fidelity, use their Q8_K_XL.

## Usage

```
llama-server \
  --model Tiel-Coder-35B-A3B-Q6_0-ROCmFPX-AGENT.gguf \
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
