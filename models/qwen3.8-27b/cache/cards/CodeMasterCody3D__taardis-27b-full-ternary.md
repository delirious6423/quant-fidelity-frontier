---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
  - ternary
  - quantization
  - integer
  - gguf
  - llama.cpp
  - taardis
library_name: gguf
---

![TAARDIS](TAARDIS-new.png)

# TAARDIS-27B — Full-Ternary Integer (V2)

**Ternary Adaptive Alignment & Rotation for Dense Integer Stacking.**

A 27-billion-parameter transformer at **1.75 bits per weight — 5.90 GB** —
where **every weight is a ternary integer** `{-1, 0, +1} × scale`: body,
attention, MLP, **LM head and embedding table included**, with norms and
group scales on the integer grid too (balanced-ternary digit stacks). And
V2 ships the pipeline's correction system: **The Doctors** — 496 cross-layer
low-rank ternary branches that ride alongside the frozen weights and cancel
propagated quantization error.

| file | size | what it is |
|---|---|---|
| **TAARDIS-27B-Full-Ternary-V2-1.75bit.gguf** | **5.90 GB** | the model, 1.75 bpw (base-3 five-trit pack) |
| **doctors/TAARDIS-27B-Doctors-V3.lora.gguf** | **0.32 GB** | the corrections, **all-ternary** — load with `--lora` (fork ≥ `c4c56a5`) |
| doctors/TAARDIS-27B-Doctors-V2.lora.gguf | 0.92 GB | same corrections, f16 container — for older fork builds |
| TAARDIS-27B-Full-Ternary-V1.gguf | 7.16 GB | same states at 2.125 bpw (2-bit pack), kept for compatibility |

**Wikitext perplexity (c512, 274 chunks, identical binary/kernels/text):**

| configuration | PPL |
|---|---|
| V1 / V2 weights alone | 13.61 / 13.6114 |
| weights + The Doctors (**recommended**) | **11.8346** |

The 1.75-bit file is a **lossless repack** of the 2.125-bit one — same ternary
states, same scales byte-for-byte, just a tighter numeral system (five trits
per byte instead of four 2-bit codes). Verified by full decode-back of every
block plus the perplexity equality above.

---

## V1 vs V2 — same model, two containers

**They are the same weights.** V2 is a lossless repack of V1: identical
ternary states and identical scales, byte for byte — only the numeral system
of the container changes. Wikitext agrees to four decimals (13.6110 vs
13.6114). Pick by *where you run it*, not by quality.

| | **V1** | **V2** |
|---|---|---|
| file | `TAARDIS-27B-Full-Ternary-V1.gguf` | `TAARDIS-27B-Full-Ternary-V2-1.75bit.gguf` |
| tensor type | `Q1_0_g128` — four 2-bit codes per byte | `Q1_T_g128` — five base-3 trits per byte |
| bits / weight | **2.125** | **1.75** |
| size | 7.17 GB | **5.90 GB** |
| CPU decode (AVX2) | **fastest** — the 2-bit unpack is ~2.75× cheaper | slower (base-3 unpack) |
| GPU decode (fused kernels, Blackwell) | 101 t/s | 90 t/s |
| best for | **CPU-only machines**, max speed | **GPU / tight VRAM / small downloads** |

Both take the **same Doctors adapters** — the corrections don't care which
container the weights live in.

**Run V1 on a CPU** (the AVX2 ternary kernels; ~3 t/s on a 12-thread Ryzen 3600, 27B in ~8 GB of RAM):
```bash
./build/bin/llama-cli -m TAARDIS-27B-Full-Ternary-V1.gguf \
    --lora doctors/TAARDIS-27B-Doctors-V3.lora.gguf \
    -t $(nproc) -c 4096 --repeat-penalty 1.3 -p "Q: Why is the sky blue? A:"
```

**Run V2 on a GPU** (fused ternary GEMV + ternary KV cache; 5.9 GB of weights):
```bash
./build/bin/llama-cli -m TAARDIS-27B-Full-Ternary-V2-1.75bit.gguf \
    --lora doctors/TAARDIS-27B-Doctors-V3.lora.gguf \
    -ngl 99 -c 8192 -ctk q1_t_g128 -ctv q1_t_g128 -fa on \
    --repeat-penalty 1.3 -p "Q: Why is the sky blue? A:"
```

**Small GPU (e.g. 6–8 GB)?** Keep the FFN weights in system RAM and put
attention + the KV cache on the card:
```bash
./build/bin/llama-cli -m TAARDIS-27B-Full-Ternary-V1.gguf \
    --lora doctors/TAARDIS-27B-Doctors-V3.lora.gguf \
    -ngl 99 -ot "blk\.\d+\.ffn_.*=CPU" -c 4096 --repeat-penalty 1.3 -p "..."
```
Measured on an **AMD RX 5600 XT (6 GB) + Ryzen 3600**, built with `-DGGML_HIP=ON`:
2.85 t/s CPU-only → **4.15 t/s** with this split, perplexity **bit-identical** to the CPU
run. The same fork builds for CUDA, ROCm/HIP and AVX2 CPU with no source changes.

**Check either file yourself:** `llama-perplexity -m <file> -f wiki.test.raw -c 512`
— both print ~13.61 alone and ~11.83 with the Doctors.

## vs Ternary-Bonsai-27B (PrismML)

Measured head-to-head on the same binary, kernels and text:

| | **TAARDIS-27B V2** | Ternary-Bonsai-27B |
|---|---|---|
| ternary GGUF size | **5.90 GB (1.75 bpw)** | 7.17 GB (2.125 bpw) |
| size *with* corrections | **6.22 GB** (V3) | — |
| wikitext c512 PPL | **11.8346** (with Doctors) | 11.01 |
| norms + group scales | **integer grid (k8/k6 digit stacks)** | FP16 |
| head + embedding | ternary | ternary |
| ternary KV-cache option | **yes — 1.75 bits/value** | no |
| conversion recipe | **open** (fork + tools published) | closed |
| team | **one person, 51 days** | funded team |

PrismML shipped Bonsai-27B on **July 4, 2026**. This project started from an
empty folder on **July 14 — 51 days (7 weeks and 2 days) before this release**,
built solo on free-tier Colab/Kaggle GPUs and a home desktop. Bonsai's quality
still leads by a few percent — they train their ternary weights; this pipeline
is post-training conversion plus trained corrections — but the corrected
TAARDIS stack is **smaller than their model alone**, more integer, and the
recipe is open.

---

## ⚠️ Requires the TAARDIS fork of llama.cpp

The weights live in a **rotated basis** (block-Hadamard) and the runtime must
rotate activations to match. **Stock llama.cpp will load the file and produce
garbage** (perplexity ≈ 1,260,000). Use the fork:

```bash
git clone -b q1_0_g128-port https://github.com/CodeMasterCody3D/taardis-llama.cpp llama.cpp
cd llama.cpp
```

**Build (CPU, AVX2 ternary kernels):**
```bash
cmake -B build -DLLAMA_CURL=OFF
cmake --build build -j --target llama-cli llama-server llama-perplexity
```

**Build (CUDA):**
```bash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF
cmake --build build -j --target llama-cli llama-server llama-perplexity
```
*(`75` = T4/RTX 20xx, `80` = A100, `86` = RTX 30xx, `89` = RTX 40xx.)*

**Run — recommended setup (V2 + the Doctors):**
```bash
./build/bin/llama-cli -m TAARDIS-27B-Full-Ternary-V2-1.75bit.gguf \
    --lora doctors/TAARDIS-27B-Doctors-V3.lora.gguf \
    -t $(nproc) -c 4096 --repeat-penalty 1.3 \
    -p "Q: Why is the sky blue? A:"
```
One file is the model, the other is its medicine. Leave `--lora` off and you
get the uncorrected model exactly; load it and all 496 branches apply at scale
1.0. The rotation is applied automatically from GGUF metadata.

---

## GPU speed (fused ternary GEMV)

The fork's CUDA path runs decode through **fused ternary GEMV kernels** (fork
commit `89187fb`+): the packed trits are read directly and dotted against
int8 activations with `dp4a` — no fp16 intermediate. Measured with
`llama-bench -ngl 99 -p 512 -n 128` on an NVIDIA RTX PRO 6000 (Blackwell):

| file | decode (tg128) | prompt (pp512) |
|---|---|---|
| V1 — Q1_0_g128, 2.125 bpw | **101 t/s** | 2690 t/s |
| V2 — Q1_T_g128, 1.75 bpw | **90 t/s** | 2350 t/s |

Before these kernels both decoded at ~8 t/s on the same GPU. The 1.75-bit
file decodes its base-3 trits through a 243-entry lane lookup table in a
warp-uniform kernel (no divergence), landing within 11% of the 2-bit pack
while reading 18% fewer bytes — on bandwidth-bound GPUs (T4-class) the
smaller file is expected to close that gap or lead. Both kernels are validated
bit-exact against the CPU reference by `test-backend-ops`.

## The Doctors

The correction mechanism: **cross-layer, jointly-trained low-rank ternary
branches** (DOCTOR: Downstream-Oriented Coordinated Ternary Output Repair)
that cancel the *propagated* quantization error — measured 3.3× more
effective than per-layer correction on held-out data. They ride inside the
TAARDIS and heal the damage: 496 branches, ranks allocated 8…256 per matmul
by measured benefit, packed as a llama.cpp-native LoRA with the basis
rotation folded in offline.

**V3 — the Doctors are ternary too.** Each branch is ternarized per rank
component (one scale per rank column of A / rank row of B). V3 folds A's
scale into B's row scale and ships `B` as `Q1_0_g128` blocks and `A` as pure
`{-1,0,+1}` (2-bit packed where rank ≥ 128, f16 containers of ±1/0 values
below that): **920 MB → 323 MB, same function** (wikitext 10.7300 vs V2's
10.7365 on the same 4 chunks — fp16 scale rounding). It declares
`adapter.type = taardis-lora`: the fork feeds it the block-Hadamard-rotated
activation it was trained on, and **older builds refuse it loudly** instead
of silently applying it in the wrong basis (that would cost ~1.6×). Requires
fork commit `c4c56a5` or later; V2 stays for older builds.

**Why a sidecar instead of one file:** a low-rank correction *cannot* be
folded into a ternary base without pushing the weights off the integer grid —
merging would de-ternarize the model. Riding as a branch is the
mathematically honest architecture, and it means you can toggle the
correction on and off and measure exactly what it buys (11.8346 vs 13.6114).

**They also stop thinking loops.** Qwen3.8's `xhigh` reasoning effort at the
model's own recommended sampling (temp 1.0, top-p 0.95, top-k 20, no repeat
penalty) is where low-bit models are most prone to degenerating into
repetition. Measured on a hard reasoning question ("how many trailing zeros
does 1000! have?"):

| config | outcome |
|---|---|
| V2 + Doctors V3 | closed `</think>` on its own at 4,373 tokens (5% repeat-rate) and answered |
| V2 alone (no Doctors) | **hard loop** — the same sentence repeated ~150 times, never closed the tag |

The answer with Doctors was still wrong (arithmetic slipped inside the
thinking, not a format failure) — the Doctors are not claimed to fix
reasoning correctness here, only the **stop discipline**: with them, the
model reliably finishes; without them, it can get stuck. A controlled
comparison against the FP16 teacher under the same settings is still
outstanding.

---

## Ternary-integer KV cache (optional)

The fork also ships **ternary KV-cache types**, so the *runtime state* can be
integer too. Select per-tensor with `-ctk`/`-ctv`. Measured on this 27B:

| KV type | flag | bits/value | PPL cost | KV @ 1M ctx | model + 1M ctx |
|---|---|---|---|---|---|
| **f16** | *(default)* | 16 | — | 68.7 GB | 74.6 GB |
| **q4_0** | `q4_0` | 4.5 | **+0.16%** | 19.3 GB | 25.2 GB |
| **q1_0_g128** | `q1_0_g128` | 2.125 | +11.4% | 9.1 GB | 15.0 GB |
| **q1_t_g128 (k1)** | `q1_t_g128` | 1.75 | +11.2% | **7.4 GB** | **13.3 GB** |

*("model + ctx" columns are weights + KV cache only; add ~2 GB of compute buffers at `-b 256` for the real peak -- see the T4 measurement below.)*

```bash
# q4_0 KV — near-free quality, 3.6× smaller cache. RECOMMENDED default:
./build/bin/llama-cli -m TAARDIS-27B-Full-Ternary-V2-1.75bit.gguf \
    --lora doctors/TAARDIS-27B-Doctors-V2.lora.gguf -ctk q4_0 -ctv q4_0 -c 8192 -p "..."

# k1 ternary KV — MAXIMUM compression. On a 16 GB card (T4, measured):
#   512K tokens fits in 12.2 GB and decodes at 6.4 t/s.
./build/bin/llama-cli -m TAARDIS-27B-Full-Ternary-V2-1.75bit.gguf \
    --lora doctors/TAARDIS-27B-Doctors-V3.lora.gguf -ctk q1_t_g128 -ctv q1_t_g128 -c 524288 -p "..."

# 1M tokens needs ~15.4 GB (weights+Doctors 6.2 + KV 7.2 + compute buffers ~2) --
# doesn't fit a 16 GB card at -b 256. Two honest options if you need the full 1M:
#   a) a bigger card (measured clean on a 97 GB Blackwell), or
#   b) --no-kv-offload: the KV cache stays in system RAM, decode goes through
#      the CPU attention path (slower, but it fits by construction).
```

> **✅ GPU-resident ternary KV (CUDA) — as of fork commit `e638dc1`.** The
> ternary cache types now have CUDA write (`set_rows`, with the same Lloyd
> scale refinement as the CPU path) and flash-attention read kernels. Validated
> on an NVIDIA Blackwell: cache written by the GPU scores **13.29 vs 13.28** for
> the CPU-written cache (0.06%), and a **1,000,000-token `q1_t_g128` cache was
> allocated on-GPU with the model decoding through it.** Measured cost on the
> V2+Doctors stack: **+11.7%** perplexity vs f16 KV (8 chunks).
>
> **Measured on a 16 GB card (Tesla T4, `-b 256`), weights fully on the GPU:**
>
> | config | max context that fits + decodes | peak VRAM | decode |
> |---|---|---|---|
> | V2 + Doctors V3 | **524,288 tokens** | 12.2 GB | 6.4 t/s |
> | V2 alone (no Doctors) | **786,432 tokens** | 14.8 GB | 6.9 t/s |
>
> The Doctors cost ~262K tokens of context on a 16 GB card (their weights are
> only 0.3 GB, but that's enough to tip the compute-buffer math). **Neither
> configuration reaches 1,000,000 tokens on a 16 GB card with the weights fully
> resident.** An earlier draft of this card claimed 1M fits a 16 GB card as-is;
> that was wrong and has been corrected here.
>
> **The full 1,000,000 tokens DOES fit a 16 GB card — the right way to do it
> is `--no-kv-offload`, not FFN offload.** This model is a hybrid: only
> **16 of its 64 layers** are real attention layers with a growing KV cache
> (the other 48 are Gated DeltaNet -- linear attention with a small
> *fixed-size* recurrent state, unaffected by context length). Keep every
> weight on the GPU and move only the KV cache to system RAM, and just those
> 16 layers pay a PCIe round trip per token instead of the whole model:
> ```bash
> ./build/bin/llama-cli -m TAARDIS-27B-Full-Ternary-V2-1.75bit.gguf \
>     --lora doctors/TAARDIS-27B-Doctors-V3.lora.gguf \
>     -ngl 99 --no-kv-offload -c 1000000 \
>     -ctk q1_t_g128 -ctv q1_t_g128 -fa on -p "..."
> ```
> Measured, full 1,000,000 tokens, same T4:
>
> | config | peak VRAM | decode |
> |---|---|---|
> | V2 + Doctors V3 | **11.0 GB** | **4.7 t/s** |
> | V2 alone (no Doctors) | 10.7 GB | 5.1 t/s |
>
> 6x faster than moving the FFN instead (0.8 t/s, see below), and with 4+ GB of
> VRAM still free -- this is a usable interactive speed, not just an offline
> batch mode. **This is the recommended way to run 1M tokens on a 16 GB card.**
>
> A worse alternative also fits, for the record: moving the FFN weights to
> system RAM instead (`-ot "blk\.\d+\.ffn_.*=CPU"`) also gets you the full
> 1M, at 13.7-13.8 GB peak but only **0.8 t/s** -- the FFN is most of the
> model's weight bytes, so nearly everything round-trips over PCIe every
> token. Only useful for a build-once/query-many cache or offline scoring.

**The honest trade-off:** the ternary KV types cost about **+11% perplexity**.
On a 27B that already fits in memory, use `q4_0` (+0.16%). The ternary KV's
home is the regime where fp16/q4 *can't fit at all* — million-token contexts,
big batches, 120B-class models — where a 9× smaller cache is the difference
between running and not running. Choose deliberately.

---

## Notes & honesty

- **Research artifact.** Aggressive compression (27B → 5.90 GB); expect
  quality below the fp16 original. The Doctors close part of the gap
  (13.61 → 11.8346); parity is the roadmap, not the present.
- **Values are integer; compute is not yet.** Every stored parameter sits on
  the ternary-integer grid; the forward pass still dequantizes to fp16 for
  the matmuls. A fused ternary kernel is future work.
- **Reproduce:** `llama-perplexity -m <model> [--lora <doctors>] -f wiki.test.raw
  -c 512`. Rotation off (`LLAMA_FORGE_ROT_DISABLE=1`) explodes perplexity to
  ~1.26M — proof the rotation is load-bearing, and that stock llama.cpp
  cannot honestly run this file.

## License & attribution

**TAARDIS-27B is a derivative of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**,
created by the Qwen team (Alibaba Cloud) and released under the **Apache License 2.0**.
A copy of that license is included in this repository as [`LICENSE`](LICENSE).

The base checkpoint's weights were **modified** by the TAARDIS pipeline
(ternarization, block-Hadamard rotation, balanced-ternary integer conversion,
and low-rank ternary corrections); TAARDIS does **not** retrain the model from
scratch. This release is **not endorsed by or affiliated with** Alibaba Cloud
or the Qwen team.

| component | author |
|---|---|
| Base architecture & checkpoint | Qwen team, Alibaba Cloud — Apache 2.0 |
| TAARDIS conversion / representation pipeline | Cody Dixon |
| Fork implementation & ternary kernels | Cody Dixon |
| The Doctors (correction system) | Cody Dixon |
| Benchmarks & measurements | Cody Dixon |

**Statement of changes (Apache 2.0 §4b):** the base weights were converted to
a full-ternary integer representation at 1.75 bits/weight with per-linear
block-Hadamard rotation, k8/k6 integer norms and scales, and 496 low-rank
ternary correction branches, as described above.

## Citation

TAARDIS pipeline & The Doctors — Cody Dixon, 2026.
Fork: <https://github.com/CodeMasterCody3D/taardis-llama.cpp> (branch `q1_0_g128-port`).
