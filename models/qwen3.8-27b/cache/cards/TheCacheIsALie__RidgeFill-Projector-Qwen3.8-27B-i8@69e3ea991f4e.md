---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - ridgefill
  - llama.cpp
  - gguf
  - prefill-acceleration
  - qwen3.8
---

<p align="center"><img src="https://huggingface.co/TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8/resolve/main/assets/ridgefill-banner.webp" width="100%" alt="RidgeFill"></p>

<p align="center"><b>RidgeFill maps for Qwen3.8-27B: split 16 of 64, for the llama.cpp fork.</b></p>
<p align="center"><b>1.25 to 1.31× faster time to first token at 16K to 120K tokens. RULER and LongBench v2 change by −1.2 to −1.4 points, each 95 % interval including zero.</b></p>

<p align="center">
<a href="https://github.com/Approximate-Science/ridgefill-data">Data and code</a> ·
<a href="https://huggingface.co/TheCacheIsALie">All maps</a> ·
<a href="https://github.com/Approximate-Science/llama.cpp-ridgefill">llama.cpp fork</a> ·
<a href="#citation">Citation</a>
</p>

## What this is

The per-layer maps that let the [RidgeFill llama.cpp fork](https://github.com/Approximate-Science/llama.cpp-ridgefill) skip most of Qwen3.8-27B's upper layers during
prefill. Layers 0 to 15 run for every prompt row. For the other prompt rows, except an exact tail and a selected set of
exact rows, the maps predict the inputs of layers 16 to 63 from the state entering layer 16. Generation after the
prompt is exact. This is not a language model: it loads with `--ridgefill` next to an unmodified Qwen3.8-27B GGUF.
Output is not identical to stock llama.cpp.

## Files

| File | Bytes | SHA256 |
|---|---|---|
| `ridgefill-projector-qwen3.8-27b-i8-S16.gguf` | 1,452,188,032 | `8646d41a5a28d60ef53cc5e1fee4b2bfb20d0775db88d3f89d0731ad9f6b629d` |

The GGUF is the file the paper's 27B runs loaded. It holds the 48 maps in llama.cpp's Q8_0 format, converted from the
int8 release below, and per-layer terminal-state corrections, which the measured configuration does not use
(`--ridgefill-alpha 0`).

`ridgefill-projector-qwen3.8-27b-i8/` holds the int8 release the GGUF was converted from by the fork's
`convert_ridgefill_to_gguf.py`: 48 maps (`proj8.L16` to `proj8.L63`, 1.2 GiB; int8 with one bf16 scale per 128
weights of each row and a bf16 bias), `correction.safetensors`,
`rowsel.safetensors`, `ridgefill.json`, `LICENSE` and `SHA256SUMS`. The files are byte for byte the ones the
conversion read, so `ridgefill.json` keeps fields written before this release. Every file's sha256 is also in the data
repository's `provenance/external.tsv`.

## Use

```sh
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 --local-dir models
hf download TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8 ridgefill-projector-qwen3.8-27b-i8-S16.gguf --local-dir models
```

Build the fork and start `llama-server` with the settings in
[RIDGEFILL.md](https://github.com/Approximate-Science/llama.cpp-ridgefill/blob/main/RIDGEFILL.md) (Quick start, Qwen3.8-27B). Prompts shorter than 8,192 tokens run
exactly. Draft models and MTP are refused while RidgeFill is enabled.

## Results

One A100 PCIe 40 GB, CUDA. Stock is upstream llama.cpp 1242be66f with the same flags, without RidgeFill. Time to first
token is the server's prompt time, median of three warm requests.

| Context | Speedup, maps in VRAM | Speedup, maps streamed from host | Extra VRAM, in VRAM / streamed |
|---|---|---|---|
| 16,384 | 1.25× | 1.19× | 1,592 / 268 MiB |
| 32,768 | 1.30× | 1.27× | 1,802 / 478 MiB |
| 65,536 | 1.31× | 1.30× | 2,270 / 944 MiB |
| 120,000 | 1.28× | 1.29× | 3,014 / 1,732 MiB |

| Benchmark | Stock | RidgeFill | Δ [95 % CI] |
|---|---|---|---|
| RULER 16K, 13 tasks × 20 | 85.69 | 84.51 | −1.19 [−3.13, +0.79] |
| RULER 32K, 13 tasks × 20 | 87.85 | 86.42 | −1.42 [−3.21, +0.35] |
| LongBench v2, official CoT, full length, 503 items | 38.57 | 37.18 | −1.39 [−4.97, +2.19] |

After the prompt, next-token agreement with stock over 100 LongBench v2 documents per length is 94.6 % at 16K and
93.6 % at 32K, with median KL of 0.0040 and 0.0042 nats per token.

## How RidgeFill works

<p align="center"><img src="https://raw.githubusercontent.com/Approximate-Science/ridgefill-data/v1.1/figures/fig_fill_vs_kva.png" width="85%" alt="Two ways to fill a late layer's cache from a middle layer's state: trained key and value heads, and RidgeFill's closed-form map to the layer's input"></p>

The lower layers run exactly up to a split. For every prompt row except the exact ones, a ridge map per later layer
predicts that layer's input from the split layer's state, and the layer writes its cache from the prediction (b)
instead of running in full: its own key and value projections run unchanged, and on a Gated DeltaNet layer so does
its state update. The maps are fitted once, in closed form, from one pass over calibration text. The last few thousand
prompt tokens, any selected exact rows and every decoding step run the full model. Nothing is trained and no token is
dropped. The design comes from DeepSeek-V4.1-Flash, where the projection is trained into the model; (a) is kishida's
Q3-8B-KVA-Projector, which retrofitted it onto Qwen3-8B with trained key and value heads.

## Citation

The paper, *RidgeFill: Training-Free Late-Layer Cache Fill for Faster Prefill* by Dylan Johnston and Robert P Smith, is in
preparation. Until it is out, cite the data release:

```bibtex
@misc{johnston2026ridgefilldata,
  author  = {Johnston, Dylan and Smith, Robert P},
  title   = {{RidgeFill}: data and code for ``Training-Free Late-Layer Cache Fill for Faster Prefill''},
  year    = {2026},
  version = {1.1},
  url     = {https://github.com/Approximate-Science/ridgefill-data}
}
```

An earlier technical note is archived at [doi:10.5281/zenodo.23179168](https://doi.org/10.5281/zenodo.23179168). The
paper replaces some of its numbers; `SUPERSEDED.md` in the data repository lists which.
