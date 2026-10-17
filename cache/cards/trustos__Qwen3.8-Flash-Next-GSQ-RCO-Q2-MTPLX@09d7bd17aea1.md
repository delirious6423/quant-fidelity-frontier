---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mtplx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mtplx
- mlx
- apple-silicon
- macos
- speculative-decoding
- multi-token-prediction
- mtp
- moe
- qwen
- qwen3.8
- qwen3.8-flash-next
- flash-next
- qwen4
- gsq
- rco
- gsq-rco
- ist-daslab
- 2-bit
- quantization
- mixed-precision
- vision
- multimodal
- local-llm
- m5-max
---

# Qwen3.8-Flash-Next · GSQ-RCO Q2 · MTPLX

Qwen's 125B-A6B Flash-Next for [MTPLX](https://github.com/youssofal/MTPLX) on Apple Silicon, with
native multi-token-prediction drafting and vision. The routed experts (120.8B of the ~125B trunk
parameters) are taken **bit for bit** from IST Austria DASLab's
[GSQ-RCO Q2_0 GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF). Everything else
comes from the MTPLX
[Optimized Speed pack](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed).

**~45 GiB peak while serving, against ~80 GiB for the 4-bit MTPLX pack, with MTP decode as fast or
faster in our runs.** The speed comes from MTPLX's fused stage-3 MoE verify lane, which this repo
extends to 2-bit experts (`mtplx-q2-stage3/`, installed once, see below).

| | |
|---|---|
| Download | 77.3 GB (includes the 32 GB n-gram table, streamed from SSD) |
| Peak while serving | 45 GiB (measured, M5 Max 128 GB) |
| Routed experts | MLX affine 2-bit, group 64 (ISTA GSQ-RCO Q2_0, lossless repack) |
| Dense trunk, MTP head, vision, n-gram table | from the MTPLX Optimized Speed pack (4-bit g32 / 8-bit g64 / bf16) |
| Context window | 262,144 tokens |
| MTP depth | 3 |
| Vision | yes (333-tensor vision tower, image + text input) |
| Runtime | MTPLX 2.12.2 (tested) |
| Tested on | MacBook Pro M5 Max, 128 GB. Smaller Macs are untested. |

## Use it

```bash
pip install mtplx                                   # or the MTPLX app
mtplx pull trustos/Qwen3.8-Flash-Next-GSQ-RCO-Q2-MTPLX
sh ~/.mtplx/models/trustos--Qwen3.8-Flash-Next-GSQ-RCO-Q2-MTPLX/mtplx-q2-stage3/install.sh   # once
mtplx serve --model trustos/Qwen3.8-Flash-Next-GSQ-RCO-Q2-MTPLX
```

The model load should print:

```text
[gsq-mtplx] stage-3 installed for q2/g64 routed experts: layers=48 ... max_abs_diff=0.0 ... mode=auto
```

The OpenAI-compatible server lists the model as `qwen3.8-flash-next-gsq-rco-q2-mtplx`. Use the
path where `mtplx pull` put the pack; `~/.mtplx/models` is MTPLX's default.

### Why the one-time install step

MTPLX 2.12.2 runs its fused "stage-3" MoE verify lane only for 4-bit/group-32 experts. The lane is a
two-kernel routing head, a paired gate/up GLU kernel, and one dispatch for routed-down, the
slot-ordered reduce, the shared add and the hyper residual. Without this step, stock MTPLX runs the
pack on stock `gather_qmm`: about 10 % slower MTP decode (97.8 vs 107.9 tok/s), same output.

`mtplx-q2-stage3/install.sh` adds one `.pth` file to MTPLX's Python environment. It does not modify
MTPLX itself. From then on every MTPLX process (CLI, `serve`, the app) arms the lane for this pack
automatically. Details:

- **Exact.** The 2-bit/group-64 kernels reproduce MLX 0.32.2's stock 2-bit arithmetic bit for bit,
  and MTPLX's own installer checks every layer against the stock path.
- **Scoped.** Every other model runs exactly as before, and any stage-3 flag you set explicitly
  wins.
- **Fail-safe.** If a future MTPLX changes the internals it relies on, it prints a warning and
  falls back to the stock path.
- **Controls.** `install.sh --status` shows the state and `install.sh --uninstall` removes it. For
  a single launch without installing, use `sh mtplx-q2-stage3/bin/mtplx-gsq` with the usual `mtplx`
  arguments.

## Speeds

MacBook Pro M5 Max, 128 GB, MTPLX 2.12.2. `mtplx chat --profile turbo --mtp --reasoning off`, code
prompt, 512 tokens, seed 1. Other apps' compositing kept the GPU about 35–40 % busy.

**Stage-3 lane on vs off, same pack, three alternating rounds.** Both arms produced identical
output (358 of 456 drafts accepted):

| | Round 1 | Round 2 | Round 3 | Verify ms/call | Peak |
|---|---|---|---|---|---|
| **This pack (stage-3 lane, `install.sh`)** | **108.4** | **107.7** | **107.6** | **25.1** | **45 GiB** |
| This pack, stock path (`MTPLX_QWEN4_M4_STAGE3=0`) | 98.0 | 97.5 | 97.8 | 28.3 | 45 GiB |

**Against the 4-bit MTPLX Optimized Speed pack** (earlier session, arms alternated, two rounds, GPU
15–25 % busy):

| | Round 1 | Round 2 | Peak |
|---|---|---|---|
| **This pack (stage-3 lane)** | **107.1** | **111.3** | **45 GiB** |
| MTPLX Optimized Speed (4-bit) | 95.1 | 87.7 | 80 GiB |

The 4-bit pack's best run on this test on a quieter GPU was 107.2 tok/s. Plain autoregressive decode
is faster on this pack (42.1 vs 36.5 tok/s), because fewer bytes are read per token.

**Runtime contract** (`mtplx forge verify --stamp`, `mtplx serve` lane, stage-3 on, sampled at
temperature 1.0, 2 samples): 48.7 tok/s autoregressive, 95.7 tok/s MTP depth 3. These rows sample
different text on every run, so compare them only with other forge stamps.

## How it is built

ggml `Q2_0` stores 64 weights per block as an fp16 scale `d` and 2-bit codes `q`, with
`w = (q − 1)·d`. That is exactly MLX affine 2-bit with group size 64 (`scale = d`, `bias = −d`), and
the packed bytes are reused as-is. Against `mx.dequantize`, the repack is bit-exact with fp16 scales.
The scales are stored as bf16 (the pack's dtype), which rounds each scale by at most ≈0.2 %.

The build itself:

- All 144 routed-expert tensors (`ffn_{gate,up,down}_exps`, 48 layers) from ISTA's
  `Q2_0/…-00001-of-00002.gguf` (rev `ed59f92`) replace the pack's
  `mlp.switch_mlp.{gate,up,down}_proj`. Shapes, orientation and expert order match one to one.
- Everything else is the MTPLX Optimized Speed pack (rev `34ae535`): attention, GDN, shared experts,
  router, hyper-connections, QSA indexer, `embed_tokens` / `lm_head`, the MTP head
  (`mtp.safetensors`), the vision tower (`model-vision.safetensors`) and the PLE n-gram table
  (`ngram-table.safetensors`, 4-bit g32).
- `config.json → mtplx_recipe.routed_experts` records the expert source and method. The runtime
  contract was re-stamped for these exact bytes.

## Quality

**Not evaluated.** ISTA's evaluation covers their own Q2_0 GGUF, which pairs these experts with
their own GSQ-quantized dense layers. This pack pairs the same experts with MTPLX's 4/8-bit dense
layers, a combination nobody has benchmarked. Signals from our runs so far:

- MTP acceptance on the code prompt matched the 4-bit pack (79 % vs 80 %); on one prose sample it
  was lower (43 % vs 53 %).
- Code, prose and image answers were coherent and on-task.

## Credits

- **Qwen**: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). The upstream
  card is preserved as `README-upstream-qwen.md`.
- **IST Austria DASLab**: GSQ and RCO, and the GSQ-RCO Q2_0 experts used here.
- **MTPLX** (Youssof Altoukhi): the runtime and the Optimized Speed pack that the non-expert tensors
  come from. `mtplx-q2-stage3` builds on MTPLX's stage-3 kernels (Apache-2.0).

This is an independent community build, not affiliated with or endorsed by Qwen, IST Austria or
MTPLX.

## License

Qwen Community License 1.0, inherited from the base model; see `LICENSE`. It includes conditions for
large-scale commercial use and a separate-license requirement for Model-as-a-Service and AI Work
Assistant businesses. The `mtplx-q2-stage3/` code is Apache-2.0 and builds on MTPLX (Apache-2.0).

## Citation

If you use the GSQ-RCO weights or methods, please cite both papers:

```bibtex
@article{gsq2026,
  title  = {GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling},
  author = {Dadgarnia, Alireza and Tabesh, Soroush and Nikdan, Mahdi and Helcig, Michael and Kurtic, Eldar and Kleinegger, Maximilian and Alistarh, Dan},
  journal= {arXiv preprint arXiv:2604.18556},
  year   = {2026}
}
@article{rco2026,
  title  = {Model Compression with Exact Budget Constraints via Riemannian Manifolds},
  author = {Helcig, Michael and Alistarh, Dan},
  journal= {arXiv preprint arXiv:2605.00649},
  year   = {2026}
}
```
