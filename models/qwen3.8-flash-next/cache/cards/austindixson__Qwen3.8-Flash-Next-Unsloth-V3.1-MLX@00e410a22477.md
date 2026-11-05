---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- quantized
- moe
- qwen4_exp
- unsloth
- imatrix
---

# Qwen3.8-Flash-Next, MLX (mlx-serve) port with the Unsloth Dynamic V3.1 bit map

**Tested end-to-end with mlx-serve 26.9.3 on Apple M3 Max, 128 GB unified memory:** loaded and served real chat/tool-call generation, passed the long-context QSA smoke, and recorded the speed comparison below.

An MLX affine mixed-precision quant of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(`model_type: qwen4_exp`) for [mlx-serve](https://github.com/ddalcu/mlx-serve). It was converted from the
original BF16 weights (not from a GGUF) using

- the per-tensor bit allocation of **Unsloth Dynamic V3.1**
  ([method](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/dynamic-3.1-ggufs),
  [GGUFs](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)), re-expressed as MLX affine widths and group sizes, and
- the **Unsloth importance matrix** from that GGUF repo, remapped to the HF tensor layout, to weight the quantization fit.

This is a port of the recipe, not of Unsloth's files: GGUF K-quant types do not exist in MLX, so each Unsloth
type was mapped to the MLX affine width and group size of similar size. It is not an Unsloth release and Unsloth did not
review it.

Credit: the base model is Qwen's; the V3.1 method and the imatrix are Unsloth's; the streaming converter is
ddalcu's (`tests/convert_qwen38_flash_next.py` in the mlx-serve repository, commit `7ff28ae`).

## Recipe

| tensors | MLX affine width | Unsloth V3.1 counterpart |
|---|---|---|
| routed experts gate/up, layers 0-47 except layer 2 | 4-bit, group 64 (47 layers) | Q4_K |
| routed experts gate/up, layer 2 | 5-bit, group 64 | Q5_K |
| routed experts down, 43 layers | 5-bit, group 32 | Q5_1 |
| routed experts down, layers 2, 4, 30, 46, 47 | 8-bit, group 64 | Q8_0 |
| attention, GDN, indexer, shared experts, hyper-connections | 8-bit, group 64 | Q8_0 |
| token embedding, lm_head | 8-bit, group 64 | Q8_0 |
| n-gram embedding table (`ngram_table.bin`) | 4-bit, group 32 | n/a (kept as in other mlx-serve packs) |
| routers, norms, convs, SSM state | bf16 | |
| vision tower (`model-vision.safetensors`) | bf16 | |

Quantized with the imatrix-weighted affine fit (`--nonexpert-bits 8 --embed-bits 8 --ngram-bits 4`, per-layer expert
widths from an allocation file). Layer numbers are the HF layer indices. The widths in the pack were checked against
the allocation file (144 expert tensors checked, 0 mismatches; embed and lm_head 8-bit).

## Size

116,714,668,611 bytes on disk (116.7 GB, 108.7 GiB), of which `ngram_table.bin` is 32.0 GB (memory-mapped by
mlx-serve, not resident) and the vision tower 0.9 GB. The speed bench measured 24.0 GB server RSS for this pack versus 55.2 GB for the comparison live 4/8 server; these are live-process observations, not a controlled wired-memory comparison.

## Usage (mlx-serve)

Tested with **mlx-serve 26.9.3** on **Apple M3 Max, 128 GB unified memory**:

```bash
export MLX_SERVE_CACHE_LIMIT=1073741824
mlx-serve --model <local-path-to-this-repo> --serve --host 127.0.0.1 --port 8080   --log-file off --mtp --mtp-depth 4 --pld --pld-draft-len 5   --kv-quant 4 --ctx-size 150000 --prefix-cache-mem 4GB --prefix-cache-disk 4GB   --no-vision --skip-mem-preflight --no-prevent-sleep --prefix-cache-entries 0
```

The measured run used these flags; `--no-vision` skips the vision tower. MTP and PLD are enabled above; their isolated contribution has not been measured.

## Results

Smoke and speed are measured. KLD/top-1, the full G2 and optimization are still in progress; no claim of a finished quality win is made.

**G2 weight-space error (EARLY, partial: 54 of 131 BF16 shards, covering layers 0, 1 and 10-18 plus the dense tensors of layer 19; layers 2-9 and 20-47, and with them all five 8-bit-down layers 2, 4, 30, 46, 47, are not in this subset).**
Imatrix-weighted relative squared error, sum om (w_hat - w)^2 / sum om w^2, against the BF16 weights, lower is better.
Experts are sampled (8 per bank per layer) and dense tensors are row-subsampled (at most 2048 rows). "live 4/8" is the common
mixed 4-bit experts / 8-bit rest MLX pack recipe with plain (unweighted) quantization; "iQ-4.7bpw recipe" is the same
widths as live with an imatrix-weighted fit, scored here with the Unsloth imatrix. These are recipe re-quantizations of
the sampled tensors, not scores of other published packs. The embed row is not in the early subset.

### By role (all layers; imatrix-weighted relative error, lower is better)

| role | live 4/8 | iQ-4.7bpw recipe | B (V3.1 map) | B vs live |
|---|---|---|---|---|
| dense | 9.290e-05 | 5.772e-05 | 5.772e-05 | -37.9% |
| expert_down | 8.395e-03 | 6.269e-03 | 1.058e-03 | -87.4% |
| expert_gate_up | 8.318e-03 | 6.448e-03 | 6.448e-03 | -22.5% |

<details><summary>Per layer, B vs live (early)</summary>

(expert_gate_up / expert_down / dense: B vs live %)

| layer | gate_up | down | dense |
|---|---|---|---|
| 0 | -29.8% | -90.3% | -38.1% |
| 1 | -26.9% | -89.2% | -37.3% |
| 10 | -21.7% | -86.9% | -37.2% |
| 11 | -21.0% | -86.9% | -38.9% |
| 12 | -21.7% | -86.9% | -37.2% |
| 13 | -22.5% | -87.3% | -37.5% |
| 14 | -22.0% | -86.6% | -38.7% |
| 15 | -24.0% | -87.4% | -38.4% |
| 16 | -20.9% | -86.6% | -39.4% |
| 17 | -20.7% | -87.0% | -37.3% |
| 18 | -22.0% | -86.8% | -37.1% |
| 19 | - | - | -36.8% |

</details>

G2 is a weight-space proxy for output error, not a quality benchmark.

| measurement | status |
|---|---|
| Real load + generation smoke on this pack | PASS: short chat, tool call and ~18k-character QSA probe returned 200 on default kernels; no ERROR/Traceback/panic. Six of seven behavior auto-checks passed; arithmetic was unscored at the 200-token cap on both arms, not marked wrong. |
| Measured speed, this pack vs comparison live 4/8 | Decode 56.1 vs 58.8 tok/s (3 x 256-token prompts, greedy); TTFT 14.68 vs 15.08 s at ~8k tokens; 68.50 vs 60.01 s at ~32k tokens; MTP accepts/round 1.685 vs 1.723. Cold prefix cache; same mlx-serve 26.9.3 flags. Decode, not a speed win. |
| KLD and top-1 agreement vs BF16 (A/B against the common 4/8 pack and the iQ-4.7bpw pack) | UNMEASURED - pending |
| Full G2 (all 131 shards, including the embedding and flagged layers) | UNMEASURED - pending |

## License

The weights are derived from Qwen3.8-Flash-Next and are distributed under the **Qwen Community License 1.0**; the
`LICENSE` file from the base repository is included unchanged. Its conditions apply to this derivative, including
keeping the copyright and permission notice and the separate-license requirement for Model-as-a-Service and AI Work
Assistant businesses. Read it before commercial use.
