---
license: apache-2.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags: [nvfp4, quantized, qwen3.8, tandemllm, gptq, speculative-decoding]
---
# Qwen3.8-27B in NVFP4 for TandemLLM

These are 4-bit weights for Qwen3.8-27B, built for the [TandemLLM](https://github.com/0xBakeer/TandemLLM) engine on a DGX Spark. They are the NVFP4 set published with the paper *StairCut: sizing speculative draft trees on a measured verification-cost staircase* (Khaled Bakeer, 2026; Zenodo: [10.5281/zenodo.23045275](https://doi.org/10.5281/zenodo.23045275)). The paper's PDF is also in this repository: [`paper/StairCut.pdf`](paper/StairCut.pdf). We quantised them once, from Qwen's BF16 release `Qwen/Qwen3.8-27B`. The weight set TandemLLM served before this one came from Qwen's FP8 release, so every weight in it was rounded twice. On four held-out texts, the new full set loses 16 % to 49 % less than the old one.

![Speed by kind of text with StairCut on these weights](media/content_speed.gif)

## Intended use

Serving Qwen3.8-27B with TandemLLM on one NVIDIA DGX Spark (GB10), and reproducing the paper's measurements. The files are not a standalone checkpoint: they are overlays in TandemLLM's own NVFP4 layout and need Qwen's FP8 release beside them. Other engines cannot load them without a converter. The model's abilities, limits and risks are Qwen3.8-27B's; see its model card.

## What is in it

The files are overlays. TandemLLM loads Qwen's FP8 release `Qwen/Qwen3.8-27B-FP8` and replaces the projections named in a file with their NVFP4 form. Everything else stays as Qwen shipped it: the embedding, every norm, the convolution weights, `in_proj_a`, `in_proj_b` and the recurrent state.

| file | projections | size |
|-|-|-|
| `mlp.safetensors` | `gate_proj`, `up_proj`, `down_proj` of all 64 layers | 9.63 GB |
| `gdn.safetensors` | `in_proj_qkv`, `in_proj_z`, `out_proj` of the 48 Gated DeltaNet layers | 3.11 GB |
| `attn.safetensors` | `q_proj`, `k_proj`, `v_proj`, `o_proj` of the 16 attention layers | 0.94 GB |
| `head-fp8.safetensors` | `lm_head` in e4m3, one fp32 scale per vocabulary row | 1.27 GB |
| `mix-1gb.safetensors` | 341 of the 400 projections; the other 59 stay FP8 | 12.41 GB |
| `mix-3.16gb.safetensors` | 240 of the 400 projections; the other 160 stay FP8 | 9.63 GB |

The profiles:

| profile | `--nvfp4` | bytes a decode step reads |
|-|-|-|
| NVFP4 | `mlp,gdn,attn` | 15.01 GB |
| mix 1 GB | `mix-1gb` | 16.00 GB |
| balanced | `mlp` | 18.17 GB |
| mix 3.16 GB | `mix-3.16gb` | 18.17 GB |
| FP8 | none, and no `--fp8-head` | 26.93 GB |

Every profile except FP8 also loads `head-fp8.safetensors` with `--fp8-head`.

The format is the engine's own NVFP4 layout. Each weight is a 4-bit e2m1 code, packed two to a byte with the low nibble first. Each run of 16 weights along a row shares one e4m3 scale, and each tensor has one fp32 scale (`weight_scale_2`). Tensor names are `layers.N.<module>.<proj>.weight`, `.weight_scale` and `.weight_scale_2`, the same as the set served before, so the engine loads these without a change. The kernels are weights-only: activations stay bf16.

## How we built them

1. Download `Qwen/Qwen3.8-27B` at revision `1d4bf0f`. Run 149,682 calibration tokens through the engine in BF16: 74,519 tokens of Python (the repository's `bench/calib.txt` plus 37 Python standard-library modules), 59,163 tokens of English Wikipedia and 16,000 tokens of German Wikipedia. `calibration/manifest.json` lists every document.
2. Record, for every projection, the second moment of its input over those tokens (`H = X^T X`, 101 GB in fp32, taken in five passes of 13 layers).
3. Quantise every projection with GPTQ on the NVFP4 grid. At the first column of each group of 16, a search over five scale sizes (1.00 to 0.80 of the group's largest weight) picks the group's e4m3 scale. Each column is then rounded, and its error is pushed onto the columns not yet rounded, weighted by the inverse of `H`. On the calibration inputs this leaves 46 % less output error than the scale search alone (relative error 0.0024 against 0.0044).
4. Build the e4m3 head from the BF16 `lm_head`. The BF16 and FP8 releases carry the same `lm_head` bit for bit, and this head differs from the old served head in 1 row of 248,320.
5. Measure how far the next-token distribution moves when one piece of one layer alone goes to NVFP4: 256 pieces, on 8,192 tokens that neither the calibration nor the gate reads. The output projections of the recurrent layers cost the most per byte, then the attention output projections. Layers 35 to 41 are the most sensitive part of the model. A mix keeps the pieces with the highest cost per byte in FP8 until a byte budget is spent: 0.99 GB for `mix-1gb`, and 3.16 GB, the bytes the balanced profile spends, for `mix-3.16gb`.

The whole build is one job on one RTX PRO 6000 (96 GB) and took 49 minutes. The code is in the [TandemLLM repository](https://github.com/0xBakeer/TandemLLM): `tools/calib_corpus.py`, `tools/quant_nvfp4.py build`, `tools/quant_head.py`, `tools/quant_sensitivity.py` and `tools/quality_gate.py --plan`.

## Quality against BF16

Every row is the change in loss per token against Qwen's BF16 weights, on the same tokens, in one process, teacher-forced. Lower is better. The texts are held out from the calibration and from the mix selection: `bench/heldout_prose.txt` (1,521 tokens), `bench/heldout_code.txt` (8,002 tokens of Triton's Python source), and two further splits of 10,500 tokens of Python and 11,999 tokens of Wikipedia prose.

| weights | prose | code | more code | more prose | GB a step |
|-|-|-|-|-|-|
| Qwen FP8, BF16 head | +0.0079 | +0.0032 | +0.0050 | +0.0028 | 26.93 |
| Qwen FP8, e4m3 head | +0.0092 | +0.0029 | +0.0051 | +0.0027 | 25.65 |
| old NVFP4, from FP8 | +0.0304 | +0.0220 | +0.0530 | +0.0286 | 15.01 |
| new NVFP4, from BF16 | +0.0234 | +0.0184 | +0.0272 | +0.0174 | 15.01 |
| new `mix-1gb` | +0.0144 | +0.0145 | +0.0211 | +0.0174 | 16.00 |
| old balanced, from FP8 | +0.0239 | +0.0144 | +0.0372 | +0.0197 | 18.17 |
| new balanced, from BF16 | +0.0136 | +0.0144 | +0.0245 | +0.0159 | 18.17 |
| new `mix-3.16gb` | +0.0090 | +0.0084 | +0.0185 | +0.0132 | 18.17 |

TandemLLM's own gate reads the first 2,048 tokens of each built-in text and allows +0.05 nats on the worse one. Against BF16 the new full set scores +0.0221 prose and +0.0426 code and passes. The old full set scores +0.0284 and +0.0507 and misses by 0.0007; against the FP8 release, which it was built from and gated on, it passed.

The new full set, the new balanced set and the old full set also wrote up to 900 tokens greedily on five prompts outside the calibration: a passage of prose, a formal letter, a SQL schema, a German explanation and a proof. All 15 generations stopped at their own end, after 224 to 849 tokens. The share of unique 4-grams was 0.939 to 1.000 for the new sets and 0.941 to 1.000 for the old one. The mixes have not written free text yet.

`quality/` holds the raw numbers: `gate.json` and `gate2.json` from the build job, `gate-wide.json` for the table above, `sens.json` for the per-piece costs and `report.json` for the per-projection output errors.

## Speed

The paper measured the full set (`mlp`, `gdn`, `attn` and the FP8 head) on one DGX Spark, one request at a time, on the benchmark row `serve-single-i256-o256-v1` (256 prompt tokens, 256 new tokens, 50 requests, greedy). With the persistent lookup store off, TandemLLM with StairCut reads 49.89 tok/s, against 13.69 tok/s for plain greedy decoding on the same weights. The other profiles have not been measured on these weights.

## How to use with TandemLLM

```bash
hf download Qwen/Qwen3.8-27B-FP8
hf download z-lab/Qwen3.8-27B-DFlash2
hf download 0xBakeer/TandemLLM-Qwen3.8-27B-NVFP4 --local-dir ~/tandem/nvfp4
git clone https://github.com/0xBakeer/TandemLLM && cd TandemLLM

python server/app.py --port 8000 --max-len 32768 --drafter dflash2 --dflash2-path greedy \
    --nvfp4 ~/tandem/nvfp4/mlp.safetensors,~/tandem/nvfp4/gdn.safetensors,~/tandem/nvfp4/attn.safetensors \
    --fp8-head ~/tandem/nvfp4/head-fp8.safetensors
```

That serves with the released base drafter. For the served profile with StairCut and the two fine-tuned drafters (`0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b8` and `-b16`), set `NV`, `HEAD`, `CKPT8` and `CKPT16` in `ops/serve.env` to the downloaded files and run `bash ops/start.sh`. The repository's README has the details.

## Limits

These numbers are loss and argmax agreement on four texts, not task accuracy. The per-piece costs do not add up: the 256 pieces sum to 3.4 times the cost of quantising all of them at once, so a mix is a ranking of pieces, and the gate, not the estimate, says what a mix costs.

## License and attribution

The weights are derived from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team (Alibaba Cloud), released under the Apache License 2.0, and they are distributed under the same license. `LICENSE` is Qwen's license file, unchanged. The change from Qwen's release is the one described above: the projections were quantised to NVFP4 and the output head to e4m3. Keep this notice and the license when you pass them on. The calibration used Wikipedia text (CC BY-SA 4.0) and the Python standard library (PSF license) only to compute statistics; none of that text is in this repository.

## Citation

```bibtex
@misc{bakeer2026staircut,
  title  = {StairCut: sizing speculative draft trees on a measured verification-cost staircase},
  author = {Bakeer, Khaled},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.23045275},
  url    = {https://doi.org/10.5281/zenodo.23045275}
}
```
