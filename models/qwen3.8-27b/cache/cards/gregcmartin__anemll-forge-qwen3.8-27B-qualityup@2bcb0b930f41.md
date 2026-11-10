---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: text-generation
library_name: anemll-forge
tags:
  - coreai
  - ane
  - quantized
  - speculative-decoding
  - dflash2
  - kv-cache
---

# Qwen3.8-27B qualityup for the M6 Neural Engine

**A community build made with [anemll-forge-qualityup](https://github.com/gregcmartin/anemll-forge-qualityup), a copy of [ANEMLL Forge](https://github.com/Anemll/anemll-forge). It is not an official ANEMLL release.**

These weights follow ANEMLL release 0.2's recipe: GPTQ, then quantization-aware training (QAT) against BF16 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). There is one change. The MLP matrices of layers 24–63 use **4-bit lookup tables** instead of 3-bit vector tables. Those layers hold most of the MLP quantization error.

The packages use release 0.2.1's graph and run on the **M6 only, with macOS 27**:
- INT8 key/value cache (`kv8`);
- 8-bit attention, with FP8 softmax and PV on M6;
- a transposed key cache.

Context entries are **8K, 16K and 32K**. The included drafter is release 0.2.1's body-only DFlash2 package, which drafts through this target's own head.

## Measured on an M6 Mac mini (32 GB), 9 November 2026

Both models ran on the same machine with the same prompts and the same drafter. Release 0.2 is ANEMLL's published package, with its V8 cache.

**Quality**

Metric: KL-512 against BF16 on 64 prompts (40,090 positions), at the 8K entry. The bootstrap resamples the 64 prompts 20,000 times.

| | qualityup | release 0.2 | qualityup − release 0.2, paired 95% CI |
| --- | ---: | ---: | --- |
| Mean KL | **0.0500** | 0.0515 | −0.0044 to +0.0013 (not significant) |
| Median KL | 0.0093 | 0.0093 | |
| p99 KL | **0.445** | 0.513 | |
| Top-1 agreement | **91.75%** | 91.42% | +0.06 to +0.60 points (significant) |
| Perplexity (BF16: 2.149) | 2.141 | 2.144 | |

- The gain in top-1 agreement is real.
- The largest errors (p99) are 13% lower.
- The gain in mean KL is within noise.

On CUDA, QAT brought this export to a mean KL of 0.030. The ANE graph's own approximations (tanh SiLU, INT8 attention, quantized cache) add error on top of the weights. On the M6 that extra error narrows the gap between the two weight sets.

**Speed**

Method:
- 3 prompts each of prose and code at about 2K and 31.5K tokens;
- greedy decoding, 256 output tokens, thinking off;
- no prompt-cache reuse, 32K cap.

| | qualityup | release 0.2 |
| --- | ---: | ---: |
| Prefill, 2K prompts | 291 tok/s | 300 tok/s |
| Prefill, 31.5K prompts | 269 tok/s | 280 tok/s |
| Verify call (1 anchor + 7 drafts), 2K context | 114.5 ms | 106.0 ms |
| Verify call, 31.5K context | 125.0 ms | 119.8 ms |
| Decode, 2K (tokens per verify) | 27.8 tok/s (3.18) | 30.9 tok/s (3.27) |
| Decode, 31.5K (tokens per verify) | 24.7 tok/s (3.09) | 24.6 tok/s (2.95) |

Decode speed depends on how many drafts are accepted, and that varies with the text each model generates. The verify-call time is the like-for-like cost: about 8% slower at 2K and 4% slower at 31.5K.

**Memory**

Measured at the 32K cap. Total = wired memory above the 2.2 GB no-model baseline, plus the server process footprint.

| | qualityup (kv8 cache) | release 0.2 (V8 cache) |
| --- | ---: | ---: |
| Wired after load | 18.6 GB | 16.0 GB |
| Peak wired, 31.5K prompts | 21.3 GB | 22.5 GB |
| Peak server footprint | 2.5 GB | 3.1 GB |
| **Total at peak** | **21.6 GB** | 23.4 GB |

Release 0.2 used more memory at long context because its V8 cache keeps keys in FP16. Release 0.2.1 has release 0.2's weights with the kv8 cache, and its notes report 19.3 GB with this method. By that comparison, qualityup costs about 2.3 GB more, though the two were not measured side by side. Swap did not grow during the runs.

## Files

- [coreai/](coreai): 16 target chunks, the output head (four-bit, trained during QAT) and the manifest. Weight formats:
  - MLP layers 0–23: three-bit vector tables;
  - MLP layers 24–63: four-bit tables;
  - token mixers: release 0.2's formats, with rank-64 low-rank factors;
  - online Hadamard rotations on the MLP and token-mixer inputs.

  [MODIFICATIONS.md](MODIFICATIONS.md) has the details.
- [model/](model): tokenizer and configuration files, and the FP16 host embedding table. These are identical to release 0.2's.
- [drafter/all3/](drafter/all3): release 0.2.1's body-only DFlash2 drafter, unchanged.
- [export/qualityup_full/](export/qualityup_full): the quantized export the packages were converted from (20 GiB, 129 safetensors files). It has per-layer lookup tables, indices, scales, rotation seeds and low-rank factors, plus the four-bit head. You only need it to rebuild the packages, not to run the model.
- [release.json](release.json): SHA-256 and size of every file, and the target/drafter pairing.
- [config.json](config.json): a copy of `model/config.json` for Hub discovery.

## Use

You need an M6 Mac with macOS 27 and about 22 GB of memory free. Use the qualityup source at commit `1134662` or later; upstream ANEMLL Forge does not accept this bundle's body-only drafter entry. To set up:

1. Set up the inference environment and build the Swift bridge as described in the [README](https://github.com/gregcmartin/anemll-forge-qualityup#readme).
2. Run:

```sh
git clone https://github.com/gregcmartin/anemll-forge-qualityup.git && cd anemll-forge-qualityup

python forge.py download --repo gregcmartin/anemll-forge-qwen3.8-27B-qualityup \
  --revision FULL_COMMIT_HASH --runtime coreai --output ~/Models/qwen38-qualityup

python forge.py serve --runtime coreai --model ~/Models/qwen38-qualityup/model \
  --build ~/Models/qwen38-qualityup/coreai --ctx 32768
```

- `download` checks every file against `release.json`. Add `--include-export` to also fetch and check `export/` (20 GiB).
- `serve` picks up the drafter in `drafter/all3/` automatically.
- The first start compiles the packages for the ANE: about 22 s per chunk, or 6–7 minutes in all on M6.

**Rebuild the packages from the export.** Use a conversion environment (`requirements-conversion.txt`) and release 0.2's numerics with the M6-only 8-bit attention. `MODEL` needs `config.json` and the checkpoint's small per-layer tensors (norms, conv1d, A_log, dt_bias, in_proj_a/b, q/k norms) plus the final norm; the export supplies the quantized matrices and head. Steps: [pipelines/qualityup/README.md](https://github.com/gregcmartin/anemll-forge-qualityup/blob/main/pipelines/qualityup/README.md#4-convert-compile-and-gate-on-the-m6-macos-27).

```sh
MODEL=<checkpoint dir> EXPORT_DIR=<bundle>/export/qualityup_full OUT=<builds> \
SILU=tanh MLP_SILU=tanh GDN_SQ=16 GDN_SV=64 MLP_DS=1 QCONV_INT8=0 \
ATT_S8_UNIT=0.25 ATT_S8B_UNIT=0.25 ATT_PF8_UNIT=0.015625 ATT_INT8MM=s8,s8b,sm8,pvf8 KV_KEYS_T=1 \
  python coreai/qwen38_coreai_build.py all --kv-cache-dtype kv8 --ctx 8192,16384,32768 --pctx 8192,16384,32768
```

## Limitations

- **Chip and OS:** M6 only, macOS 27. The packages have no M5 function set.
- **Context:** contexts above 32K are not included.
- **Quality evidence:** quality is measured as KL and top-1 agreement against BF16 on 64 prompts. It is not a task benchmark, and long-context quality is not established.
- **Out of scope:** vision and MTP are not part of this text-generation bundle.

## Training

The weights were trained on 1× RTX PRO 6000 (96 GB) in about 6 hours of pipeline time, for $12.85. The data and steps are in [pipelines/qualityup](https://github.com/gregcmartin/anemll-forge-qualityup/tree/main/pipelines/qualityup).

## Attribution

The target is Qwen3.8-27B by the **Qwen Team, Alibaba Cloud** (Copyright 2026 Alibaba Cloud, Apache-2.0, license in [LICENSE](LICENSE)).

The drafter derives from [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2), which credits the z-lab DFlash2 donor and the Prism Bonsai training target. Its license and notice are in `drafter/all3/`.

The conversion tooling, recipe and drafter package are from ANEMLL Forge (MIT). See [NOTICE](NOTICE). No affiliation with or endorsement by Qwen, Alibaba Cloud, ProCreations, z-lab, Prism ML, ANEMLL or Apple is claimed.
