---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - tenstorrent
  - blackhole
  - p150
  - strata
  - moe
  - bfp4
  - gptq
---

# Qwen3.8-Flash-Next Coder for the Tenstorrent P150 (Strata `tt` backend)

Block-float weights that let [Strata](https://github.com/Niko1221/Strata)'s `tt` backend run Qwen3.8-Flash-Next
Coder on one Tenstorrent Blackhole P150 card:

- the routed experts of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  (IQ1_M), 48 layers x 256 experts x gate/up/down, re-quantized with GPTQ to Tenstorrent's BFP4 (`bfp4_b`) format,
  in the exact byte layout the P150 kernels read;
- the precision/placement plan `bfp4-tiered-2.4e+10`: which experts live in the card's DRAM and which in pinned host
  memory;
- the MTP draft layer's 512 routed experts in BFP4 and their routing profile (speculative decoding).

**This is not a standalone model.** It holds no tokenizer, attention, embedding or LM-head weights: those come from
the GGUF that Strata's setup downloads. The files are only useful to the Strata `tt` backend (the `fnx` engine in
Strata's `tt/` folder) and are fetched by it at a pinned commit (`tt/artifacts.json`, `tt/tools/tt_artifacts.py fetch`).

## Files (340 files, 59.8 GB)

Paths are relative to the backend's data folder (`FNX_DATA`).

| path | what |
|---|---|
| `quant/store/pool/L{00..47}.b4.npy` | GPTQ-BFP4 rows of every routed expert of the layer: uint8 [768, 921600], one row per (expert, matrix), W^T [in, out] with one shared exponent per 16 output channels (48 files, ~708 MB each) |
| `quant/store/pool/L{00..47}.b4idx.npy` | int32 [768, 2]: (expert, matrix) of each row; matrix 0 gate, 1 up, 2 down |
| `quant/store/plans/bfp4-tiered-2.4e+10/` | the plan: per layer `fmt` (all 4 = BFP4), `row` (pool row of each matrix), `res` (device-resident or host), `plan.json` |
| `mtp/experts-bfp4-search.u8` | the MTP draft layer's 512 routed experts: uint8 [512, 3, 921600] BFP4 rows |
| `mtp/expert-profile.npy` | int64 [512]: how often the drafter routes to each expert (the 254 hottest stay in device DRAM) |
| `quant-iq3s/residents/L{00..47}.npy` | **full model (IQ3_S)**: GPTQ-BFP4 values of the 8,823 routed experts the card holds (155-221 per layer): uint8 [3 n, 921600], row 3 i + m = matrix m (gate, up, down) of the layer's i-th resident expert, W [out, in] with one shared exponent per 16 input columns (the layout the card keeps them in) |
| `quant-iq3s/residents/L{00..47}.ids.npy`, `manifest.json` | int32 [n]: those experts (ascending); per-layer SHA-1 + settings |

## How it was made

- **Source**: the GGUF shards of ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF at revision
  `5348543e0147355ac9cbcb031184a3546350988e`, IQ1_M (the revision Strata pins), a quantization of
  [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) that keeps 256 of its 512 routed experts.
- **Calibration** (`fnx.quant.calib_data`): 128 sequences x 2048 tokens (+16 held out), chat 48 (UltraChat 200k
  train_sft), code 48 (Python, C/C++/CUDA, TypeScript/JavaScript, Rust, Go source trees), instruct 16
  (databricks-dolly-15k), docs 16 (Markdown documentation); disjoint from the quality-gate evaluation set.
- **GPTQ-BFP4** (`fnx.quant.pipeline quantize --weighted 0 --n-min 8192`): a float32 PyTorch forward of the GGUF
  (dequantized weights) over the calibration set, layer by layer on one RTX 3090 (62 minutes for 48 layers); per
  expert input Hessians of its routed rows (gate/up share one; a layer-mean prior up to 8192 rows for rarely routed
  experts), then GPTQ with act-order, 1% damping and a min-MSE shared exponent per 16-value block (clips -1..2),
  emitted as bit-exact tt-metal `bfp4_b` blocks.
- **Plan** (`fnx.quant.alloc --tiered 2.4e10`): every expert BFP4; the 8680 most-routed experts (calibration
  routing counts, 85.9% of all routings, 24.0 GB) are resident in the P150's DRAM, the other
  3608 are read from pinned host memory over PCIe by the MoE kernels.
- **MTP drafter**: the MTP layer of Qwen/Qwen3.8-Flash-Next as Strata packs it (Q2_0 experts), converted to BFP4 with
  a per-block min-MSE exponent (`fnx.quant.methods.search`); the routing profile comes from draft rounds on
  calibration text on the P150 (`fnx/tests/test_mtp.py profile`).
- **Full model (IQ3_S) card residents** (`fnx.quant.residents`): the same calibration set through a float32 forward
  of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  IQ3_S (revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, all 512 experts), Hessians of the 8,823 most-routed
  experts only (the ones the card holds; the rest run on the CPU from the GGUF as they are), the same prior and
  damping; GPTQ with the block exponent fixed at each 16-column block's first column and whole blocks in act-order
  (`fnx.quant.methods.gptq_rows`); 30 minutes on one RTX 3090.  Quality gate on the P150 (cpu MoE with these
  residents): KL .0279 (.0437 with min-MSE BFP4 of the GGUF values), top-1 .942 (.925), perplexity -0.66%
  (-2.21%), GSM8K-50 45/50 (45/50); same speed.

`tt/tools/tt_artifacts.py regenerate` in Strata reruns these steps (CUDA GPU with >= 16 GB).

## Quality and speed (P150, Strata `tt` backend, default configuration)

Quality gate against the published GGUF evaluated in float32 (teacher-forced chat/code/doc records + GSM8K), with
these BFP4 experts on the device (gated 2026-10-07 on the 22.0 GB placement of the same rows; the placement does not
change the numerics):

| metric | P150 |
|---|---|
| perplexity change, all records | +0.09% (chat +0.37%, code -0.18%) |
| top-1 agreement | 0.937 |
| KL divergence | 0.038 |
| GSM8K, first 50 | 45/50 (Strata's CUDA engine on an RTX 3090: 44/50) |

Decode speed on one P150, traced, chat / code / doc prompts: 30.9 / 29.2 / 26.6 tok/s plain, 46.9 / 50.9 / 50.0
tok/s with MTP speculative decoding (2026-10-08). Context up to 32,768 tokens; text only.

## License

These weights are a derivative of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and are
distributed under the **Qwen Community License 1.0** (the [LICENSE](LICENSE) file here, copied unchanged from the
base model repository). Its condition 1 requires that copyright and permission notice in all copies. Its condition 2
applies to every downstream user: a licensee (or affiliate) that runs a *Model as a Service* or *AI Work Assistant*
business must obtain a separate license from Qwen before using the weights or derivatives for any commercial purpose
(internal use that exposes nothing to third parties excepted).

The routed experts were re-quantized from ISTA-DASLab's
[Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
(GSQ-RCO expert pruning and quantization by IST Austria's DASLab, published under Apache-2.0); credit for the
expert selection and the IQ1_M quantization goes to them. The MTP drafter weights come from the Qwen base checkpoint.
