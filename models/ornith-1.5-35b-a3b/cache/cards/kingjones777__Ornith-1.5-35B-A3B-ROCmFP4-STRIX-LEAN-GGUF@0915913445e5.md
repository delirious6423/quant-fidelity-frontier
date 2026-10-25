---
license: apache-2.0
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
  - ornith-ai/Ornith-1.5-35B-A3B-GGUF
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags:
  - mtp
  - speculative-decoding
  - gguf
  - rocmfp4
  - rocmfpx
  - strix-halo
  - gfx1151
  - amd
  - rocm
  - quantized
  - ornith
  - multimodal
  - lean
---

# Ornith-1.5-35B-A3B — LEAN — ROCmFP4 / ROCmFPX + MTP GGUF

## New file (2026-10-25): `Ornith-1.5-35B-A3B-Q4_0_ROCMFP4_LEAN-v1.gguf`

The file already in this repository is a STRIX_LEAN build whose embedded MTP block does not draft on Super ROC. This v1 LEAN file was built with the `qwen35moe-v1` protection (output and embedding at Q6_K; the ten `blk.40` draft weights, `eh_proj` and `ssm_alpha` / `ssm_beta` at Q8_0) over an agentic imatrix weighting. Its embedded MTP block drafts on Super ROC at depth 2.

**How it was built:**
- Tool: `gufo-quant`, recipe LEAN, weighted with the agentic imatrix.
- Protection `qwen35moe-v1` (15 rules): `output.weight` and `token_embd.weight` at Q6_K, ten exact `blk.40` draft weights at Q8_0, `eh_proj` and `ssm_alpha` / `ssm_beta` at Q8_0.
- The protect file is `tools/gufo-quant/protect/qwen35moe-v1.txt` in the Super ROC repository; the exact command line was not recorded.

**Measured on Super ROC** (AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.13.0; one request at a time, temperature 0, cold prompt cache):

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1773.7 tok/s |
| Generation after that prompt (up to 256 tokens) | 74.6 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 96.4 tok/s (92.6-101.3) |
| Drafted tokens accepted on those prompts | 933/1176 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

**Quality:** Perplexity 10.2407 on a 96-chunk long-form corpus at context 512 (1.01727 times the stock Unsloth UD-Q4_K_XL file, 95% interval 1.01073-1.02385), KL divergence 0.08350 and top-1 agreement 85.83 % against a Q8_0 reference (the stock file: 0.02707 and 91.79 %). Measured on an earlier Super ROC build.

**Run it with Super ROC** (https://github.com/kingjones30/superroc):
```sh
GPU_MAX_HW_QUEUES=2 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Ornith-1.5-35B-A3B-Q4_0_ROCMFP4_LEAN-v1.gguf --served-model-name ornith-1.5-35b --context 32768 --think off --speculative mtp -d 2 --min-draft-tokens 1
```


**The size-optimised 4-bit** of `ornith-ai/Ornith-1.5-35B-A3B` — ftype 106 `Q4_0_ROCMFP4_STRIX_LEAN`, built from Ornith's own BF16 GGUF and **with the output head explicitly protected at Q6_K**.

Built for **AMD Strix Halo (gfx1151)** — Ryzen AI MAX+ 395, 128 GB unified memory, ROCm 7.2.4 — using the [ROCmFPX](https://github.com/charlie12345/ROCmFPX) llama.cpp fork, which adds AMD-native FP4/FP8 tensor types that mainline llama.cpp does not have.

> ⚠️ These files require a **ROCmFPX-capable llama.cpp build**. They will **not** load in
> stock llama.cpp / Ollama / LM Studio — the `Q4_0_ROCMFP4_*` and `Q*_0_ROCMFPX*` tensor
> types are not in mainline.

## Variants in this repo

| file | ftype | size | BPW | `token_embd` | `output.weight` | decode | decode +MTP |
|---|---|---|---|---|---|---|---|
| `Ornith-1.5-35B-A3B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` | 106 | 17.88 GiB | 4.32 | q5_K | **q6_K** | **62.83 t/s** | 54.95 t/s (0.87×) |

**Which to pick:** LEAN is the **smallest** file here (17.88 GiB, 4.32 BPW) and — **served without a drafter** — also the **fastest** at **62.83 t/s**, edging FAST by 0.2%. It reaches that by taking Q5_K token embeddings while keeping the **output head at Q6_K**. If you serve with MTP, see the decision table: the ranking changes.

## Head protection — verified in the file, not assumed

`tie_word_embeddings` is **false** on this model, so `output.weight` is a real standalone tensor and `--output-tensor-type` genuinely bites. Every artifact here was re-opened after quantization and its header read back:

| ftype | `token_embd.weight` | `output.weight` |
|---|---|---|
| 102 | q6_K | **q6_K** |
| 114 | q8_0 | **q8_0** |
| 111 | q8_0 | **q8_0** |
| 115 | q8_0 | **q8_0** |

⚠️ For contrast, the other public ROCmFP4 build of this model (`julianmb/Ornith-1.5-35B-A3B-ROCmFP4-GGUF`, ftype 106) ships **Q5_K token embeddings and a 4-bit `output.weight`**, while its card states "FP16 embedding/norm preservation". We read both headers with two independent parsers. Head protection is not implied by an ftype name — it has to be requested and then verified in the file.

## Measured — not estimated

Hardware: **AMD Ryzen AI MAX+ 395 (Strix Halo, gfx1151)**, 128 GB unified, ROCm 7.2.4. Idle box, **2 warm-ups discarded**, **median of 5**, 300 tokens, identical prompt across every sample.

| quant | run 1 | run 2 | run 3 | run 4 | run 5 | median |
|---|---|---|---|---|---|---|
| 106 | 62.88 | 62.86 | 62.83 | 62.83 | 62.83 | **62.83** |

### Speculative decoding (MTP)

This model ships its **draft head inside the base weights** — `qwen35moe.nextn_predict_layers = 1`, tensors under `blk.40.nextn.*`. Enable it with `--spec-type draft-mtp --spec-draft-ngl 999` and **no `--model-draft`**.

⚠️ Do **not** pass `mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF` as a draft model. It is the same 753 tensors with 32 bytes of extra metadata — loading it as a drafter loads a **second full 35B**.

⛔ **MTP measured a net loss on every tier here** (0.89–0.94×). The flags are documented so you can turn it on; we are not selling it as faster. Best n-max on the 4-bit was **3** (56.05 t/s) — note that n=2 had *higher* acceptance (0.927 vs 0.913) and was **slower**, so rank on t/s, not acceptance.

## Source — byte-verified, not re-converted

Quantized from `ornith-ai/Ornith-1.5-35B-A3B-GGUF` → `Ornith-1.5-35B-BF16.gguf`, **71,066,994,240 bytes**, sha256 `a3ee48dd8f05d10f529aa8ca8b9e080082c38910909638d226001b74e3307591`. The vision projector `mmproj-Ornith-1.5-35B-BF16.gguf` (**902,822,016 bytes**, sha256 `d9ce31026d1cb1f3f8d5152e2e2a014d9d2b302b6c93a7dc07bb0a0487f52837`) is included.

Both were byte-verified against the Hub before quantization. **No re-conversion from safetensors.**

## Verification

Every artifact: loaded at `-ngl 999 -c 4096 -fit off -fa on`, **3/3 correctness** (17×23 → 391, capital of Japan → Tokyo, days in 2024 → 366) asserted on both `content` and `reasoning_content` with `finish_reason=stop`, and **4/4 vision** on a four-quadrant colour image via the mmproj.

File sizes were checked against `--dry-run` projections: the header delta is a **constant 10.48 MiB** across all artifacts (spread 0.005 MiB), which is the signature of complete, untruncated files.

⚠️ **Bandwidth note:** this is a **256-expert MoE with ~3B active**. Effective bandwidth must be computed against the *active* weight (~1.72 GB at 4.58 BPW), not the 20.3 GB file. At 60.34 t/s that is ~104 GB/s — the file size is not the bus.
