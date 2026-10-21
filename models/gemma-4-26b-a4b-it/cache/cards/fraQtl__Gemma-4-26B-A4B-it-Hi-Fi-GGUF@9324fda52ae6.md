---
library_name: gguf
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
language:
  - en
tags:
  - gguf
  - q4_k_m
  - llama.cpp
  - imatrix
  - calibration-aware-quantization
  - moe
  - gemma-4
  - gemma-4-26b-a4b
  - quantized
  - fraqtl
inference: false
---

# Gemma-4-26B-A4B Q4_K_M GGUF — fraQtl Hi-Fi (calibration-aware quantization)

Gemma-4-26B-A4B Q4_K_M GGUF built with calibration-aware quantization — a split result, stated up front: **−9.2% code/math KLD and −9.3% general KLD** to the original bf16 model versus the pinned community Q4_K_M baseline, top-1 agreement ahead on both slices, **AND a GSM8K loss (59.5% vs 68.5%, outside the tie band)**, disclosed in full below. This artifact optimizes distributional fidelity to the original model at the same size class; on this model, that did not translate to GSM8K. Every number is measured, pinned, and reproducible from the provenance table.

## Model Overview
![Fidelity receipts chart](kld_chart.png)

### Description

This repository contains a GGUF quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (a mixture-of-experts model, ≈4B active parameters) produced with fraQtl's calibration-aware, MoE-aware per-tensor quantization: the same llama.cpp Q4_K machinery everyone uses, with per-tensor precision decisions (265 tensor-type overrides) driven by a calibration analysis instead of fixed defaults.

| Metric (vs original bf16 model) | fraQtl Hi-Fi | bartowski Q4_K_M (pinned) | Delta |
|---|---:|---:|---|
| KLD code/math ↓ (3-run, drift 0.0) | **0.5433** | 0.5984 | **−9.2%** |
| KLD general ↓ (3-run, drift 0.0) | **0.8484** | 0.9351 | **−9.3%** |
| top-1 agreement, code/math ↑ | **0.803** | 0.791 | ahead |
| top-1 agreement, general ↑ | **0.694** | 0.682 | ahead |
| GSM8K, 0-shot instructed chain-of-thought, n=200 ↑ | 59.5% (119/200) | **68.5%** (137/200) | **−9.0pp — LOSS** (tie band ±4.2pp) |
| File size | 17,117,713,344 B | 17,035,038,112 B | +0.49% (size-class match, not byte-identical) |

**Who should use which:** if your workload depends on GSM8K-style chain-of-thought math accuracy, use the community baseline — it measured better here. If you want the closest distributional match to the original bf16 model at this size class (sampling behavior, general chat/tool token agreement, distillation-style uses), this artifact measured ahead on every fidelity metric we ran.

Fidelity metric: symmetric top-20 KL divergence on Q8 teacher support, mean over positions; teacher is a Q8_0 quantization of the ORIGINAL bf16 checkpoint. Lower is better.

### License/Terms of Use

Apache 2.0 for the quantization artifacts in this repository. The model weights derive from `google/gemma-4-26B-A4B-it` and remain subject to Google's Gemma terms ([license link](https://ai.google.dev/gemma/docs/gemma_4_license)). Nothing here modifies the base model's own usage terms.

### Use Case

llama.cpp / llama-cpp-python inference of Gemma-4-26B-A4B-it at 4-bit class size where distributional fidelity to the original model matters. See the split result above before choosing this artifact for math-benchmark-critical workloads.

### Release Date

Hugging Face 07/2026 via this repository.

### References

- Companion on-device model: [fraQtl/Gemma-4-E2B-it-Hi-Fi-GGUF](https://huggingface.co/fraQtl/Gemma-4-E2B-it-Hi-Fi-GGUF)
- Serving lane (KV-cache compression for vLLM, same receipt discipline): [Qwen3-4B kit](https://huggingface.co/fraQtl/qwen3-4b-instruct-2507-kv-sidecars) · [Mistral-7B kit](https://huggingface.co/fraQtl/mistral-7b-instruct-v0.3-kv-sidecars)
- All fraQtl artifacts: [huggingface.co/fraQtl](https://huggingface.co/fraQtl)

## Model Architecture

**Base model:** `google/gemma-4-26B-A4B-it` — a mixture-of-experts instruction-tuned model (26B total, ≈4B active parameters per token). Converted from the pinned bf16 checkpoint with llama.cpp's `convert_hf_to_gguf.py --outtype bf16`, then quantized with `llama-quantize` using 265 per-tensor `--tensor-type` overrides selected by fraQtl's calibration-aware, MoE-aware analysis (recipe label: `restore_deep`, 384K-token calibration arm). How the per-tensor decisions are made is not published; the recipe hash, the imatrix, and every artifact hash are, so the result is verifiable end to end.

The GGUF's embedded chat template is the pinned Gemma jinja template (sha256 `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0`), patched post-quantization with tensor bytes asserted byte-identical.

## Input / Output

- **Input:** text (Gemma chat format)
- **Output:** text
- **Context length:** as the base model; calibration used ctx 2048.

## Calibration Dataset

- **Dataset:** fraQtl Apex code+math packet (sha256 `9bec84a28dcb0c940047e6084561a00857fd610b8a1e148cc38e27929e0a7e02`), decontaminated against the eval slices.
- **Data collection method:** Automated (curated public code/math text).
- **Properties:** importance matrix computed with `llama-imatrix` at ctx 2048 on the BF16 GGUF substrate; the shipped arm uses the 384K-token calibration budget (187 chunks), selected by a receipted three-budget sweep (128K/256K/384K — 384K measured best on code/math KLD: 0.7391 / 0.7274 / 0.7162 on the sweep recipe). The exact `imatrix.dat` ships in this repository (sha256 `b8260ada80053d2cd6a9d23d164e8b8fe51a7c71795c32fc5f7ac66449280051`).

## Usage

Receipt-validated stack: llama.cpp @ `78d2f524682d9fee790a6460c93d018dafeb5229` (2026-07-05) / `llama-cpp-python==0.3.33` (vendors that commit). Newer releases should work; the pin is what every number on this page was measured with.

```bash
pip install llama-cpp-python==0.3.33
hf download fraQtl/Gemma-4-26B-A4B-it-Hi-Fi-GGUF Gemma-4-26B-A4B-it-fraQtl-HiFi-Q4_K_M.gguf --local-dir .
```

```python
from llama_cpp import Llama

llm = Llama(model_path="Gemma-4-26B-A4B-it-fraQtl-HiFi-Q4_K_M.gguf", n_ctx=4096, seed=0)
out = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
    temperature=0.0,
)
print(out["choices"][0]["message"]["content"])
```

Or with llama.cpp directly:

```bash
llama-cli -m Gemma-4-26B-A4B-it-fraQtl-HiFi-Q4_K_M.gguf -cnv -p "You are a helpful assistant."
```

**Chat template / turn-token note:** the pinned Gemma chat template is embedded in the GGUF — use `create_chat_completion` (or `llama-cli -cnv`) and let the runtime apply it. Gemma turns are delimited by `<start_of_turn>user` / `<start_of_turn>model` and terminated by `<end_of_turn>`, and the assistant role is named `model`, not `assistant`. If you hand-roll prompts, make sure `<end_of_turn>` is in your stop set: stacks that only stop on `<eos>` will run past the turn boundary and generate spurious extra turns.

## Evaluation

All evaluation is fidelity to the ORIGINAL `gemma-4-26B-A4B-it` (the axis a user migrating from the bf16 model experiences), on an A100-80GB with seed 0, deterministic settings, 3 independent KLD runs (run-to-run drift 0.0). Eval slices: code/math 400-record held-out (sha256 `cce6860274ab715e3ab8493c9f817b1c29ff99c148f83a25de8944e027db1665`); general chat/tool/wikitext 400-record (sha256 `b10a79caf2c17cc10cd1edcae44d4655278baf5b09b6a867b4d3ade2f996b276`).

### Fidelity to the original model (KLD, lower is better)

| Arm | KLD code/math ↓ | KLD general ↓ | top-1 (code/math) | top-1 (general) |
|---|---:|---:|---:|---:|
| bartowski Q4_K_M (community, pinned) | 0.5984 (3-run) | 0.9351 (3-run) | 0.791 | 0.682 |
| unsloth qat UD-Q4_K_XL | 1.1830 (3-run) | 1.2261 (3-run) | 0.638 | 0.574 |
| google QAT q4_0 | 1.2357 (3-run) | 1.2791 (3-run) | 0.627 | 0.563 |
| **fraQtl Hi-Fi (this repo)** | **0.5433 (3-run, −9.2%)** | **0.8484 (3-run, −9.3%)** | **0.803** | **0.694** |

**QAT context (read before comparing QAT rows):** the QAT GGUFs descend from a **retrained** QAT checkpoint, not from the original `gemma-4-26B-A4B-it`. Their ≈1.2–1.3 KLD values in this table measure distance from the ORIGINAL model — the axis this table is about — not "QAT quality" on its own axis; against their own retrained parent they would score far better. Both statements are true simultaneously; they answer different questions.

### GSM8K — disclosed loss

| Metric | fraQtl Hi-Fi | bartowski Q4_K_M | Verdict |
|---|---:|---:|---|
| GSM8K, **0-shot instructed chain-of-thought**, n=200, seed 0, greedy | 59.5% (119/200) | 68.5% (137/200) | **LOSS** — diff −9.0pp, outside the ±4.2pp tie band |

Stated plainly: this artifact loses GSM8K to the community baseline, with statistical significance at this sample size. We do not offset the KLD wins against this; both results stand as measured. The protocol label is exact: 0-shot with a chain-of-thought instruction; no few-shot exemplars; n=200, not the full set. Wikitext PPL was not run on this artifact (lean scope).

### Methodological note: how this artifact was chosen

Three receipted iterations, each a single-variable causal experiment:

1. **Build + calibration-budget sweep:** one recipe, three imatrix budgets (128K/256K/384K tokens); 384K measured best; ship candidates quantized from pinned inputs.
2. **Recipe iteration (single variable):** a targeted test of whether 40 deep-layer precision demotions caused the GSM8K gap — reverting them (recipe `restore_deep`, this artifact) improved both KLD slices further but GSM8K did not recover. Finding, disclosed as a result in its own right: **a KLD improvement does not guarantee task-benchmark parity.**
3. **Calibration iteration (single variable):** same recipe, only the calibration data changed (code/math packet → mixed chat/math/code/wiki packet). KLD got worse against the code/math-calibrated arm and GSM8K dropped a further 9pp — proving calibration data is causally potent on task benchmarks, and that the shipped arm was the better of the two.

The shipped artifact is the best measured arm across those experiments, published with its losses intact. Receipt files are named in the provenance table.

### Reproducibility

Deterministic eval rig: 3 independent KLD runs are byte-identical (drift 0.0). Verdicts are by mean; no confidence-interval claim is made on KLD. GSM8K uses the tie-band discipline above (band 1.96·sqrt((p₁(1−p₁)+p₂(1−p₂))/n) = ±4.2pp at n=200).

## Provenance

| Field | Value |
|---|---|
| Base model | `google/gemma-4-26B-A4B-it` @ `5305c1e72ea29c01f31a81230d52b375ba88b409` |
| Base checkpoint sha256 | `model-00001-of-00002.safetensors` = `1127684971bbca40465435a5cad69d67ad603bf5e61c6dfd5561fae4a3bcfdb3`; `model-00002-of-00002.safetensors` = `aab47033e1e8a492ef8e581efae1cf36478d0433567e7729b3c1728bc8970db7` |
| BF16 GGUF substrate | sha256 `c9393491480ea3415dfbf3b26f10f27cf77b92c746cfd0f2b799ec04363731e3` (50.505 GB) |
| Conversion + quantization | llama.cpp @ `78d2f524682d9fee790a6460c93d018dafeb5229` (2026-07-05); `convert_hf_to_gguf.py --outtype bf16` then `llama-quantize` with 265 per-tensor `--tensor-type` overrides (type assert: ALL_MATCH) |
| Recipe | `restore_deep`, sha256 `ae2919b73c48af666b72cc7a256e7523ad280d0372f618828e3e3134c71559b4` |
| Calibration | fraQtl Apex code+math packet (sha256 `9bec84a28dcb0c940047e6084561a00857fd610b8a1e148cc38e27929e0a7e02`), `llama-imatrix` ctx 2048, BF16 substrate, 384K-token budget (187 chunks; selected by receipted 128K/256K/384K sweep) |
| imatrix (ships in this repo) | `imatrix.dat` sha256 `b8260ada80053d2cd6a9d23d164e8b8fe51a7c71795c32fc5f7ac66449280051` |
| GGUF artifact | `Gemma-4-26B-A4B-it-fraQtl-HiFi-Q4_K_M.gguf` — 17,117,713,344 B, sha256 `3dd83b024970aba6b483fee1e3b4a500991663496bac3963d2f79db57741ae79` |
| Chat template | pinned Gemma jinja, sha256 `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0` (patched post-quantization; tensor bytes asserted identical) |
| Eval runtime | `llama-cpp-python==0.3.33` (vendored llama.cpp `78d2f5246`), A100-80GB, seed 0, deterministic (3-run drift 0.0) |
| Fidelity metric | symmetric top-20 KLD on Q8 teacher support, mean over positions; teacher = `unsloth/gemma-4-26B-A4B-it-GGUF` Q8_0 @ `c462057f7ed65ccdb7f7e0778fae67894d425d92` (sha256 `b26c56ea4bf724b4efa0161bb4615e974847ec64450a4dd49da8712614a128c7`), quantized from the ORIGINAL bf16 |
| Eval slices | code/math 400-record held-out (sha256 `cce6860274ab715e3ab8493c9f817b1c29ff99c148f83a25de8944e027db1665`); general chat/tool/wikitext 400-record (sha256 `b10a79caf2c17cc10cd1edcae44d4655278baf5b09b6a867b4d3ade2f996b276`) |
| Baseline | `bartowski/google_gemma-4-26B-A4B-it-GGUF` Q4_K_M @ `fabed3e586120477355eea23b92644540a79ce2f` — 17,035,038,112 B, sha256 `e718536fe9b4bd505b07d44ded8f1595053a5d5407315bccf555ce592f33c140` |
| QAT arms (context rows) | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf` @ `21bfe2a8c89118c9a1a2aa242934fc4d1c0fff15` (lfs sha256 `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`); `unsloth/gemma-4-26B-A4B-it-qat-GGUF` UD-Q4_K_XL @ `c1f25db7cf31985b52caa1db777eb72d17ca1c7c` (lfs sha256 `dcf179a91153e3a7ece792e48ef872180d9d6ef9b7677f0a0bd3e83cfe624d5e`) |
| Receipts | build `gemma4_hifi_build_2026-07-17.json`; recipe iteration `gemma4_hifi_iterate_2026-07-20.json`; calibration iteration `gemma4_hifi_calibtest_2026-07-21.json`; probe `gemma4_probe_eval_2026-07-10.json` |
| Drift | 3-run KLD drift 0.0 (deterministic rig) |

## Model Limitations

- **GSM8K loss:** −9.0pp vs the community Q4_K_M baseline (59.5% vs 68.5%, outside the ±4.2pp band). If chain-of-thought math accuracy is your gating metric, the baseline measured better. This is the headline limitation and it is not hedged.
- **KLD-task gap:** our own experiments show KLD improvement does not guarantee task-benchmark parity on this model (see Methodological note). Distributional fidelity is what this artifact optimizes and wins; it is not a claim of across-the-board task superiority.
- **Size:** 17,117,713,344 B — +0.49% vs the baseline's 17,035,038,112 B. Same size class, not byte-identical.
- **Wikitext PPL:** not run on this artifact (lean scope); no claim either way.
- **Calibration domain:** the imatrix is code+math (the receipted better arm; a mixed-domain calibration arm measured worse on both axes). Calibration is not domain-neutral by construction.
- **Quantization is lossy** by construction; the evidence standard here is measured distributional fidelity with pinned receipts, not bit-identity.
- The base model's own limitations and biases are inherited unchanged.

## Files

| File | What | sha256 |
|---|---|---|
| `Gemma-4-26B-A4B-it-fraQtl-HiFi-Q4_K_M.gguf` | Quantized model (17.12 GB, recipe `restore_deep`, 384K-token Apex imatrix) | `3dd83b024970aba6b483fee1e3b4a500991663496bac3963d2f79db57741ae79` |
| `imatrix.dat` | Calibration importance matrix, 384K arm (reproducibility) | `b8260ada80053d2cd6a9d23d164e8b8fe51a7c71795c32fc5f7ac66449280051` |

## More from fraQtl

Weight quantization is one lane — the companion on-device build is [fraQtl/Gemma-4-E2B-it-Hi-Fi-GGUF](https://huggingface.co/fraQtl/Gemma-4-E2B-it-Hi-Fi-GGUF). The serving lane — KV-cache compression sidecars for vLLM with per-user retrieval verification — lives at [fraQtl/qwen3-4b-instruct-2507-kv-sidecars](https://huggingface.co/fraQtl/qwen3-4b-instruct-2507-kv-sidecars) and [fraQtl/mistral-7b-instruct-v0.3-kv-sidecars](https://huggingface.co/fraQtl/mistral-7b-instruct-v0.3-kv-sidecars). Same standard per artifact: pinned provenance, measured numbers, results reported in both directions. Org page: [huggingface.co/fraQtl](https://huggingface.co/fraQtl).

## The fraQtl ladder

| Tier | Job | Example |
|---|---|---|
| **Hi-Fi** | maximum fidelity at standard 4-bit size | this repo's Q4_K_M |
| **Hi-Fi Phone** | fits phone/laptop RAM, fidelity-first | [E2B Phone, 2.86 GB](https://huggingface.co/fraQtl/Gemma-4-E2B-it-Hi-Fi-GGUF) |
| **SmartEdge** | smallest footprint that survives | [E2B IQ3XXS, 2.45 GB](https://huggingface.co/fraQtl/Gemma-4-E2B-it-Hi-Fi-GGUF) |

Same discipline at every tier: pinned provenance, measured numbers, losses disclosed.
