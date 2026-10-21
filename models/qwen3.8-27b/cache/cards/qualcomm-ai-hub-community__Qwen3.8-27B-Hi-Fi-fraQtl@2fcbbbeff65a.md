---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- gguf
- quantized
- calibration-aware
- long-context
---

# Qwen3.8-27B Hi-Fi (fraQtl) — calibration-aware Q4_K_M GGUF

**Model Conversion Contributor: fraQtl** — built by fraQtl.
Canonical repo + full receipts: https://huggingface.co/fraQtl/Qwen3.8-27B-Hi-Fi-GGUF

**Text-only GGUF** (the vision tower is not converted). Iso-size Q4_K_M-class
build of `Qwen/Qwen3.8-27B` — byte-matched to the leading community Q4_K_M
(within 0.001%) with fraQtl calibration-aware per-tensor quantization.

## Measured (all reproducible from the canonical repo's receipts pack)

- **Fidelity (KLD to original weights, lower = closer):** −45.3% on code/math
  (0.123 vs 0.225) and −18.0% on general (0.231 vs 0.282) vs the community
  canonical at identical size.
- **Long-context retrieval:** perfect on every cell of every run through the
  131,072-token stretch rows (all arms needle-verified; native context 262,144).
- **Task benchmarks — ties, stated as ties:** GSM8K 95.0% vs 94.0%,
  MATH-500 87.6% vs 88.2% — every pairwise difference inside the 95% CI.
- **Disclosed losses:** Wikitext-2 perplexity +0.7% worse than canonical;
  top-1 agreement on general −0.2pp. Stated here, not buried.

## Notes

- 16.5GB at ~4.8 bpw — Snapdragon X-Elite / AI-PC class rather than phone class.
- Community builders credited: comparison arms are unsloth's UD-Q4_K_M
  (canonical) and ggml-org's Q4_K_M — both well-made; ties are ties.
- Every number re-runnable: eval slices, teacher dumps, needle grid, and a
  single-file llama.cpp runner ship in the canonical repo's receipts pack.
- Artifact sha256: `43bd8f235bd7843b6bb06d632235b3e2430c23ad3f54b2fb1e56f35e74eebd8d` (byte-verified pre- and post-upload).
