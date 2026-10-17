---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- awq
- autoround
- experimental
---

# Qwen3.8-27B selective AWQ A45 (experimental)

**Weight shards are not yet available here.** Hugging Face rejected the private model upload because this account has reached its private repository storage limit. The local shards passed integrity checks and remain stored locally. The report below is a preliminary coding check.

This is a selectively quantized AWQ/GEMM version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Selected matrices use 4-bit asymmetric groups of 128. The export retains 992 byte-exact BF16 tensors, including protected components and MTP weights. QuantTrio's recipe informed the tensor allocation; this is a separate AutoRound calibration, not a claim that the numerical weights are identical.

The chat template comes from [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), revision `855bffc49448e299789730ff92c9b8d834d6cc14`, Jinja SHA-256 `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`. The quick test covered `medium` and `xhigh` reasoning effort.

## Preliminary quality check

- **21 matched coding tasks**, each in medium and xhigh modes: **42** responses per model.
- Selective AWQ passed **40/42**; B0 passed **40/42**. Newly lost B0 coding passes: **0**.
- Both runs used the same frozen task bank, prompts, sampling, patched template and grader. Artifact receipts were checked for every included response.
- The earlier candidate attempt stopped after 333 saved records. The fresh B0 attempt stopped after 42 saved records when the proxy stream lacked a final usage or finish marker. This comparison uses only matching complete responses from those attempts. See `quick-coding-check.json`.

**This is a preliminary coding check, not a full quality validation.** It covers the first 21 coding tasks only. Tool, JSON, multilingual, full-bank coding, BF16 KL/fidelity, long-context, vision, MTP and energy checks remain open. No overall quality parity or improvement is claimed. The model is private while those checks remain pending.

Quantization itself finished in 7h11m within its 10-hour limit; all 8 weight shards and tensor integrity checks passed. The [private recipe repo](https://github.com/groxaxo/qwen38-selective-awq-v2) contains the implementation and evaluation protocol.
