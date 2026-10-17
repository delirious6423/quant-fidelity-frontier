---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- prism-ml/Ternary-Bonsai-2-27B-gguf
library_name: transformers
pipeline_tag: text-generation
tags:
- quantized
- w4a16
- gptq
- compressed-tensors
- speculative-decoding
- dflash2
- bonsai
- ternary
---

# Bonsai-2 27B — W4A16 (community conversion)

W4A16 GPTQ conversion of [Prism ML's Bonsai-2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
(the 1.72-bpw ternary retrain of Qwen3.8-27B) into a vLLM-loadable checkpoint,
so the weights can run the full vLLM feature set — including native DFlash2
speculative decoding in vLLM ≥ 0.28.0 — outside the llama.cpp-only ternary
format. **Community conversion, not an official Prism ML release.**

| | |
|---|---|
| Parameters | 27.36B (24.35B language + 2.54B embed/lm_head + 0.46B vision + MTP module) |
| Checkpoint | 19.5 GB (W4A16 pack-quantized language model; int8 lm_head/embed/MTP; visual tower + MTP in bf16) |
| Context | 262,144 tokens |
| License | Apache 2.0 |

## Provenance

The source is the `F16.gguf` from
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
@ `6ed5e12bf84b7a63069882c91dd9e9218647d17b`, whose weights are stored in a
Hadamard-rotated, v-grouped, delta-normed layout for llama.cpp's custom
kernels. Conversion inverts that layout deterministically:

1. **Hadamard unrotation** — `W_hf = W_stored @ blockdiag(H_sylv·diag(s_b)/√1024)`
   per 1024-block of the input dim; signs from the file's own
   `prism.hadamard.*` manifest.
2. **GDN v-layout reorder** — the 48 v-heads move from rep-major (3×16) to
   group-major (16×3) in `attn_qkv` (v segment), `attn_gate`, `ssm_alpha`,
   `ssm_beta`, `ssm_a`, `ssm_dt`, `ssm_conv1d` (v part) and `ssm_out`
   (columns), mirroring the Bonsai-demo `runtime.py` `reorder()`.
3. **Delta norms** — `attn_norm`, `post_attention_norm`, `output_norm`,
   `attn_q_norm`, `attn_k_norm` are stored as `g − 1` in the GGUF; restored
   to `g` (the runtime adds the 1 inside its fused RMSNorm+rotate kernel).
4. **Grafts** — `model.visual.*` and `mtp.*` are absent from the ternary
   GGUF and are copied verbatim (bf16) from Qwen/Qwen3.8-27B. The MTP module
   is what DFlash2 speculative decoding drafts with.

Verified: NLL 1.169 on a held-out sentence (base Qwen3.8-27B: 1.518 through
the same harness), coherent greedy generation, structural index match.

## Quantization

GPTQ W4A16 via llmcompressor 0.13 (pack-quantized compressed-tensors):
256 samples × 1024 tokens of open_platypus, symmetric group-128. Then the
syv-ai single-user prepare chain: int8 (group-128) lm_head, embed_tokens and
MTP module in place, plus the 40960-token draft head
(`mtp.draft_lm_head`, 213 MB) for DFlash2 speculative decoding. Per-layer
GPTQ round-trip relative errors 0.6–0.8%. The vision tower stays bf16.

## Serving

vLLM ≥ 0.28.0 (DFlash2 is native there; no patch needed for the drafter):

```bash
vllm serve /path/to/Bonsai-2-27B-W4A16 \
  --speculative-config '{"method":"dflash2","method_config":{"draft_tokens":7}}'
```

or the syv-ai single-user launcher (`SPEC=dflash2 CTX=fast`), which also
enables lookup-augmented drafting.

Measured on 1× NVIDIA CMP 170HX (SM80, 180 W): **155.7 tok/s** single-stream
decode (256-token cohort), **251.6 tok/s** on a 900-token cohort (adaptive
draft depth), prefill 1876 tok/s @ ~6.6k prompt, DFlash2 acceptance 4.17
tokens/draft (59.6% per position) with the base-calibrated drafter at zero
recalibration. Full receipts, protocol and the ternary-llama.cpp comparison
lane: [PixelML/club-170hx PR #49](https://github.com/PixelML/club-170hx/pull/49)
and [#47](https://github.com/PixelML/club-170hx/pull/47).

## Notes and limitations

- Quality benchmarks are the upstream Bonsai-2 numbers (84.78 thinking-mode
  average, community-reported) plus GPTQ error on top; no independent eval
  suite was run for this conversion.
- The checkpoint keeps Qwen3.8-27B's vision tower; the ternary retrain
  covers the language model only.
- `mtp.*` and `model.visual.*` are required at serve time for DFlash2 and
  image input respectively — do not strip them.
- If you serve the DFlash2 drafter
  ([syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)),
  note it was calibrated on the base model's hidden states; acceptance
  against this retrain measured 59.6% per position with no recalibration.
