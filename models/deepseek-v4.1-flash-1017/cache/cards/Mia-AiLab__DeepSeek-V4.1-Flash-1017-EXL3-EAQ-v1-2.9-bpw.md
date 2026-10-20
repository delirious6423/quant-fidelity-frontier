---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - eaq
  - quantized
  - moe
  - deepseek
  - multimodal
---

# DeepSeek-V4.1-Flash-1017 · EXL3 · EAQ v1 · 2.9 bpw

EAQ v1 (EXL3 format) quantization of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision `1bae0e10`, the refreshed V4.1-Flash weights), converted with exllamav3 `v1.6.1` plus the EAQ patch set (converter version 1.6.1) and a port of the DeepSeek-V4.1 architecture (see Conversion). Teacher-forced next-token fidelity to the original source weights was measured in a shared evaluation port; see **Quality vs the original model**.

This is **not** a drop-in replacement for the original FP8/FP4 checkpoint. The Engram embedding tables are **not** in this repo: you still need shards 47 and 48 from the base model (about 95 GiB each).

**Format: plain EXL3 (EAQ v1).** EAQ v1 currently uses the standard EXL3 format unchanged: `quant_method` is `exl3`, codebook `mul1`, `out_scales=always`, and the tensor keys and the main-model per-tensor bit widths are identical to the standard EXL3 2.9 bpw builds of this model (checked against [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw): 192,495 tensor keys; the widths of all 46,702 main-model quantized tensors match; tokenizer files byte-identical). The 46 MTP / DSpark dense tensors differ: 4 bit here, 6 bit in the LibertAI build. What EAQ v1 changes is only how the weights were produced (the EAQ conversion; see Conversion). Any runtime that loads EXL3 DeepSeek-V4.1 checkpoints should load this one without EAQ-specific code, including the TensorFold DeepSeek-V4.1-Flash-1017 2x DGX Spark recipe (set its `MODEL_ID` to this repo).

| | |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Format | EXL3 (`trellis` / `suh` / `svh` / `mul1`), codebook **mul1**, `out_scales=always` |
| Average bpw | **2.90** over the body (integer widths; routed experts 3 bit, layers 18-22 at 2 bit; attention 5, shared experts 4-5, indexer wk 8) |
| Head bits | 6 |
| MTP / DSpark bits | 4 |
| Size | 196 GiB (210.6 GB, `total_size` in the index) in 39 shards (the previous 2.9 bpw build of the V4.1-Flash base: 210.7 GB in 39 shards) |
| DSpark | In-checkpoint (`dspark_block_size=5`, 128 draft experts / top-3, layers 37-39) |
| Vision | Included (tower left native) |
| Engram tables | **Not quantized.** Use original shards **47** and **48** of the base model |

## Quality vs the original model

These are teacher-forced next-token fidelity measurements against the original `1bae0e10` source weights evaluated in the same custom exllamav3 DeepSeek-V4.1 port. Both sides receive the same token IDs. KL is measured in nats in the direction reference || quantized model; lower is closer to this reference. Top-1 agreement measures whether the two distributions have the same highest-probability token, not whether a response or tool call is correct.

**Short context: 100 evaluation conversations**, each at most 4,029 tokens, containing third-party text rendered with the DS chat template. This run was measured on an RTX PRO 6000. Results cover 89,880 scored assistant tokens after the assistant header through EOS, excluding the generation prompt. KL uses the full 129,280-token vocabulary. Means are token-weighted; brackets are 95% percentile intervals from 5,000 conversation-bootstrap resamples. The all-conversations interval resamples all 100 conversations; each stratum interval resamples within that stratum.

| Evaluation stratum | Conversations | KL (nats; 95% CI) | Top-1 agreement (95% CI) |
|---|---:|---|---|
| All | 100 | 0.0137 [0.0115, 0.0165] | 96.6% [96.2%, 96.9%] |
| Agentic | 14 | 0.0068 [0.0058, 0.0084] | 98.0% [97.6%, 98.3%] |
| Code | 13 | 0.0074 [0.0061, 0.0085] | 98.2% [97.4%, 98.7%] |
| Tool calling | 15 | 0.0113 [0.0076, 0.0160] | 97.3% [96.4%, 98.1%] |
| Chat | 10 | 0.0357 [0.0196, 0.0546] | 94.0% [92.1%, 95.7%] |
| Multilingual | 14 | 0.0306 [0.0239, 0.0368] | 95.5% [94.7%, 96.1%] |
| Thinking agentic | 6 | 0.0076 [0.0061, 0.0094] | 97.3% [96.8%, 97.8%] |
| Thinking code | 8 | 0.0075 [0.0070, 0.0083] | 97.1% [96.7%, 97.4%] |
| Thinking tool calling | 8 | 0.0186 [0.0152, 0.0217] | 95.2% [94.0%, 96.9%] |
| Thinking chat | 4 | 0.0178 [0.0166, 0.0196] | 95.2% [94.8%, 96.1%] |
| Thinking STEM | 8 | 0.0134 [0.0117, 0.0152] | 95.9% [95.3%, 96.4%] |

Thinking strata are separate from the other named strata; all are included in the pooled row. Stratum estimates have small sample sizes, and the four-conversation bootstrap is only indicative. On the short-context scored assistant tokens, the reference's top-1 token appears in EAQ v1 2.9 bpw's top-5 on 99.98% of tokens.

![KL to the original by workload, EAQ v1 2.9 bpw vs LibertAIDAI](images/kl_by_workload.png)

**Long agentic sessions: eight synthetic sessions** rendered with the DS chat template, containing real tool calls and results: four sessions at 8.0–8.8k tokens and four at 22.9–31.2k, with 89 assistant turns in total. Thinking was off, and every assistant turn is a scripted tool call. These were scored with teacher forcing on the same RTX PRO 6000, with the same assistant-token scoring as the short-context set, not evaluated as autonomous generated rollouts. Each length group has only four sessions; turns and tokens are not independent session replicates.

| Scored subset | Sessions | KL (nats) | Top-1 agreement |
|---|---:|---:|---:|
| All scored tokens, 8.0–8.8k sessions | 4 | 0.0060 | 99.1% |
| All scored tokens, 22.9–31.2k sessions | 4 | 0.0051 | 98.8% |
| Tool-call tokens, 22.9–31.2k sessions | 4 | 0.0015 | 99.7% |

Tool-call rows measure next-token fidelity on the scored tool-call subset, not tool-call success. In these eight sessions, mean KL by token-position bin was 0.0059 (0–4k), 0.0070 (4–8k), 0.0066 (8–16k), and 0.0037 (16–31.2k). These aggregate bin means do not show a sustained increase in this sample; the later bins contain only the four longer sessions. They do not establish stability on other tasks or at longer contexts.

![KL to the original by token position in the long sessions](images/kl_by_position.png)

**Comparison with [LibertAIDAI 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), revision `0be63c33`.** The LibertAIDAI checkpoint was scored as a second student against the same teacher, on the same inputs, in the same evaluation port and hardware, with `REF_QUANT=1` on both sides.

![KL to the original model: EAQ v1 2.9 bpw vs LibertAIDAI](images/kl_overview.png)

Short context, the same 100 conversations (89,880 scored tokens; means are token-weighted; brackets are 95% percentile intervals from 5,000 paired conversation-bootstrap resamples):

| | EAQ v1 2.9 bpw | LibertAIDAI |
|---|---|---|
| KL (nats) | 0.01374 [0.01146, 0.01654] | 0.01545 [0.01318, 0.01821] |
| Top-1 agreement | 96.59% [96.21%, 96.94%] | 96.23% [95.84%, 96.60%] |
| Paired KL difference (EAQ v1 2.9 bpw minus LibertAIDAI) | -0.00171 [-0.00204, -0.00135] | |
| Paired top-1 difference (EAQ v1 2.9 bpw minus LibertAIDAI; percentage points) | +0.36 [+0.21, +0.51] | |

Long sessions, the same eight sessions as above:

| Scored subset | EAQ v1 2.9 bpw: KL / top-1 | LibertAIDAI: KL / top-1 |
|---|---|---|
| All scored tokens, 8.0–8.8k sessions | 0.0060 / 99.1% | 0.0095 / 98.7% |
| All scored tokens, 22.9–31.2k sessions | 0.0051 / 98.8% | 0.0069 / 98.7% |
| Tool-call tokens, 22.9–31.2k sessions | 0.0015 / 99.7% | 0.0025 / 99.5% |

The measured paired KL differences (EAQ v1 2.9 bpw minus LibertAIDAI) were -0.0035 nats in the shorter group and -0.0017 nats in the longer group. They are reported from the paired analysis rather than recomputed from rounded table entries. On both sets EAQ v1 2.9 bpw had lower observed KL to the shared teacher; on the long sessions the sample is small and synthetic. The differences are small (on the short-context set, 0.0017 nats of KL and 0.4 points of top-1), and they do not establish generally better task performance. The comparison is between two finished checkpoints, not a controlled ablation of conversion methods.

Scope and implementation limits: the teacher is the original source checkpoint run through this port, not the official reference implementation or TensorFold engine. Source FP8/FP4 weights are dequantized for FP16 matmuls; the reference's dynamic FP8 activation quantization is not modeled. KV/index rounding is enabled on both sides (`REF_QUANT=1`), but these rounding helpers have not been validated bit-for-bit against the reference kernels. Port comparisons against the fork cover only layers 0–3; parity of layers 4–39 with the reference implementation has not been established. Shared implementation defects can be hidden by agreement between teacher and students.

These fidelity tests do not measure free-generation behavior, downstream task accuracy, vision quality, or MTP/DSpark quality or speculative-decoding behavior. No context above the longest measured session, 31.2k tokens, was evaluated. The evaluation port fails above 32,768 tokens in layers 20–39; this is a limitation of the evaluation implementation, not a demonstrated context limit of the checkpoint.

## Engram tables (required)

EXL3 quantized only the Engram `wkv` linears. The FP8 row tables stay in the original DeepSeek tree:

- `model-00047-of-00048.safetensors` (~95 GiB)
- `model-00048-of-00048.safetensors` (~95 GiB)

Take them from **the same revision** (`1bae0e10`) of the base model, not from the earlier V4.1-Flash repo: the 1017 refresh changed the weights.

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --include "model-00047-of-00048.safetensors" \
  --include "model-00048-of-00048.safetensors" \
  --local-dir ./DeepSeek-V4.1-Flash-1017
```

Point the runtime at that directory (`engram_table_dir` in `config.json`). This repo ships `engram_table_dir` as `null` in `config.json`; set it to your local copy of the base model directory.

## Conversion (what was actually done)

- Source: the FP8 (dense, 32x32 blocks, ue8m0) + FP4 (experts) 1017 checkpoint, dequantized to FP16 on load (bit-exact against the DeepSeek reference decode on 11 sampled tensors; FP4 differs from the reference table only in the sign of zero).
- Converter: clean upstream exllamav3 v1.6.1 + EAQ patches, plus a port of `deepseek_v41` (attention with shared compressed KV, two-level indexer, Engram, shifted mHC pre-mix, DSpark drafter, vision) from the MiaAI-Lab fork. The port was checked against the fork's forward on real layer 0-3 weights (strict router-set match failed because the earlier fork rounds the router `gate.bias` to fp16; with that rounding emulated the port matches the fork: router set 96.8-99.6 %, MoE rel L2 7e-5 on identical-expert tokens; **fixed in this build**: the converter keeps the router `gate.bias` in fp32, bit-equal to the source, where the earlier fork rounded it to fp16).
- Allocation: the exllamav3 integer allocator at 2.9 bpw (integer widths, high-quality promotion), frozen as an explicit per-tensor recipe (sha256 `5e6a394f...`); per-tensor widths equal those of the earlier [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) build for all 46,702 comparable tensors. MTP uniformly 4 bit. The pack has 43 extra tensors versus the earlier build (`*.ffn.gate.bias_vl`, the vision router selection bias of the 1017 source); no tensor is missing.
- Smoke test (conversion machine, pack only): the 197 GB pack loads module by module with finite logits and greedy decoding of a sanity prompt answers correctly. This only shows the pack is not broken.
- Quality: see **Quality vs the original model** for source-weight fidelity and the limited comparison with LibertAIDAI's 1017 checkpoint. No quality comparison with the earlier V4.1-Flash base checkpoint is reported here.

## Download

```bash
hf download Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw \
  --local-dir ./DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw
```

## Runtime

Stacks that load EXL3 (`trellis` / `suh` / `svh` / `mul1`) with DeepSeek-V4.1 support:

- **TensorFold** with the DeepSeek-V4.1-Flash-1017 2x DGX Spark recipe (the same EXL3 path as other EXL3 2.9 bpw builds).
- **ExLlamaV3 / TabbyAPI** once the architecture is supported in your build.

Official sampling from the base card: temperature 1.0, top_p 0.95, context up to 1M tokens. Thinking defaults on. KV cache is the native V4.1 FP4 layout; do not force an unrelated `fp8` KV dtype.

## Base model

DeepSeek-V4.1-Flash is a multimodal MoE (552B backbone) with CED, CSA2, Engram memory and in-checkpoint DSpark. Architecture, evals and prompt format live on the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Scores there are for the unquantized model, not this build.

## Credits

- **[DeepSeek-AI](https://huggingface.co/deepseek-ai)**: the base model, [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).
- **[turboderp](https://github.com/turboderp-org)**: the EXL3 format and the [exllamav3](https://github.com/turboderp-org/exllamav3) converter this build is based on (MIT).
- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and contributors: the runtime.
- **[LibertAIDAI](https://huggingface.co/LibertAIDAI)**: authors of the EXL3 2.9 bpw comparison checkpoint; revision `0be63c33` was evaluated on the eight synthetic long sessions reported above.
- `chat_template.jinja` adapted from [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) (comment lines removed).

## License

MIT, inherited from the base model. Quantization adds no extra restrictions.

Quantized by [MiaAI Lab](https://x.com/MiaAI_lab). Format and tooling: [exllamav3](https://github.com/turboderp-org/exllamav3) (turboderp).
