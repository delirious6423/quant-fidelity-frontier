---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - compressed-tensors
  - w4a16
  - int4
  - vllm
  - gemma4
  - moe
---

# gemma-4-26B-A4B-it-W4A16

Int4 weight-only quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it),
in compressed-tensors format for vLLM. **15.65 GB**, down from 51.61 GB — a 70% reduction,
and the whole model fits a 24 GB card at its full 256k context.

Unofficial and unaffiliated with Google. All model capabilities, evaluations and
limitations belong to the original model card — see the base model for those.

Unlike the 12B and 31B, Google publishes **no `qat-w4a16-ct` build at this size**. Its own
4-bit release for the 26B A4B is `qat-q4_0-gguf`, which is llama.cpp-only, and NVIDIA's
`NVFP4` build needs a Blackwell card. This repository fills that gap: the same weights in
the compressed-tensors format vLLM serves natively, on any GPU of compute capability 7.5
or above. If you specifically want QAT quality under vLLM, you would be relying on a
third-party conversion rather than a Google release.

## What was changed

Weights were quantized from bfloat16 to **int4, group size 64, symmetric, weight-only**
(activations stay 16-bit) using `llmcompressor.model_free_ptq`. No calibration data was
used and the model was never loaded — the quantizer operates directly on the safetensors.
Architecture, tokenizer, chat template and processor config are the vendor's, unmodified.

11,725 Linear modules were converted, covering 83.1% of the output's bytes:

| component | precision | source | quantized |
|---|---|---|---|
| MoE experts (128 per layer × 30 layers) | int4 g64 | 45.68 GB | **12.13 GB** |
| attention projections (115 modules) | int4 g64 | 2.22 GB | 0.59 GB |
| shared-expert MLP (90 modules) | int4 g64 | 1.07 GB | 0.28 GB |
| `embed_tokens` (tied to the output head) | bfloat16 | 1.48 GB | 1.48 GB |
| vision tower + `embed_vision` | bfloat16 | 1.15 GB | 1.15 GB |
| routers | bfloat16 | 0.02 GB | 0.02 GB |
| norms, layer scalars | bfloat16 | 0.001 GB | 0.001 GB |
| **total** | | **51.61 GB** | **15.65 GB** |

This is the largest reduction in the collection for a simple reason: 88.5% of the source
checkpoint is expert weight, and essentially all of it quantizes.

**Group size 64, not the usual 128.** The expert `down_proj` takes a 704-wide input and the
shared MLP's takes 2112; neither is divisible by 128. At the default group size those
layers cannot be quantized and the build collapses to a few percent of bytes. 704 and 2112
are both divisible by 64.

Left at bfloat16:

- **`vision_tower`, `embed_vision`** — the tower's `intermediate_size` is 4304, not
  divisible by 64, so int4 Marlin-style kernels cannot serve it; vLLM's Gemma 4 loader has
  an explicit guard for this case.
- **routers** — `router.proj` is built in vLLM as a `GateLinear` that takes no
  `quant_config` at all and emits fp32 logits, because the top-k kernel needs fp32 for
  stable routing. A quantized router would simply fail to load. It is 0.02 GB across all
  30 layers, so there is nothing to gain either.
- **`embed_tokens`** — precision-sensitive, and it *is* the output head here
  (`tie_word_embeddings: true`; no `lm_head` tensor exists).

This model has **no audio tower** (`audio_config: null`) — Gemma 4 ships audio only on
E2B, E4B and 12B.

### Checkpoint layout

The source stores experts fused as 3-D tensors (`experts.gate_up_proj` (128, 1408, 2816),
`experts.down_proj` (128, 2816, 704)). `model_free_ptq` splits them into per-expert 2-D
weights before quantizing, so this checkpoint ships **11,520 individually quantized expert
modules** (`…experts.{id}.gate_proj.weight_packed`, `up_proj`, `down_proj`) rather than
fused 3-D blocks. vLLM handles both layouts explicitly — its Gemma 4 loader carries a
dedicated path for "CompressedTensors-format AWQ/W4A16 `_packed`, `_scale`" expert names —
so no conversion is needed. It does mean the tensor count is high (35,923) and the index
file is correspondingly large.

## Usage

```bash
vllm serve GotoAI-Inc/gemma-4-26B-A4B-it-W4A16 \
  --max-model-len 131072 \
  --enable-auto-tool-choice --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

Do not pass `--quantization`; compressed-tensors is detected from `config.json`. The int4
W4A16 scheme uses Marlin kernels and runs on compute capability 7.5 and above.

**Version note.** Gemma 4 needs `transformers >= 5.10.1`, but vLLM ≤ 0.27.1 cannot serve
Gemma 4 with `transformers >= 5.15`: at 5.15 the config turns `global_head_dim` into a
per-layer `head_dim` override, and older vLLM reads `head_dim` globally, raising
`AmbiguousGlobalPerLayerAttributeError` at engine-config time. Use either
`transformers < 5.15` with vLLM 0.25.1–0.27, or **vLLM >= 0.28**, which handles both
layouts.

### Fitting the card

30 layers: 25 sliding-attention (window 1024, 8 KV heads, `head_dim` 256) and 5 global
(every 6th, 2 KV heads, `global_head_dim` 512). The sliding layers are bounded by the
window at ~0.2 GB per sequence no matter how long the context; only the 5 global layers
grow, at ~20 KB/token — unusually cheap:

| context | KV cache | + weights |
|---|---|---|
| 32k | ~0.9 GB | ~16.5 GB |
| 128k | ~2.8 GB | ~18.5 GB |
| 256k (max) | ~5.4 GB | ~21.1 GB |

That is what makes a 24 GB card viable at full context. `--language-model-only` frees the
1.15 GB tower (vLLM skips tower weights when every modality limit is zero) plus the
multimodal profiling headroom if you need more. Note that only ~3.8B of the 25.2B
parameters are active per token, so throughput is far better than the footprint suggests.
This is arithmetic from `config.json`, not a measured deployment.

Speculative decoding has a vendor drafter, `google/gemma-4-26B-A4B-it-assistant`
(`Gemma4AssistantForCausalLM`, 4 layers). vLLM normalizes it to its `gemma4_mtp` path,
which produces one draft token per forward:

```bash
--speculative-config '{"method": "mtp", "model": "google/gemma-4-26B-A4B-it-assistant", "num_speculative_tokens": 1}'
```

Not smoke-tested here.

### Thinking

The chat template defaults `enable_thinking` to **false**, so this model does *not* think
unless asked. Both knobs are template variables passed through `chat_template_kwargs`:

```json
{"chat_template_kwargs": {"enable_thinking": true}}      // injects <|think|> into the system turn
{"chat_template_kwargs": {"preserve_thinking": true}}    // keep thinking on tool-call turns
```

With thinking on, reasoning is emitted as `<|channel>thought … <channel|>`, which
`--reasoning-parser gemma4` splits into `reasoning_content`. Per the base model card,
thinking from earlier turns should *not* be replayed into history — except on tool-call
turns, which is what `preserve_thinking` keeps.

Sampling, per the base model card: `temperature=1.0, top_p=0.95, top_k=64`. Place image
content *before* the text in a prompt. The visual token budget is configurable
(70/140/280/560/1120, default 280) — lower it for video and captioning, raise it for OCR
and document parsing.

## Reproducing this checkpoint

Built with [llm-quantizer](https://github.com/gotoai/llm-quantizer):

```bash
./llmq.py run --profile gemma-4-26b-a4b-it
```

which re-shards the source — it ships as 2 shards, the larger 49.9 GB — into 15 pieces of
~4 GB, then:

```python
# llmcompressor==0.13.1a20260814, compressed-tensors==0.18.1a20260818,
# transformers==5.15.1, torch==2.13.0
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="gemma-4-26B-A4B-it-resharded",
    save_directory="gemma-4-26B-A4B-it-W4A16",
    scheme="W4A16",
    group_size=64,
    ignore=["re:.*vision.*", "re:.*audio.*", "re:.*router.*",
            "re:.*layernorm_\\d+$", "lm_head", "re:.*embed_tokens.*"],
    device="cuda:0",
)
```

A job holds one shard at a time, so after re-sharding the build peaks at a few GB of VRAM
rather than the model size — no large GPU required.

The `re:.*layernorm_\d+$` entry is not optional. compressed-tensors auto-skips norms with a
literal `module_name.endswith("norm")` test, which Gemma 4 MoE's suffixed norms miss —
`post_feedforward_layernorm_1`, `post_feedforward_layernorm_2` and
`pre_feedforward_layernorm_2`, 90 one-dimensional tensors in all. Without that pattern they
reach the quantizer and it aborts with `expected 2D linear weight`.

## Evaluation

**No benchmarks have been run.** Data-free round-to-nearest quantization degrades quality
more than a calibrated (GPTQ/AWQ) or QAT build; how much, for your task, is unmeasured
here. Treat published Gemma 4 26B A4B numbers as describing the bf16 model, not this one.

Two reasons to be more careful than usual with an MoE at 4 bits: the routers stay 16-bit
here, so expert *selection* is unchanged, but every expert's weights are quantized without
calibration, and rarely-activated experts get no more attention than hot ones. If you
measure anything, measure it on your own traffic.

## License

Apache 2.0, inherited from the base model — see `LICENSE` and Google's
[Gemma 4 license page](https://ai.google.dev/gemma/docs/gemma_4_license). The base
repository ships no LICENSE file, so the Apache-2.0 text is included here for
redistribution. "Gemma" is Google's mark; this repository is not endorsed by or
affiliated with Google.
