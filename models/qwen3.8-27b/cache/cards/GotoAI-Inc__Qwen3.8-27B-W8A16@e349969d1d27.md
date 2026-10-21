---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - compressed-tensors
  - w8a16
  - int8
  - vllm
  - qwen3
---

# Qwen3.8-27B-W8A16

Int8 weight-only quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
in compressed-tensors format for vLLM. **31.59 GB**, down from 55.56 GB — it fits a 48 GB
card with room for a long context window, or a 40 GB card with a moderate one.

This is the fidelity-first build. Int8 round-to-nearest stays far closer to the bfloat16
weights than int4 does, at ~1.6x the footprint of the
[int4 W4A16 sibling](https://huggingface.co/GotoAI-Inc/Qwen3.8-27B-W4A16) (19.42 GB). If
you are targeting a 24 GB card, use that one; use this one when you have the VRAM and want
the least quality loss quantization can give without calibration.

Unofficial and unaffiliated with Alibaba/Qwen. All model capabilities, evaluations and
limitations belong to the original model card — see the base model for those.

## What was changed

Weights were quantized from bfloat16 to **int8, group size 128, symmetric, weight-only**
(activations stay 16-bit) using `llmcompressor.model_free_ptq`. No calibration data was
used and the model was never loaded — the quantizer operates directly on the safetensors.
Architecture, tokenizer, chat template and processor configs are the vendor's, unmodified.

496 Linear modules were converted, covering 78.3% of the checkpoint's bytes:

| component | precision | size |
|---|---|---|
| language-model linears (64 layers) | int8 g128 | 24.73 GB (78.3%) |
| `embed_tokens` + `lm_head` (untied) | bfloat16 | 5.09 GB (16.1%) |
| vision tower (`model.visual`) | bfloat16 | 0.92 GB (2.9%) |
| MTP speculator head (`mtp.*`) | bfloat16 | 0.85 GB (2.7%) |
| conv1d kernels, norms, biases | bfloat16 | 0.004 GB |
| **total** | | **31.59 GB** |

Four things are deliberately left at 16-bit:

- **`model.visual.*`** — vLLM builds multimodal towers with `quant_config=None`, so a
  checkpoint carrying quantized vision weights cannot be loaded.
- **`mtp.*`** — the built-in multi-token-prediction speculator head, loaded through vLLM's
  speculative-decoding path rather than the main stack.
- **`linear_attn.conv1d`** — 3-D causal-convolution kernels in the gated-delta-net blocks,
  shape `(10240, 1, 4)`. Not Linear layers, and quantizers reject them outright.
- **`lm_head` + `embed_tokens`** — precision-sensitive, and `lm_head` is untied here.

The linear-attention **projections** (`in_proj_*`, `out_proj`) *are* quantized; only the
convolution kernels beside them are excluded.

## Usage

Runs on released vLLM — the architecture has been supported since **0.25.1**, so no
nightly build is required:

```bash
vllm serve GotoAI-Inc/Qwen3.8-27B-W8A16 \
  --max-model-len 65536 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

Do not pass `--quantization`; compressed-tensors is detected from `config.json`. The int8
W8A16 scheme uses Marlin kernels and runs on compute capability 7.5 and above.

### Fitting the card

31.59 GB of weights leave the rest of the card for the KV cache. With 4 KV heads at
`head_dim` 256 and only 16 of the 64 layers using full attention, the KV cache costs
**~64 KB/token** — about 2 GB at 32k and 4 GB at 64k. On a 48 GB card that is comfortable
at 64k and beyond; on a 40 GB card, budget for roughly 32k, or add `--language-model-only`
(frees the ~0.92 GB vision tower plus its profiling headroom) for more. This is arithmetic
from `config.json`, not a measured deployment.

### Controlling reasoning depth

The chat template defaults to `reasoning_effort='xhigh'`, which produces long deliberation.
Both knobs below are template variables, passed through `chat_template_kwargs`:

```json
{"chat_template_kwargs": {"reasoning_effort": "low"}}   // xhigh (default) | medium | low
{"chat_template_kwargs": {"enable_thinking": false}}    // skip thinking entirely
```

Set a server-wide default with
`--default-chat-template-kwargs '{"reasoning_effort": "low"}'`; request-level values still
win. `preserve_thinking: false` drops earlier turns' thinking from history, which matters
for long multi-turn sessions.

### Context

262144 tokens natively. The base model card documents a YaRN recipe for 1M tokens via
`--hf-overrides` plus `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`; that is not configured here, and
RoPE scaling costs quality at short contexts, so enable it only if you need it.

## Reproducing this checkpoint

Built with [llm-quantizer](https://github.com/gotoai/llm-quantizer):

```bash
./llmq.py run --profile qwen3.8-27b
```

W8A16 is the profile's default scheme, so no `--scheme` flag is needed. The command is
equivalent to:

```python
# llmcompressor==0.13.1a20260814, compressed-tensors==0.18.1a20260818,
# transformers==5.15.1, torch==2.13.0
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="Qwen/Qwen3.8-27B",
    save_directory="Qwen3.8-27B-W8A16",
    scheme="W8A16",
    ignore=["re:.*visual.*", "re:.*mtp.*", "re:.*\\.conv1d$",
            "lm_head", "re:.*embed_tokens.*"],
    device="cuda:0",
)
```

The source ships as 18 shards of ~4 GB, and a job holds one shard at a time, so the build
peaks at a few GB of VRAM — no re-sharding needed and no large GPU required.

## Evaluation

**No benchmarks have been run.** Data-free round-to-nearest quantization degrades quality
more than a calibrated (GPTQ/AWQ) or QAT build; how much, for your task, is unmeasured
here. Int8 degrades far less than int4 — that is the reason this build exists — but "less"
is not "none". Treat the published Qwen3.8 numbers as describing the bfloat16 model, not
this one.

For an agentic model the informative checks are well-formed `reasoning_content` and clean
multi-step tool calls rather than perplexity: structured emission degrades before fluency
does.

## License

Apache 2.0, inherited from the base model — the vendor's `LICENSE` is included unmodified.
"Qwen" is Alibaba's mark; this repository is not endorsed by or affiliated with Alibaba.
