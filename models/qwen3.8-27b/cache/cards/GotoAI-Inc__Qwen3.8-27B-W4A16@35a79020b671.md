---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - compressed-tensors
  - w4a16
  - int4
  - vllm
  - qwen3
---

# Qwen3.8-27B-W4A16

Int4 weight-only quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
in compressed-tensors format for vLLM. **19.42 GB**, down from 55.56 GB — it fits a 24 GB
card with room for a useful context window.

Unofficial and unaffiliated with Alibaba/Qwen. All model capabilities, evaluations and
limitations belong to the original model card — see the base model for those.

## What was changed

Weights were quantized from bfloat16 to **int4, group size 128, symmetric, weight-only**
(activations stay 16-bit) using `llmcompressor.model_free_ptq`. No calibration data was
used and the model was never loaded — the quantizer operates directly on the safetensors.
Architecture, tokenizer, chat template and processor configs are the vendor's, unmodified.

| component | precision | size |
|---|---|---|
| language-model linears (64 layers) | int4 g128 | 12.56 GB (64.7%) |
| `embed_tokens` + `lm_head` (untied) | bfloat16 | 5.09 GB (26.2%) |
| vision tower (`model.visual`) | bfloat16 | 0.92 GB (4.7%) |
| MTP speculator head (`mtp.*`) | bfloat16 | 0.85 GB (4.4%) |
| conv1d kernels, norms, biases | bfloat16 | 0.01 GB |
| **total** | | **19.42 GB** |

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
vllm serve GotoAI-Inc/Qwen3.8-27B-W4A16 \
  --max-model-len 65536 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

Do not pass `--quantization`; compressed-tensors is detected from `config.json`. The int4
W4A16 scheme uses Marlin kernels and runs on compute capability 7.5 and above.

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

which is equivalent to:

```python
# llmcompressor==0.13.1a20260814, compressed-tensors==0.18.1a20260818,
# transformers==5.15.1, torch==2.13.0
from llmcompressor import model_free_ptq

model_free_ptq(
    model_stub="Qwen/Qwen3.8-27B",
    save_directory="Qwen3.8-27B-W4A16",
    scheme="W4A16",
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
here. Treat the published Qwen3.8 numbers as describing the bfloat16 model, not this one.

For an agentic model the informative checks are well-formed `reasoning_content` and clean
multi-step tool calls rather than perplexity: structured emission degrades before fluency
does.

## License

Apache 2.0, inherited from the base model — the vendor's `LICENSE` is included unmodified.
"Qwen" is Alibaba's mark; this repository is not endorsed by or affiliated with Alibaba.
