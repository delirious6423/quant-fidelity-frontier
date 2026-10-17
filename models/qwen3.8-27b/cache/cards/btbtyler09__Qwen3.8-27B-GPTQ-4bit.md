---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
license: apache-2.0
tags:
- qwen3.8
- dense
- gptq
- foem
- quantized
- 4-bit
- vllm
- vision
- multimodal
- mtp
---

# Qwen3.8-27B GPTQ 4-bit

GPTQ 4-bit quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B-parameter dense multimodal model. Uses **FOEM** (First-Order Error Matters, AAAI 2026) for outlier-aware error compensation, enabling pure 4-bit quantization without per-module FP16 exclusions — this checkpoint lands at **+0.81% perplexity** over BF16.

Includes the full vision encoder and MTP (Multi-Token Prediction) module for image understanding and speculative decoding support.

## Model Overview

- **Architecture**: Qwen3_5ForConditionalGeneration (multimodal: text + vision; dense sibling of `qwen3_5_moe`)
- **Total parameters**: ~27B
- **Layers**: 64 (48 linear-attention + 16 full-attention, repeating 3:1 pattern)
- **Hidden size**: 5120, intermediate size: 17408 (dense MLP — no MoE)
- **Context length**: 262,144 tokens
- **Vision encoder**: 27-block ViT, BF16 (333 tensors)
- **MTP module**: 1-layer speculative decoding head, BF16 (15 tensors)

## Quantization Details

All quantizable Linear modules in the text decoder are quantized to INT4 using GPTQ + FOEM. The vision encoder, MTP module, norms, embeddings, and LM head remain at BF16/FP16 for quality preservation.

| Component | Precision | Notes |
|-----------|-----------|-------|
| `mlp.{gate_proj, up_proj, down_proj}` | INT4 (GPTQ + FOEM) | All 64 layers |
| `self_attn.{q,k,v,o}_proj` | INT4 (GPTQ + FOEM) | 16 full-attention layers |
| `linear_attn.{in_proj_qkv, in_proj_z, out_proj}` | INT4 (GPTQ + FOEM) | 48 linear-attention layers (GatedDeltaNet) |
| `linear_attn.{in_proj_a, in_proj_b}` | FP16 | Tiny projections, kept at full precision |
| Vision encoder (`model.visual.*`) | BF16 | 333 tensors, full precision |
| MTP module (`mtp.*`) | BF16 | 15 tensors, full precision |
| Embeddings, LM head, norms | FP16/BF16 | Full precision |

**GPTQ + FOEM configuration:**
- **Bits**: 4
- **Group size**: 32
- **Symmetric**: Yes
- **desc_act**: No
- **true_sequential**: Yes
- **act_group_aware**: Yes
- **mse**: 2.0 (activation-weighted MSE for outlier handling)
- **FOEM**: alpha=0.25, beta=0.2 (FOEM + GPTAQ defaults)
- **Fallback**: RTN at 0.5% calibration-coverage threshold

### What FOEM does

FOEM (First-Order Error Matters; [Zheng et al., AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40123)) augments the standard GPTQ Hessian-based weight update with a first-order error compensation term. In pure-4-bit settings, this is the difference between a usable checkpoint and one that drifts on long-tail tokens — on the architecturally identical Qwen3.6-27B, FOEM cut perplexity degradation from +8.45% (vanilla GPTQ 4-bit) to +1.95%, and on this model the same recipe lands at +0.81%, while keeping the same 4-bit format and vLLM kernel compatibility.

### Calibration

- **Dataset**: Mixed — evol-codealpaca-v1 (code) + C4 (general English text)
- **Samples**: 256, binned uniformly across context lengths 256–2048 tokens
- **Quantizer**: [GPTQModel](https://github.com/modelcloud/gptqmodel) v6.0.3
- Note: this is general-purpose calibration. Calibrating on wikitext directly would yield lower wikitext perplexity but worse out-of-distribution performance; we optimized for the latter.

The exact quantization script is included in this repo as `quantize.py`.

## Model Size

| Version | Size | Compression |
|---------|------|-------------|
| BF16 (original)  | ~56 GB | — |
| **GPTQ 4-bit FOEM** | **21 GB** | **2.6×** |

The 21 GB total includes the BF16 vision encoder (~1.2 GB) and BF16 MTP head (~0.85 GB) which are kept at full precision. The quantized text decoder alone is ~19 GB.

## Perplexity

Evaluated on wikitext-2-raw-v1 (test set), seq_len=2048, stride=512:

| Model | Perplexity | Degradation |
|-------|------------|-------------|
| BF16 (original)              | 6.4457 | — |
| **GPTQ 4-bit FOEM** (this)   | **6.4982** | **+0.81%** |

## Usage

### vLLM (Recommended for Serving)

```bash
vllm serve btbtyler09/Qwen3.8-27B-GPTQ-4bit \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 262144 \
  --dtype float16 \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image": 2}'
```

| Parameter | Description |
|-----------|-------------|
| `--tensor-parallel-size 4` | Shard across 4 GPUs (adjust to your setup) |
| `--gpu-memory-utilization 0.95` | Use 95% of GPU VRAM for KV cache + weights |
| `--max-model-len 262144` | Full 256K context window support |
| `--dtype float16` | Run in FP16 (required for ROCm GPTQ kernels) |
| `--skip-mm-profiling` | Skip multimodal memory profiling at startup |
| `--limit-mm-per-prompt '{"image": 2}'` | Allow up to 2 images per request |

> **vLLM bug workaround (may apply)**: Up through at least vLLM 0.19.x, `Qwen3_5TextConfig` defines `ignore_keys_at_rope_validation` as a `list` instead of a `set`, causing a `TypeError` during config parsing. Apply this patch before serving if you hit the error:
>
> ```python
> python3 -c "
> for f in [
>     '/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5.py',
>     '/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5_moe.py',
> ]:
>     t = open(f).read()
>     t = t.replace(
>         'ignore_keys_at_rope_validation\"] = [\n            \"mrope_section\",\n            \"mrope_interleaved\",\n        ]',
>         'ignore_keys_at_rope_validation\"] = {\n            \"mrope_section\",\n            \"mrope_interleaved\",\n        }')
>     open(f,'w').write(t)
>     print('Patched', f)
> "
> ```

### Vision Example (via OpenAI API)

```python
import base64, requests

with open("image.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "btbtyler09/Qwen3.8-27B-GPTQ-4bit",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Describe what you see in this image."},
    ]}],
    "max_tokens": 1024,
})
print(response.json()["choices"][0]["message"]["content"])
```

### GPTQModel / transformers

Loads natively under GPTQModel v6.0.3+ via the upstream `Qwen3_5QModel` definition (which uses `AutoModelForImageTextToText` and the multimodal `model.language_model.layers.*` weight prefix). No checkpoint patching needed.

```python
from gptqmodel import GPTQModel
model = GPTQModel.load("btbtyler09/Qwen3.8-27B-GPTQ-4bit", trust_remote_code=True)
```

Note: transformers 5.x does not instantiate the MTP submodule for this architecture, so the `mtp.*` tensors are ignored (with a warning) when loading via transformers/GPTQModel. They are included in the checkpoint for engines that use them for speculative decoding.

## Technical Notes

Qwen3.8-27B is a dense multimodal model — it shares the `Qwen3_5ForConditionalGeneration` wrapper with the MoE-based Qwen3.5/3.6-35B-A3B but uses a standard dense MLP in every decoder layer instead of an expert mixture. The text decoder alternates 3 linear-attention (GatedDeltaNet) layers with 1 full-attention layer, repeated 16 times for 64 total layers.

The vision encoder (27-block ViT) and MTP speculative decoding module are preserved at full BF16 precision from the original model. Only the text decoder's quantizable Linear modules are converted to INT4.

## Credits

- **Base Model**: [Qwen](https://huggingface.co/Qwen) — Qwen3.8-27B
- **Quantization**: GPTQ + FOEM via [GPTQModel](https://github.com/modelcloud/gptqmodel) v6.0.3
- **FOEM paper**: Zheng et al., "First-Order Error Matters: Accurate Compensation for Quantized Large Language Models", AAAI 2026
- **Quantized by**: [btbtyler09](https://huggingface.co/btbtyler09)

## License

This model inherits the [Apache 2.0 license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) from the base model.
