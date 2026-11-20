---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- lfm2
- moe
- nf4
- bitsandbytes
- qlora
- lora
- cpu
---

# LFM2.5-24B-A2B, NF4 experts (training format for CPU QLoRA)

A 4-bit copy of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) made for one purpose:
**LoRA / QLoRA fine-tuning on machines without a GPU.** The mixture-of-experts weights (about 95% of the 24B
parameters) are stored as 4-bit NF4; everything else is unchanged. The whole model then fits in about 15 GB of RAM,
so an adapter can be trained on an ordinary CPU server.

This is an independent derivative. It is not made, endorsed or supported by Liquid AI. All rights in the original
model remain with Liquid AI, Inc. It is distributed under the **LFM Open License v1.0** (copy in [`LICENSE`](LICENSE)).

## What was changed

| File | Status |
|---|---|
| `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | **unmodified**, byte-identical to the original repository |
| `nf4_cache/rest.safetensors` | **modified (re-packaged):** every non-expert tensor of the original (embeddings, attention, convolution, norms, router, dense layers 0-1), values unchanged, in their original dtypes, in one file |
| `nf4_cache/layer_XX.safetensors` (38 files) | **modified (quantized):** the expert weights of MoE layer XX, quantized from bf16 to 4-bit NF4 |
| `load_nf4.py`, `README.md`, `NOTICE` | **added** |
| original `model-*.safetensors` shards, `model.safetensors.index.json`, figures, original README | not included (the weights are replaced by `nf4_cache/`) |

How the experts were quantized: each expert matrix (`feed_forward.experts.{e}.w1/w2/w3`) was taken from the original
bf16 weights and quantized with `bitsandbytes.functional.quantize_4bit(w, blocksize=64, quant_type="nf4",
compress_statistics=False)`. Each layer file stores the packed weights (`{w}.{e}.q`, uint8) and the absmax scales
(`{w}.{e}.a`, float32). NF4 quantization is lossy: the experts are close to, not equal to, the original values.

## How to use it: LoRA on CPU

Not a drop-in for `AutoModelForCausalLM.from_pretrained`: load it with [`load_nf4.py`](load_nf4.py), which builds
the model and plugs the 4-bit experts in.

```python
import torch
from peft import LoraConfig, get_peft_model
from load_nf4 import load_model

torch.set_num_threads(16)                       # CPU cores
model = load_model("path/to/this/folder")      # ~15 GB RAM
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0,      # dropout 0: see notes
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"],  # attention + convolution projections
    task_type="CAUSAL_LM"))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
# ...then a normal training loop in bf16; only the LoRA weights (~0.06% of the parameters) are trained.
```

Notes from our use:
- **Memory:** ~15 GB for the weights; training long rows (several thousand tokens) used ~25-32 GB of RAM in total with
  gradient checkpointing and logits computed only at the trained positions.
- **Speed:** on a 16-core x86 server, roughly 100 tokens/s of training at 2,048-token rows.
- **Dropout must be 0:** with gradient checkpointing, random dropout changes the recomputed activations, so tokens
  can be routed to different experts in the backward pass.
- **Experts per token:** the config uses 4; if you serve the model with fewer (llama.cpp `--moe-k`), train with the same
  number (`load_model(path, experts=3)`).
- **Serving the adapter:** convert it with llama.cpp's `convert_lora_to_gguf.py --base <this folder>` and apply the
  resulting `lora.gguf` on the official GGUF model
  ([LiquidAI/LFM2.5-24B-A2B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF)).
- Tested with torch 2.15.0 (CPU), transformers 5.22.1, peft 0.21.2, bitsandbytes 0.50.2.

## Limitations

- Lossy 4-bit experts: use the original model where full precision matters.
- Meant for training adapters, not for fast inference: for inference use the official GGUF files.
- Inherits all limitations of the original model; see the [original model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).

## License

LFM Open License v1.0, the license of the original model. A copy is in [`LICENSE`](LICENSE); attribution and the list of
modified files are in [`NOTICE`](NOTICE) and above.
