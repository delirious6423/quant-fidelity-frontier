---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- ko
tags:
- tenstorrent
- tensix
- tt-metal
- ttnn
- block-floating-point
- bfloat4_b
- bfloat8_b
- quantized
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Ornith 1.5 35B A3B text decoder for Tenstorrent Tensix</h1>

[![Base model](https://img.shields.io/badge/base-ornith--ai%2FOrnith--1.5--35B--A3B-blue)](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
[![Formats](https://img.shields.io/badge/formats-BFP4-5B2C83)](#folders)
[![License](https://img.shields.io/badge/license-MIT-green)](https://huggingface.co/Xylolabs/Ornith-1.5-35B-A3B-Tenstorrent-Tensix/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-tensix--bfp--gptq-black)](https://github.com/XylolabsInc/tensix-bfp-gptq/tree/v0.1.0)
[![Tenstorrent Tensix](https://img.shields.io/badge/Tenstorrent-Tensix-5B2C83)](https://docs.tenstorrent.com)

</div>

These weights from [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) are quantized for Tenstorrent Tensix NPUs, with the files in `text-bfp4/`. Vision and MTP weights are omitted, and drafting is disabled.

## Folders

| Folder | Formats | Size (GB, packed) | Mean KL to BF16 | Answers (33 + 7) |
| --- | --- | --- | --- | --- |
| `text-bfp4/` | Routed experts bfloat4_b; attention/shared experts/LM head bfloat8_b; the rest bf16 | 21.218 | 0.058006 | 31/33 + 7/7 |
| Original BF16 reference | - | - | 0 | 31/33 + 7/7 |
| Matched Q4_K_M | - | - | 0.079161 | 31/33 + 7/7 |

KL covers 8,188 positions from English/Korean raw texts.

## How to run

```bash
hf download Xylolabs/Ornith-1.5-35B-A3B-Tenstorrent-Tensix --include "text-bfp4/*" --local-dir Ornith-1.5-35B-A3B-Tenstorrent-Tensix
```

Use Transformers 5.23.0 and PyTorch.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Ornith-1.5-35B-A3B-Tenstorrent-Tensix/text-bfp4")
model = AutoModelForCausalLM.from_pretrained("Ornith-1.5-35B-A3B-Tenstorrent-Tensix/text-bfp4", dtype="bfloat16", device_map="auto")
messages = [{"role": "user", "content": "What is the capital of France? Reply with the city name only."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True, enable_thinking=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Run on Tensix

Pack one routed-expert matrix:

```python
import json
import torch
import ttnn
from safetensors import safe_open

d = "Ornith-1.5-35B-A3B-Tenstorrent-Tensix/text-bfp4"
name = "model.layers.0.mlp.experts.down_proj"
shard = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"][name]
with safe_open(f"{d}/{shard}", "pt") as f:
    w = f.get_tensor(name)[0].t().contiguous()
device = ttnn.open_device(device_id=0)
w_tt = ttnn.as_tensor(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                      device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
assert torch.equal(ttnn.to_torch(w_tt).float(), w.float())
ttnn.close_device(device)
```

```bash
python Ornith-1.5-35B-A3B-Tenstorrent-Tensix/text-bfp4/pack_tensix.py Ornith-1.5-35B-A3B-Tenstorrent-Tensix/text-bfp4
python Ornith-1.5-35B-A3B-Tenstorrent-Tensix/text-bfp4/verify_grid.py Ornith-1.5-35B-A3B-Tenstorrent-Tensix/text-bfp4
```

Add `--device` to round-trip the packed values.

## Updates

- 2026-11-28: Card text trimmed.
- 2026-11-27: Renamed from Ornith-1.5-35B-A3B-Tenstorrent-Tensix-BFP4; files now in text-bfp4/.
- 2026-11-27: Weights now the Experts mix: KL 0.0870 to 0.0580, 20.99 to 21.22 GB.

## License

The pinned Ornith model card declares MIT; its linked license file is absent upstream. `LICENSE` contains the
full standard MIT terms naming the original model authors. Inherited Qwen3.5 components retain Apache 2.0 and
Copyright 2026 Alibaba Cloud; `UPSTREAM_QWEN_LICENSE` is the exact original license. `NOTICE` records both the
inheritance and our text-only changes. Generic Xylolabs helpers use Apache 2.0 (`CODE_LICENSE`). `pack_tensix.py` and `verify_grid.py` are [tensix-bfp-gptq v0.1.0](https://github.com/XylolabsInc/tensix-bfp-gptq/tree/v0.1.0/tools)'s tools, unchanged.

Each folder carries its own `LICENSE` and `NOTICE`.

## Citation

Cite the [original Ornith 1.5 release](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).
