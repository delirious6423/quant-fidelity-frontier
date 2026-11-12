---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
- tenstorrent
- tensix
- tt-metal
- ttnn
- tt-forge
- block-floating-point
- bfloat4_b
- quantized
- gptq
- qwen3_5_moe
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Ornith-1.5-35B-A3B for Tenstorrent Tensix (BFP4)</h1>

[![Base model](https://img.shields.io/badge/base-ornith--ai%2FOrnith--1.5--35B--A3B-blue)](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
[![Formats](https://img.shields.io/badge/formats-BFP4-5B2C83)](#formats)
[![License](https://img.shields.io/badge/license-MIT-green)](https://huggingface.co/Xylolabs/Ornith-1.5-35B-A3B-Tensix-BFP4/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-tensix--bfp--gptq-black)](https://github.com/XylolabsInc/tensix-bfp-gptq)
[![Tenstorrent Tensix](https://img.shields.io/badge/Tenstorrent-Tensix-5B2C83)](https://docs.tenstorrent.com)

`tenstorrent` · `tensix` · `tt-metal` · `ttnn` · `tt-forge` · `block-floating-point` · `bfloat4_b` · `quantized`

</div>

[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) with its linear layers
quantized to `ttnn.bfloat4_b` (BFP4) with GPTQ, for Tenstorrent Tensix NPUs (Wormhole, Blackhole). The quantized matrices are stored as bf16 values on the BFP4 grid. All 20,790 individual matrices pack bit for bit. Made by Xylolabs.

## Formats

| Tensors | Format |
|---|---|
| Language-model attention, DeltaNet, shared-expert and routed-expert projection weights | bfloat4_b, 4.5 bits a weight |
| Router, shared-expert gate, LM head and embedding | bf16 |
| Other unquantized language-model parameters | bf16 |
| Vision and MTP modules | bf16, unchanged |

4.84 bits a weight on average, 20.99 GB packed. `quant4.json` lists each projection's format. Quantized with
GPTQ (damping 0.1) and a per-block exponent search (offsets -1 and 0), except the routed expert down
projections, which use plain GPTQ with round-to-nearest for experts with insufficient calibration tokens.

Only the language model was quantized and evaluated.
Vision and MTP tensors remain unchanged. Average bits and packed sizes cover the language model, excluding
those modules. The Q4_K_M row gives the GGUF file size and average bits.

## Quickstart

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer

repo = "Xylolabs/Ornith-1.5-35B-A3B-Tensix-BFP4"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(repo, dtype="bfloat16", device_map="auto")

messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The whole checkpoint is retained. The text quickstart uses the native conditional-generation class and was
checked with Transformers 5.22.2 on CPU. Vision inputs and MTP were not evaluated.

## Run on Tensix

Tested with ttnn 0.81.1 on ttsim, Tenstorrent's Blackhole simulator; not on hardware. Pack one matrix and
check that no value changed (TT-NN reads weights as [in, out]):

```python
import json
import torch
import ttnn
from safetensors import safe_open

d = "Ornith-1.5-35B-A3B-Tensix-BFP4"  # hf download Xylolabs/Ornith-1.5-35B-A3B-Tensix-BFP4 --local-dir Ornith-1.5-35B-A3B-Tensix-BFP4
name = "model.language_model.layers.0.linear_attn.in_proj_a.weight"
fmt = json.load(open(f"{d}/quant4.json"))["layers"]["0"]["linear_attn.in_proj_a"]["format"]
dtype = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[fmt]
shard = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"][name]
with safe_open(f"{d}/{shard}", "pt") as f:
    w = f.get_tensor(name).t().contiguous()

device = ttnn.open_device(device_id=0)
w_tt = ttnn.as_tensor(w, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                      device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
assert torch.equal(ttnn.to_torch(w_tt).float(), w.float())
ttnn.close_device(device)
```

`python pack_tensix.py Ornith-1.5-35B-A3B-Tensix-BFP4` checks every matrix this way (`--device` also
round-trips it through device DRAM). `python verify_grid.py Ornith-1.5-35B-A3B-Tensix-BFP4` checks the grid
with torch alone.

tt-transformers and tt-forge were not run with this checkpoint.

## Evaluation

| | Bits a weight | Size (GB) | Mean KL to BF16 | Top-1 (%) | Answers right | Lost |
|---|---:|---:|---:|---:|---:|---:|
| BF16 ([ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/10fbf86fed7ecee4a061f8b499a618f46001cac1)) | 16 | 69.32 | 0 | 100 | 31 of 33 | - |
| **this model** | 4.84 | 20.991 | 0.0870 | 87.10 | 32 of 33 | 0 |
| Q4_K_M (llama.cpp) | 4.89 | 21.713 | 0.0792 | 87.63 | 31 of 33 | 0 |

KL and top-1 cover 8,188 positions of English and Korean text (WikiText-2 test, Korean Wikipedia), as plain
text. Answers are 33 short prompts with automatic checks. "Lost" means right in BF16 but wrong or broken here;
near ties are not counted. Q4_K_M was made from the same checkpoint with llama.cpp v0.9.0 (`cd959bae`) and no
importance matrix. Our bar: mean KL at most 1.5 times Q4_K_M's and at
most one more lost answer. Passed: 1.10 times Q4_K_M's KL, 0 lost against 0.

These numbers measure closeness to BF16, not task quality; no device speed was measured. For the model itself,
see the [base card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

## License

MIT, as declared by the original model card. Its linked LICENSE file is absent.
`LICENSE` contains the canonical MIT terms; `NOTICE` identifies the source revision and changes.
The Xylolabs packing/checking scripts use Apache-2.0 (`LICENSE-CODE`).
Not affiliated with or endorsed by Tenstorrent.

## Citation

```bibtex
@misc{ornith_1_5,
    title = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
    url = {https://ornith.ai/ornith_1_5.html},
    author = {{Ornith Team}},
    year = {2026}
}
```
