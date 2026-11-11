---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- tenstorrent
- tensix
- tt-metal
- ttnn
- tt-forge
- block-floating-point
- bfloat4_b
- bfloat8_b
- quantized
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Qwen3.8-27B for Tenstorrent Tensix (BFP4)</h1>

[![Base model](https://img.shields.io/badge/base-Qwen%2FQwen3.8--27B-blue)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![Formats](https://img.shields.io/badge/formats-BFP4%20%2F%20BFP8%20head-5B2C83)](#formats)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://huggingface.co/Xylolabs/Qwen3.8-27B-Tensix-BFP4/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-tensix--bfp--gptq-black)](https://github.com/XylolabsInc/tensix-bfp-gptq)
[![Tenstorrent Tensix](https://img.shields.io/badge/Tenstorrent-Tensix-5B2C83)](https://docs.tenstorrent.com)

`tenstorrent` · `tensix` · `tt-metal` · `ttnn` · `tt-forge` · `block-floating-point` · `bfloat4_b` · `bfloat8_b` · `quantized`

</div>

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with its linear layers quantized to
`ttnn.bfloat4_b` (BFP4) with GPTQ and its LM head to `ttnn.bfloat8_b` (BFP8) with round to nearest, for
Tenstorrent Tensix NPUs (Wormhole, Blackhole). The files are bf16, with every value already on the BFP4 grid,
so TT-NN packs them as they are. Made by Xylolabs.

## Formats

| Tensors | Format |
|---|---|
| attention q, k, v, o (16 layers) | bfloat4_b, 4.5 bits a weight |
| Gated DeltaNet q, k, v, z, a, b, output (48 layers) | bfloat4_b, 4.5 bits a weight |
| MLP gate, up, down (64 layers) | bfloat4_b, 4.5 bits a weight |
| LM head | bfloat8_b, 8.5 bits a weight |
| embedding, norms and every other tensor | bf16, as in the base model |
| `mtp`, `visual` | bf16, unchanged |

5.23 bits a weight on average, 17.60 GB packed. `quant4.json` lists each projection's format. The bfloat4_b
tensors were quantized with GPTQ and a per-block exponent search, calibrated on 65,536 tokens of WikiText-2
and Korean Wikipedia; the bfloat8_b tensors with round to nearest.

Only the language model was quantized and evaluated.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Xylolabs/Qwen3.8-27B-Tensix-BFP4"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16", device_map="auto")

messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Run on Tensix

Tested with ttnn 0.81.1 on ttsim, Tenstorrent's Blackhole simulator; not on hardware. Pack one matrix and
check that no value changed (TT-NN reads weights as [in, out]):

```python
import json
import torch
import ttnn
from safetensors import safe_open

d = "Qwen3.8-27B-Tensix-BFP4"  # hf download Xylolabs/Qwen3.8-27B-Tensix-BFP4 --local-dir Qwen3.8-27B-Tensix-BFP4
name = "model.language_model.layers.0.mlp.down_proj.weight"
fmt = json.load(open(f"{d}/quant4.json"))["layers"]["0"]["mlp.down_proj"]["format"]
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

`python pack_tensix.py Qwen3.8-27B-Tensix-BFP4` checks every matrix this way (`--device` also round-trips it
through device DRAM). `python verify_grid.py Qwen3.8-27B-Tensix-BFP4` checks the grid with torch alone. The
same values also pack exactly into `ttnn.bfloat8_b`.

**tt-metal qwen38_27b_qb2 demo (v0.81.1).** tt-transformers does not support this architecture; tt-metal's
[qwen38_27b_qb2 demo](https://github.com/tenstorrent/tt-metal/tree/v0.81.1/models/demos/qwen38_27b_qb2) runs
it tensor-parallel on four Blackhole chips (QuietBox 2) and refuses other hardware. Set
`QWEN_PRECISION_CONFIG=baseline`, which matches this checkpoint; the shipped `config/precision.json` would
re-quantize the head to bfloat4_b. Its decoder packed every projection of layers 0 and 3 exactly; the head and
a full run need the four chips and were not tested.

```
# build tt-metal v0.81.1 as the demo's README describes, then
export MODEL_WEIGHTS_DIR=/path/to/Qwen3.8-27B-Tensix-BFP4
export QWEN_PRECISION_CONFIG=baseline
python models/demos/qwen38_27b_qb2/demo/text_demo.py --full --length 128 --generate 128 --output qwen38.json
```

**tt-forge (tt-xla).** Not run with this checkpoint yet.

## Evaluation

| | Bits a weight | Size (GB) | Mean KL to BF16 | Top-1 (%) | Answers right | Lost |
|---|---:|---:|---:|---:|---:|---:|
| BF16 ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0)) | 16 | 53.79 | 0 | 100 | 40 of 40 | - |
| **this model** | 5.23 | 17.596 | 0.02783 | 93.19 | 38 of 40 | 0 |
| Q4_K_M (llama.cpp) | 4.92 | 16.811 | 0.02084 | 93.50 | 40 of 40 | 0 |

KL and top-1 cover 4,094 positions of English and Korean text (WikiText-2 test, Korean Wikipedia), as plain
text. Answers are 33 short prompts with automatic checks, plus the 7 reasoning ones again with thinking on (40
in all). "Lost" means right in BF16 but wrong or broken here; near ties are not counted. The two reasoning
prompts it misses with thinking off come out right with thinking on, so neither counts as lost. Q4_K_M was
made from the same checkpoint with llama.cpp v0.9.0 (`cd959bae`) and no importance matrix; its sha256 is
`66927ba20072bb323235927c4d1b54b860dce33ace2b801982a9febddb46c07a`. Our bar: mean KL at most 1.5 times
Q4_K_M's and at most one more lost answer. Passed: 1.34 times Q4_K_M's KL, 0 lost against 0.

These numbers measure closeness to BF16, not task quality; no device speed was measured. For the model itself,
see the [base card](https://huggingface.co/Qwen/Qwen3.8-27B).

## License

Apache 2.0, as the base model (`LICENSE`). The changes are listed in `NOTICE`. `pack_tensix.py` and
`verify_grid.py` are by Xylolabs under the same license. Provided as is, without warranty. Not affiliated with
or endorsed by Tenstorrent.

## Citation

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```
