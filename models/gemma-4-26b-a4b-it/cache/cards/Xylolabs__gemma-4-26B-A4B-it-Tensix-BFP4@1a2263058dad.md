---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
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
- quantized
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>gemma-4-26B-A4B-it for Tenstorrent Tensix (BFP4)</h1>

[![Base model](https://img.shields.io/badge/base-google%2Fgemma--4--26B--A4B--it-blue)](https://huggingface.co/google/gemma-4-26B-A4B-it)
[![Formats](https://img.shields.io/badge/formats-BFP4-5B2C83)](#formats)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-Tensix-BFP4/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-tensix--bfp--gptq-black)](https://github.com/XylolabsInc/tensix-bfp-gptq)
[![Tenstorrent Tensix](https://img.shields.io/badge/Tenstorrent-Tensix-5B2C83)](https://docs.tenstorrent.com)

`tenstorrent` · `tensix` · `tt-metal` · `ttnn` · `tt-forge` · `block-floating-point` · `bfloat4_b` · `quantized`

</div>

[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) with its linear layers quantized
to `ttnn.bfloat4_b` (BFP4) with GPTQ, for Tenstorrent Tensix NPUs (Wormhole, Blackhole). The files are bf16,
with every value already on the BFP4 grid, so TT-NN packs them as they are. Made by Xylolabs.

## Formats

| Tensors | Format |
|---|---|
| attention q, k, v, o, MLP gate, up, down and routed experts down, gate, up (30 layers) | bfloat4_b, 4.5 bits a weight |
| router per-expert scale and router (30 layers) | bf16 |
| LM head (tied to the embedding), embedding, norms and every other tensor | bf16, as in the base model |
| `embed_vision`, `vision_tower` | bf16, unchanged |

4.84 bits a weight on average, 15.27 GB packed. `quant4.json` lists each projection's format. Quantized with
GPTQ and a per-block exponent search, calibrated on 65,536 tokens of WikiText-2 and Korean Wikipedia. The
routed experts' gate and up share one Hessian per layer; each expert's down takes its own, or uses round to
nearest when it saw too few calibration tokens.

Only the language model was quantized and evaluated.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Xylolabs/gemma-4-26B-A4B-it-Tensix-BFP4"
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

d = "gemma-4-26B-A4B-it-Tensix-BFP4"  # hf download Xylolabs/gemma-4-26B-A4B-it-Tensix-BFP4 --local-dir gemma-4-26B-A4B-it-Tensix-BFP4
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

`python pack_tensix.py gemma-4-26B-A4B-it-Tensix-BFP4` checks every matrix this way (`--device` also
round-trips it through device DRAM). `python verify_grid.py gemma-4-26B-A4B-it-Tensix-BFP4` checks the grid
with torch alone. The same values also pack exactly into `ttnn.bfloat8_b`.

**tt-metal Gemma 4 demo (v0.81.1).** tt-transformers does not support Gemma 4. tt-metal's [Gemma 4
demo](https://github.com/tenstorrent/tt-metal/tree/v0.81.1/models/demos/gemma4) runs 26B-A4B tensor-parallel
on four or eight Blackhole chips (QuietBox 2, LoudBox) or on a Wormhole T3K, not on one chip, and it has no
bfloat4_b option. With no entry for this repository in its `precision_overrides.json`, every module loads in
bf16, which is exact. This entry packs attention, the dense MLP and the experts in bfloat8_b, also exactly.
That was checked on the first 2 layers on one simulated chip; the multi-chip split was not run.

```
"gemma-4-26B-A4B-it-Tensix-BFP4": {"default": {"attention": "bfp8", "shared_mlp": "bfp8", "experts": "bfp8"}}
```

**tt-forge (tt-xla).** Not run with Gemma 4 yet. By default it keeps the weights in bf16, which is exact; its
`bfp_bf4` option re-quantizes them in PyTorch's [out, in] layout.

## Evaluation

| | Bits a weight | Size (GB) | Mean KL to BF16 | Top-1 (%) | Answers right | Lost |
|---|---:|---:|---:|---:|---:|---:|
| BF16 ([google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52)) | 16 | 50.47 | 0 | 100 | 33 of 33 | - |
| **this model** | 4.84 | 15.271 | 0.14638 | 87.24 | 33 of 33 | 0 |
| Q4_K_M (llama.cpp) | 5.32 | 16.796 | 0.1249 | 88.16 | 33 of 33 | 0 |

KL and top-1 cover 8,048 positions of English and Korean text (WikiText-2 test, Korean Wikipedia), in the chat
template. Answers are 33 short prompts with automatic checks. "Lost" means right in BF16 but wrong or broken
here; near ties are not counted. Q4_K_M was made from the same checkpoint with llama.cpp v0.9.0 (`cd959bae`)
and no importance matrix; its sha256 is `2289deaba79c84dfa5727922ac0e7aee2a4807cc5b438faebdb75644e4856c7c`.
Our bar: mean KL at most 1.5 times Q4_K_M's and at most one more lost answer. Passed: 1.17 times Q4_K_M's KL,
0 lost against 0.

These numbers measure closeness to BF16, not task quality; no device speed was measured. For the model itself,
see the [base card](https://huggingface.co/google/gemma-4-26B-A4B-it).

## License

Apache 2.0, as the base model's card says; its repository has no LICENSE file, so `LICENSE` holds the standard
text. The changes are listed in `NOTICE`. `pack_tensix.py` and `verify_grid.py` are by Xylolabs under the same
license. Provided as is, without warranty. Not affiliated with or endorsed by Tenstorrent.

## Citation

```bibtex
@misc{gemmateam2026gemma4,
      title={Gemma 4 Technical Report},
      author={Gemma Team},
      year={2026},
      eprint={2607.02770},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.02770},
}
```
