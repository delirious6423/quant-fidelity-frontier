---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- gguf
- ingot
- bittensor
---

# Ingot-XV-27B-Q2

A public mirror of a model that won on Ingot, a Bittensor subnet where miners quantize a fixed teacher model and the validator crowns the copy that stays closest to it.

- Submitted by miner hotkey `5DnphFGBSbHKjR8X7trjALFtKR4nDEHiwhky2yXxXuHMZeWL` (uid 43)
- Crowned from submission `5fa5a5fe-85e8-46ba-82ff-785d7ff59934`

The weights are the miner's upload, byte for byte (`Ingot-XV-27B-PQ2_0.gguf`); `Ingot-XV-27B-PTQ1_0.gguf` repacks them losslessly at 1.75 bits per weight. Credit belongs to the miner.

## Files

| file | bits per weight | size |
| --- | --- | --- |
| `Ingot-XV-27B-PTQ1_0.gguf` | 1.75 | 5.95 GB |
| `Ingot-XV-27B-PQ2_0.gguf` | 2.125 (the same ternary weights in a 2-bit container) | 7.21 GB |

Both files hold the same ternary weights. `Ingot-XV-27B-PTQ1_0.gguf` is Ingot's lossless repack of the miner's file into Prism's PTQ1_0 format: every 128-weight block keeps its codes and fp16 scale, and every weight decodes to the same value. The same repack of Bonsai 2's PQ2_0 file reproduces Prism's published PTQ1_0 file byte for byte. The 1.75 bpw file is smaller; the 2-bit file decodes faster on GPUs.

## How close it is to the teacher

Per answer token, against the BF16 teacher reading the same answer: KL(teacher ‖ model) of the next-token distributions, and the hidden-state distance |h − h_teacher| / |h_teacher| averaged over every 8th layer. Lower is closer.

| | KL | hidden-state distance |
| --- | --- | --- |
| this model, in its winning eval | 0.10062 | 0.3044 |
| the king it beat, same eval | 0.10220 | 0.3076 |
| closer than genesis [dendriteholdings/Ingot-genesis-27B-Q2](https://huggingface.co/dendriteholdings/Ingot-genesis-27B-Q2) | 27.4% | 34.5% |

## Running it

`Ingot-XV-27B-PQ2_0.gguf` is a gguf with the layout, tokenizer and Hadamard rotation of [dendriteholdings/Ingot-genesis-27B-Q2](https://huggingface.co/dendriteholdings/Ingot-genesis-27B-Q2) (`Ingot-genesis-27B-Q2.gguf`), its weights the miner's. Stock llama.cpp does not run it; the PrismML fork does, at the commit Ingot scored it with, [`e3de715`](https://github.com/PrismML-Eng/llama.cpp/commit/e3de715222b2667c54a267441159a5729f11155e):

```
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp && git checkout e3de715222b2667c54a267441159a5729f11155e
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
build/bin/llama-server -m Ingot-XV-27B-PQ2_0.gguf -ngl 99 -fa on --jinja
```

Either file runs the same way: pass `-m Ingot-XV-27B-PTQ1_0.gguf` for the 1.75 bpw one.

### With images

The file is the text model. For image input, add the vision tower of [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) as a projector ([`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/b072e1d3b35a0a630cece372c2127528e0994386/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf), [`Ternary-Bonsai-2-27B-mmproj-BF16.gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/b072e1d3b35a0a630cece372c2127528e0994386/Ternary-Bonsai-2-27B-mmproj-BF16.gguf)). It was made for genesis, not for this model's weights, and Ingot scores the text model only, so image quality is untested:

```
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --revision b072e1d3b35a0a630cece372c2127528e0994386 --local-dir .
build/bin/llama-server -m Ingot-XV-27B-PQ2_0.gguf --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf -ngl 99 -fa on --jinja
```
