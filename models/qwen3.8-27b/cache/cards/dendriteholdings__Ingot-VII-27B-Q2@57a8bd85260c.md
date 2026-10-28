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

# Ingot-VII-27B-Q2

A public mirror of a model that won on Ingot, a Bittensor subnet where miners quantize a fixed teacher model and the validator crowns the copy that stays closest to it.

- Submitted by miner hotkey `5HKne1BL4xabTsbdSyMDK3Q17W8VeuN4CNUbjozffniU11sT` (uid 187)
- Crowned from submission `e7360cf1-719e-4c5c-a8a7-5de3d51a5ac1`

The weights are the miner's upload, byte for byte. Credit belongs to the miner.

## How close it is to the teacher

Per answer token, against the BF16 teacher reading the same answer: KL(teacher ‖ model) of the next-token distributions, and the hidden-state distance |h − h_teacher| / |h_teacher| averaged over every 8th layer. Lower is closer.

| | KL | hidden-state distance |
| --- | --- | --- |
| this model, in its winning eval | 0.10548 | 0.3297 |
| the king it beat, same eval | 0.10738 | 0.3349 |
| closer than genesis [dendriteholdings/Ingot-genesis-27B-Q2](https://huggingface.co/dendriteholdings/Ingot-genesis-27B-Q2) | 17.5% | 27.2% |

## Running it

`Ingot-VII-27B-PQ2_0.gguf` is a gguf with the layout, tokenizer and Hadamard rotation of [dendriteholdings/Ingot-genesis-27B-Q2](https://huggingface.co/dendriteholdings/Ingot-genesis-27B-Q2) (`Ingot-genesis-27B-Q2.gguf`), its weights the miner's. Stock llama.cpp does not run it; the PrismML fork does, at the commit Ingot scored it with, [`e3de715`](https://github.com/PrismML-Eng/llama.cpp/commit/e3de715222b2667c54a267441159a5729f11155e):

```
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp && git checkout e3de715222b2667c54a267441159a5729f11155e
cmake -B build -DGGML_CUDA=ON && cmake --build build -j
build/bin/llama-server -m Ingot-VII-27B-PQ2_0.gguf -ngl 99 -fa on --jinja
```

### With images

The file is the text model. For image input, add the vision tower of [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) as a projector ([`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/b072e1d3b35a0a630cece372c2127528e0994386/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf), [`Ternary-Bonsai-2-27B-mmproj-BF16.gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/b072e1d3b35a0a630cece372c2127528e0994386/Ternary-Bonsai-2-27B-mmproj-BF16.gguf)). It was made for genesis, not for this model's weights, and Ingot scores the text model only, so image quality is untested:

```
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --revision b072e1d3b35a0a630cece372c2127528e0994386 --local-dir .
build/bin/llama-server -m Ingot-VII-27B-PQ2_0.gguf --mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf -ngl 99 -fa on --jinja
```

The same weights as an MLX pack for Apple Silicon: [dendriteholdings/Ingot-VII-27B-MLX-2bit](https://huggingface.co/dendriteholdings/Ingot-VII-27B-MLX-2bit).
