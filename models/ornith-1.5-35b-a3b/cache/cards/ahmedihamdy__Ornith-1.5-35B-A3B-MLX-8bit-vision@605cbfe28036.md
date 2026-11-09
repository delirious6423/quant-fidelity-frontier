---
library_name: mlx
license: mit
pipeline_tag: image-text-to-text
base_model: ornith-ai/Ornith-1.5-35B-A3B
language:
- en
tags:
- mlx
- vision
- image-text-to-text
- multimodal
- 8bit
- qwen3.5
- moe
- apple-silicon
- astronomical
---

# Ornith-1.5-35B-A3B-MLX-8bit-vision

**Ornith 1.5 35B A3B can see again, on your Mac.** This is [ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit) with its original vision tower from [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) back in place, so it takes images and text in 37.74 GB.

It is a mixture-of-experts model: 35B parameters in all, with 8 of its 256 experts (about 3B parameters) at work for each token, which is why it answers quickly for its size.

The 6-bit sibling, [ahmedihamdy/Ornith-1.5-35B-A3B-MLX-6bit-vision](https://huggingface.co/ahmedihamdy/Ornith-1.5-35B-A3B-MLX-6bit-vision), is about 9 GB lighter.

Built with, and made for, **[Astronomical](https://aosama.github.io/astronomical/)**.

## Run it with Astronomical

[**Download Astronomical for Mac**](https://aosama.github.io/astronomical/) (Apple silicon, macOS 14 and later).

Astronomical is a free, open source app and command-line toolbox for running open models on your own Mac. You set one memory ceiling, and every request is answered right there on your machine, with no cloud account. On a Mac with less memory than this model needs, Astronomical can stream a mixture-of-experts model's experts from the SSD, which is slower but still runs.

1. Install Astronomical from [the site](https://aosama.github.io/astronomical/).
2. Download this model:

   ```bash
   hf download ahmedihamdy/Ornith-1.5-35B-A3B-MLX-8bit-vision --local-dir models/Ornith-1.5-35B-A3B-MLX-8bit-vision
   ```

3. Add the absolute path of that `models` folder to `runtime.model_directories` in Astronomical's config.
4. Show it a picture:

   ```bash
   astronomical respond "What is in this image?" --image image.png --model Ornith-1.5-35B-A3B-MLX-8bit-vision
   ```

## What makes this build special

- **Ornith's own 8-bit text weights, byte for byte.** Nothing was requantized. The experts are at 8-bit and the routers at 8-bit, exactly as Ornith shipped them.
- **Ornith's original vision tower.** All 27 blocks and 447M parameters, at full bf16 precision.
- **Proven to belong together, not assumed.** A tower only works with the text weights it was trained beside, so the pair was checked first: 24 of 24 sampled text tensors agree with the base, and the tower width and image tokens match.
- **Tested on real images.** Shown a screenshot of the Astronomical app, it read the model name, the size on disk and the GPU utilization exactly, and it described a second picture's text, colours and shapes correctly.
- **Every file is accounted for.** `astronomical-build.json` records both source commits and the SHA-256 of each weight and settings file, so anyone can check this build or repeat it.

In mlx-vlm 0.7.12 on an Apple silicon Mac, this build peaked at 39.07 GB of memory and generated 53 to 60 tokens per second.

This is a community build, not an official Ornith release. All credit for the model itself goes to its authors.

| Source | Commit |
|---|---|
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit` | `02440c39bdf7365c494a7f55f2a8b104ba87562f` |
| `ornith-ai/Ornith-1.5-35B-A3B` | `10fbf86fed7ecee4a061f8b499a618f46001cac1` |

## Build your own with the Astronomical CLI

This model was made with one command, and you can do the same for other models. The `vision` commands come with Astronomical 0.2.90 and later:

```bash
# Does this model have a vision tower? Answered in about a second, without downloading the weights.
astronomical vision inspect ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit

# Does this tower belong with these weights?
astronomical vision match ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit ornith-ai/Ornith-1.5-35B-A3B

# Build the model. It proves the match first and refuses a pair that does not belong together.
hf download ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit --revision 02440c39bdf7365c494a7f55f2a8b104ba87562f
astronomical vision add PATH_FROM_HF_DOWNLOAD --tower ornith-ai/Ornith-1.5-35B-A3B@10fbf86fed7ecee4a061f8b499a618f46001cac1 --output Ornith-1.5-35B-A3B-MLX-8bit-vision
```

## Also runs in mlx-vlm

```python
from mlx_vlm import apply_chat_template, generate, load
from mlx_vlm.utils import load_config

model_path = "ahmedihamdy/Ornith-1.5-35B-A3B-MLX-8bit-vision"
model, processor = load(model_path)
config = load_config(model_path)
prompt = apply_chat_template(processor, config, "What is in this image?", num_images=1)
reply = generate(model, processor, prompt, ["image.png"], max_tokens=1500, temperature=0.0)
print(reply.text)
```

## Layout

The tower is in `model-vision.safetensors`, its own file beside Ornith's seven unchanged text shards, and `model.safetensors.index.json` lists it. Most MLX models keep the tower in the same shards as the text weights. Here it has its own file, so the text shards stay identical to the source and can be checked against their hashes.

## License

MIT, following the license stated on [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B). See that repository for the authors and terms.
