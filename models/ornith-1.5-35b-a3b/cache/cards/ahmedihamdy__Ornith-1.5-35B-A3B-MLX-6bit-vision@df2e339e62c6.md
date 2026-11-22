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
- 6bit
- qwen3.5
- moe
- apple-silicon
- astronomical
---

# Ornith-1.5-35B-A3B-MLX-6bit-vision

**Ornith 1.5 35B A3B can see again, on your Mac.** This is [ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit) with its original vision tower from [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) back in place, so it takes images and text in 29.08 GB.

It is a mixture-of-experts model: 35B parameters in all, with 8 of its 256 experts (about 3B parameters) at work for each token, which is why it answers quickly for its size.

The 8-bit sibling, [ahmedihamdy/Ornith-1.5-35B-A3B-MLX-8bit-vision](https://huggingface.co/ahmedihamdy/Ornith-1.5-35B-A3B-MLX-8bit-vision), is closest to the original, for about 9 GB more memory.

Made for **[Astronomical](https://getastronomical.com/)**, the free, open source Mac app that runs open models privately on your own Mac: nothing you type leaves it.

## Run it with Astronomical

[**Download Astronomical for Mac**](https://getastronomical.com/) (Apple silicon, macOS 14 and later). You tell it how much memory a model may use, and it keeps the rest of your Mac usable.

### Get the model from the app

1. Open Astronomical. In the **Astronomical** menu at the top of the screen, choose **Open Observatory**.
2. Go to **Models**, find **Ornith-1.5-35B-A3B-MLX-6bit-vision**, and click **Get**. Astronomical downloads about 27.1 GB and checks every file before it uses the model.
3. In the chat window, choose it from the model menu beside the message box, and start typing.

### Or use it from Terminal

First add the `astronomical` command: click Astronomical's menu-bar item, click the **⋯** button, and choose **Install “astronomical” command in terminal…**. Then:

```bash
# Download it from the catalog.
astronomical models download Ornith-1.5-35B-A3B-MLX-6bit-vision

# Show it a picture.
astronomical respond "What is in this image?" --image photo.png --model Ornith-1.5-35B-A3B-MLX-6bit-vision

# Start a coding agent that runs on it.
astronomical launch claude --model Ornith-1.5-35B-A3B-MLX-6bit-vision
```

Any other tool that speaks the OpenAI or Anthropic API can use it too: the [local API page](https://getastronomical.com/local-api.html) has the settings for each client.

## What makes this build special

- **Ornith's own 6-bit text weights, byte for byte.** Nothing was requantized. The experts are at 6-bit and the routers at 8-bit, exactly as Ornith shipped them.
- **Ornith's original vision tower.** All 27 blocks and 447M parameters, at full bf16 precision.
- **Proven to belong together, not assumed.** A tower only works with the text weights it was trained beside, so the pair was checked first: 24 of 24 sampled text tensors agree with the base, and the tower width and image tokens match.
- **Tested on real images.** Shown a screenshot of the Astronomical app, it read the model name, the size on disk and the GPU utilization exactly, and it described a second picture's text, colours and shapes correctly.
- **Every file is accounted for.** `astronomical-build.json` records both source commits and the SHA-256 of each weight and settings file, so anyone can check this build or repeat it.

In mlx-vlm 0.7.12 on an Apple silicon Mac, this build peaked at 30.41 GB of memory and generated 65 to 73 tokens per second.

This is a community build, not an official Ornith release. All credit for the model itself goes to its authors.

| Source | Commit |
|---|---|
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit` | `585b7867b0517980293ece857b26d64e84491352` |
| `ornith-ai/Ornith-1.5-35B-A3B` | `10fbf86fed7ecee4a061f8b499a618f46001cac1` |

## Build your own with the Astronomical CLI

This model was made with one command, and you can do the same for other models. The `vision` commands come with Astronomical 0.2.90 and later:

```bash
# Does this model have a vision tower? Answered in about a second, without downloading the weights.
astronomical vision inspect ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit

# Does this tower belong with these weights?
astronomical vision match ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit ornith-ai/Ornith-1.5-35B-A3B

# Build the model. It proves the match first and refuses a pair that does not belong together.
hf download ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit --revision 585b7867b0517980293ece857b26d64e84491352
astronomical vision add PATH_FROM_HF_DOWNLOAD --tower ornith-ai/Ornith-1.5-35B-A3B@10fbf86fed7ecee4a061f8b499a618f46001cac1 --output Ornith-1.5-35B-A3B-MLX-6bit-vision
```

## Also runs in mlx-vlm

```python
from mlx_vlm import apply_chat_template, generate, load
from mlx_vlm.utils import load_config

model_path = "ahmedihamdy/Ornith-1.5-35B-A3B-MLX-6bit-vision"
model, processor = load(model_path)
config = load_config(model_path)
prompt = apply_chat_template(processor, config, "What is in this image?", num_images=1)
reply = generate(model, processor, prompt, ["image.png"], max_tokens=1500, temperature=0.0)
print(reply.text)
```

## Layout

The tower is in `model-vision.safetensors`, its own file beside Ornith's six unchanged text shards, and `model.safetensors.index.json` lists it. Most MLX models keep the tower in the same shards as the text weights. Here it has its own file, so the text shards stay identical to the source and can be checked against their hashes.

## License

MIT, following the license stated on [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B). See that repository for the authors and terms.
