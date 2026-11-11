---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags: [mlx, moe, on-device, ios, ownowl]
---

# Gemma 4 26B A4B (25B, 3.8B active) for OwnOwl (iPhone)

Files for the **OwnOwl** iPhone app, which runs this model fully on the phone, without internet.
The app downloads this repository on first launch when you pick the model.

- Base model: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), license **apache-2.0** (see the base model for its full license terms).
- Weights: 4-bit MLX quantization of the base model (group size 64), stored unchanged:
  `nonexpert.safetensors` holds every tensor except the routed experts; `pack/` holds the routed experts,
  byte-for-byte, laid out for direct reads from the phone's flash storage.
- Languages: multilingual.
- `reference.json`: fixed test prompts with reference output tokens, used by the app's self-test.

This is a repackaging for an app, not a new model. All credit for the model goes to its authors.
