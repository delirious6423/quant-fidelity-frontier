---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags: [tokenwing, run-my-ai, gguf, moe]
---
# Qwen3.8 Flash Next for TokenWing

Qwen3.8 Flash Next (by Qwen) in the layout [TokenWing](https://github.com/runmyai/tokenwing) streams from your SSD, so
this mixture-of-experts model runs on ordinary PCs, even with less memory than the model's size.

The easiest way to use it is the [Run My AI](https://runmyai.org) app: it downloads this model and sets everything up for
your PC.

- Weights: Grawnch's WV-v5 3.5bpw quantization ([Grawnch/Qwen3.8-Flash-Next-WV-v5-3.5bpw-GGUF](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-v5-3.5bpw-GGUF)), unchanged; the first file's layout is changed so experts can be read
  straight from the SSD. Next to it: TokenWing's data files for this model (correction, VRAM list, usage seed).
- License: Qwen Community License 1.0, as the original model (the full text is in LICENSE). Credit: Qwen, and Grawnch for
  the quantization.
