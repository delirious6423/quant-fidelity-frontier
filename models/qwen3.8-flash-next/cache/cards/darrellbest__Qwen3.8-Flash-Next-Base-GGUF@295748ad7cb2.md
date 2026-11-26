---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
---

# Qwen3.8-Flash-Next-Base-GGUF

> Unofficial community quantization; not affiliated with or endorsed by Qwen or Alibaba.

The untuned Q8_0 reference GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). It
is the source the [Magma-Coder](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Magma-Coder-GGUF) builds are
calibrated and quantized from.

## Files

All of these are mirrored unchanged from **Unsloth**
([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), same Qwen Community
License 1.0):

| File | What it is |
|---|---|
| `Q8_0/` (6 shards, 188 GB) | Unsloth's Q8_0 build. All 512 experts in all 48 layers; the n-gram (PLE) table in Q8_0 |
| `MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf` | The multi-token-prediction head, in BF16 |
| `mmproj-F16.gguf` | The vision projector |

## License

Qwen Community License 1.0 (see `LICENSE`, included verbatim). In short:

- **Clause 1:** keep the copyright and license notice in all copies. A commercial product or service with more than
  100M monthly active users or US$20M monthly revenue must prominently display the model name.
- **Clause 2:** a licensee (or affiliate) running a **Model-as-a-Service** or **AI Work Assistant** business (an
  AI-coding or office-productivity assistant) needs a **separate license from Qwen** before any commercial Use.
  Purely internal use that exposes nothing to third parties is exempt.
- **No warranty:** the software and its outputs are provided "as is".

This summary isn't legal advice; the `LICENSE` text governs.

## Credits

- **Model:** Qwen3.8-Flash-Next by Qwen ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).
- **GGUF conversion, Q8_0, MTP and mmproj files:** Unsloth ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)).
