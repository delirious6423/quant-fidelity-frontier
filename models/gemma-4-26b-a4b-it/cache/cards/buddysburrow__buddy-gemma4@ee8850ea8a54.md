---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
- gguf
- gemma4
---
# Buddy's brain: Gemma 4 26B-A4B

The on-device model used by the iOS app **Buddy**. The app downloads these files and runs them on the phone.

| File | What |
|---|---|
| `buddy-gemma4.resident.gguf` | The always-used part (from Unsloth's UD-IQ2_M, 5-bit) |
| `buddy-gemma4.experts` | The experts, read from flash (from Unsloth's UD-IQ3_S) |
| `buddy-gemma4.json` | Where each expert sits in the experts file |
| `buddy-gemma4-26b-eyes-Q8_0.gguf` | Vision projector (ggml-org's mmproj Q8_0) |

Made from Google's Gemma 4 26B-A4B (Apache 2.0) via [Unsloth](https://huggingface.co/unsloth)'s and
[ggml-org](https://huggingface.co/ggml-org)'s GGUF files, unchanged except for the split. Licensed Apache 2.0.
