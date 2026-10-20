---
pipeline_tag: image-text-to-text
base_model:
- google/gemma-4-26B-A4B-it
---
These are **MXFP4** quantizations of the model [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)

## Quick Start
1. Download the latest release of **llama.cpp**.
2. Download your preferred model variant from below.
3. For the `mmproj` file, it is recommended to use the **F32 version** for the best visual processing results. F32 > BF16 > F16

## Which version should I choose?
All variants use **MXFP4** for the MoE (Mixture of Experts) weights to keep the model efficient. The difference lies in how the remaining tensors are handled:

| Variant | Quality | Performance | Size | Recommendation |
| :--- | :--- | :--- | ---: | :--- |
| **BF16** | ⭐⭐⭐ | Variable* | 15.80GiB | Best for maximum accuracy; original unquantized weights. |
| **F16** | ⭐⭐ | Fast | 15.80GiB | Great alternative if BF16 is slow on your hardware. |
| **Q8** | ⭐ | Fastest | 14.36GiB | Balanced performance and memory usage. |

**Note:** On some older architectures, BF16 may be slower than F16.  
Check that your GPU supports native BF16

Read the [guide from unsloth](https://unsloth.ai/docs/models/gemma-4) in order to set up the model's recommended settings:  

### The official chat template has been updated from Google. If you do not want to download the model again, you can just tell llama.cpp to use the new chat template.
