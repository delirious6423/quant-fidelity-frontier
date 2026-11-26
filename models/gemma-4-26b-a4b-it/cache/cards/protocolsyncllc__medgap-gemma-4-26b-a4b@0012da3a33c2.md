---
license: apache-2.0
datasets:
- protocolsyncllc/fda-warning-letters
language:
- en
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
tags:
- fda warning letters
- medical devices
- gemini-4-26b-a4b
library_name: unsloth
---
## FDA Warning Letters Gemini 4 26B A4B
This is a fine tune of Gemini 4 26B A4B on FDA warning letters for medical devices.

**Base model:** [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)  
**Dataset:** medical-devices.jsonl  
**Hardware:** 3x NVIDIA GeForce RTX 5060 Ti  
**Library:** Unsloth  
**Epochs**: 1  
**Technique**: LoRA  
**Learning Rate**: 2e-4  
**Inference GGUF:** [medgap-gemma-4-26b-a4b_q8_0.gguf](https://huggingface.co/protocolsyncllc/medgap-gemma-4-26b-a4b/blob/main/medgap-gemma-4-26b-a4b_q8_0.gguf)  
