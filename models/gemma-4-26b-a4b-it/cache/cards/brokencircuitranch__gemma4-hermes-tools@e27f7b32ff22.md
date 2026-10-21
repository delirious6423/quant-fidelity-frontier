---
license: gemma
language:
- en
base_model:
- google/gemma-4-26B-A4B-it
library_name: transformers
tags:
- unsloth
- gemma
- tool-use
- function-calling
- qlora
---


Fine-tuned version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) for reliable tool use and function calling.  
                                                                                                                                                   
  ## Training                                                                                                                                      
                                                               
  - **Base model:** google/gemma-4-26B-A4B-it (Mixture of Experts)                                                                                 
  - **Fine-tuning framework:** [Unsloth](https://github.com/unslothai/unsloth)
  - **Hardware:** NVIDIA A100 80GB (HuggingFace Space)                                                                                             
  - **Method:** QLoRA (4-bit) → merged to 16-bit                                                                                                   
                                                                                                                                                   
  ## Training Data                                                                                                                                 
                                                                                                                                                   
  - [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) — 1,893 examples of         
  structured tool use and function calling in Hermes format                                                                               
  - [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) — 5,000 sampled examples for general instruction following and
   reasoning                                                                                                                                       
                                                                                                                                                   
  Total: 6,893 examples, 2 epochs                              
                                                                                                                                                   
  ## Training Results                                          
                                                                                                                                                   
  | Step | Loss |                                              
  |------|------|
  | 10 | 1.825 | 
  | 50 | 0.374 |
  | 200 | 0.196 |                                                                                                                                  
  | 500 | 0.110 |
  | 862 | 0.113 |                                                                                                                                  
                                                               
  Final training loss: 0.224                                                                                                                       
                                                                                                                                                   
  ## Intended Use
                                                                                                                                                   
  Designed for agentic pipelines requiring reliable structured tool call generation. Tested with Ollama for local inference.
                                                                                                                                                   
  ## Files
                                                                                                                                                   
  - `model-0000x-of-00002.safetensors` — merged 16-bit weights 
  - `gemma4-hermes-tools-Q4_K_M.gguf` — quantized for local inference via Ollama/llama.cpp                                                         
                                                                                          
  ## License                                                                                                                                       
                                                               
  Inherits [Gemma Terms of Use](https://ai.google.dev/gemma/terms)