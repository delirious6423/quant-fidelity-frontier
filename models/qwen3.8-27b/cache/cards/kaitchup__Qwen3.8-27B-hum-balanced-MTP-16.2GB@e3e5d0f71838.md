---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
tags:
- autoround
- llm-compressor
---

<div align="center">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/64b93e6bd6c468ac7536607e/mj6xac74jHGLqymiovObc.png" 
    alt="The Kaitchup -- AI on a Budget" 
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://kaitchup.substack.com/subscribe"><strong>Subscribe and Support</strong></a>
  </div>
</div>



This is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized with [AutoRound](https://github.com/intel/auto-round). Model packed by [llm-compressor](https://github.com/vllm-project/llm-compressor). 

Quantization ("balanced" settings):
* INT data types
* ~3.5 bits for linear layers
* 8-bit LM head
* BF16 MTP layers
* 8-bit activations

The model is compatible with vLLM (tested: v0.26+). Made and Tested with an RTX Pro 6000 (Google Colab's G4). 
Currently under evaluation: [Subscribe](https://kaitchup.substack.com/subscribe) to be notified when I publish the results.

- **Made by:** [The Kaitchup](https://kaitchup.substack.com/)


## Instructions 


```
vllm serve [this_model_id] \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 
```

Add --speculative-config '{"method":"mtp","num_speculative_tokens":X}' for MTP, where good values for X are between 2 and 6 (depending on your tasks and concurrency).








