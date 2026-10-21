---
license: apache-2.0
base_model:
- coder3101/Qwen3.8-27B-heretic-ara
- Qwen/Qwen3.8-27B
---
I wanted a high quality coding model that doesn't give me AI safety lectures. So I quantized a heretic model with big coding/conversational/Japanese (for my instruction app)
datasets and copied layer config and MTP heads from unsloth/Qwen3.8-27B-NVFP4. llmcompressor wuantization script is included in model folder. Tested in Zoo code / 
chat / Japanese teacher app, seems to work well. I also finetuned drafter to the best of my ability to work with heretic / NVFP4 variant, getting 82.1% acceptance rate on
first position in my tasks.

I have uploaded a separate DSpark MTP head in https://huggingface.co/catplusplus/Qwen3.8-27B-heretic-ara-DSpark. It provides significantly
faster decode, but does not work with turboquant kv cache, so original MTP head uses less memory for given context length, both choices
have pros and cons.

I run this model as follows (uses local scripts, just illustrating vLLM arguments)

dockless run -d --name seeker-inference -e CUDA_HOME=/usr/local/cuda-13.0 -e C_INCLUDE_PATH=/usr/local/cuda-13.0/include -e LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia -e FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc -e VLLM_USE_FLASHINFER_MOE_FP4=0 ~/bin/unglitched_vllm --trust-remote-code --served-model-name Nikola --gpu-memory-utilization 0.675 --port 9000 --enable-auto-tool-choice --kv-cache-dtype turboquant_4bit_nc --tool-call-parser qwen3_coder --reasoning-parser qwen3 --mm-encoder-tp-mode data --mm-processor-cache-type shm  --enable-prefix-caching --max_num_batched_tokens 8192 --max-num-seqs 8 --cudagraph-capture-sizes 1 2 4 8 --mm-processor-kwargs '{"max_pixels": 3584000}' --enable-chunked-prefill --performance-mode interactivity --attention-backend TURBOQUANT --async-scheduling --moe-backend cutlass --compilation-config '{"mode": 0, "cudagraph_mode": 0}' --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' --model "${@:-/home/olegk/Nikola/models/Qwen3.8-27B-heretic-ara-NVFP4-MTP-V2}"

