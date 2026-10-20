---
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
tags:
- qwen3.5
- moe
- gguf
- vision
- multimodal
- genesis
- hermes
- agentic
language:
- en
- zh
- multilingual
datasets:
- NousResearch/hermes-function-calling-v1
pipeline_tag: image-text-to-text
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
---

<img width="600px" src="ornith_logo.png">

# 🌟 Ornith-1.5-35B-A3B -> Genesis Hermes

An experimental model processed via an algorithm and created upon request from China.

> ⚡ If you like this Genesis LLM release you can donate to me via [Hipolink](https://hipolink.net/luffythefox) or:

> USDT (TRC20): `TGa4KTwHfF6zDBsLUBEjd1f1KdeAFwYUks`

> USDT (ERC20): `0x93F4019E0aa85d8078F56B3D3176Fab5Dfa79924`

> USDT (SOL): `BorkkyPDG4aRN2op8c5SQF5NithX38U4sh7wDAWQhYyX`

> and support future Genesis LLM development.

> ⚡ **Why Genesis project exists?** During training, **ALL** models don't just learn knowledge - they also accumulate random noise in their tensors. This noise builds up and creates something I call the **Noise Gate** - a fundamental barrier that stops LLM models from learning further and makes them unstable, verbose, and prone to hallucinations. My approach reduces this noise. It repairs the signal without touching the learned knowledge and gradient. The result is a model that consistent in performance, context clarity and following instructions, because it's no longer fighting its own internal chaos.

> **What is Genesis?** Genesis is post training data regeneration and calibrarion algorythm for neural networks (LLM) in GGUF format that I made with AI help during almost half a year of development. It's optimized, architecture independent, works with any model in GGUF format and based on mathematical statistics. I don't train or finetune models, I repair **purity of signal** in them instead on Google Collab Free on Tesla T4 GPU via Python based on how models learns information. On first stage I scan ssm_conv1d tensors in model, they handle long context memory. I repair balance between heads in them. On second stage I scan model and detect noise in tensors via custom SVD. During scanning I exclude token_embd.weight, output.weight, ffn_gate_inp_shexp.weight, 1D tensors, bias and norms. Then I reduce training noise in tensors via custom SVD with preserved training data, 99% of siginal and learned gradient. On third stage, I scan blocks in model via chunks via 3 parameters and pick best one that fits to weight distribution in tensor. Best picked chunk replaces zero chunks in broken tensor without touching learned structure in model

Model is based on [Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) base.

And [DJLougen/hermes-qwen3.5-35b-a3b-GGUF](https://huggingface.co/DJLougen/hermes-qwen3.5-35b-a3b-GGUF) finetune for Hermes agent.

I transferred data from finetune on Hermes dataset (around 2k blocks from two FFN expert tensors) to [Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) censored base.

LLM models often have:

- **Saturated weights**: the model's activations are stuck, gradients vanish, outputs degrade.
- **Scale mismatches**: one layer's weights are 10× larger than its peers for no good reason.
- **Mean drift**: weight distributions shifted positive or negative, breaking symmetry assumptions.
- **Zero blocks**: zero blocks corrupt the signal, turning training into noise amplification.
- **Training Noise:** training noise increase randomness and ruins model output quality.

My approach fixes all of that without retraining - pure numerical surgery on the raw bytes of the file.

**Quantization script available here: https://pastebin.com/hXhcMJn9**

Feel free to do your own quants if you want.

## Recommended Settings for best perfomance on APEX quant

Chat template: [chat_template.jinja](https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V7-GGUF/raw/main/chat_template.jinja) thanks to [froggeric](https://huggingface.co/froggeric) and [qweefchief](https://huggingface.co/qweefchief)

Set K Cache Quantization Type and V Cache Quantization Type to F16. 

Set Number of layers for which to force MoE weights onto CPU to 40.

Set GPU offload to maximum. Set number of active experts to 8.

For best model stability and first experience I recommend starting from this string in your System Prompt with enabled thinking and nothing else: 

`You are Qwen (Tongyi Qianwen), a large language model developed by Alibaba Group's Tongyi Lab.`

If you want to bring more creativity to model use this System Prompt with `agent` identity: [link](https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V5-GGUF/discussions/7#6a6277b134def1392a9bd10f)

Or this System Prompt with `assistant` identity: [System_Prompt_Creative.txt](https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V5-GGUF/raw/main/System_Prompt_Creative.txt)

**Thinking mode (coding):**
- Hermes agent: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.05, seed=42, presence_penalty=disabled, repeat_penalty=1.05`
- Coding/precise tasks: `temperature=0.6, top_p=0.95, top_k=20, min_p=0, seed=42, presence_penalty=disabled, repeat_penalty=disabled`
- General: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.05, seed=42, presence_penalty=disabled, repeat_penalty=disabled`

**Non Thinking mode (creative):**
- General: `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, seed=42, presence_penalty=disabled, repeat_penalty=disabled`

For agentic tasks you can use this System Prompt: 

`You are Qwen (Tongyi Qianwen), a large language model developed by Alibaba Group's Tongyi Lab. You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema>.`

And this fix: [link to discussion](https://huggingface.co/ornith-ai/Ornith-1.0-35B-GGUF/discussions/46#6a5f80e69e4b977617f0f409)

And commands from this dataset: [hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)

## Benchmarks:

<img style="width: 100%; max-width: 900px;" src="ornith_35b_eval.png" alt="Ornith 1.5 35B Benchmark Results" title="Ornith 1.5 35B Benchmark Results">

**Links:**
- [Original censored model](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)
- [Quantization Script with Unsloth profiles support](https://pastebin.com/hXhcMJn9)

## Specs

- 35B total parameters, ~3B active per forward pass (MoE)
- 256 experts, 8 routed + 1 shared per token
- Hybrid architecture: Gated DeltaNet linear attention + full softmax attention (3:1 ratio)
- 40 layers, pattern: 10 × (3 × DeltaNet-MoE + 1 × Attention-MoE)
- 262K native context (extendable to 1M with YaRN)
- Natively multimodal (text, image, video)
- 248K vocabulary, 201 languages
- Base model. [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)

---

## Compatibility

Works with llama.cpp, LM Studio, koboldcpp, and other GGUF-compatible runtimes.