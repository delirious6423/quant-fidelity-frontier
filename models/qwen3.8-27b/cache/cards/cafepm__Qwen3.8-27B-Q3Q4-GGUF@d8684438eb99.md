---
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
tags:
- unsloth
- RSIT
- Infiniteons
---
<center><a href="https://ai.tnt.chat" target="_blank">
  <img src="https://ai.tnt.chat/assets/tnt.chat.shape-white.bg-transp-Dj8x3xII.svg" width="30%">
</a></center>

<img src="https://notoken.cloud/assets/qwen38-artificial_analysis_v42.jpg" width="100%">

Use this model for **FREE** in your **Local VS Code**, even without a GPU card at ~10 tokens/s, via Google Colab (colab as endpoint):
https://colab.research.google.com/gist/masterofrisk/7d7e037afca618112a1ac9df404fa9e0/free-qwen-3-8-27b-q2q4-colab_tnt_bridge.ipynb

<center><a href="https://notoken.cloud" target="_blank">
  <img src="https://ai.tnt.chat/article/notoken.png" width="100%">
</a>
<a href="https://notoken.cloud" target="_blank">  
<h3 style="color:mediumpurple">Run Qwen3.8-27B MXFP4 on the Cloud, from a dedicated GPU, ZERO Data Retention, from $0.69/h</h3>
</a></center>

---

**Quantized from `unsloth/Qwen3.8-27B-GGUF` (BF16) using the Infiniteon Algebra associator to guide mixed-precision allocation: 20% of the weights (the shared expert and attention tensors) are quantized to 4-bit (Q4_K), and the remaining 80% to 3-bit (Q3_K). This achieves Q4-level generation quality at near-Q3 size, fitting a 27B-parameter model inside 24 GB VRAM, and the Q2Q4 format inside a 16 GB VRAM.**

**Why this works:** The associator of the Infiniteon algebra identifies which weight tensors carry structural "gap-crossing" information — the tensors that connect diverse token neighborhoods. By allocating 4-bit precision to these structurally critical tensors and 3-bit to the rest, the model preserves the directional diversity needed for coherent generation while keeping the overall size near Q3.

**Research and credits:** This research was conducted by TNT.Chat (https://ai.tnt.chat) as part of our LOCAL AI effort to bring larger models onto consumer hardware. The target is a capable model running on 16 GB VRAM for Agentic usage with a large context window. Try the Q2Q4 version of this model with only 13 GB in size.

**Theory references:**

Pinto Martins, C. F. (2026). *Ramsey Statistics And Infiniteons Theory, Volume I.* Zenodo. DOI: https://doi.org/10.5281/zenodo.19329589

Pinto Martins, C. F. (2026). *Ramsey Statistics And Infiniteons Theory, Volume II.* Zenodo. DOI: https://doi.org/10.5281/zenodo.19330373


**The mmproj GGUF has BF16 format only.**

The Q2Q4 format follows the same 20% Q4 and 80% Q2 structure ← Smallest size.

**We tested the three MIX20 formats** on our `tnt-bench`, a 51-question benchmark created to evaluate human interaction through direct questions to the models. The goal is to provide a first glance at how each model behaves before diving deeper into long-horizon tasks.

Qwen3.8 27B was a winner in both versions, Q2Q4 and M_Q3Q4, but the M_Q3Q4 retained remarkable precision, if you have more than 16GB of VRAM, that version should be used.

Our overall pick for a 16 GB GPU is the Q2Q4 format, with the `mmproj` offloaded to the CPU.

For more details about `tnt-bench` please check our [website](https://ai.tnt.chat/models).

---

# Read our How to [Run Qwen3.8-27B Guide!](https://unsloth.ai/docs/models/qwen3.8)
<div>
  <p style="margin: 0 0 0px 0; margin-top: 0px;">
    <em>This GGUF uses Unsloth Dynamic V3.0 (preview) for SOTA quantization performance.</em>
  </p>
  <div style="display: flex; gap: 5px; align-items: center; margin-bottom: 0px;">
    <a href="https://github.com/unslothai/unsloth/">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="133">
    </a>
    <a href="https://discord.gg/unsloth">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/Discord%20button.png" width="173">
    </a>
    <a href="https://unsloth.ai/docs/models/qwen3.8">
      <img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png" width="143">
    </a>
  </div>
    <ul style="margin: 0;">
    <li>Developer Role Support so Qwen3.8 can work in agentic tools like Codex and more!</li>
    <li>MTP for fast inference is available.</li>
    <li>Qwen3.8 can now be run and fine-tuned in <a href="https://unsloth.ai/docs/new/desktop">Unsloth Desktop</a> with <strong>Thinking toggles</strong>. <a href="https://unsloth.ai">Download</a> for Mac, Windows and Linux.</a>.</li>
    <li>Tool calling improvements: Makes parsing nested objects to make tool calling succeed more.</li>
   <li>See below for 4-bit Qwen3.8-27B run inside of Unsloth Desktop:</li>
</div>

<img width="600" alt="qwen3.8 unsloth desktop" src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2FSqxs6NjShWrLfRKhDy1m%2Fvolcano%202.gif?alt=media&token=395274a0-b437-403a-8a01-8e8502f9d225" />


---

# Qwen3.8-27B

Following the widespread community adoption of the Qwen3.5 and Qwen3.6 series, we are pleased to introduce Qwen3.8, the most capable generation in the Qwen open-model family to date.

Built on the architectural foundation of Qwen3.5, Qwen3.8 delivers substantial gains across coding, professional work, research, and long-horizon agentic tasks. Qwen3.8-27B brings these advances to a compact, deployment-friendly dense model: a native vision-language model that understands images and videos, with flexible thinking control, designed to carry complex, multi-step tasks through to completion with greater reliability.

## Qwen3.8 Highlights

Qwen3.8-27B features the following enhancements:
- **Core Capabilities**: Comprehensive improvements across coding, professional work, research, and long-horizon agentic tasks.
- **Agent Execution**: Stronger autonomous planning and better handling of environment feedback, leading to more reliable end-to-end task completion.
- **Downstream Compatibility**: Broader support for popular harnesses and development tools, making it easier to integrate into your existing stack.
- **Flexible Thinking Control**: Thinking mode is on by default and can be disabled per request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from historical messages is retained via `preserve_thinking`.
- **Vision-Language Understanding**: Native support for image and video understanding, from STEM diagrams and documents to hour-scale videos.


## Model Overview

- Type: Causal Language Model with Vision Encoder
- Training Stage: Pre-training & Post-training
- Language Model
    - Number of Parameters: 27B
    - Hidden Dimension: 5120
    - Token Embedding: 248,320 (Padded)
    - Number of Layers: 64
    - Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
    - Gated DeltaNet:
        - Number of Linear Attention Heads: 48 for V and 16 for QK
        - Head Dimension: 128
    - Gated Attention:
        - Number of Attention Heads: 24 for Q and 4 for KV
        - Head Dimension: 256
        - Rotary Position Embedding Dimension: 64
    - Feed Forward Network:
        - Intermediate Dimension: 17,408
    - LM Output: 248,320 (Padded)
    - MTP (Multi-Token Prediction): trained with multiple steps
- Context Length: 262,144 natively and extensible up to 1,000,000 tokens.

## Best Practices

To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**: We suggest using the following sets of sampling parameters:  
    
    - Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
    - Instruct (or non-thinking) mode: `temperature=0.7`, `top_p=0.80`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
    
    For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetition. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

2. **Adequate Output Length**: To optimize performance on agentic tasks, we recommend allocating sufficient output length to allow the model to generate detailed and comprehensive responses. For frameworks that support separate token limits for internal reasoning and final outputs, we suggest the following configuration within the 1M context length:
    
    - Reasoning Content: Set the maximum output length to 262,144 tokens.
    - Final Response: Set the maximum output length to 131,072 tokens.

    These settings provide the necessary capacity for complex reasoning while ensuring ample space for high-quality final deliverables.

3. **Processing Ultra-Long Texts**: Qwen3.8-27B natively supports context lengths of up to 262,144 tokens. For long-horizon tasks where the total length (including both input and output) exceeds this limit, we recommend using RoPE scaling techniques to handle long texts effectively, e.g., YaRN.

4. **Long Video Understanding**: To optimize inference efficiency for plain text and images, the `size` parameter in the released `video_preprocessor_config.json` is conservatively configured. It is recommended to set the `longest_edge` parameter in the video_preprocessor_config file to 469,762,048 (corresponding to 224k video tokens) to enable higher frame-rate sampling for hour-scale videos and thereby achieve superior performance. For example,
    ```json
    {"longest_edge": 469762048, "shortest_edge": 4096}
    ```

## Citation

If you find our work helpful, feel free to give us a cite.


```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```