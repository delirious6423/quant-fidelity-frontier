---
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
tags:
- unsloth
- heretic
- uncensored
- decensored
- abliterated
- ara-lora
- gguf
pipeline_tag: image-text-to-text
---
# This is a decensored version of [unsloth/Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B), made using [Heretic](https://heretic-project.org) v1.4.0+custom with the Arbitrary-Rank Ablation (ARA) method using a LoRA adapter and row-norm preservation

## Abliteration parameters

| Parameter | Value |
| :-------- | :---: |
| **start_layer_index** | 9 |
| **end_layer_index** | 51 |
| **preserve_good_behavior_weight** | 1.0000 |
| **steer_bad_behavior_weight** | 0.3027 |
| **overcorrect_relative_weight** | 0.9481 |
| **neighbor_count** | 1 |

## Performance

| Metric | This model | Original model ([unsloth/Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B)) |
| :----- | :--------: | :---------------------------: |
| **Keywords** | 0/100 | 100/100 |
| **KL divergence** | 0.0528 | 0 *(by definition)* |

Note: Performance testing, including the measurement of refusal rates, was conducted using Japanese datasets.

## ⚠️ Important Notice

This model has undergone substantial reduction of its safety alignment. As a result, it is more likely than standard models to generate harmful, inaccurate, biased, offensive, or otherwise inappropriate content.

### Intended Use

For research and experimentation only, including safety research, alignment studies, and red-teaming. Please avoid deploying it in public or end-user-facing services.

### User Responsibility

All outputs should be treated as untrusted and independently verified before use. Users are solely responsible for:

* Evaluating the accuracy and suitability of generated content
* Implementing appropriate safeguards and human oversight
* Complying with applicable laws, regulations, licenses, and ethical standards

Use of this model is entirely at your own risk.

### Disclaimer

OS-Software provides this model without warranties of any kind and assumes no liability for any direct or indirect damages, losses, misuse, or legal consequences arising from its use.

## Acknowledgements

Thanks to the base model developers, [p-e-w](https://github.com/p-e-w) for Heretic, and the wider open-source community.

This is a derivative work released under the base model’s applicable license. All rights to the base model remain with their respective owners.

-----


# Read our How to [Run Qwen3.8-27B Guide!](https://unsloth.ai/docs/models/qwen3.8)
<div>
  <p style="margin: 0 0 0px 0; margin-top: 0px;">
    <em>See <a href="https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf">Unsloth Dynamic 2.0 GGUFs</a> for our quantization benchmarks.</em>
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
    <li>Qwen3.8 can now be run and fine-tuned in <a href="https://unsloth.ai/docs/new/desktop">Unsloth Desktop</a>. <a href="https://unsloth.ai/docs/models/qwen3.8">Read our guide</a>.</li>
    <li>Tool calling improvements: Makes parsing nested objects to make tool calling succeed more.</li>
   <li>See below for 1-bit Qwen3.8 run inside of Unsloth:</li>
</div>

<img width="600" alt="qwen3.8 unsloth desktop" src="https://3215535692-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FxhOjnexMCB3dmuQFQ2Zq%2Fuploads%2Fb3IiYJzHzsp2698xTim4%2Fgiffyy%20gf.gif?alt=media&token=f1bda1a1-b81f-43e2-ba10-dbdc9a29c0c0" />


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