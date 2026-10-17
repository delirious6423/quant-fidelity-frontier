---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
- abliterated
- uncensored
- huihui
- qwen3

---

# huihui-ai/Huihui-Qwen3.8-27B-abliterated


This is an uncensored version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) created with abliteration (see [remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) to know more about it).
This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens.

## Note
The first 15 layers were retained without ablation.
This model card modified from the original upload. Although the first 15 layers are without ablation, the model still is very compliant.
Moreso than any attempts with Obliteratus or Heretic, which were ineffective for me.
I presume Huihui-AI has a rather heavily modified version of Sumandora's github repository.
All model quantizations tested with the same mmproj.gguf for multi-modal image use.

## llama.cpp

Please use the latest version of [llama.cpp](https://github.com/ggml-org/llama.cpp)

My llama.cpp settings to yarn the context window to 1 million, use llama-bench to tweak the settings for your hardware's best performance.

```
[Qwen3.8-abliterated-Q8_0]
model = Huihui-Qwen3.8-abliterated-Q8_0.gguf
mmproj = mmproj.gguf
ctx-size = 1000000
rope-scaling = yarn
rope-scale = 4.0
spec-type = draft-mtp
spec-draft-n-max = 2
temp = 1.0
top-p = 0.95
top-k = 20
batch-size = 2048
ubatch-size = 2048
jinja = true

```

### Usage Warnings


 - **Risk of Sensitive or Controversial Outputs**: This model’s safety filtering has been significantly reduced, potentially generating sensitive, controversial, or inappropriate content. Users should exercise caution and rigorously review generated outputs.

 - **Not Suitable for All Audiences**: Due to limited content filtering, the model’s outputs may be inappropriate for public settings, underage users, or applications requiring high security.

 - **Legal and Ethical Responsibilities**: Users must ensure their usage complies with local laws and ethical standards. Generated content may carry legal or ethical risks, and users are solely responsible for any consequences.

 - **Research and Experimental Use**: It is recommended to use this model for research, testing, or controlled environments, avoiding direct use in production or public-facing commercial applications.

 - **Monitoring and Review Recommendations**: Users are strongly advised to monitor model outputs in real-time and conduct manual reviews when necessary to prevent the dissemination of inappropriate content.

 - **No Default Safety Guarantees**: Unlike standard models, this model has not undergone rigorous safety optimization. huihui.ai bears no responsibility for any consequences arising from its use.


### Donation
##### Your donation helps us continue our further development and improvement, a cup of coffee can do it.
- bitcoin:
```
  bc1q6nvh39fcmy0de0ezepnn2z0rn4dme9yjal77ah 
```

