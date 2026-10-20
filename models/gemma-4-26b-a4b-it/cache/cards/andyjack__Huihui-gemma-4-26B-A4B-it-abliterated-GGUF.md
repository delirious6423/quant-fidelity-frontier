---
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: any-to-any
base_model:
- google/gemma-4-26B-A4B-it
tags:
- abliterated
- uncensored
---

# huihui-ai/Huihui-gemma-4-26B-A4B-abliterated


This is an uncensored version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) created with abliteration (see [remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) to know more about it).
This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens.  I created Q8 and Q6 quantizations for almost perfect full quality.  I made Q4_KM and MXFP4_MOE quantizations to run on a 16 GB or 24 GB graphics card.  All models have been tested with the mmproj-BF16.gguf for multi-modal capability.

**Note** For this model, both the thinking mode and the non-thinking mode have been completely abliterated. the first 5 layers have not been abliterated, may contain warning information, but no refusal will be made..

No ablation was performed on the 256 experts per layer.

## llama.cppp

Please use the latest version of llama.cpp
 
```
../../workspace/llama.cpp/build/bin/llama-server \
  -m Huihui-gemma-4-26B-A4B-it-abliterated-Q4_K_M.gguf \
  --mmproj mmproj-BF16.gguf \
  --host 0.0.0.0 \
  --port 11434 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --n-gpu-layers 99 \
  --split-mode layer \
  --no-mmap \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 64 \
  --repeat-penalty 1.0 \
  -c 256000 \
  -b 2048 \
  -ub 2048 \
  --parallel 1 

```


### Usage Warnings


 - **Risk of Sensitive or Controversial Outputs**: This model’s safety filtering has been significantly reduced, potentially generating sensitive, controversial, or inappropriate content. Users should exercise caution and rigorously review generated outputs.

 - **Not Suitable for All Audiences**: Due to limited content filtering, the model’s outputs may be inappropriate for public settings, underage users, or applications requiring high security.

 - **Legal and Ethical Responsibilities**: Users must ensure their usage complies with local laws and ethical standards. Generated content may carry legal or ethical risks, and users are solely responsible for any consequences.

 - **Research and Experimental Use**: It is recommended to use this model for research, testing, or controlled environments, avoiding direct use in production or public-facing commercial applications.

 - **Monitoring and Review Recommendations**: Users are strongly advised to monitor model outputs in real-time and conduct manual reviews when necessary to prevent the dissemination of inappropriate content.

 - **No Default Safety Guarantees**: Unlike standard models, this model has not undergone rigorous safety optimization. huihui.ai nor I bear any responsibility for any consequences arising from its use.


### Donation
##### Your donation helps us continue our further development and improvement, a cup of coffee can do it.
- bitcoin:
```
  bc1q6nvh39fcmy0de0ezepnn2z0rn4dme9yjal77ah
```

