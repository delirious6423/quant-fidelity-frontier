---
library_name: transformers
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- abliterated
- uncensored
- huihui
- qwen3
- unsloth
- llama.cpp
- qwen3_8
- moe
- gsq
- rco
- reasoning
- efficient-thinking
- token-efficient
- post-training
- Strata
---

# huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF


This is an uncensored version of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) created with abliteration (see [remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers) to know more about it).
This is a crude, proof-of-concept implementation to remove refusals from an LLM model without using TransformerLens.

# Last Update
We used calibration3.txt to generate imatrix_micro3.gguf, and then based on that produced 
[unsloth-UD-Q4_K_XL-V2](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF/tree/main/unsloth-UD-Q4_K_XL-V2), 
with the PPL value decreasing slightly.

The llama.cpp version used for testing is [llama.cpp-b11540](https://github.com/ggml-org/llama.cpp/releases/tag/b11540).


# Note
This version Compared to the [first version](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF), the ablation emphasis has been adjusted, and none of the experts have been ablated. Suitable for novel writing and programming.

## Strata

The newly added **Swift**  series come from [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF).

The **unsloth** series come from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). 

This **unsloth** and **Swift** series supports the latest versions of both [Strata](https://github.com/Niko1221/Strata) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

Strata has byte-level and SHA256 verification for the Unsloth series.   
The simplest method is to modify the verify_sha256 function in the setup.py file so that it returns immediately, then no verification is needed.

```
def verify_sha256(s: Path, size: int, sha: str) -> None:
    return
```

This is just a test/validation.


## llama.cpp
GGUFs come from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)

Use the latest [llama.cpp](https://github.com/ggml-org/llama.cpp),
```
llama-cli -m huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -c 262144
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
  bc1qqnkhuchxw0zqjh2ku3lu4hq45hc6gy84uk70ge
```
- Support our work on [Ko-fi](https://ko-fi.com/huihuiai)!

