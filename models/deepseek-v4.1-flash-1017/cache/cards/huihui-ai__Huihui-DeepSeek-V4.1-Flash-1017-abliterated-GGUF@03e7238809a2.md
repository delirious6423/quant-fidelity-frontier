---
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- deepseek
- deepseek_v41
- unsloth
- abliterated
- uncensored
- huihui
---

# huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated

This is an uncensored version of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) created with abliteration.

This ablation was performed entirely using llama.cpp's GGUF model, without relying on Transformers, 

For the specific ablation method, please refer to [cvector-generator](https://github.com/ggml-org/llama.cpp/blob/master/tools/cvector-generator/README.md).

## Note

The **UD-** series come from [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF). 
Only layers **11 to 28 (0-based indexing)** have been ablated.  
The expert system was not ablated. Only 18 weights were ablated in total (the 18 attn_output_b tensors); everything else was left untouched.

## llama.cpp
```

apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev git -y
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp fetch origin pull/30817/head:deepseek41
git -C llama.cpp checkout deepseek41
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp

llama-cli -m huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf -c 262144
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

