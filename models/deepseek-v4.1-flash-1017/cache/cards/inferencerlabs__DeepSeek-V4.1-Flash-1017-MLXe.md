---
language: en
pipeline_tag: image-text-to-text
tags:
- quantized
- mlx
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: mlx
---
# DeepSeek-V4.1-Flash-1017
See DeepSeek-V4.1-Flash-1017 in action: [demonstration videos](https://youtube.com/xcreate)
   
#### Tested with [Inferencer](https://inferencer.com)
- M3 Ultra 512GB: ~15.9 tokens/s @ 1000 tokens ~278 GiB

<p style="margin-bottom:0px;">
This build repacks the language model's existing quantized weights, with the vision layers quantized to 9 bits and the engrams offloaded to storage to reduce the memory footprint. This configuration empirically yielded the highest quality in our testing.
</p>

![Screenshot 1](https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/o6qS3AH3mIv8O4RbWuvTL.jpeg)
![Screenshot 2](https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/bSLGJNG93dm-eXuzR3kMm.jpeg)

##### Quantized with a modified version of [MLX](https://github.com/ml-explore/mlx)
##### For more details see our [demonstration videos](https://youtube.com/xcreate) or visit [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

## Disclaimer

We are not the creator, originator, or owner of any model listed. Each model is created and provided by third parties. Models may not always be accurate or contextually appropriate. You are responsible for verifying the information before making important decisions. We are not liable for any damages, losses, or issues arising from its use, including data loss or inaccuracies in AI-generated content.
