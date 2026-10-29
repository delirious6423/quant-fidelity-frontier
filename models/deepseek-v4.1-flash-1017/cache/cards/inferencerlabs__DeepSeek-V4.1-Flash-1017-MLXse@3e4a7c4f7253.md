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
   
#### Tested with [Inferencer 2.5.0](https://inferencer.com)
- M1 Max 64GB: ~1.5 tokens/s
- M5 Max 128GB: ~6 tokens/s

<p style="margin-bottom:0px;">
This build repacks the model's existing quantized weights and offloads the engrams and experts to storage for running at full precision on low memory devices.
</p>

![Screenshot 1](https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/o6qS3AH3mIv8O4RbWuvTL.jpeg)
![Screenshot 2](https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/bSLGJNG93dm-eXuzR3kMm.jpeg)

##### Quantized with a modified version of [MLX](https://github.com/ml-explore/mlx)
##### For more details see our [demonstration videos](https://youtube.com/xcreate) or visit [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

## Disclaimer

We are not the creator, originator, or owner of any model listed. Each model is created and provided by third parties. Models may not always be accurate or contextually appropriate. You are responsible for verifying the information before making important decisions. We are not liable for any damages, losses, or issues arising from its use, including data loss or inaccuracies in AI-generated content.
