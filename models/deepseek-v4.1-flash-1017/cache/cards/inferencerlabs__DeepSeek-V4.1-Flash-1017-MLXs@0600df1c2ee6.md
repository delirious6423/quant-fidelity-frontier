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

For improved performance, download and enable a compatible [MTP speculative decoder](https://huggingface.co/models?search=inferencerlabs/deepseek-v4.1-flash-1017-mtp-mlx).
   
### Tested with [Inferencer v2.5.3](https://inferencer.com)
##### M3 Ultra without MTP
- 500 Tokens: ~19.9 tokens/s ~280 GiB
- 5,000 Tokens: ~19.5 tokens/s ~280 GiB

##### M3 Ultra with MTP
- 500 Tokens: ~42.1 tokens/s ~288 GiB
- 5,000 Tokens: ~35.8 tokens/s ~288 GiB

<p style="margin-bottom:0px;">
This build repacks the language model's existing quantized weights, with the vision layers quantized to 9 bits and the engrams streamed from storage to reduce the memory footprint. This configuration empirically yielded the highest quality in our testing.
</p>

![Screenshot 1](https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/o6qS3AH3mIv8O4RbWuvTL.jpeg)
![Screenshot 2](https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/bSLGJNG93dm-eXuzR3kMm.jpeg)

##### Quantized with a modified version of [MLX](https://github.com/ml-explore/mlx)
##### For more details see our [demonstration videos](https://youtube.com/xcreate) or visit [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

## Disclaimer

We are not the creator, originator, or owner of any model listed. Each model is created and provided by third parties. Models may not always be accurate or contextually appropriate. You are responsible for verifying the information before making important decisions. We are not liable for any damages, losses, or issues arising from its use, including data loss or inaccuracies in AI-generated content.
