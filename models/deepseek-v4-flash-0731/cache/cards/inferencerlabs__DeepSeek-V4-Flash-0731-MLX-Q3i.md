---
language: en
tags:
- quantized
- mlx
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
---
**See DeepSeek-V4-Flash-0731 MLX in action - [demonstration videos](https://youtube.com/xcreate)**
   
#### Tested with an M3 Ultra 512 GiB using [Inferencer app](https://inferencer.com)
- Text inference: ~39.7 tokens/s @ 500 tokens ~104.6 GiB

<p style="margin-bottom:0px;">
This build uses the data-agnostic INF method to compresses the model to operate within the memory constraints of a 128 GiB RAM system while maintaining response coherence.

Consider increasing the available RAM for inferencing with:
- ```sudo sysctl iogpu.wired_limit_mb=118000```
</p>

![Screenshot](https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/wOCIKD8XeGCmFKM4ZEmDE.jpeg)

##### Quantized with a modified version of [MLX](https://github.com/ml-explore/mlx)
##### For more details see our [demonstration videos](https://youtube.com/xcreate) or visit [DeepSeek-V4-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-0731).

## Disclaimer

We are not the creator, originator, or owner of any model listed. Each model is created and provided by third parties. Models may not always be accurate or contextually appropriate. You are responsible for verifying the information before making important decisions. We are not liable for any damages, losses, or issues arising from its use, including data loss or inaccuracies in AI-generated content.
