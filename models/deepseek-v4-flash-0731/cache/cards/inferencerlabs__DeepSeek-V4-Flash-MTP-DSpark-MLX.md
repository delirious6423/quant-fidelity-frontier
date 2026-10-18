---
language: en
tags:
- mlx
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
library_name: mlx
pipeline_tag: text-generation
---
# DeepSeek-V4-Flash-0731 DSpark
See DeepSeek-V4-Flash with MTP in action: [demonstration videos](https://youtube.com/xcreate)

This draft model contains the extracted **DSpark** layers to be used alongside the [DeepSeek-V4-Flash-0731-MLX](https://huggingface.co/models?search=inferencerlabs/deepseek-v4-flash-0731-mlx) model as a speculative decoder for improved performance.

#### Tested on an M3 Ultra 512 GiB RAM using [Inferencer app v2.3.1](https://inferencer.com)
### Tetris HTML
<table style="border-collapse: collapse; border: none; text-align:left; margin-top:10px; margin-bottom:0px;">
  <tr><td><strong>Text inference</strong></td><td>~31 tokens/s @ 500 tokens ~148.5 GiB</td></tr>
  <tr><td><strong>Text inference (with MTP)</strong></td><td>~35.8 tokens/s @ 500 tokens ~152.9 GiB</td></tr>
  <tr><td><strong>Text inference (with DSpark)</strong></td><td>~53.1 tokens/s @ 500 tokens ~165.6 GiB</td></tr>
</table>

### Tell me about Paris
<table style="border-collapse: collapse; border: none; text-align:left; margin-top:10px; margin-bottom:0px;">
  <tr><td><strong>Text inference</strong></td><td>~30.5 tokens/s @ 500 tokens ~148.5 GiB</td></tr>
  <tr><td><strong>Text inference (with MTP)</strong></td><td>~37.3 tokens/s @ 500 tokens ~152.9 GiB</td></tr>
  <tr><td><strong>Text inference (with DSpark)</strong></td><td>~31.3 tokens/s @ 500 tokens ~165.6 GiB</td></tr>
</table>

<p style="margin-bottom:0;padding-bottom:0;">Enable speculative decoding in Inference Contols:</p>
<div style="margin-left:10px;">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/weJgx7wZI7VgOdQe827yS.jpeg"
    style="display:block;margin:0;">
</div>

##### Extracted with a modified version of [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
##### For more details see our [demonstration videos](https://youtube.com/xcreate) or visit [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash).

## Disclaimer

We are not the creator, originator, or owner of any model listed. Each model is created and provided by third parties. Models may not always be accurate or contextually appropriate. You are responsible for verifying the information before making important decisions. We are not liable for any damages, losses, or issues arising from its use, including data loss or inaccuracies in AI-generated content.
