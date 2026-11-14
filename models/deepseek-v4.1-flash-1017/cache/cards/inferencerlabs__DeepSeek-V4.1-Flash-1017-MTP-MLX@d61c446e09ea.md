---
language: en
tags:
- mlx
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: mlx
pipeline_tag: text-generation
---
# DeepSeek-V4.1-Flash-1017 MTP
See DeepSeek-V4.1-Flash-1017 with MTP in action: [demonstration videos](https://youtube.com/xcreate)

This draft model contains the extracted **DSpark MTP** layers to be used alongside the [DeepSeek-V4.1-Flash-1017-MLX](https://huggingface.co/models?search=inferencerlabs/deepseek-v4.1-flash-1017-mlx) model as a speculative decoder for improved performance.

#### Tested with [Inferencer v2.5.3](https://inferencer.com)
### Tetris HTML (No Thinking) / M3 Ultra / 500 Tokens
<table style="border-collapse: collapse; border: none; text-align:left; margin-top:10px; margin-bottom:0px;">
  <tr><td><strong>Text inference</strong></td><td>~19.9 tokens/s ~280 GiB</td></tr>
  <tr><td><strong>Text inference (with DSpark)</strong></td><td>~42.1 tokens/s ~288 GiB</td></tr>
</table>

### Tetris HTML (Low Thinking) / M3 Ultra / 500 Tokens
<table style="border-collapse: collapse; border: none; text-align:left; margin-top:10px; margin-bottom:0px;">
  <tr><td><strong>Text inference</strong></td><td>~19.8 tokens/s ~280 GiB</td></tr>
  <tr><td><strong>Text inference (with DSpark)</strong></td><td>~31.6 tokens/s ~288 GiB</td></tr>
</table>

### Tetris HTML (Low Thinking) / M3 Ultra / 5,000 Tokens
<table style="border-collapse: collapse; border: none; text-align:left; margin-top:10px; margin-bottom:0px;">
  <tr><td><strong>Text inference</strong></td><td>~19.5 tokens/s ~280 GiB</td></tr>
  <tr><td><strong>Text inference (with DSpark)</strong></td><td>~35.8 tokens/s ~288 GiB</td></tr>
</table>

### Tell me about Paris / M3 Ultra / 500 Tokens
<table style="border-collapse: collapse; border: none; text-align:left; margin-top:10px; margin-bottom:0px;">
  <tr><td><strong>Text inference</strong></td><td>~19.9 tokens/s ~280 GiB</td></tr>
  <tr><td><strong>Text inference (with DSpark)</strong></td><td>~20.7 tokens/s ~288 GiB</td></tr>
</table>

<p style="margin-bottom:0;padding-bottom:0;">Enable speculative decoding in Inference Contols:</p>
<div style="margin-left:10px;">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/weJgx7wZI7VgOdQe827yS.jpeg"
    style="display:block;margin:0;">
</div>

##### Extracted with a modified version of [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
##### For more details see our [demonstration videos](https://youtube.com/xcreate) or visit [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

## Disclaimer

We are not the creator, originator, or owner of any model listed. Each model is created and provided by third parties. Models may not always be accurate or contextually appropriate. You are responsible for verifying the information before making important decisions. We are not liable for any damages, losses, or issues arising from its use, including data loss or inaccuracies in AI-generated content.
