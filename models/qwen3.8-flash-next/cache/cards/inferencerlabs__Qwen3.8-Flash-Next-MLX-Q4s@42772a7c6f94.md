---
language: en
library_name: mlx
tags:
- quantized
- mlx
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
---
# Qwen3.8-Flash-Next
See Qwen3.8-Flash-Next in action: [demonstration videos](https://youtube.com/xcreate)

For improved performance, download and enable a compatible [MTP speculative decoder](https://huggingface.co/models?search=inferencerlabs/qwen3.8-flash-next-mtp-mlx).

#### Tested with [Inferencer v2.4.6](https://inferencer.com)
- M3 Max 128GB: ~45 tokens/s @ 1000 tokens ~70.5 GiB
- M5 Max 128GB (with MTP): ~67.5 tokens/s @ 1000 tokens ~70.6 GiB

<p style="margin-bottom:0px;">
This build quantizes the majority of the model to 4 bits and important layers to 9 bits and above for good accuracy, while streaming n-grams from storage to reduce the memory footprint.
</p>

![Screenshot 1](https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/JK5jPxzNdBTvLXcNJ8wyi.jpeg)

![Screenshot 2](https://cdn-uploads.huggingface.co/production/uploads/6930102f5a301d8b708c299c/oepXbLQ375_EWD3a_zKE5.jpeg)

<table style="border-collapse: collapse; border: none; text-align:center; margin-top:10px; margin-bottom:0px;">
<thead>
<tr><th>Quantization (bpw)</th><th>Perplexity</th><th>Token Accuracy</th><th>Missed Divergence</th></tr>
</thead>
<tbody>
<tr><td><strong>Q3</strong></td><td>168.0</td><td>43.45%</td><td>72.57%</td></tr>
<tr><td><strong>Q4</strong></td><td>1.33593</td><td>91.65%</td><td>17.28%</td></tr>
<tr><td><strong>Q5</strong></td><td>1.23437</td><td>95.05%</td><td>17.28%</td></tr>
<tr><td><strong>Q6</strong></td><td>1.21875</td><td>96.65%</td><td>12.03%</td></tr>
<tr><td><strong>Q8</strong></td><td>1.21875</td><td>97.65%</td><td>9.92%</td></tr>
<tr><td><strong>Q9</strong></td><td>1.20312</td><td>97.80%</td><td>9.60%</td></tr>
<tr><td><strong>Base</strong></td><td>1.20312</td><td>100%</td><td>0.000%</td></tr>
</tbody>
</table>

<ul>
<li style="margin:0;">Perplexity: Measures the confidence for predicting base tokens (lower is better)</li>
<li style="margin:0;">Token Accuracy: The percentage of correctly generated base tokens</li>
<li style="margin:0;">Missed Divergence: Measures severity of misses; how much the token was missed by</li>
</ul>

##### Quantized with a modified version of [MLX](https://github.com/ml-explore/mlx)
##### For more details see [demonstration videos](https://youtube.com/xcreate) or visit [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## Disclaimer

We are not the creator, originator, or owner of any model listed. Each model is created and provided by third parties. Models may not always be accurate or contextually appropriate. You are responsible for verifying the information before making important decisions. We are not liable for any damages, losses, or issues arising from its use, including data loss or inaccuracies in AI-generated content.