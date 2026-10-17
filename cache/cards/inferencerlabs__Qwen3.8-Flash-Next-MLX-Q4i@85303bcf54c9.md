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

#### Tested on a M5 Max using [Inferencer app](https://inferencer.com)
<table style="border-collapse: collapse; border: none; text-align:left; margin-top:10px; margin-bottom:0px;">
  <tr><td><strong>Multimodal inference</strong></td><td>~57.5 tokens/s ~98.7 GiB</td></tr>
</table>

For larger context windows consider increasing the available RAM for inferencing with: ```sudo sysctl iogpu.wired_limit_mb=118000```

![Screenshot](https://cdn-uploads.huggingface.co/production/uploads/688479d616f1ec82fa645019/dR5SdcR-vfZxqCugcPt8W.jpeg)

<p style="margin-bottom:0px;">
<strong>Q4-INF uses the data-agnostic INF method tuned to yield maximum general accuracy within a 128 GiB memory budget.</strong>

Due to system memory and time constraints, the figures shown below are from the conversion of GLM 5.1.
</p>
<table style="border-collapse: collapse; border: none; text-align:center; margin-top:10px; margin-bottom:0px;">
<thead>
<tr><th>Quantization (bpw)</th><th>Perplexity</th><th>Token Accuracy</th><th>Missed Divergence</th></tr>
</thead>
<tbody>
<tr><td><strong>Q3</strong></td><td>168.0</td><td>43.45%</td><td>72.57%</td></tr>
<tr><td><strong>Q4</strong></td><td>1.33593</td><td>91.65%</td><td>17.28%</td></tr>
<tr><td><strong>Q4-INF</strong></td><td>1.21093</td><td>97.70%</td><td>10.65%</td></tr>
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