---
language: en
library_name: mlx
tags:
- quantized
- mlx
base_model:
- google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
---
# Gemma-4-26B-A4B
See gemma-4-26B-A4B in action: [demonstration video](https://youtu.be/S0oxWHGtg2w)

For improved performance, download and enable a compatible [MTP speculative decoder](https://huggingface.co/models?search=inferencerlabs/gemma-4-26B-A4B-mtp-mlx).

#### Tested on a M3 Ultra 512GB RAM using [Inferencer app](https://inferencer.com)
- Text inference: ~57.7 tokens/s @ 1000 tokens (debug build)
- Vision inference: ~51.7 tokens/s @ 1000 tokens
- Vision inference (with MTP): ~93.5 tokens/s (available from v1.11.5)
- Memory usage: ~26.7 GiB

<p style="margin-bottom:0px;">
<strong>9bpw quant typically achieves near lossless accuracy in our coding test</strong>
</p>
<table style="border-collapse: collapse; text-align:center; margin-top:10px; margin-bottom:0px;">
<thead>
<tr><th>Quantization (bpw)</th><th>Perplexity</th><th>Token Accuracy</th><th>Missed Divergence</th></tr>
</thead>
<tbody>
<tr><td><strong>q4.5</strong></td><td>1.32812</td><td>90.5%</td><td>26.44%</td></tr>
<tr><td><strong>q5.5</strong></td><td>1.23437</td><td>95.4%</td><td>16.03%</td></tr>
<tr><td><strong>q6.5</strong></td><td>1.21875</td><td>96.85%</td><td>12.55%</td></tr>
<tr><td><strong>q8.5</strong></td><td>1.21875</td><td>97.65%</td><td>9.92%</td></tr>
<tr><td><strong>q9</strong></td><td>1.21093</td><td>97.95%</td><td>9.61%</td></tr>
<tr><td><strong>Base</strong></td><td>1.20312</td><td>100.0%</td><td>0.000%</td></tr>
</tbody>
</table>

- Perplexity: Measures the confidence for predicting base tokens (lower is better)
- Token Accuracy: The percentage of correctly generated base tokens
- Missed Divergence: Measures severity of misses; how much the token was missed by

##### Quantized with a modified version of [MLX](https://github.com/ml-explore/mlx)
##### For more details see [demonstration video](https://youtu.be/S0oxWHGtg2w) or visit [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

## Disclaimer

We are not the creator, originator, or owner of any model listed. Each model is created and provided by third parties. Models may not always be accurate or contextually appropriate. You are responsible for verifying the information before making important decisions. We are not liable for any damages, losses, or issues arising from its use, including data loss or inaccuracies in AI-generated content.