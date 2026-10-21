---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: Vultr
license: mit
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - deepseek4
---

# DeepSeek V4 Flash 0731, GGUF

DeepSeek V4 Flash 0731 is a large-scale Mixture-of-Experts model optimized for ultra-long-context reasoning, efficient inference, and enhanced agentic capabilities. Its architecture combines CSA and HCA attention with manifold-constrained hyper-connections. It features FP4 expert weights, FP8 quantization, and a speculative decoding module for faster generation.

This repository holds Vultr's GGUF quantizations of it. They are hybrid quantizations: rather than applying one precision to the whole network, each layer's precision is chosen from a measured importance score, so the bits go where they change the output most.

## Model Overview

<div class="dark:hidden" style="overflow-x:auto;">
<table style="display:table;width:auto;max-width:100%;border-collapse:collapse;border:none;border-radius:0;">
<thead>
<tr>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;border-right:1px solid #FFFFFF;">Attribute</th>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;">Value</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Base model</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731">deepseek-ai/DeepSeek-V4-Flash-0731</a></td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Architecture</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">deepseek4, Mixture-of-Experts</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Parameters</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">284B total</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Active parameters</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">approximately 13B</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Layers</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">43</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Hidden size</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">4,096</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Attention heads</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">64</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Experts</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">256 routed, 6 selected per token</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Context length</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">1M tokens</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Vocabulary</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">129,280</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">License</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;"><a href="https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md">MIT</a></td>
</tr>
</tbody>
</table>
</div>
<div class="hidden dark:block" style="overflow-x:auto;">
<table style="display:table;width:auto;max-width:100%;border-collapse:collapse;border:none;border-radius:0;">
<thead>
<tr>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;border-right:1px solid #1A3169;">Attribute</th>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;">Value</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Base model</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731">deepseek-ai/DeepSeek-V4-Flash-0731</a></td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Architecture</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">deepseek4, Mixture-of-Experts</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Parameters</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">284B total</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Active parameters</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">approximately 13B</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Layers</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">43</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Hidden size</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">4,096</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Attention heads</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">64</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Experts</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">256 routed, 6 selected per token</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Context length</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">1M tokens</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Vocabulary</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">129,280</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">License</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;"><a href="https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md">MIT</a></td>
</tr>
</tbody>
</table>
</div>

## Quantizations

<div class="dark:hidden" style="overflow-x:auto;">
<table style="display:table;width:auto;max-width:100%;border-collapse:collapse;border:none;border-radius:0;">
<thead>
<tr>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;border-right:1px solid #FFFFFF;">File</th>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;border-right:1px solid #FFFFFF;">Type</th>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;border-right:1px solid #FFFFFF;">Size</th>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;">KLD vs original</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">DeepSeek-V4-Flash-0731-MXFP4-*.gguf (2 parts)</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">MXFP4</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">154.99 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.0132</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">DeepSeek-V4-Flash-0731-IQ3_M.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">IQ3_M</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">136.11 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.1423</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">DeepSeek-V4-Flash-0731-IQ3_S.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">IQ3_S</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">134.68 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.1445</td>
</tr>
</tbody>
</table>
</div>
<div class="hidden dark:block" style="overflow-x:auto;">
<table style="display:table;width:auto;max-width:100%;border-collapse:collapse;border:none;border-radius:0;">
<thead>
<tr>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;border-right:1px solid #1A3169;">File</th>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;border-right:1px solid #1A3169;">Type</th>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;border-right:1px solid #1A3169;">Size</th>
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;">KLD vs original</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">DeepSeek-V4-Flash-0731-MXFP4-*.gguf (2 parts)</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">MXFP4</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">154.99 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.0132</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">DeepSeek-V4-Flash-0731-IQ3_M.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">IQ3_M</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">136.11 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.1423</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">DeepSeek-V4-Flash-0731-IQ3_S.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">IQ3_S</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">134.68 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.1445</td>
</tr>
</tbody>
</table>
</div>

KL divergence measures how far a quantized model's next-token probabilities drift from those of the original model as released (its routed experts are already in 4-bit MXFP4), averaged over every token of the evaluation text. The lower the value, the closer the quantized model is to the original; zero would mean identical output.

## Evaluation

Every variant in this repository was evaluated against the original model as released (its routed experts are already in 4-bit MXFP4) for perplexity, KL divergence and top-token agreement, under identical conditions. The full results are maintained on the model's page in the Vultr Model Library:

**https://docs.vultr.com/models/deepseek-v4/deepseek-v4-flash-0731?tab=quantized**

That page is the reference for these files. The summary above is an excerpt from it.

HellaSwag and WinoGrande are not included: llama.cpp cannot yet score multiple-choice tasks on this architecture.

## Run

    llama-server -hf vultr/DeepSeek-V4-Flash-0731-GGUF:MXFP4

The model's chat template is embedded in every file, so no extra prompt formatting is needed.

The repository also includes `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`, the model's DSpark draft model, for optional speculative decoding with any quant above:

    llama-server -hf vultr/DeepSeek-V4-Flash-0731-GGUF:MXFP4 --spec-type draft-dspark

## License

These files are derived from [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) and are distributed under the same license, [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md). The model is the work of DeepSeek; Vultr produced the quantizations.
