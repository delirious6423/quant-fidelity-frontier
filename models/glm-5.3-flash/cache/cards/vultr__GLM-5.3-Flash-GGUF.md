---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: Vultr
license: mit
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - gguf
  - llama.cpp
  - glm5-next
---

# GLM 5.3 Flash, GGUF

GLM 5.3 Flash is a natively multimodal model designed for efficient, high-performance agentic workloads, coding, and long-context applications. It is delivering better performance than GLM 5.2 while substantially reducing serving costs. The model uses a hybrid sparse architecture combining linear attention with DeepSeek-style sparse attention, alongside Manifold-Constrained Hyper-Connections (mHC) for improved scaling efficiency. It has multimodal capabilities for demanding agentic workloads.

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
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;"><a href="https://huggingface.co/zai-org/GLM-5.3-Flash">zai-org/GLM-5.3-Flash</a></td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Architecture</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">glm5-next, Mixture-of-Experts</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Parameters</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">320B total</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Active parameters</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">18B</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Layers</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">45</td>
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
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">288 routed, 8 activated per token</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Context length</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">up to 1M tokens</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Vocabulary</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">154,880</td>
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
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;"><a href="https://huggingface.co/zai-org/GLM-5.3-Flash">zai-org/GLM-5.3-Flash</a></td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Architecture</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">glm5-next, Mixture-of-Experts</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Parameters</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">320B total</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Active parameters</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">18B</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Layers</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">45</td>
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
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">288 routed, 8 activated per token</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Context length</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">up to 1M tokens</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Vocabulary</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">154,880</td>
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
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;">KLD vs bf16</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">GLM-5.3-Flash-Q5_K_M-*.gguf (2 parts)</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">Q5_K_M</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">244.70 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.0265</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">GLM-5.3-Flash-Q4_K_M-*.gguf (2 parts)</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">Q4_K_M</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">208.16 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.0468</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">GLM-5.3-Flash-IQ3_M-*.gguf (2 parts)</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">IQ3_M</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">150.74 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.1506</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">GLM-5.3-Flash-IQ3_S.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">IQ3_S</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">149.57 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.1611</td>
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
<th scope="col" style="padding:12px;text-align:left;background:#031059;color:#FFFFFF;border:none;">KLD vs bf16</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">GLM-5.3-Flash-Q5_K_M-*.gguf (2 parts)</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">Q5_K_M</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">244.70 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.0265</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">GLM-5.3-Flash-Q4_K_M-*.gguf (2 parts)</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">Q4_K_M</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">208.16 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.0468</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">GLM-5.3-Flash-IQ3_M-*.gguf (2 parts)</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">IQ3_M</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">150.74 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.1506</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">GLM-5.3-Flash-IQ3_S.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">IQ3_S</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">149.57 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.1611</td>
</tr>
</tbody>
</table>
</div>

The vision encoder is in `GLM-5.3-Flash-mmproj-f16.gguf` (1.13 GB). llama.cpp downloads it automatically with the command under Run, so image input works without extra steps.

KL divergence measures how far a quantized model's next-token probabilities drift from those of the original bf16 model, averaged over every token of the evaluation text. The lower the value, the closer the quantized model is to the original; zero would mean identical output.

## Evaluation

Every variant in this repository was evaluated against the original bf16 model for perplexity, KL divergence, top-token agreement, HellaSwag and WinoGrande, under identical conditions. The full results are maintained on the model's page in the Vultr Model Library:

**https://docs.vultr.com/models/zai-glm-53/glm-5.3-flash?tab=quantized**

That page is the reference for these files. The summary above is an excerpt from it.

## Run

    llama-server -hf vultr/GLM-5.3-Flash-GGUF:Q4_K_M

The model's chat template is embedded in every file, so no extra prompt formatting is needed.

The repository also includes `mtp-GLM-5.3-Flash-Q8_0.gguf`, the model's multi-token prediction (MTP) head, for optional speculative decoding with any quant above:

    llama-server -hf vultr/GLM-5.3-Flash-GGUF:Q4_K_M --spec-type draft-mtp

## License

These files are derived from [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) and are distributed under the same license, [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md). The model is the work of Z.ai; Vultr produced the quantizations.
