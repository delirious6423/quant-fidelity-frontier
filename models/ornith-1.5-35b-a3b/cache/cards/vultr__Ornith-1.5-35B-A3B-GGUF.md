---
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
quantized_by: Vultr
license: mit
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - qwen35moe
---

# Ornith 1.5 35B A3B, GGUF

Ornith 1.5 35B A3B is the mid-size Mixture-of-Experts member of the Ornith-1.5 family, designed for coding and agentic workloads. Its end-to-end self-improvement approach jointly optimizes task generation, scaffold construction, and solution rollouts through reinforcement learning. The Qwen3.5-based architecture features hybrid linear and full attention.

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
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;"><a href="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B">ornith-ai/Ornith-1.5-35B-A3B</a></td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Architecture</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">qwen35moe, Mixture-of-Experts</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Active parameters</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">~3B activated per token</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Layers</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">40</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Hidden size</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">2,048</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Attention heads</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">16</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Key-value heads</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">2</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Experts</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">256 experts with 8 activated per token</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Context length</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">262,144 tokens</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Vocabulary</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">248,320</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">License</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;"><a href="https://choosealicense.com/licenses/mit/">MIT</a></td>
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
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;"><a href="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B">ornith-ai/Ornith-1.5-35B-A3B</a></td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Architecture</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">qwen35moe, Mixture-of-Experts</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Active parameters</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">~3B activated per token</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Layers</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">40</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Hidden size</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">2,048</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Attention heads</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">16</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Key-value heads</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">2</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Experts</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">256 experts with 8 activated per token</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Context length</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">262,144 tokens</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Vocabulary</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">248,320</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">License</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;"><a href="https://choosealicense.com/licenses/mit/">MIT</a></td>
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
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Ornith-1.5-35B-A3B-Q5_K_M.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">Q5_K_M</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">27.19 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.0349</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Ornith-1.5-35B-A3B-Q4_K_M.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">Q4_K_M</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">23.26 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.0595</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Ornith-1.5-35B-A3B-IQ3_M.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">IQ3_M</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">16.97 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.1399</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;border-left:1px solid #C9F4FF;">Ornith-1.5-35B-A3B-IQ3_S.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#EEFBFF;color:#031059;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">IQ3_S</th>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">16.77 GB</td>
<td style="padding:10px 12px;text-align:left;border-bottom:1px solid #C9F4FF;border-right:1px solid #C9F4FF;border-radius:0;">0.1504</td>
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
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Ornith-1.5-35B-A3B-Q5_K_M.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">Q5_K_M</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">27.19 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.0349</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Ornith-1.5-35B-A3B-Q4_K_M.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">Q4_K_M</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">23.26 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.0595</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Ornith-1.5-35B-A3B-IQ3_M.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">IQ3_M</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">16.97 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.1399</td>
</tr>
<tr>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;border-left:1px solid #1A3169;">Ornith-1.5-35B-A3B-IQ3_S.gguf</th>
<th scope="row" style="padding:10px 12px;text-align:left;font-weight:400;background:#172147;color:#C9F4FF;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">IQ3_S</th>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">16.77 GB</td>
<td style="padding:10px 12px;text-align:left;color:#B9C5DE;border-bottom:1px solid #1A3169;border-right:1px solid #1A3169;border-radius:0;">0.1504</td>
</tr>
</tbody>
</table>
</div>

The vision encoder is in `Ornith-1.5-35B-A3B-mmproj-f16.gguf` (899 MB). llama.cpp downloads it automatically with the command under Run, so image input works without extra steps.

KL divergence measures how far a quantized model's next-token probabilities drift from those of the original bf16 model, averaged over every token of the evaluation text. The lower the value, the closer the quantized model is to the original; zero would mean identical output.

## Evaluation

Every variant in this repository was evaluated against the original bf16 model for perplexity, KL divergence, top-token agreement, HellaSwag and WinoGrande, under identical conditions. The full results are maintained on the model's page in the Vultr Model Library:

**https://docs.vultr.com/models/deepreinforceai-ornith-15/ornith-1.5-35b-a3b?tab=quantized**

That page is the reference for these files. The summary above is an excerpt from it.

## Run

    llama-server -hf vultr/Ornith-1.5-35B-A3B-GGUF:Q4_K_M

The model's chat template is embedded in every file, so no extra prompt formatting is needed.

The repository also includes `mtp-Ornith-1.5-35B-A3B-Q8_0.gguf`, the model's multi-token prediction (MTP) head, for optional speculative decoding with any quant above:

    llama-server -hf vultr/Ornith-1.5-35B-A3B-GGUF:Q4_K_M --spec-type draft-mtp

## License

These files are derived from [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) and are distributed under the same license, [MIT](https://choosealicense.com/licenses/mit/). The model is the work of Ornith; Vultr produced the quantizations.
