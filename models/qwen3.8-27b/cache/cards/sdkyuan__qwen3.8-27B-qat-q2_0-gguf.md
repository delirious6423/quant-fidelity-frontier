---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - q2_0
  - qat
  - llama.cpp
  - reasoning
  - code
---

# Qwen3.8-27B-QAT-Q2_0

This is a 2-bit (Q2_0) **quantization-aware-trained (QAT)** checkpoint of Qwen3.8-27B for llama.cpp at 8.76 GB.

Quantized by Samuel Yuan \<sdkyuan@mit.edu\>.

> [!Note]
> This quantization was heavily focused on preserving code and reasoning performance. As such, multimodal and raw fact recall performance are slightly degraded.
>
> Interestingly, being able to preserve reasoning and code performance at the expense of Wikitext recall performance during quantization via QAT seems to be in line with the "cognitive core" hypothesis.
>
> A Q1_0 version and improved Q2_0 quant are in the works.

Unlike most community quants that use Post-Training Quantization (PTQ) (i.e. Unsloth), this is a quantization-aware-trained model.

Hopefully this quant makes Qwen3.8 27b-level coding and reasoning more accessible to those with <12-16 GB cards or Macs, since every little bit saved in model size means more room for KV cache and faster decode on memory bandwidth bound devices!

## Experiments

We measure **Top-1 agreement** (how often the next-token argmax matches the FP16 teacher's) and **mean KL divergence** (nats/token, lower is better) on held-out chat-formatted slices and on wikitext-2. 

<table style="border-collapse:separate;border-spacing:0;font-family:ui-monospace,'SF Mono',SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace;font-size:0.84em;border:2px solid #d9d2f5;border-radius:10px;overflow:hidden">
  <thead><tr>
    <th style="background:#f6f4fe;color:#4c3bb5;padding:10px 14px;text-align:left;vertical-align:bottom;font-size:0.78em;font-weight:600;letter-spacing:.02em;border:0;border-bottom:2px solid #d9d2f5;line-height:1.45">Model</th>
    <th style="background:#f6f4fe;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:bottom;font-size:0.78em;font-weight:600;letter-spacing:.02em;border:0;border-bottom:2px solid #d9d2f5;line-height:1.45">Size</th>
    <th style="background:#f6f4fe;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:bottom;font-size:0.78em;font-weight:600;letter-spacing:.02em;border:0;border-bottom:2px solid #d9d2f5;line-height:1.45">Reasoning<br><span style='font-weight:500;opacity:.75'>top-1 (&uarr;)</span><br><span style='font-weight:500;opacity:.75'>KL (&darr;)</span></th>
    <th style="background:#f6f4fe;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:bottom;font-size:0.78em;font-weight:600;letter-spacing:.02em;border:0;border-bottom:2px solid #d9d2f5;line-height:1.45">Code<br><span style='font-weight:500;opacity:.75'>top-1 (&uarr;)</span><br><span style='font-weight:500;opacity:.75'>KL (&darr;)</span></th>
    <th style="background:#f6f4fe;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:bottom;font-size:0.78em;font-weight:600;letter-spacing:.02em;border:0;border-bottom:2px solid #d9d2f5;line-height:1.45">Tool calling<br><span style='font-weight:500;opacity:.75'>top-1 (&uarr;)</span><br><span style='font-weight:500;opacity:.75'>KL (&darr;)</span></th>
    <th style="background:#f6f4fe;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:bottom;font-size:0.78em;font-weight:600;letter-spacing:.02em;border:0;border-bottom:2px solid #d9d2f5;line-height:1.45">Wikitext-2<br><span style='font-weight:500;opacity:.75'>top-1 (&uarr;)</span><br><span style='font-weight:500;opacity:.75'>KL (&darr;)</span></th>
  </tr></thead>
  <tbody>
  <tr>
    <td style="background:#ece7fd;color:#4c3bb5;padding:10px 14px;text-align:left;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45;font-weight:600">Qwen3.8-27B-QAT-Q2_0</td>
    <td style="background:#ece7fd;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45;font-weight:600">8.76 GB</td>
    <td style="background:#ece7fd;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45;font-weight:600">94.7%<br><span style='opacity:.72;font-size:0.92em'>0.029</span></td>
    <td style="background:#ece7fd;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45;font-weight:600">93.3%<br><span style='opacity:.72;font-size:0.92em'>0.064</span></td>
    <td style="background:#ece7fd;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45;font-weight:600">78.7%<br><span style='opacity:.72;font-size:0.92em'>0.638</span></td>
    <td style="background:#ece7fd;color:#4c3bb5;padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45;font-weight:600">78.3%<br><span style='opacity:.72;font-size:0.92em'>0.302</span></td>
  </tr>
  <tr>
    <td style="padding:10px 14px;text-align:left;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">Unsloth UD-Q2_K_XL</td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">9.83 GB</td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">94.4%<br><span style='opacity:.72;font-size:0.92em'>0.032</span></td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">93.1%<br><span style='opacity:.72;font-size:0.92em'>0.061</span></td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">82.5%<br><span style='opacity:.72;font-size:0.92em'>1.026</span></td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">87.5%<br><span style='opacity:.72;font-size:0.92em'>0.098</span></td>
  </tr>
  <tr>
    <td style="padding:10px 14px;text-align:left;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">Unsloth UD-IQ2_S</td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">8.37 GB</td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">92.9%<br><span style='opacity:.72;font-size:0.92em'>0.052</span></td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">91.5%<br><span style='opacity:.72;font-size:0.92em'>0.089</span></td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">78.9%<br><span style='opacity:.72;font-size:0.92em'>1.271</span></td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;border-bottom:1.5px solid #d9d2f5;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">84.5%<br><span style='opacity:.72;font-size:0.92em'>0.148</span></td>
  </tr>
  <tr>
    <td style="padding:10px 14px;text-align:left;vertical-align:middle;border:0;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">Unsloth UD-IQ3_XXS</td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">10.93 GB</td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">95.7%<br><span style='opacity:.72;font-size:0.92em'>0.018</span></td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">94.6%<br><span style='opacity:.72;font-size:0.92em'>0.038</span></td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">84.0%<br><span style='opacity:.72;font-size:0.92em'>0.929</span></td>
    <td style="padding:10px 14px;text-align:right;vertical-align:middle;border:0;font-variant-numeric:tabular-nums;white-space:nowrap;line-height:1.45">90.0%<br><span style='opacity:.72;font-size:0.92em'>0.064</span></td>
  </tr>
  </tbody>
</table>

Slices correspond to:
- *Reasoning*: `nvidia/Nemotron-Post-Training-Dataset-v1 (math)`, `open-thoughts/OpenThoughts3-1.2M`, `nvidia/OpenMathReasoning (cot)`
- *Code*: `nvidia/OpenCodeReasoning`, `open-r1/codeforces-cots`, `nvidia/Nemotron-Post-Training-Dataset-v1 (code)`, `bigcode/the-stack-smol`
- *Tool calling*: `Salesforce/xlam-function-calling-60k`, `togethercomputer/glaive-function-calling-v2-formatted`, `togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories`, `nvidia/Nemotron-Post-Training-Dataset-v1 (tool_calling)`
- *Wikitext-2*: `Salesforce/wikitext` (wikitext-2-raw-v1, test)

So this quantization exceeds the reasoning and code performance of SOTA PTQ quants over 1 GB larger, but does indeed suffer lower Wikitext accuracy. The comparatively lower tool calling performance may be due to our data blend, and we are looking to release a improved version with more focus on that as well for local agentic use cases.

## Methods

We applied the Efficient QAT (https://arxiv.org/abs/2407.11062) recipe with some improvements to the E2E-QP step (KD from teacher instead of CE on dataset) and data blend.

We have included a mmproj file for vision. Again, note that multimodal capabilities may be slightly degraded.


## Run it


```
./llama-cli \
  -m qwen38-27b-qat-q2_0.gguf \
  --jinja -cnv \
  --reasoning-budget -1 \
  -c 32768 -ngl 99 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --flash-attn on \
  --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 \
  --no-mmproj
```
or 
```
./llama-server \
  --model qwen38-27b-qat-q2_0.gguf \
  --jinja \
  --reasoning-budget -1 \
  --ctx-size 32768 \
  --host 127.0.0.1 \
  --port 8033 \
  -ngl 99 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --flash-attn on \
  --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 \
  --no-mmproj
```

Requires a llama.cpp build with Q2_0 support.


## License

apache-2.0 inherited from the base model. 

Feel free to reach out if you have any quesions about the model.
