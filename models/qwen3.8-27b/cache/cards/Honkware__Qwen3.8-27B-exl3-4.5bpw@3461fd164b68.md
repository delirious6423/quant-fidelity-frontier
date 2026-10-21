---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: Honkware
library_name: exllamav3
pipeline_tag: text-generation
tags:
  - exl3
  - exllamav3
  - quantized

quantization_format: exl3
bits_per_weight: 4.5
---

<div align="center">

# Qwen3.8 · 27B

<sub><code>EXL3</code> &nbsp;·&nbsp; <b>4.5&nbsp;bpw</b> &nbsp;·&nbsp; 18.7&nbsp;GB &nbsp;·&nbsp; Dense</sub>

<br/>

[![format](https://img.shields.io/badge/format-EXL3-c63010?style=for-the-badge&labelColor=14120e)](https://github.com/turboderp-org/exllamav3)
[![bpw](https://img.shields.io/badge/bpw-4.5-6b8a76?style=for-the-badge&labelColor=14120e)](#quants)
[![size](https://img.shields.io/badge/size-18.7_GB-6b8a76?style=for-the-badge&labelColor=14120e)](#quants)
![codebook](https://img.shields.io/badge/codebook-mul1-6b8a76?style=for-the-badge&labelColor=14120e)
[![arch](https://img.shields.io/badge/arch-Qwen3_5_27B-c63010?style=for-the-badge&labelColor=14120e)](https://huggingface.co/Qwen/Qwen3.8-27B)

[![base model](https://img.shields.io/badge/Base-Qwen%2FQwen3.8--27B-2a2620?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![quantized by](https://img.shields.io/badge/Quantized_by-Honkware-2a2620?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/Honkware)
[![collection](https://img.shields.io/badge/All_bpws-Collection-c63010?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/collections/Honkware/qwen38-27b-exl3-6a7f89eeca48ae23da6520d1)

</div>

---

> [!NOTE]
> An [ExLlamaV3](https://github.com/turboderp-org/exllamav3) build of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at **4.5 bits per weight**. See [Quants](#quants) for sibling repos at other bit&#8209;widths or browse the [collection](https://huggingface.co/collections/Honkware/qwen38-27b-exl3-6a7f89eeca48ae23da6520d1).

## Quants

<div align="center">

| BPW &nbsp; | &nbsp; Head bits &nbsp; | &nbsp; Calibration rows &nbsp; | &nbsp; Size &nbsp; | &nbsp; Status |
| :---: | :---: | :---: | ---: | :--- |
| 3.8 | 8 | 250 | 16.6&nbsp;GB | [link](https://huggingface.co/Honkware/Qwen3.8-27B-exl3-3.8bpw) |
| **4.5** | 8 | 250 | **18.7&nbsp;GB** | <kbd>this repo</kbd> |

</div>

## Inference

<table>
  <thead>
    <tr>
      <th align="left" width="32%">Loader</th>
      <th align="left">Use it for</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://github.com/theroyallab/tabbyAPI"><b>TabbyAPI</b></a></td>
      <td>OpenAI&#8209;compatible HTTP server. Drop&#8209;in for OpenAI clients.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/oobabooga/text-generation-webui"><b>text&#8209;generation&#8209;webui</b></a></td>
      <td>Local chat UI. Pick the <i>ExLlamaV3</i> loader from the model dropdown.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/turboderp-org/exllamav3"><b>ExLlamaV3</b></a></td>
      <td>Direct Python API for embedding the model in your own code or pipeline.</td>
    </tr>
  </tbody>
</table>

## Download

```bash
pip install -U huggingface_hub

hf download \
  Honkware/Qwen3.8-27B-exl3-4.5bpw \
  --local-dir ./Qwen3.8-27B-exl3-4.5bpw
```

<details>
<summary><b>Quantization recipe</b> &nbsp;<sub>(advanced, embedded in <code>quantization_config.json</code>)</sub></summary>

<br/>

| Setting | Value |
| :--- | :--- |
| Format | `EXL3` |
| Bits per weight | `4.5` |
| Head bits | `8` |
| Calibration rows | `250` |
| Codebook | `mul1` |
| Out&#8209;scales | `always` |
| Parallel mode | `enabled` |

The codebook is recorded in the weights, so loaders pick it up with no configuration. `mul1` needs **ExLlamaV3 v0.0.3** or newer; an older build ignores the marker and decodes the weights with the wrong codebook.

Loaded automatically by every ExLlamaV3 loader; reproduced here for searchability.

</details>

## License &amp; use

> [!IMPORTANT]
> Use and license **follow the [base model](https://huggingface.co/Qwen/Qwen3.8-27B)**.
> Quantization adds no additional restrictions. Refer to the upstream repository for terms, citation, and safety documentation.

---

<div align="center">
  <sub><i>Quantized with <a href="https://github.com/Honkware/blockquant"><b>BlockQuant</b></a> &nbsp;·&nbsp; convention&nbsp;<code>{org}/{model}-exl3-{bpw}bpw</code></i></sub>
</div>
