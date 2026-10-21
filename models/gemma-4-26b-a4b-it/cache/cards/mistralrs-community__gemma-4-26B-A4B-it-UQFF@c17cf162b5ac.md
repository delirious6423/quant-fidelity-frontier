---
tags:
  - uqff
  - mistral.rs
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
---

# `google/gemma-4-26B-A4B-it`, UQFF quantization

Generated with [mistral.rs](https://github.com/EricLBuehler/mistral.rs) 0.9.0. Documentation: [UQFF docs](https://ericlbuehler.github.io/mistral.rs/guides/perf/use-uqff/).

1) **Flexible** 🌀: Multiple quantization formats in *one* file format with *one* framework to run them all.
2) **Reliable** 🔒: Compatibility ensured with *embedded* and *checked* semantic versioning information from day 1.
3) **Easy** 🤗: Download UQFF models *easily* and *quickly* from Hugging Face, or use a local file.
4) **Customizable** 🛠️: Make and publish your own UQFF files in minutes.

## Install

Install [mistral.rs](https://github.com/EricLBuehler/mistral.rs) ([full guide](https://ericlbuehler.github.io/mistral.rs/guides/install/)):

**Linux/macOS:**
```
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

**Windows (PowerShell):**
```
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

## Examples

**Note:** AFQ variants are optimized for Apple Silicon / Metal.

|Quantization|Command|
|--|--|
|AFQ2|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff afq2-0.uqff`|
|AFQ3|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff afq3-0.uqff`|
|AFQ4|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff afq4-0.uqff`|
|AFQ6|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff afq6-0.uqff`|
|AFQ8|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff afq8-0.uqff`|
|Q2K|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff q2k-0.uqff`|
|Q3K|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff q3k-0.uqff`|
|Q4K|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff q4k-0.uqff`|
|Q5K|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff q5k-0.uqff`|
|Q6K|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff q6k-0.uqff`|
|Q8_0|`mistralrs run -m mistralrs-community/gemma-4-26B-A4B-it-UQFF --from-uqff q8_0-0.uqff`|
