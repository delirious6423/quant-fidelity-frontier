---
base_model: 
  - google/gemma-4-26B-A4B-it
language:
  - en
model_creator: Google
model_name: google/gemma-4-26B-A4B-it
model_type: gemma4
quantized_by: s3dev-ai
tags:
  - text-generation
---

# Overview

This model repository provides various quantisations of the following [base model](https://huggingface.co/google/gemma-4-26B-A4B-it), in GGUF format.
- google/gemma-4-26B-A4B-it

# Model Description

For a full model description, please refer to the [base model's](https://huggingface.co/google/gemma-4-26B-A4B-it) card.

This model, and subsequent quantisations, have been converted directly from the author's base model *unaltered*.

## How are the GGUF files created?
After cloning the author's original base model repository, [`llama.cpp`](https://github.com/ggml-org/llama.cpp) is used to convert the model to GGUF format, using `--outtype=bf16` to preserve the original model's 16-bit fidelity. 

Finally, for each subsequent quantisation level, `llama.cpp`'s `llama-quantize` executable is called using the BF16 GGUF file as the source file.

# Quantisation
The purpose of this repository is to provide *unaltered*  quantisations of the author's base model. This section is designed to help the user visualise the difference in quantisation levels, in efforts to assist in model (quantisation) selection.

## Comparison Statistics
To aid a user in model/quantisation selection, the team has created the following statistics specifically for comparing the similarity scores across quantisation runs.

The dataset against which each run was conducted is composed of 175 question/answer pairs, divided amongst 7 topics, specifically designed to test a quantisation's processing ability. The test dataset was created by Mistral Large (via [Le Chat](https://chat.mistral.ai/chat)) using prompts explicitly stating the requirement for the question/answer pairs to be designed for Mistral model quantisation testing. 

The similarity scores used by these statistics were calculated as the cosine similarity between the embedding of the 'gold standard' answer provided in the dataset, and the embedding of the response from the quantised model. The embedding model used in these tests is the [all-MiniLM-L6-v2 Q8_0](https://huggingface.co/s3dev-ai/all-MiniLM-L6-v2-gguf).

### Range
The range graph below illustrates how the range of similarity scores varies amongst the quantisation levels. Included in the range stats are the:

- Minimum scores
- Maximum scores
- Mean scores
- Score distribution (KDE)
- Outliers 

<!-- Range image -->
<div align="center">
  <img src="imgs/gemma-4-26b-range.png" alt="Stats Graph: Range" width="90%">
</div>

### Mean
The mean graph below illustrates how the mean similarity scores (when grouped by 'topic') vary amongst the quantisation levels.

<!-- Mean image -->
<div align="center">
  <img src="imgs/gemma-4-26b-mean.png" alt="Stats Graph: Mean" width="90%">
</div>

### Standard Deviation
The standard deviation graph below illustrates the how spread of similarity scores vary amongst the quantisation levels, when grouped by the test dataset's 'topic' categories.

<!-- StdDev image -->
<div align="center">
  <img src="imgs/gemma-4-26b-stddev.png" alt="Stats Graph: StdDev" width="90%">
</div>

### Kernel Density Estimate
The KDE graph below illustrates the how distribution of similarity scores vary amongst the quantisation levels.

<!-- KDE image -->
<div align="center">
  <img src="imgs/gemma-4-26b-kde.png" alt="Stats Graph: KDE" width="90%">
</div>

