---
base_model: google/gemma-4-26B-A4B-it
language:
- en
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
mradermacher:
  readme_rev: 1
quantized_by: mradermacher
---
## About

<!-- ### quantize_version: 2 -->
<!-- ### output_tensor_quantised: 1 -->
<!-- ### convert_type: hf -->
<!-- ### vocab_type:  -->
<!-- ### tags: nicoboss -->
<!-- ### quants:  Q2_K IQ3_M Q4_K_S IQ3_XXS Q3_K_M small-IQ4_NL Q4_K_M IQ2_M Q6_K IQ4_XS Q2_K_S IQ1_M Q3_K_S IQ2_XXS Q3_K_L IQ2_XS Q5_K_S IQ2_S IQ1_S Q5_K_M Q4_0 IQ3_XS Q4_1 IQ3_S -->
<!-- ### quants_skip:  -->
<!-- ### skip_mmproj:  -->
weighted/imatrix quants of https://huggingface.co/google/gemma-4-26B-A4B-it

<!-- provided-files -->

***For a convenient overview and download list, visit our [model page for this model](https://hf.tst.eu/model#gemma-4-26B-A4B-it-i1-GGUF).***

static quants are available at https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-GGUF

**This is a vision model - mmproj files (if any) will be in the [static repository](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-GGUF).**
## Usage

If you are unsure how to use GGUF files, refer to one of [TheBloke's
READMEs](https://huggingface.co/TheBloke/KafkaLM-70B-German-V0.1-GGUF) for
more details, including on how to concatenate multi-part files.

## Provided Quants

(sorted by size, not necessarily quality. IQ-quants are often preferable over similar sized non-IQ quants)

| Link | Type | Size/GB | Notes |
|:-----|:-----|--------:|:------|
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.imatrix.gguf) | imatrix | 0.2 | imatrix file (for creating your own quants) |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ1_S.gguf) | i1-IQ1_S | 8.4 | for the desperate |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ1_M.gguf) | i1-IQ1_M | 8.8 | mostly desperate |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ2_XXS.gguf) | i1-IQ2_XXS | 9.4 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ2_XS.gguf) | i1-IQ2_XS | 9.9 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ2_S.gguf) | i1-IQ2_S | 10.0 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ2_M.gguf) | i1-IQ2_M | 10.5 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q2_K.gguf) | i1-Q2_K | 10.7 | IQ3_XXS probably better |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q2_K_S.gguf) | i1-Q2_K_S | 10.7 | very low quality |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ3_XXS.gguf) | i1-IQ3_XXS | 11.4 | lower quality |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ3_XS.gguf) | i1-IQ3_XS | 11.7 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ3_S.gguf) | i1-IQ3_S | 12.3 | beats Q3_K* |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q3_K_S.gguf) | i1-Q3_K_S | 12.3 | IQ3_XS probably better |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ3_M.gguf) | i1-IQ3_M | 12.5 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q3_K_M.gguf) | i1-Q3_K_M | 13.4 | IQ3_S probably better |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q3_K_L.gguf) | i1-Q3_K_L | 13.9 | IQ3_M probably better |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-IQ4_XS.gguf) | i1-IQ4_XS | 14.0 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q4_0.gguf) | i1-Q4_0 | 14.6 | fast, low quality |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q4_K_S.gguf) | i1-Q4_K_S | 15.6 | optimal size/speed/quality |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q4_1.gguf) | i1-Q4_1 | 16.1 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q4_K_M.gguf) | i1-Q4_K_M | 16.9 | fast, recommended |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q5_K_S.gguf) | i1-Q5_K_S | 18.1 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q5_K_M.gguf) | i1-Q5_K_M | 19.2 |  |
| [GGUF](https://huggingface.co/mradermacher/gemma-4-26B-A4B-it-i1-GGUF/resolve/main/gemma-4-26B-A4B-it.i1-Q6_K.gguf) | i1-Q6_K | 22.7 | practically like static Q6_K |

Here is a handy graph by ikawrakow comparing some lower-quality quant
types (lower is better):

![image.png](https://www.nethype.de/huggingface_embed/quantpplgraph.png)

And here are Artefact2's thoughts on the matter:
https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9

## FAQ / Model Request

See https://huggingface.co/mradermacher/model_requests for some answers to
questions you might have and/or if you want some other model quantized.

## Thanks

I thank my company, [nethype GmbH](https://www.nethype.de/), for letting
me use its servers and providing upgrades to my workstation to enable
this work in my free time. Additional thanks to [@nicoboss](https://huggingface.co/nicoboss) for giving me access to his private supercomputer, enabling me to provide many more imatrix quants, at much higher quality, than I would otherwise be able to.

<!-- end -->
